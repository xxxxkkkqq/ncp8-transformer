"""Acceptance that the ISA is stated exactly once.

The three implementations previously transcribed the encoding independently, and every
divergence found so far came from a copy that was updated while its twins were not. These
checks treat isa_table.py as the definition and demand that the reference's dispatch, the
tensor decode ROMs, the Triton front end and the assembler all agree with it, code point by
code point, rather than agreeing with each other by coincidence.

Run: python3 test_spec_conformance.py
"""
from __future__ import annotations

import re
from pathlib import Path

import isa_table as ISA
import circuit_torch as CT
import circuit_triton as CTT
from golden_sim import MachineError, NCP8, asm

PC_SITES = (0, 256)
IMAGE = 0x0110
CFG = ISA.MachineConfig

def rows():

    out = [("single", op, ISA.SINGLE[op]) for op in sorted(ISA.SINGLE)]
    out += [("escape", sub, ISA.ESCAPE[sub]) for sub in sorted(ISA.ESCAPE)]
    return out

def row_length(space, row):
    head = 1 if space == "single" else ISA.PREFIX_BYTES
    return head + row["l"]

_OPERANDS = {
    "a16": lambda row, end: (end & 0xFF, end >> 8),
    "i16": lambda row, end: (0x00, 0x01),
    "i8": lambda row, end: (0x03,),
    "roff": lambda row, end: (0x02,),
    "off": lambda row, end: (0x00,),
    "k": lambda row, end: (0x00,),
    "": lambda row, end: (),
}

def case(space, code, row, pc):

    end = pc + row_length(space, row)
    head = (code,) if space == "single" else (ISA.ESCAPE_PREFIX, code)
    body = _OPERANDS[row["kind"]](row, end)
    img = bytearray(IMAGE)
    for i, v in enumerate(head + tuple(body)):
        img[pc + i] = v
    cfg = CFG(vec={k: end for k in range(ISA.VEC_COUNT)}, winlo=0x00, winhi=0x10)
    R = [1, 2, 3, 4]
    HL, DE, SP = 8, 8, 2048
    data = bytearray(CT.DATA_SIZE)
    if row["alu"] == "JPHL":
        HL = end
    if row["alu"] == "RET":
        data[SP] = end >> 8
        data[SP + 1] = end & 0xFF
    return bytes(img), R, HL, DE, SP, data, cfg

def step(space, code, row, pc):

    img, R, HL, DE, SP, data, cfg = case(space, code, row, pc)
    g = NCP8(img, data=data, config=cfg)
    g.load_state(R, HL, DE, SP, 0, 0, 0, PC=pc)
    g.step()
    return g

def _cols(rom):
    return [list(col) for col in rom]

def test_torch_module_roms_match_table():

    want_single = _cols(ISA.single_rom())
    got_single = [CT._ALU.cpu().tolist(), CT._S0.cpu().tolist(),
                  CT._S1.cpu().tolist(), CT._LEN.cpu().tolist()]
    for name, want, got in zip(("alu", "s0", "s1", "len"), want_single, got_single):
        assert got == want, ("circuit_torch", "single-byte ROM", name,
                             _first_diff(want, got, "opcode"))
    want_esc = _cols(ISA.escape_rom())
    got_esc = [CT._ALU2.cpu().tolist(), CT._S02.cpu().tolist(),
               CT._S12.cpu().tolist(), CT._LX2.cpu().tolist()]
    for name, want, got in zip(("alu", "s0", "s1", "extra"), want_esc, got_esc):
        assert got == want, ("circuit_torch", "escape ROM", name,
                             _first_diff(want, got, "subcode"))

def test_torch_live_tensors_match_table():

    alu, s0, s1, ln = ISA.single_rom()
    alu2, s02, s12, lx2 = ISA.escape_rom()
    oh = lambda col: [[1 if v == k else 0 for k in range(4)] for v in col]
    c = CT.TorchCircuit(bytes([0x00]))
    for name, want in (("alu_t", alu), ("s0_t", s0), ("ln_t", ln),
                       ("alu2_t", alu2), ("lx2_t", lx2),
                       ("OH_S0", oh(s0)), ("OH_S1", oh(s1)),
                       ("ohs0e", oh(s02)), ("ohs1e", oh(s12))):
        got = getattr(c, name).cpu().tolist()
        assert got == want, ("circuit_torch", "live machine", name,
                             _first_diff(want, got, "row"))
    assert c.KR.cpu().tolist() == list(range(ISA.K)), "circuit_torch selector space size"
    assert int(CT.BAD) == ISA.BAD and int(CT.K) == ISA.K

def _first_diff(want, got, axis):

    for i, (w, g) in enumerate(zip(want, got)):
        if w != g:
            return f"{axis} {i:#04x}: table {w}, implementation {g}"
    return "no difference"

def test_triton_decode_matches_table():

    alu1, s01, s11, ln1 = ISA.single_rom()
    for op in range(256):
        eop, d, s, ln = (int(v) for v in CTT._dec_first.fn(op))
        assert eop == op, ("circuit_triton", "single-byte", hex(op),
                           f"effective opcode {eop:#x}, table {op:#x}")
        for field, got, want in (("s0", d, s01[op]), ("s1", s, s11[op]),
                                 ("length", ln, ln1[op])):
            assert got == want, ("circuit_triton", "single-byte", hex(op),
                                 f"{field} {got}, table {want}")
    alu2, s02, s12, lx2 = ISA.escape_rom()
    for sub in range(256):
        eop, d, s, lx = (int(v) for v in CTT._dec_esc.fn(sub))
        row = ISA.ESCAPE.get(sub)
        want = ISA.ESC_EOP_ID["BAD" if row is None else row["alu"]]
        assert eop == want, ("circuit_triton", "escape", hex(sub),
                             f"effective opcode {eop:#x}, table {want:#x}")
        for field, got, exp in (("s0", d, s02[sub]), ("s1", s, s12[sub]),
                                ("extra bytes", lx, lx2[sub])):
            assert got == exp, ("circuit_triton", "escape", hex(sub),
                                f"{field} {got}, table {exp}")

def test_triton_escape_ids_match_table():

    for name, want in sorted(ISA.ESC_EOP_ID.items()):
        key = "ESC_" + name
        assert hasattr(CTT, key), f"circuit_triton does not define {key}"
        got = int(getattr(CTT, key))
        assert got == want, f"circuit_triton {key} = {got:#x}, table says {want:#x}"
    assert int(CTT.ESCAPE_PREFIX) == ISA.ESCAPE_PREFIX
    assert int(CTT.PREFIX_BYTES) == ISA.PREFIX_BYTES

def test_triton_import_guard_is_live():

    CTT._check_decode_against_table()
    assert issubclass(CTT.DecodeTableMismatch, Exception)

def test_every_assigned_code_point_has_a_mnemonic():
    for space, code, row in rows():
        assert isinstance(row["mnem"], str) and row["mnem"], (space, hex(code), row)
        assert row["alu"] in ISA.ALU_ID, (space, hex(code), row["alu"])
        assert 0 <= row["s0"] <= 3 and 0 <= row["s1"] <= 3, (space, hex(code))
        assert 0 <= row["l"] <= 2, (space, hex(code), row["l"])

def _must_raise(fn, code, exc):

    try:
        value = fn(code)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}({code!r}) returned {value!r}, "
                         f"expected {exc.__name__}")

def test_every_unassigned_code_point_raises():
    for op in ISA.unassigned_single():
        _must_raise(ISA.single_row, op, ISA.UndefinedCode)
        _must_raise(ISA.length, op, ISA.UndefinedCode)
    for sub in ISA.unassigned_escape():
        _must_raise(ISA.escape_row, sub, ISA.UndefinedCode)
        _must_raise(ISA.escape_length, sub, ISA.UndefinedCode)
    for space, code, row in rows():
        if space == "single":
            assert ISA.length(code) == 1 + row["l"]
            assert ISA.operands(code) == (row["s0"], row["s1"])
            assert ISA.mnemonic(code) == row["mnem"]
        else:
            assert ISA.escape_length(code) == ISA.PREFIX_BYTES + row["l"]
            assert ISA.escape_operands(code) == (row["s0"], row["s1"])
            assert ISA.escape_mnemonic(code) == row["mnem"]

def test_escape_subcode_space_is_accounted_for_exactly():

    assigned = set(ISA.ESCAPE)
    planned = set(ISA.V4_RESERVED)
    free = set(ISA.unassigned_escape())
    bank = {s for s in assigned if 0xB0 <= s <= 0xBD}
    assert len(bank) == 14, [hex(b) for b in sorted(bank)]
    assert {ISA.ESCAPE[s]["alu"] for s in bank} == {
        "LDM", "STM", "LDMW_DE_HL", "LDMW_HL_DE", "STMW_HL_DE", "STMW_DE_HL",
        "MOV_MB_HL", "MOV_HL_MB"}
    branch = {s for s in assigned if 0x64 <= s <= 0x67}
    assert [ISA.ESCAPE[s]["alu"] for s in sorted(branch)] == ["JS", "JNS", "VS", "VC"]
    assert len(planned) == 8, sorted(hex(p) for p in planned)
    assert planned <= free, [hex(p) for p in planned - free]
    assert assigned | free == set(range(256))
    assert not assigned & free
    assert len(assigned) + len(planned) + len(free - planned) == 256
    ISA.check_structure()

def test_single_byte_space_is_accounted_for_exactly():
    assigned = set(ISA.SINGLE)
    unassigned = set(ISA.unassigned_single())
    assert len(assigned) == 240, len(assigned)
    assert ISA.ESCAPE_PREFIX not in assigned and ISA.ESCAPE_PREFIX not in unassigned
    assert len(assigned | unassigned) == 255
    assert not assigned & unassigned
    assert sorted(unassigned) == list(range(0x71, 0x80)), sorted(hex(u) for u in unassigned)

def test_selector_spaces_do_not_collide():

    single = {r["alu"] for r in ISA.SINGLE.values()}
    esc = {r["alu"] for r in ISA.ESCAPE.values()}
    assert not (single & esc), sorted(single & esc)
    assert "BAD" not in single and "BAD" not in esc

def test_table_declares_every_selector_the_roms_use():
    alu, s0, s1, ln = ISA.single_rom()
    alu2, s02, s12, lx2 = ISA.escape_rom()
    names = {ISA.ALU_NAMES[i] for i in alu + alu2}
    assert names <= set(ISA.ALU_ID)
    assert max(alu + alu2) < ISA.K and min(alu + alu2) >= 0
    assert all(0 <= v <= 3 for v in s0 + s1 + s02 + s12)
    assert min(ln) >= 1 and 1 <= max(ln) <= 3 and 0 <= max(lx2) <= 2

def test_golden_instruction_length_matches_table():

    for space, code, row in rows():
        want = row_length(space, row)
        for pc in PC_SITES:
            g = step(space, code, row, pc)
            got = g.PC - pc
            assert got == want, (
                f"golden_sim length: {space} {code:#04x} ({row['mnem']}) at PC={pc:#04x} "
                f"advanced {got}, table says {want}")

def test_golden_register_field_matches_table():

    for space, code, row in rows():
        for pc in PC_SITES:
            img, R, HL, DE, SP, data, cfg = case(space, code, row, pc)
            g = NCP8(img, data=data, config=cfg)
            g.load_state(R, HL, DE, SP, 0, 0, 0, PC=pc)
            g.step()
            changed = [i for i in range(4) if g.r[i] != R[i]]
            assert set(changed) <= {row["s0"]}, (
                f"golden_sim operand field: {space} {code:#04x} ({row['mnem']}) at "
                f"PC={pc:#04x} wrote r{changed}, table selects destination r{row['s0']}")

def _refused(g, pc, code, label):

    msg = None
    try:
        g.step()
    except MachineError as e:
        msg = str(e)
    assert msg is not None, (
        f"golden_sim executed {label} {code:#04x} at PC={pc:#04x}; "
        f"the decode table does not assign it")
    assert f"{code:#04x}" in msg, (
        f"{label} {code:#04x} at PC={pc:#04x} raised for the wrong reason: "
        f"{ascii(msg)}")
    assert g.PC == pc and g.tick == 0 and g.r == [1, 2, 3, 4], (
        f"{label} {code:#04x} at PC={pc:#04x} did not leave the tick atomic: "
        f"PC={g.PC:#04x} tick={g.tick}")

def test_golden_refuses_every_unassigned_code_point():

    for op in ISA.unassigned_single():
        for pc in PC_SITES:
            img = bytearray(IMAGE)
            img[pc] = op
            g = NCP8(bytes(img))
            g.load_state([1, 2, 3, 4], 8, 8, 2048, 0, 0, 0, PC=pc)
            _refused(g, pc, op, "unassigned opcode")
    for sub in ISA.unassigned_escape():
        for pc in PC_SITES:
            img = bytearray(IMAGE)
            img[pc], img[pc + 1] = ISA.ESCAPE_PREFIX, sub
            g = NCP8(bytes(img))
            g.load_state([1, 2, 3, 4], 8, 8, 2048, 0, 0, 0, PC=pc)
            _refused(g, pc, sub, "reserved subcode")

def test_golden_dispatches_every_assigned_selector():

    handled = set()
    import inspect
    import re
    src = inspect.getsource(NCP8._step_inner)
    for names in re.findall(r"sel in \(([^)]*)\)", src):
        handled.update(re.findall(r'"(\w+)"', names))
    handled.update(re.findall(r'sel == "(\w+)"', src))
    assigned = {r["alu"] for r in ISA.SINGLE.values()} | {r["alu"] for r in ISA.ESCAPE.values()}
    missing = sorted(assigned - handled)
    assert not missing, f"golden_sim has no branch for selectors {missing}"

def test_assembler_width_matches_table():

    for alu, src in sorted(_ENCODE.items()):
        img = asm(src)
        row = _BY_ALU[alu]
        want = row_length(row["space"], row)
        assert len(img) == want, (
            f"assembler emitted {len(img)} bytes for {src!r} ({alu}), "
            f"the decode table reads {want}")
        assert ISA.instruction_length(img) == want, (
            f"{src!r} does not decode to {alu}")

_ENCODE = {
    "JMP": "JMP 0x0100", "JZ": "JZ 0x0100", "JNZ": "JNZ 0x0100", "JC": "JC 0x0100",
    "JNC": "JNC 0x0100", "CALL": "CALL 0x0100", "DJNZ": "DJNZ r0, 0x0100",
    "LDI_HL": "LDI HL, 0x0100", "LDI_DE": "LDI DE, 0x0100",
    "ADDI_HL": "ADDI HL, r2", "ADDI_DE": "ADDI DE, r2",
    "LDI": "LDI r1, 0x02", "ADDI": "ADDI r1, 0x02", "SUBI": "SUBI r1, 0x02",
    "ADCI": "ADCI r1, 0x02",
    "DIV": "DIV r1, r2", "MOD": "MOD r1, r2", "CMP": "CMP r1, r2", "MULH": "MULH r1, r2",
    "NOT": "NOT r1", "NEG": "NEG r1", "ROL": "ROL r1", "ROR": "ROR r1",
    "LDX": "LDX r1, [HL+5]", "STX": "STX [HL-5], r1", "ADD_SP": "ADD SP, -4",
    "EXT": "EXT 3", "STC": "STC [HL], r1", "LDC": "LDC r1, [HL]",
    "MOVW_HL_DE": "MOVW HL, DE", "MOVW_DE_HL": "MOVW DE, HL",
    "MOVW_HL_SP": "MOVW HL, SP", "MOVW_DE_SP": "MOVW DE, SP",
    "MOVW_SP_HL": "MOVW SP, HL", "MOVW_SP_DE": "MOVW SP, DE",
    "PUSHW_HL": "PUSHW HL", "PUSHW_DE": "PUSHW DE", "POPW_HL": "POPW HL",
    "POPW_DE": "POPW DE", "STW_HLDE": "STW [HL], DE", "STW_DEHL": "STW [DE], HL",
    "LDW_DEHL": "LDW DE, [HL]", "LDW_HLDE": "LDW HL, [DE]",
    "ADD_HLDE": "ADD HL, DE", "SUB_HLDE": "SUB HL, DE",
    "XCHG": "XCHG HL, DE",
    "HALT": "HALT", "NOP": "NOP", "INC_HL": "INC HL", "DEC_HL": "DEC HL",
    "INC_DE": "INC DE", "CLC": "CLC", "OUTM": "OUTM", "OUTDE": "OUTDE", "RET": "RET",
    "JPHL": "JPHL", "GETPC": "GETPC r2", "GETSP": "GETSP r2", "GETF": "GETF r2",
    "AND": "AND r1, r2", "OR": "OR r1, r2", "XOR": "XOR r1, r2", "MUL": "MUL r1, r2",
    "ADD": "ADD r1, r2", "SUB": "SUB r1, r2", "ADC": "ADC r1, r2", "SBB": "SBB r1, r2",
    "MOV": "MOV r1, r2", "SHL": "SHL r2", "SHR": "SHR r2", "TST": "TST r2",
    "MOV_R_HL": "MOV r2, [HL]", "MOV_HL_R": "MOV [HL], r2",
    "MOV_R_DE": "MOV r2, [DE]", "MOV_DE_R": "MOV [DE], r2",
    "PUSH": "PUSH r2", "POP": "POP r2", "OUT": "OUT r2", "IN": "IN r2",
}

_BY_ALU = {}
for _space, _code, _row in rows():
    _BY_ALU.setdefault(_row["alu"], dict(_row, space=_space))

def implementation_root():

    return Path(__file__).resolve().parent

def implementation_sources():

    return sorted(p for p in implementation_root().rglob("*.py")
                  if not p.name.startswith("test_"))

def tick_budget_spellings():

    value = int(ISA.TICK_BUDGET_DEFAULT)
    return sorted({str(value), "{:,}".format(value).replace(",", "_")},
                  key=len, reverse=True)

def tick_budget_restatements():

    pattern = re.compile(r"(?<![\d_])(" + "|".join(tick_budget_spellings())
                         + r")(?![\d_])")
    root = implementation_root()
    out = []
    for path in implementation_sources():
        if path.name == "isa_table.py":
            continue
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                out.append(f"{path.relative_to(root)}:{no}: {line.strip()}")
    return out

def test_tick_budget_is_written_as_a_number_only_in_the_table():

    found = tick_budget_restatements()
    scanned = implementation_sources()
    assert scanned, "the scan found no implementation file to read"
    assert any(p.name == "isa_table.py" for p in scanned), \
        "the scan does not reach the file that defines the constant"
    assert not found, (
        f"isa_table.TICK_BUDGET_DEFAULT states the default tick budget once; these "
        f"files write the same number as a literal: {found}")

CHECKS = (
    test_torch_module_roms_match_table,
    test_torch_live_tensors_match_table,
    test_triton_decode_matches_table,
    test_triton_escape_ids_match_table,
    test_triton_import_guard_is_live,
    test_every_assigned_code_point_has_a_mnemonic,
    test_every_unassigned_code_point_raises,
    test_escape_subcode_space_is_accounted_for_exactly,
    test_single_byte_space_is_accounted_for_exactly,
    test_selector_spaces_do_not_collide,
    test_table_declares_every_selector_the_roms_use,
    test_golden_instruction_length_matches_table,
    test_golden_register_field_matches_table,
    test_golden_refuses_every_unassigned_code_point,
    test_golden_dispatches_every_assigned_selector,
    test_assembler_width_matches_table,
    test_tick_budget_is_written_as_a_number_only_in_the_table,
)

def run_all():
    for fn in CHECKS:
        fn()
        print(f"  {fn.__name__} ok")
    print(f"spec conformance: all {len(CHECKS)} checks passed")

if __name__ == "__main__":
    print("decode-table spec conformance:")
    run_all()