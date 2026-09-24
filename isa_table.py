"""The NCP-8 decode table: one statement of what each code point means.

Every fact about an encoding lives here once - the mnemonic, the ALU selector, the two
operand selectors, the instruction length - for the single-byte opcodes and for the
`0x70` escape space. The reference simulator dispatches on the selector this table gives
it, the tensor implementation builds its decode ROMs from these rows, and the Triton
implementation checks its in-kernel decode against them for every code point before it
will load.

Self-test: `python3 isa_table.py` compares the table against the ROMs and dispatch of the
implementations as they are actually loaded, and reports the escape subcodes still free.
"""

DATA_SIZE = 4096
ESCAPE_PREFIX = 0x70
PREFIX_BYTES = 2

class DecodeTableError(Exception):

    pass

class UndefinedCode(DecodeTableError):

    pass

ALU_NAMES = (
    "NOP", "ADD", "SUB", "ADC", "SBB", "MOV", "TST", "SHL", "SHR", "DJNZ", "CLC",
    "LDI", "ADDI", "SUBI", "ADCI",
    "MOV_R_HL", "MOV_HL_R", "MOV_R_DE", "MOV_DE_R",
    "PUSH", "POP", "OUT", "IN",
    "INC_HL", "DEC_HL", "INC_DE",
    "OUTM", "OUTDE",
    "JMP", "JZ", "JNZ", "JC", "JNC",
    "CALL", "RET",
    "LDI_HL", "LDI_DE", "ADDI_HL", "ADDI_DE",
    "HALT", "BAD",
    "JPHL", "GETPC", "GETSP", "GETF",
    "AND", "OR", "XOR", "MUL",
    "DIV", "MOD", "CMP",
    "NOT", "NEG", "ROL", "ROR",
    "ADD_HLDE", "SUB_HLDE", "XCHG", "EXT",
    "STC", "LDC",
    "MOVW_HL_DE", "MOVW_DE_HL", "MOVW_HL_SP", "MOVW_DE_SP", "MOVW_SP_HL",
    "MOVW_SP_DE",
    "PUSHW_HL", "PUSHW_DE", "POPW_HL", "POPW_DE",
    "STW_HLDE", "STW_DEHL", "LDW_DEHL", "LDW_HLDE",
    "LDX", "STX", "ADD_SP", "MULH",
)
ALU_ID = {name: i for i, name in enumerate(ALU_NAMES)}
K = len(ALU_NAMES)
BAD = ALU_ID["BAD"]

ESC_EOP_BASE = 0x100
ESC_EOP_NAMES = (
    "DIV", "MOD", "CMP", "NOT", "NEG", "ROL", "ROR",
    "ADD_HLDE", "SUB_HLDE", "XCHG", "EXT", "STC", "LDC", "BAD",
    "MOVW_HL_DE", "MOVW_DE_HL", "MOVW_HL_SP", "MOVW_DE_SP", "MOVW_SP_HL",
    "MOVW_SP_DE", "PUSHW_HL", "PUSHW_DE", "POPW_HL", "POPW_DE",
    "STW_HLDE", "STW_DEHL", "LDW_DEHL", "LDW_HLDE", "LDX", "STX", "ADD_SP", "MULH",
)
ESC_EOP_ID = {name: ESC_EOP_BASE + i for i, name in enumerate(ESC_EOP_NAMES)}

SINGLE = {}
ESCAPE = {}

def _single(op, alu, mnem, s0=0, s1=0, l=0, kind=""):
    if op in SINGLE:
        raise DecodeTableError(f"single-byte code {op:#04x} is assigned twice")
    SINGLE[op] = dict(op=op, space="single", alu=alu, mnem=mnem, s0=s0, s1=s1,
                      l=l, kind=kind)

def _escape(sub, alu, mnem, s0=0, s1=0, l=0, kind=""):
    if sub in ESCAPE:
        raise DecodeTableError(f"escape subcode {sub:#04x} is assigned twice")
    ESCAPE[sub] = dict(op=sub, space="escape", alu=alu, mnem=mnem, s0=s0, s1=s1,
                       l=l, kind=kind)

for _i, (_alu, _mn) in enumerate([("HALT", "HALT"), ("NOP", "NOP"), ("INC_HL", "INC HL"),
                                  ("DEC_HL", "DEC HL"), ("INC_DE", "INC DE"),
                                  ("CLC", "CLC"), ("OUTM", "OUTM"),
                                  ("OUTDE", "OUTDE"), ("RET", "RET")]):
    _single(_i, _alu, _mn)
for _i, _alu in enumerate(["JMP", "JZ", "JNZ", "JC", "JNC", "CALL"]):
    _single(0x09 + _i, _alu, _alu, l=2, kind="a16")
_single(0x0F, "LDI_HL", "LDI HL", l=2, kind="i16")
_single(0x10, "LDI_DE", "LDI DE", l=2, kind="i16")
_single(0x11, "ADDI_HL", "ADDI HL", l=1, kind="roff")
_single(0x12, "ADDI_DE", "ADDI DE", l=1, kind="roff")
_single(0x13, "JPHL", "JPHL")
for _base, _alu in ((0x14, "GETPC"), (0x18, "GETSP"), (0x1C, "GETF")):
    for _k in range(4):
        _single(_base | _k, _alu, f"{_alu} r{_k}", _k, _k)
for _base, _alu in ((0x20, "AND"), (0x30, "OR"), (0x40, "XOR"), (0x50, "MUL"),
                    (0x80, "ADD"), (0x90, "SUB"), (0xA0, "ADC"), (0xB0, "SBB"),
                    (0xC0, "MOV")):
    for _f in range(16):
        _single(_base + _f, _alu, f"{_alu} r{(_f >> 2) & 3}, r{_f & 3}",
                (_f >> 2) & 3, _f & 3)
for _base, _alu in ((0x60, "SHL"), (0x64, "SHR"), (0x68, "TST")):
    for _k in range(4):
        _single(_base | _k, _alu, f"{_alu} r{_k}", _k, _k)
for _k in range(4):
    _single(0x6C | _k, "DJNZ", f"DJNZ r{_k}", _k, _k, l=2, kind="a16")
for _base, _alu in ((0xD0, "LDI"), (0xD4, "ADDI"), (0xD8, "SUBI"), (0xDC, "ADCI")):
    for _k in range(4):
        _single(_base | _k, _alu, f"{_alu} r{_k}", _k, _k, l=1, kind="i8")
for _base, _alu, _tpl in ((0xE0, "MOV_R_HL", "MOV r{k}, [HL]"),
                          (0xE4, "MOV_HL_R", "MOV [HL], r{k}"),
                          (0xE8, "MOV_R_DE", "MOV r{k}, [DE]"),
                          (0xEC, "MOV_DE_R", "MOV [DE], r{k}"),
                          (0xF0, "PUSH", "PUSH r{k}"), (0xF4, "POP", "POP r{k}"),
                          (0xF8, "OUT", "OUT r{k}"), (0xFC, "IN", "IN r{k}")):
    for _k in range(4):
        _single(_base | _k, _alu, _tpl.replace("{k}", str(_k)), _k, _k)

for _base, _alu in ((0x00, "DIV"), (0x10, "MOD"), (0x20, "CMP")):
    for _f in range(16):
        _escape(_base + _f, _alu, f"{_alu} r{(_f >> 2) & 3}, r{_f & 3}",
                (_f >> 2) & 3, _f & 3)
for _base, _alu in ((0x40, "NOT"), (0x44, "NEG"), (0x48, "ROL"), (0x4C, "ROR")):
    for _k in range(4):
        _escape(_base | _k, _alu, f"{_alu} r{_k}", _k, _k)
for _sub, (_alu, _mn) in ((0x30, ("MOVW_HL_DE", "MOVW HL, DE")),
                          (0x31, ("MOVW_DE_HL", "MOVW DE, HL")),
                          (0x32, ("MOVW_HL_SP", "MOVW HL, SP")),
                          (0x33, ("MOVW_DE_SP", "MOVW DE, SP")),
                          (0x34, ("MOVW_SP_HL", "MOVW SP, HL")),
                          (0x35, ("MOVW_SP_DE", "MOVW SP, DE")),
                          (0x38, ("PUSHW_HL", "PUSHW HL")),
                          (0x39, ("PUSHW_DE", "PUSHW DE")),
                          (0x3A, ("POPW_HL", "POPW HL")),
                          (0x3B, ("POPW_DE", "POPW DE")),
                          (0x3C, ("STW_HLDE", "STW [HL], DE")),
                          (0x3D, ("STW_DEHL", "STW [DE], HL")),
                          (0x3E, ("LDW_DEHL", "LDW DE, [HL]")),
                          (0x3F, ("LDW_HLDE", "LDW HL, [DE]")),
                          (0x60, ("ADD_HLDE", "ADD HL, DE")),
                          (0x61, ("SUB_HLDE", "SUB HL, DE")),
                          (0x62, ("XCHG", "XCHG HL, DE"))):
    _escape(_sub, _alu, _mn)
for _k in range(4):
    _escape(0x50 | _k, "LDX", f"LDX r{_k}, [HL{{off}}]", _k, _k, l=1, kind="off")
    _escape(0x54 | _k, "STX", f"STX [HL{{off}}], r{_k}", _k, _k, l=1, kind="off")
_escape(0x58, "ADD_SP", "ADD SP, {soff}", l=1, kind="off")
_escape(0x70, "EXT", "EXT {k}", l=1, kind="k")
for _k in range(4):
    _escape(0x80 | _k, "STC", f"STC [HL], r{_k}", _k, _k)
    _escape(0x84 | _k, "LDC", f"LDC r{_k}, [HL]", _k, _k)
for _f in range(16):
    _escape(0x90 + _f, "MULH", f"MULH r{(_f >> 2) & 3}, r{_f & 3}",
            (_f >> 2) & 3, _f & 3)

V4_RESERVED = tuple(range(0xB0, 0xBE))

def single_row(op):

    row = SINGLE.get(op)
    if row is None:
        raise UndefinedCode(f"single-byte code point {op:#04x} is not assigned")
    return row

def escape_row(sub):

    row = ESCAPE.get(sub)
    if row is None:
        raise UndefinedCode(f"escape subcode {sub:#04x} is not assigned")
    return row

def length(op):

    return 1 + single_row(op)["l"]

def escape_length(sub):

    return PREFIX_BYTES + escape_row(sub)["l"]

def operands(op):

    r = single_row(op)
    return r["s0"], r["s1"]

def escape_operands(sub):

    r = escape_row(sub)
    return r["s0"], r["s1"]

def mnemonic(op):

    return single_row(op)["mnem"]

def escape_mnemonic(sub):

    return escape_row(sub)["mnem"]

def operand_kind(op):

    return single_row(op)["kind"]

def escape_operand_kind(sub):

    return escape_row(sub)["kind"]

def instruction_length(image, at=0):

    if at >= len(image):
        return None
    op = image[at]
    if op == ESCAPE_PREFIX:
        if at + 1 >= len(image):
            return None
        sub = image[at + 1]
        return None if sub not in ESCAPE else PREFIX_BYTES + ESCAPE[sub]["l"]
    return None if op not in SINGLE else 1 + SINGLE[op]["l"]

def unassigned_single():

    return tuple(o for o in range(256) if o != ESCAPE_PREFIX and o not in SINGLE)

def unassigned_escape():

    return tuple(s for s in range(256) if s not in ESCAPE)

def single_rom():

    alu = [BAD] * 256
    s0 = [0] * 256
    s1 = [0] * 256
    ln = [1] * 256
    ln[ESCAPE_PREFIX] = PREFIX_BYTES
    for op, r in SINGLE.items():
        alu[op], s0[op], s1[op], ln[op] = ALU_ID[r["alu"]], r["s0"], r["s1"], 1 + r["l"]
    return alu, s0, s1, ln

def escape_rom():

    alu = [BAD] * 256
    s0 = [0] * 256
    s1 = [0] * 256
    lx = [0] * 256
    for sub, r in ESCAPE.items():
        alu[sub], s0[sub], s1[sub] = ALU_ID[r["alu"]], r["s0"], r["s1"]
        lx[sub] = r["l"]
    return alu, s0, s1, lx

def check_structure():

    for space, table in (("single", SINGLE), ("escape", ESCAPE)):
        for code, r in sorted(table.items()):
            if not 0 <= code <= 255:
                raise DecodeTableError(f"{space} code {code} is outside one byte")
            if not isinstance(r["mnem"], str) or not r["mnem"]:
                raise DecodeTableError(f"{space} code {code:#04x} has no mnemonic")
            if r["alu"] not in ALU_ID:
                raise DecodeTableError(f"{space} code {code:#04x} selects unknown ALU "
                                       f"{r['alu']!r}")
            if r["alu"] == "BAD":
                raise DecodeTableError(f"{space} code {code:#04x} is assigned BAD")
            for field in ("s0", "s1"):
                if not 0 <= r[field] <= 3:
                    raise DecodeTableError(f"{space} code {code:#04x} {field}="
                                           f"{r[field]} is outside 0..3")
            if not 0 <= r["l"] <= 2:
                raise DecodeTableError(f"{space} code {code:#04x} has l={r['l']}, "
                                       f"outside 0..2")
    if ESCAPE_PREFIX in SINGLE:
        raise DecodeTableError("the escape prefix is not an instruction of its own")
    esc_alu = {r["alu"] for r in ESCAPE.values()}
    missing = sorted(esc_alu - (set(ESC_EOP_NAMES) - {"BAD"}))
    if missing:
        raise DecodeTableError("escape rows select effective opcodes the Triton "
                               f"vocabulary does not name: {missing}")
    unused = sorted((set(ESC_EOP_NAMES) - {"BAD"}) - esc_alu)
    if unused:
        raise DecodeTableError(f"Triton escape vocabulary names no escape row: {unused}")
    clash = [s for s in V4_RESERVED if s in ESCAPE]
    if clash:
        raise DecodeTableError("v4 reserved subcodes are already assigned: "
                               f"{[hex(c) for c in clash]}")
    if len(V4_RESERVED) != len(set(V4_RESERVED)):
        raise DecodeTableError("v4 reserved subcodes are not distinct")
    if K != len(set(ALU_NAMES)):
        raise DecodeTableError("ALU vocabulary has duplicate names")
    if len(ESC_EOP_NAMES) != len(set(ESC_EOP_NAMES)):
        raise DecodeTableError("Triton escape vocabulary has duplicate names")
    if max(ESC_EOP_ID.values()) > 0x1FF:
        raise DecodeTableError("Triton escape effective opcodes exceed the 9-bit range")
    return True

check_structure()

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import circuit_torch as CT

    print(f"table: {len(SINGLE)} single-byte codes, {len(ESCAPE)} escape subcodes")
    miss = unassigned_single()
    print("  unassigned single-byte codes:", len(miss), [hex(o) for o in miss[:4]])

    live = tuple(t.cpu().tolist() for t in (CT._ALU, CT._S0, CT._S1, CT._LEN))
    live2 = tuple(t.cpu().tolist() for t in (CT._ALU2, CT._S02, CT._S12, CT._LX2))
    want = single_rom()
    want2 = escape_rom()

    bad = []
    for k, nm in enumerate(("alu", "s0", "s1", "ln")):
        if live[k] != list(want[k]):
            for op in range(256):
                if live[k][op] != want[k][op]:
                    bad.append(("single", nm, hex(op), live[k][op], want[k][op]))
    for k, nm in enumerate(("alu", "s0", "s1", "lx")):
        if live2[k] != list(want2[k]):
            for sub in range(256):
                if live2[k][sub] != want2[k][sub]:
                    bad.append(("escape", nm, hex(sub), live2[k][sub], want2[k][sub]))
    print(f"\ntable vs live tensor ROM (assignment, length, s0/s1): {len(bad)} disagreements")
    for b in bad[:12]:
        print("   ", b)

    free = len(unassigned_escape())
    planned = [s for s in V4_RESERVED if s in ESCAPE]
    print(f"\nv4.0 bank slots 0xB0-0xBD already claimed by the table: "
          f"{[hex(p) for p in planned] or 'none'}")
    print("free escape subcodes:", free)
    print("VERDICT:", "table reproduces the live decode" if not bad
          else "TABLE DIVERGES FROM LIVE DECODE")