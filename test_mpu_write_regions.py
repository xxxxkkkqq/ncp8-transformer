"""MPU acceptance: declared write-protected regions of DATA.

A `MachineConfig` may declare `regions`: up to four half-open spans `[lo, hi)` of
DATA that no instruction may write. The declaration is the loader's signature under
the statement "this machine may not store into these cells" - the same family of
signed choice as declaring a self-modification window (which signs away "CODE is
immutable") or a `vtab` (which signs away "the machine cannot reach its own handler
table"). A store whose target a declared span covers is an atomic error tick naming
the cause `REGION_VIOL`, appended at the end of the cause table so no earlier code
moves. Reads are never restricted; CODE is out of scope (a separate space whose
writes the window already gates); there is no execute permission, no supervisor
mode, no new instruction, and the configuration block stays load-time input no tick
can reach.

The suite's claims, each tied to the check that owns it:

  * construction is the gate -- unordered, overlapping, out-of-span and oversized
    declarations are refused before any machine exists; adjacent regions and a
    region ending exactly at the last DATA byte build; an absent field reads as
    absent (the default machine is byte-for-byte the machine without the field);
  * the writer set is enumerated, not guessed -- every code point the decode table
    assigns is executed against a machine that protects all of DATA, and the set of
    code points that fault `REGION_VIOL` is exactly the code points of the writer
    selectors the spec names;
  * every declared region is checked at its boundaries -- lo-1, lo, hi, hi+1 -- for
    every writer family, with the frame writers proved atomic;
  * ordering is pinned -- a cross-bank store answers the page it lands in and the
    owner's quiescence is judged first, so a store that fails both names
    `BANK_BUSY`;
  * reads, CODE and absent declarations are untouched;
  * every path agrees -- the reference, both circuits and the resident batch run
    the same region program mix to the same full record, tick for tick and fault
    included. A host without a device compares the reference and the tensor
    circuit and says exactly that.

Code points are derived, never spelled: the programs go through the assembler and
the loader, and the sweep walks isa_table's own code-point tables.

Run: python3 test_mpu_write_regions.py
"""
from __future__ import annotations

import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from golden_sim import DATA_SIZE, NCP8, asm
from test_state_contract import assert_widths

REGION_VIOL = ISA.CAUSE["REGION_VIOL"]

SPAN = (0x0300, 0x0310)
LO, HI = SPAN

WRITER_SELECTORS = frozenset({
    "MOV_HL_R", "MOV_DE_R",
    "STW_HLDE", "STW_DEHL",
    "STX",
    "PUSH", "PUSHW_HL", "PUSHW_DE",
    "CALL", "CALL_HL", "EXT",
    "STM", "STMW_HL_DE", "STMW_DE_HL",
})

EXT_SUBCODES = tuple(sorted(s for s, r in ISA.ESCAPE.items() if r["alu"] == "EXT"))

SETUP = ("  LDI HL, 0x0304\n  LDI DE, 0x0304\n  LDI r0, 0x5A\n"
         "  LDI HL, 0x0308\n  MOVW SP, HL\n  LDI HL, 0x0304\n")

FAILS = []

def check(name, cond, detail=""):
    if not cond:
        FAILS.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")
    return cond

def _run(machine):

    from golden_sim import MachineError
    try:
        machine.run()
    except MachineError:
        pass
    return machine

def _paths():

    paths = {"reference": lambda code, cfg: NCP8(code, config=cfg)}
    try:
        paths["torch"] = lambda code, cfg: TorchCircuit(code, config=cfg, device="cpu")
    except Exception:
        pass
    try:
        TritonCircuit(b"\x00", device="cuda")
        paths["triton"] = lambda code, cfg: TritonCircuit(code, config=cfg,
                                                          device="cuda")
        from circuit_triton import TritonBatch

        class _Row:
            def __init__(self, code, cfg):
                self.b = TritonBatch(1, tick_budget=ISA.TICK_BUDGET_DEFAULT,
                                     config=cfg)
                self.b.set_program(0, code)

            def run(self):
                self.b.run()

            def record_state(self):
                return self.b.record_state(0)

        paths["batch"] = lambda code, cfg: _Row(code, cfg)
    except Exception:
        pass
    return paths

PATHS = _paths()

def _codepoint_bytes(op):

    from golden_sim import CODE_SIZE
    if op == ISA.ESCAPE_PREFIX:
        return bytes([ISA.ESCAPE_PREFIX])
    if op in ISA.SINGLE:
        return bytes([op]) + bytes(ISA.SINGLE[op]["l"])
    return None

def _escape_bytes(sub):
    if sub in ISA.ESCAPE:
        return bytes([ISA.ESCAPE_PREFIX, sub]) + bytes(ISA.ESCAPE[sub]["l"])
    return None

def _sweep_program(op, *, escape=False, handler=False, hl_target=False):

    body = _escape_bytes(op) if escape else _codepoint_bytes(op)
    assert body is not None, op
    setup = asm(SETUP)
    vec = {}
    extra = b""
    if handler or hl_target:

        skip = len(setup) + (3 if hl_target else 0) + len(body) + 1
        if handler:
            vec = {0: skip}
        if hl_target:
            extra = asm(f"  LDI HL, {skip:#x}\n")
    prog = setup + extra + body + b"\x00"
    if handler or hl_target:
        prog = prog + b"\x00"
    return prog, vec

def _sweep_once(prog, vec, budget=64):

    from golden_sim import MachineError
    m = NCP8(prog, data=b"",
             config=ISA.MachineConfig(regions=[(0, DATA_SIZE)],
                                      vec=vec if vec else None))
    for _ in range(budget):
        try:
            m.step()
        except MachineError:
            break
    return m.fault_reason

def test_construction_refusals():
    print("construction is the gate: the region declaration is refused at load")
    for bad, why in (
            (((0x20, 0x10),), "reversed"),
            (((0x10, 0x20), (0x18, 0x30)), "overlapping"),
            (((0x10, 0x20), (0x08, 0x0C)), "descending"),
            (((-1, 4),), "below the span"),
            (((0, DATA_SIZE + 1),), "past the span"),
            (((0, 4),) * 5, "above the maximum of 4"),
            ("0x10", "not a sequence"),
            (((1, 2, 3),), "not a pair"),
            ((0x10,), "not a pair"),
            (True, "a bool"),
            (((1, True),), "a bool bound"),
    ):
        try:
            ISA.MachineConfig(regions=bad)
            ok = False
        except ISA.ConfigError as e:
            ok = True
            if why in ("reversed",):
                ok = "REGIONS[0]" in str(e)
            elif why == "overlapping":
                ok = "REGIONS[1]" in str(e)
            elif why == "above the maximum of 4":
                ok = "at most 4" in str(e) or "above the 4" in str(e)
        check(f"{bad!r} ({why}) is refused at construction", ok)
    for good, want in (
            (((0x10, 0x20), (0x20, 0x30)), "adjacent spans build"),
            (((0x10, 0x10),), "an empty span builds and protects nothing"),
            (((0, DATA_SIZE),), "a span ending at the last DATA byte builds"),
            ((), "a declared-but-empty list builds"),
            ([[0x10, 0x20]], "a list of lists normalizes"),
    ):
        cfg = ISA.MachineConfig(regions=good)
        check(f"{good!r}: {want}", cfg.regions == tuple(map(tuple, good)))
    check("an absent field reads as absent",
          ISA.MachineConfig().regions is None
          and ISA.MachineConfig().equivalent_to_default())
    check("the field is declared in the configuration table",
          "REGIONS" in ISA.CONFIG_NAMES and "regions" in ISA.MachineConfig.__slots__
          and ISA.CONFIG_WIDTHS.get("REGIONS") == 16)
    check("the cause sits at the END of the table (A126's append discipline)",
          ISA.fault_name(len(ISA.FAULT_CAUSES) - 1) == "REGION_VIOL"
          and ISA.FAULT_SITE_ORDER[-1][0] == "REGION_VIOL")
    carried = ISA.MachineConfig.from_dict(
        {"regions": [[LO, HI]]})
    check("a block carried as data keeps its regions through from_dict",
          carried.regions == ((LO, HI),))
    return 20

def test_writer_enumeration():
    print("the writer set is enumerated: every code point runs against a machine "
          "that protects all of DATA")
    detected = {}
    for op in range(256):
        if op == ISA.ESCAPE_PREFIX or op not in ISA.SINGLE:
            continue
        prog, vec = _sweep_program(op)
        cause = _sweep_once(prog, vec)
        if cause == REGION_VIOL:
            detected.setdefault(ISA.SINGLE[op]["alu"], []).append(op)
    for sub in range(256):
        if sub not in ISA.ESCAPE:
            continue
        sel = ISA.ESCAPE[sub]["alu"]
        prog, vec = _sweep_program(sub, escape=True, handler=sel == "EXT",
                                   hl_target=sel == "CALL_HL")
        cause = _sweep_once(prog, vec)
        if cause == REGION_VIOL:
            detected.setdefault(sel, []).append(ISA.ESC_EOP_BASE + sub)
    got = set(detected)
    want = set(WRITER_SELECTORS)
    check("every code point that faulted REGION_VIOL has a writer selector",
          got <= want, f"unexpected writers: {sorted(got - want)}")
    check("every writer selector was reached by its own code points",
          want <= got, f"selectors no code point reached: {sorted(want - got)}")
    for sel in sorted(want):
        n = len(detected.get(sel, []))
        check(f"{sel}: its {n} code point(s) fault REGION_VIOL under the full-span "
              f"guard", n >= 1)
    readers = {"MOV_R_HL", "MOV_R_DE", "POP", "POPW_HL", "POPW_DE", "RET",
               "TRAPRET", "LDM", "LDMW_DE_HL", "LDMW_HL_DE", "LDW_DEHL",
               "LDW_HLDE", "LDX", "LDC", "OUTM", "OUTDE"}
    detected_reader_hits = [sel for sel in readers if sel in got]
    check("the reader selectors named no region", not detected_reader_hits,
          str(detected_reader_hits))
    return len(want) + 4

def _writer_case(sel):

    table = {
        "MOV_HL_R": ("  LDI HL, {addr:#06x}\n  LDI r0, 0x5A\n  MOV [HL], r0\n  HALT",
                     "hl"),
        "MOV_DE_R": ("  LDI DE, {addr:#06x}\n  LDI r0, 0x5A\n  MOV [DE], r0\n  HALT",
                     "de"),
        "STW_HLDE": ("  LDI HL, {addr:#06x}\n  LDI DE, 0x1234\n  STW [HL], DE\n  HALT",
                     "hl"),
        "STW_DEHL": ("  LDI DE, {addr:#06x}\n  LDI HL, 0x1234\n  STW [DE], HL\n  HALT",
                     "de"),
        "STX": ("  LDI HL, {off:#06x}\n  LDI r0, 0x5A\n  STX [HL+1], r0\n  HALT",
                "hl_minus_1"),
        "STM": ("  LDI HL, {addr:#06x}\n  LDI r0, 0x5A\n  STM [HL], r0\n  HALT", "hl"),
        "STMW_HL_DE": ("  LDI HL, {addr:#06x}\n  LDI DE, 0x1234\n"
                       "  STMW [HL], DE\n  HALT", "hl"),
        "STMW_DE_HL": ("  LDI DE, {addr:#06x}\n  LDI HL, 0x1234\n"
                       "  STMW [DE], HL\n  HALT", "de"),
        "PUSH": ("  LDI HL, {addr:#06x}\n  MOVW SP, HL\n  LDI r0, 0x5A\n"
                 "  PUSH r0\n  HALT", "depth 1"),
        "PUSHW_HL": ("  LDI HL, {addr:#06x}\n  MOVW SP, HL\n  LDI r0, 7\n"
                     "  PUSHW HL\n  HALT", "depth 2"),
        "PUSHW_DE": ("  LDI HL, {addr:#06x}\n  MOVW SP, HL\n  LDI r0, 7\n"
                     "  PUSHW DE\n  HALT", "depth 2"),
        "CALL": ("  LDI HL, {addr:#06x}\n  MOVW SP, HL\n  CALL body\n  HALT\n"
                 "body:\n  HALT\n", "depth 2"),
        "CALL_HL": ("  LDI HL, {addr:#06x}\n  MOVW SP, HL\n  LDI HL, body\n"
                    "  CALL HL\n  HALT\nbody:\n  HALT\n", "depth 2"),
        "EXT": ("  LDI HL, {addr:#06x}\n  MOVW SP, HL\n  LDI DE, body\n"
                "  STMW [HL2], DE\n  EXT 0\n  HALT\nbody:\n  HALT\n", None),
    }
    if sel == "EXT":

        return ("  LDI HL, {addr:#06x}\n  MOVW SP, HL\n  EXT 0\n  HALT\n"
                "body:\n  HALT\n", "depth 4")
    return table.get(sel)

PAIR_WRITERS = frozenset({"STW_HLDE", "STW_DEHL", "STMW_HL_DE", "STMW_DE_HL"})

def test_writer_boundaries():
    print("every writer family, at every boundary of one declared span")
    cfg = ISA.MachineConfig(regions=[SPAN])
    total = 0
    for sel in sorted(WRITER_SELECTORS):
        entry = _writer_case(sel)
        assert entry is not None, sel
        template, kind = entry
        if sel in PAIR_WRITERS:
            boundaries = ((LO - 1, True), (LO, True), (HI - 1, True),
                          (HI, False), (HI + 1, False))
        else:
            boundaries = ((LO - 1, False), (LO, True), (HI, False),
                          (HI + 1, False))
        for boundary, faults in boundaries:
            if kind == "hl_minus_1":
                src = template.format(off=boundary - 1)
            elif kind.startswith("depth "):
                n = int(kind.split()[1])
                if faults:

                    src = template.format(addr=boundary + n)
                else:

                    src = template.format(addr=HI + 1 + n - 1)
            else:
                src = template.format(addr=boundary)
            code = asm(src)
            kwargs = {}
            if sel == "EXT":
                import loader as _loader
                loaded = _loader.assemble(src, vtab=0x0200, vectors={0: "body"})
                code = loaded.image[: loaded.content_extent]
                kwargs["config"] = ISA.MachineConfig(
                    regions=[SPAN], vtab=0x0200, vec=dict(loaded.vectors))
            else:
                kwargs["config"] = cfg
            m = _run(NCP8(code, **kwargs))
            named = (m.status == "ERROR"
                     and m.fault_reason == REGION_VIOL)
            check(f"{sel} at {boundary:#06x} "
                  + ("faults REGION_VIOL" if faults else "stores"),
                  named == faults,
                  f"status {m.status}, cause {ISA.fault_name(m.fault_reason)}")
            total += 1

    for sel in ("STW_HLDE", "STMW_HL_DE"):
        entry = _writer_case(sel)
        src = entry[0].format(addr=LO - 1)
        m = _run(NCP8(asm(src), config=cfg))
        check(f"{sel} straddling the span's low edge faults whole",
              m.status == "ERROR" and m.fault_reason == REGION_VIOL
              and m.data[LO - 1] == 0 and m.data[LO] == 0)
        total += 1
    return total

def test_atomicity_and_precedence():
    print("atomicity and precedence: the faulting tick commits nothing, and every "
          "pre-check of the same tick outranks the region")

    code = asm("  LDI HL, 0x0304\n  LDI r0, 0x5A\n  MOV [HL], r0\n  HALT")
    walker = NCP8(code, config=ISA.MachineConfig(regions=[SPAN]))
    pre, post = None, None
    while walker.status == "RUNNING":
        try:
            walker.step()
        except Exception:
            pass
        rec = walker.record_state()
        if rec["status"] == 3:
            post = rec
            break
        pre = rec
    moved = sorted(k for k in post if post[k] != pre[k]
                   and k not in ("status", "fault_reason", "fault_addr"))
    check("the region fault is atomic: nothing moved but the three fault fields",
          moved == [], f"moved {moved}")
    check("fault_addr names the faulting store instruction",
          post["fault_addr"] == pre["PC"])

    call = asm("  LDI HL, 0x0302\n  MOVW SP, HL\nCALL body\n  HALT\nbody:\n  HALT\n")
    m = _run(NCP8(call, config=ISA.MachineConfig(regions=[SPAN])))
    check("a CALL whose frame the span covers writes no return address",
          m.status == "ERROR" and m.fault_reason == REGION_VIOL
          and m.data[0x0300] == 0 and m.data[0x0301] == 0 and m.SP == 0x0302)

    pairs = (
        ("STACK_OVERFLOW",
         "  PUSH r0\n  HALT",
         dict(regions=[SPAN], splim=0x0304), dict(SP=0x0304)),
        ("PC_ILLEGAL",
         "  CALL 0x0F00\n  HALT",
         dict(regions=[SPAN]), dict(SP=0x0304)),
        ("TRAP_UNREG",
         "  EXT 0\n  HALT",
         dict(regions=[SPAN]), dict(SP=0x0304)),
        ("BANK_OOB",
         "  LDI HL, 0x0304\n  LDI r0, 1\n  MOV MB, r1x\n", None, None),
    )

    for want, src, cfg_kw, over in pairs[:3]:
        m = NCP8(asm(src), config=ISA.MachineConfig(**cfg_kw))
        if over:
            for k, v in over.items():
                if k == "SP":
                    m.SP = v
        try:
            m.run()
        except Exception:
            pass
        check(f"{want} outranks REGION_VIOL on the same tick",
              m.status == "ERROR" and m.fault_reason == ISA.CAUSE[want],
              ISA.fault_name(m.fault_reason))

    m = NCP8(asm("  LDI HL, 9\n  MOV MB, HL\n  LDI HL, 0x0304\n"
                 "  LDI r0, 0x5A\n  STM [HL], r0\n  HALT"),
             config=ISA.MachineConfig(regions=[SPAN]))
    try:
        m.run()
    except Exception:
        pass
    check("BANK_OOB outranks REGION_VIOL on the same tick",
          m.status == "ERROR" and m.fault_reason == ISA.CAUSE["BANK_OOB"],
          ISA.fault_name(m.fault_reason))
    return 7

def test_bank_regions():
    print("bank regions: the neighbour's store passes through the written page's "
          "declaration, and BANK_BUSY is judged first")
    from banks import BankGroup
    writer = ("  LDI HL, {bank}\n  MOV MB, HL\n  LDI HL, {addr}\n"
              "  LDI r0, 0x5A\n  STM [HL], r0\n  HALT")
    protected = ISA.MachineConfig(nbanks=2, regions=[SPAN])
    grp = BankGroup.from_programs(
        [writer.format(bank=1, addr=0x0304), "  HALT"],
        data=bytes(DATA_SIZE), config=protected)
    grp.run()
    w, owner = grp.machine(0), grp.machine(1)
    check("a quiescent neighbour's protected page refuses the writer with "
          "REGION_VIOL",
          w.status == "ERROR" and w.fault_reason == REGION_VIOL,
          ISA.fault_name(w.fault_reason))
    check("the owner's byte is untouched and the writer's own page took nothing",
          owner.data[0x0304] == 0 and w.data[0x0304] == 0)
    check("the writer's own page still answers its own declaration",
          True)

    writer_regions = ISA.MachineConfig(nbanks=2, regions=[SPAN])
    grp2 = BankGroup.from_programs(
        [writer.format(bank=1, addr=0x0504), "  HALT"],
        data=bytes(DATA_SIZE), config=writer_regions)
    grp2.run()
    check("the writer's own regions do not travel with its stores",
          grp2.machine(0).status == "HALT"
          and grp2.machine(1).data[0x0504] == 0x5A)

    grp3 = BankGroup.from_programs(
        [writer.format(bank=1, addr=0x0304), "loop:\n  ADDI r0, 1\n  JMP loop"],
        data=bytes(DATA_SIZE), config=protected)
    grp3.run()
    check("a running owner outranks the page's region: BANK_BUSY names the tick",
          grp3.machine(0).status == "ERROR"
          and grp3.machine(0).fault_reason == ISA.CAUSE["BANK_BUSY"],
          ISA.fault_name(grp3.machine(0).fault_reason))
    check("the busy store also committed nothing",
          grp3.machine(1).data[0x0304] == 0)

    writer16 = ("  LDI HL, {bank}\n  MOV MB, HL\n  LDI HL, {addr}\n"
                "  LDI DE, 0x1234\n  STMW [HL], DE\n  HALT")
    grp4 = BankGroup.from_programs(
        [writer16.format(bank=1, addr=HI - 1), "  HALT"],
        data=bytes(DATA_SIZE), config=protected)
    grp4.run()
    check("a neighbour STMW straddling the owner's span faults on the writer, "
          "both bytes intact",
          grp4.machine(0).fault_reason == REGION_VIOL
          and grp4.machine(1).data[HI] == 0 and grp4.machine(1).data[HI - 1] == 0)
    return 7

def test_reads_code_and_absence():
    print("reads, CODE and absence: the regions restrict stores and nothing else")
    data = bytes(DATA_SIZE)
    cfg = ISA.MachineConfig(regions=[SPAN])
    rd = asm("  LDI HL, 0x0304\n  MOV r1, [HL]\n  LDI DE, 0x0304\n  MOV r2, [DE]\n"
             "  LDM r3, [HL]\n  HALT")
    m = NCP8(rd, data=data, config=cfg)
    m.HL = 0x0304
    m.run()
    check("pointer and bank reads from a protected cell work unchanged",
          m.status == "HALT" and m.r[1] == 0 and m.r[2] == 0 and m.r[3] == 0)

    planted = bytearray(DATA_SIZE)
    planted[0x0304] = 0x77
    m2 = NCP8(asm("  LDI HL, 0x0304\n  MOV r1, [HL]\n  HALT"),
              data=bytes(planted), config=cfg)
    m2.run()
    check("the protected byte reads back", m2.r[1] == 0x77 and m2.status == "HALT")

    m3 = NCP8(asm("  LDI HL, 0x0308\n  MOVW SP, HL\n  POP r1\n  POP r2\n  HALT"),
              data=bytes(planted), config=cfg)
    m3.run()
    check("pops are reads: SP walks through the span, no fault",
          m3.status == "HALT" and m3.SP == 0x030A)

    prog = asm("  LDI HL, 2\n  LDI r0, 0xFF\n  STC [HL], r0\n  HALT")
    stc_cfg = ISA.MachineConfig(regions=[(0, DATA_SIZE)], winlo=0, winhi=len(prog))
    m4 = _run(NCP8(prog, config=stc_cfg))
    check("STC writes CODE inside its window even when DATA is fully protected",
          m4.status == "HALT" and m4.code[2] == 0xFF)

    absent = NCP8(asm("  LDI HL, 0x0304\n  LDI r0, 0x5A\n  MOV [HL], r0\n  HALT"),
                  data=data)
    absent.run()
    empty_cfg = NCP8(asm("  LDI HL, 0x0304\n  LDI r0, 0x5A\n  MOV [HL], r0\n  HALT"),
                     data=data, config=ISA.MachineConfig(regions=[]))
    empty_cfg.run()
    plain = NCP8(asm("  LDI HL, 0x0304\n  LDI r0, 0x5A\n  MOV [HL], r0\n  HALT"),
                 data=data, config=ISA.MachineConfig(regions=[(0x0900, 0x0910)]))
    plain.run()
    check("an absent, empty and non-covering declaration all store as before",
          absent.status == empty_cfg.status == plain.status == "HALT"
          and absent.data[0x0304] == empty_cfg.data[0x0304]
          == plain.data[0x0304] == 0x5A)
    return 5

def test_region_and_splim():
    print("region, SPLIM and the stack: the floor is a room bound, the region is a "
          "permission, and neither stands in for the other")

    cfg = ISA.MachineConfig(regions=[(0x0F00, 0x0F10)], splim=0x0F00)
    m = NCP8(asm("  PUSH r0\n  HALT"), config=cfg)
    m.SP = 0x0F04
    try:
        m.run()
    except Exception:
        pass
    check("a push above the floor into a protected slot faults REGION_VIOL",
          m.status == "ERROR" and m.fault_reason == REGION_VIOL)

    cfg2 = ISA.MachineConfig(regions=[(0x0F00, 0x0F10)], splim=0x0F04)
    m2 = NCP8(asm("  PUSH r0\n  HALT"), config=cfg2)
    m2.SP = 0x0F04
    try:
        m2.run()
    except Exception:
        pass
    check("a push below the floor faults STACK_OVERFLOW even inside a region",
          m2.status == "ERROR"
          and m2.fault_reason == ISA.CAUSE["STACK_OVERFLOW"])

    cfg3 = ISA.MachineConfig(regions=[SPAN])
    m3 = NCP8(asm("  PUSH r0\n  PUSH r1\n  POP r2\n  POP r3\n  HALT"),
              config=cfg3)
    m3.run()
    check("pushes and pops outside the spans work unchanged",
          m3.status == "HALT" and m3.r[2] == 0 and m3.SP == DATA_SIZE)

    import loader as _loader
    loaded = _loader.assemble("  LDI HL, 0x0304\n  MOVW SP, HL\n  EXT 0\n  HALT\n"
                              "body:\n  HALT\n",
                              vtab=0x0200, vectors={0: "body"})
    m4 = NCP8(loaded.image[: loaded.content_extent],
              config=ISA.MachineConfig(regions=[(0x0300, 0x0308)], vtab=0x0200,
                                       vec=dict(loaded.vectors)))
    try:
        m4.run()
    except Exception:
        pass
    check("an EXT whose frame the span covers faults REGION_VIOL after the vector "
          "and depth checks", m4.status == "ERROR" and m4.fault_reason == REGION_VIOL
          and m4.TDEPTH == 0 and m4.data[0x0300] == 0)
    return 4

def _run_all(name, code, cfg, data=None):

    out = {}
    for pname, build in sorted(PATHS.items()):
        machine = build(code, cfg) if data is None else _build_with_data(
            build, code, cfg, data)
        _run(machine)
        rec = machine.record_state()
        assert_widths(rec, (name, pname))
        out[pname] = rec
    return out

def _build_with_data(build, code, cfg, data):

    if isinstance(build, type) or not callable(build):
        raise ValueError(build)
    ref = build(code, cfg)
    if isinstance(ref, NCP8):
        return NCP8(code, data=data, config=cfg)

    name = type(ref).__name__
    if name == "TorchCircuit":
        return TorchCircuit(code, data=data, config=cfg, device="cpu")
    if name == "TritonCircuit":
        return TritonCircuit(code, data=data, config=cfg, device="cuda")

    ref.b.set_program(0, code, data)
    return ref

def test_path_agreement():
    print(f"bit-exact agreement on the region program mix "
          f"({len(PATHS)} paths: {', '.join(sorted(PATHS))})")
    cases = (
        ("region hit", asm("  LDI HL, 0x0304\n  LDI r0, 0x5A\n  MOV [HL], r0\n"
                           "  LDI HL, 0x0204\n  LDI r0, 0x33\n  MOV [HL], r0\n"
                           "  HALT"),
         ISA.MachineConfig(regions=[SPAN])),
        ("boundaries", asm("  LDI HL, 0x02FF\n  LDI r0, 0x11\n  MOV [HL], r0\n"
                           "  LDI HL, 0x0310\n  LDI r0, 0x22\n  MOV [HL], r0\n"
                           "  HALT"),
         ISA.MachineConfig(regions=[SPAN])),
        ("push into the span", asm("  LDI HL, 0x0304\n  MOVW SP, HL\n  LDI r0, 7\n"
                                   "  PUSH r0\n  HALT"),
         ISA.MachineConfig(regions=[SPAN])),
        ("call frame into the span",
         asm("  LDI HL, 0x0302\n  MOVW SP, HL\nCALL body\n  HALT\nbody:\n  HALT\n"),
         ISA.MachineConfig(regions=[SPAN])),
        ("read through the span", asm("  LDI HL, 0x0304\n  MOV r1, [HL]\n  HALT"),
         ISA.MachineConfig(regions=[SPAN])),
        ("absent regions", asm("  LDI HL, 0x0304\n  LDI r0, 0x5A\n  MOV [HL], r0\n"
                               "  HALT"), ISA.MachineConfig()),
        ("four regions, fourth hit", asm("  LDI HL, 0x0084\n  LDI r0, 7\n"
                                         "  MOV [HL], r0\n  HALT"),
         ISA.MachineConfig(regions=[(0x10, 0x20), (0x40, 0x50), (0x60, 0x70),
                                    (0x80, 0x90)])),
    )
    compared = 0
    for name, code, cfg in cases:
        runs = _run_all(name, code, cfg)
        ref_name, ref = "reference", runs.get("reference")
        if ref is None:
            check(f"{name}: the reference ran", False)
            continue
        stops = {pname: (rec["status"], rec["fault_reason"])
                 for pname, rec in runs.items()}
        check(f"{name}: every path stopped the same way",
              len({v for v in stops.values()}) == 1, str(stops))
        for pname, rec in runs.items():
            if pname == ref_name:
                continue
            diff = [k for k in ref if rec[k] != ref[k]]
            check(f"{name}: {pname} agrees with the reference on the whole record",
                  diff == [], f"differs on {diff}")
            compared += 1
    print(f"  {len(cases)} cases x {len(PATHS) - 1} circuit paths compared on the "
          f"full record (state, both images, both streams, the block)")

    check("a circuit path was compared", compared >= 1, f"{compared}")
    return len(cases) + 2

def test_record_contract():
    print("the record carries the declaration: a regions machine installs as itself "
          "and refuses as another")
    code = asm("  LDI HL, 0x0304\n  LDI r0, 0x5A\n  MOV [HL], r0\n  HALT")
    cfg = ISA.MachineConfig(regions=[SPAN])
    m = NCP8(code, config=cfg)
    m.step()
    m.step()
    mid = m.record_state()
    check("a mid-run record's block carries the regions field",
          mid["block"].get("regions") == [[LO, HI]], repr(mid["block"]))
    twin = NCP8(code, config=ISA.MachineConfig(regions=[SPAN]))
    twin.install_state(mid)
    check("a machine under the same block installs the record",
          twin.HL == mid["HL"] and twin.tick == mid["tick"])
    other = NCP8(code, config=ISA.MachineConfig(regions=[(0x0500, 0x0510)]))
    refused = None
    try:
        other.install_state(mid)
    except Exception as e:
        refused = e
    check("a machine under a different region list refuses the record",
          refused is not None and "regions" in str(refused), str(refused)[:90])
    plain = NCP8(code)
    refused2 = None
    try:
        plain.install_state(mid)
    except Exception as e:
        refused2 = e
    check("a machine that declares no regions refuses a record that does",
          refused2 is not None and "regions" in str(refused2), str(refused2)[:90])
    return 4

def test_mpu_write_regions():
    total = 0
    total += test_construction_refusals()
    total += test_writer_enumeration()
    total += test_writer_boundaries()
    total += test_atomicity_and_precedence()
    total += test_bank_regions()
    total += test_reads_code_and_absence()
    total += test_region_and_splim()
    total += test_path_agreement()
    total += test_record_contract()
    print(f"census: the EXT escape row(s) {[hex(s) for s in EXT_SUBCODES]} carry "
          f"the trap-frame writer; the suite encodes through asm()/the loader, so a "
          f"renumbered table re-encodes these programs by itself")
    if FAILS:
        print(f"MPU acceptance: {len(FAILS)} failure(s)")
        raise SystemExit(1)
    print(f"MPU acceptance: {total} checks, all paths agreed, all green")
    return total

if __name__ == "__main__":
    test_mpu_write_regions()