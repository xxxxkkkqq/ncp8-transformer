"""VTAB acceptance: the declared writable vector page.

A `MachineConfig` may declare `vtab`: the DATA base of a 16-entry x 2-byte copy of the
trap vector table. Declaring it is the loader's signature under the statement "this
machine may rewrite its own handler table" - the same shape as declaring a
self-modification window signs away "CODE is immutable" - and it changes exactly one
thing: where `EXT k` reads trap `k`'s entry from. The load seeds the page once from the
block's `vec` (an absent table seeds all zeros, all unregistered); from the first tick
on the cells are ordinary DATA, so `ST`/`STMW` register and unregister handlers and
`EXT` dispatches from what the program last stored. No new instruction, no escape
subcode, no configuration block mapped into the address space: the machine still
cannot reach its configuration - only its own memory.

The suite's claims, each tied to the check that owns it:

  * construction is the gate -- an odd base and a table crossing the end of DATA
    are refused before any machine exists; a table ending exactly at the last DATA
    byte builds; an absent field reads as absent (the config-only machine is
    byte-for-byte the machine without the field);
  * seeding is a loader step, not a committed write -- STC_COUNT/STC_FIRST stay at
    reset, the seed lands little-endian (`DATA[vtab+2k]` low, `+1` high, the byte
    order STMW writes), it overrides a `data=` image that covers the cells, and an
    absent `vec` seeds sixteen unregistered entries;
  * one effective table per machine -- a page machine dispatches from DATA (runtime
    registration, unregister-to-0, k out of range, the depth limit and the frame
    tag all answered from the cells), a machine that declares no page keeps the
    block's table as the only one, and the seed is the only bridge between block
    and page;
  * every path agrees -- the reference, both circuits and the resident batch run
    the same program mix to the same full record (every component a record
    carries, block included), tick for tick and fault included. A host without a
    device compares the reference and the tensor circuit and says exactly that.

Code points are derived, never spelled: the programs go through `asm` and the
loader, which encode from isa_table, so a renumbered table re-encodes these
programs by itself.

Run: python3 test_vtab_runtime_vectors.py
"""
from __future__ import annotations

import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from golden_sim import NCP8, asm
from test_state_contract import assert_widths

VTAB = 0x0800

TRAP_UNREG = ISA.CAUSE["TRAP_UNREG"]
TRAP_DEPTH = ISA.CAUSE["TRAP_DEPTH"]
TRAP_FRAME = ISA.CAUSE["TRAP_FRAME"]

EXT_SUBCODES = tuple(sorted(s for s, r in ISA.ESCAPE.items() if r["alu"] == "EXT"))
STMW_SUBCODES = tuple(sorted(s for s, r in ISA.ESCAPE.items()
                             if r["alu"] == "STMW_HL_DE"))

PROGRAMS = {
    "register": ("  LDI HL, 0x0800\n  LDI DE, handler\n  STMW [HL], DE\n"
                 "  LDI r0, 7\n  EXT 0\n  OUT r0\n  HALT\n"
                 "handler:\n  LDI r0, 0x42\n  TRAPRET\n"),
    "unregister": ("  LDI HL, 0x0800\n  LDI DE, handler\n  STMW [HL], DE\n"
                   "  LDI DE, 0\n  STMW [HL], DE\n  EXT 0\n  OUT r0\n  HALT\n"
                   "handler:\n  LDI r0, 0x42\n  TRAPRET\n"),
    "seeded": ("  EXT 0\n  OUT r0\n  HALT\nhandler:\n  LDI r0, 0x71\n  TRAPRET\n"),
    "k_out_of_range": ("  LDI HL, 0x0800\n  LDI DE, handler\n  STMW [HL], DE\n"
                       "  EXT 16\n  OUT r0\n  HALT\n"
                       "handler:\n  LDI r0, 0x42\n  TRAPRET\n"),
    "reentry": ("  LDI HL, 0x0800\n  LDI DE, handler\n  STMW [HL], DE\n"
                "  EXT 0\n  HALT\n"
                "handler:\n  EXT 0\n  TRAPRET\n"),
    "read_back": ("  LDI HL, 0x0802\n  MOV r0, [HL]\n  OUT r0\n"
                  "  LDI HL, 0x0800\n  MOV r1, [HL]\n  OUT r1\n  HALT\n"),
}

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

def _seeded_load(name, **extra):

    import loader as _loader
    return _loader.assemble(PROGRAMS[name], vtab=VTAB, vectors={0: "handler"},
                            **extra)

def test_construction_refusals():
    print("construction is the gate: the declared page's shape is refused at load")
    for bad in (0x0801, VTAB + 1, 4095, 1):
        try:
            ISA.MachineConfig(vtab=bad)
            ok = False
        except ISA.ConfigError as e:
            ok = "odd" in str(e) and f"VTAB={bad}" in str(e)
        check(f"an odd base {bad} is refused naming the alignment rule", ok)
    for bad in (0x0FF0, 0x0FFE, 1 << 12, (1 << 16) - 30):
        try:
            ISA.MachineConfig(vtab=bad)
            ok = False
        except ISA.ConfigError as e:
            ok = "DATA" in str(e)
        check(f"a base {bad} whose table leaves DATA is refused", ok)
    edge = ISA.DATA_SIZE - ISA.VTAB_CELLS
    check(f"a table ending exactly at the last DATA byte (base {edge}) builds",
          ISA.MachineConfig(vtab=edge).vtab == edge)
    check("an absent field reads as absent",
          ISA.MachineConfig().vtab is None
          and ISA.MachineConfig().equivalent_to_default())
    check("the field is declared with the 16 bits a DATA address needs",
          ISA.CONFIG_WIDTHS.get("VTAB") == 16 and "VTAB" in ISA.CONFIG_NAMES)
    m = NCP8(asm("  HALT"), config=ISA.MachineConfig(vtab=VTAB))
    check("a machine built with a page and no vec boots with every trap "
          "unregistered",
          all(m._vector(k) == 0 for k in range(ISA.VEC_COUNT)))
    return 6

def test_seeding():
    print("seeding: a loader step into ordinary DATA, STC log untouched")
    code = asm(PROGRAMS["seeded"])
    cfg = ISA.MachineConfig(vtab=VTAB, vec={0: 5}, codelen=len(code))
    m = NCP8(code, config=cfg)
    want = ISA.vtab_seed_bytes(cfg.vectors())
    check("the seed lands little-endian at the declared base",
          bytes(m.data[VTAB: VTAB + ISA.VTAB_CELLS]) == want)
    check("the seed overrides a data= image that covers the cells",
          NCP8(code, data=b"\x99" * ISA.DATA_SIZE, config=cfg).data[VTAB] == 5)
    m2 = NCP8(code, config=ISA.MachineConfig(vtab=VTAB, vec={0: 5}, codelen=len(code)))
    check("seeding is a loader step: the self-modification log stays at reset",
          m2.stc_count == 0 and m2.stc_first == 0 and m2.status == "RUNNING")
    empty = NCP8(code, config=ISA.MachineConfig(vtab=VTAB, codelen=len(code)))
    check("an absent vec seeds sixteen zeros, all unregistered",
          bytes(empty.data[VTAB: VTAB + ISA.VTAB_CELLS]) == ISA.vtab_seed_bytes(
              ISA.DEFAULT_VECTORS)
          and all(empty._vector(k) == 0 for k in range(ISA.VEC_COUNT)))
    plain = NCP8(code, data=b"\x99" * ISA.DATA_SIZE)
    check("a machine that declares no page seeds nothing anywhere",
          plain.vtab is None and plain.data[VTAB] == 0x99
          and bytes(plain.data) == (b"\x99" * ISA.DATA_SIZE))

    code3 = asm(PROGRAMS["read_back"])
    m3 = NCP8(code3, config=ISA.MachineConfig(vtab=VTAB, vec={1: 0x0304, 0: 0x0102},
                                              codelen=len(code3)))
    m3.run()
    check("the cells are ordinary DATA a program reads (LD from VTAB+2, VTAB)",
          bytes(m3.out) == bytes([0x04, 0x02]))
    return 6

def test_runtime_registration():
    print("runtime registration: ST/STMW registers, 0 unregisters, EXT dispatches")
    code = asm(PROGRAMS["register"])
    cfg = ISA.MachineConfig(vtab=VTAB, codelen=len(code))
    m = _run(NCP8(code, config=cfg))
    check("a handler registered at runtime fires, returns through TRAPRET, and "
          "the caller's register comes back",
          bytes(m.out) == b"\x42" and m.status == "HALT" and m.TDEPTH == 0)
    store = asm("  LDI HL, 0x0800\n  LDI DE, 0x0102\n  STMW [HL], DE\n  HALT")
    mw = _run(NCP8(store, config=ISA.MachineConfig(vtab=VTAB, codelen=len(store))))
    check("the entry is the little-endian word STMW writes, read back through "
          "the machine's own dispatch",
          mw._vector(0) == 0x0102
          and bytes(mw.data[VTAB: VTAB + 2]) == b"\x02\x01")
    code2 = asm(PROGRAMS["unregister"])
    m2 = _run(NCP8(code2, config=ISA.MachineConfig(vtab=VTAB, codelen=len(code2))))
    check("an entry stored back as 0 unregisters the handler into TRAP_UNREG",
          m2.status == "ERROR" and m2.fault_reason == TRAP_UNREG
          and bytes(m2.out) == b"")
    code3 = asm(PROGRAMS["k_out_of_range"])
    m3 = _run(NCP8(code3, config=ISA.MachineConfig(vtab=VTAB, codelen=len(code3))))
    check("k >= VEC_COUNT faults TRAP_UNREG without consulting the page",
          m3.fault_reason == TRAP_UNREG and m3.status == "ERROR")
    stc = asm("  LDI HL, 0x0800\n  LDI r0, 0x11\n  STC [HL], r0\n  HALT")
    m4 = _run(NCP8(stc, config=ISA.MachineConfig(vtab=VTAB, codelen=len(stc))))
    check("the page is DATA, not CODE: STC's address bound is the program, so the "
          "table is writable only through the DATA stores",
          m4.status == "ERROR" and m4.fault_reason == ISA.CAUSE["CODE_OOB"])
    return 5

def test_depth_and_atomicity():
    print("depth and atomicity: TDLIM and the frame tag bind a runtime handler too")
    code = asm(PROGRAMS["reentry"])
    cfg = ISA.MachineConfig(vtab=VTAB, tdlim=2, codelen=len(code))
    m = _run(NCP8(code, config=cfg))
    check("a re-entering runtime handler stops at the declared depth",
          m.status == "ERROR" and m.fault_reason == TRAP_DEPTH and m.TDEPTH == 2)

    walker = NCP8(code, config=cfg)
    post, pre = None, None
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
    check("the depth fault is atomic: nothing moved but the three fault fields",
          moved == [], f"moved {moved}")
    check("fault_addr names the faulting EXT", post["fault_addr"] == pre["PC"])

    ret_code = asm("  LDI HL, 4032\n  MOVW SP, HL\n"
                   "  LDI HL, 0x0800\n  LDI DE, handler\n  STMW [HL], DE\n"
                   "  EXT 0\n  TRAPRET\n  HALT\nhandler:\n  RET\n")
    m2 = _run(NCP8(ret_code, config=ISA.MachineConfig(vtab=VTAB, codelen=len(ret_code))))
    check("the frame tag still binds: a handler left by RET faults the next "
          "TRAPRET", m2.status == "ERROR" and m2.fault_reason == TRAP_FRAME)
    return 4

def _run_all(name, code, cfg):

    out = {}
    for pname, build in sorted(PATHS.items()):
        machine = build(code, cfg)
        _run(machine)
        rec = machine.record_state()
        assert_widths(rec, (name, pname))
        out[pname] = rec
    return out

def test_path_agreement():
    print(f"bit-exact agreement on the VTAB program mix "
          f"({len(PATHS)} paths: {', '.join(sorted(PATHS))})")

    cases = []
    for name in ("register", "unregister", "k_out_of_range"):
        code = asm(PROGRAMS[name])
        cases.append((name, code, ISA.MachineConfig(vtab=VTAB)))
    seeded = _seeded_load("seeded")
    cases.append(("seeded", seeded.image[: seeded.content_extent],
                  ISA.MachineConfig(vtab=VTAB, vec=dict(seeded.vectors))))
    reentry = asm(PROGRAMS["reentry"])
    cases.append(("reentry", reentry,
                  ISA.MachineConfig(vtab=VTAB, tdlim=2)))
    read_back = asm(PROGRAMS["read_back"])
    cases.append(("read_back", read_back,
                  ISA.MachineConfig(vtab=VTAB, vec={1: 0x0304, 0: 0x0102})))
    config_only = _seeded_load("seeded")
    vtab_only = ISA.MachineConfig.from_dict(
        {k: v for k, v in config_only.config().as_dict().items()
         if k not in ("vtab", "codelen")})
    cases.append(("config_only, no page",
                  config_only.image[: config_only.content_extent], vtab_only))
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

def test_seed_is_the_only_bridge():
    print("one table per machine: the seed is the only bridge between block and page")
    seeded = _seeded_load("seeded")
    code = seeded.image[: seeded.content_extent]
    with_vec = ISA.MachineConfig(vtab=VTAB, vec=dict(seeded.vectors),
                                 codelen=len(code))
    without = ISA.MachineConfig(vtab=VTAB, codelen=len(code))
    a, b = NCP8(code, config=with_vec), NCP8(code, config=without)
    a.run()
    try:
        b.run()
    except Exception:
        pass
    check("the same program is a working trap under the seeded block and an "
          "unregistered one without the seed",
          a.status == "HALT" and b.status == "ERROR"
          and b.fault_reason == TRAP_UNREG)

    rewrote = NCP8(asm(PROGRAMS["register"]), config=ISA.MachineConfig(vtab=VTAB))
    rewrote.step()
    rewrote.step()
    rewrote.step()
    mid = rewrote.record_state()
    fresh = NCP8(asm(PROGRAMS["register"]),
                 config=ISA.MachineConfig(vtab=VTAB, codelen=len(asm(
                     PROGRAMS["register"]))))
    fresh.install_state(mid)
    check("a record installs the rewritten page: the fresh machine dispatches "
          "without storing again", fresh._vector(0) == mid["DATA"][VTAB]
          and fresh._vector(0) != 0)
    return 2

def test_vtab_runtime_vectors():
    total = 0
    total += test_construction_refusals()
    total += test_seeding()
    total += test_runtime_registration()
    total += test_depth_and_atomicity()
    total += test_path_agreement()
    total += test_seed_is_the_only_bridge()
    print(f"census: EXT escape row(s) {[hex(s) for s in EXT_SUBCODES]}, "
          f"STMW [HL], DE row(s) {[hex(s) for s in STMW_SUBCODES]}; the suite "
          f"encodes through asm()/the loader, so a renumbered table re-encodes "
          f"these programs by itself")
    if FAILS:
        print(f"VTAB acceptance: {len(FAILS)} failure(s)")
        raise SystemExit(1)
    print(f"VTAB acceptance: {total} checks, all paths agreed, all green")
    return total

if __name__ == "__main__":
    test_vtab_runtime_vectors()