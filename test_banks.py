"""Bank acceptance: what a group of machines may do to each other's DATA, measured.

One machine reaches another machine's DATA page as a bank, and only while that page's owner is
quiescent; nothing else about a neighbour is reachable at all. Each rule is checked by running
machines and comparing whole state records, so a check cannot pass by looking only at the field
a bug left alone: the refusing tick commits nothing anywhere, the refused tick does not advance,
the owner executes identically whether or not the write happened, and the reachability claim is
walked structurally over the objects a machine holds. The selector is measured as a row of the
state table on every path, and the pages a path can name are measured rather than asserted.

Run: python3 test_banks.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import disasm
import isa_table as ISA
from banks import (TICK_FIELDS, BankGroup, GROUP_MAX, GROUP_MIN, tick_once,
                   with_nbanks)
from circuit_torch import TorchCircuit
from circuit_triton import TritonBatch, TritonCircuit
from golden_sim import (DATA_SIZE, NCP8, MachineError, STATUS_CODE, STATUS_ERROR,
                        STATUS_HALT, STATUS_RUNNING, asm)

CAUSE = ISA.CAUSE
DATA = bytes(range(1, 65)) * 2
FAILS = []

FAMILY = [("  LDM r0, [HL]", 0xB0), ("  LDM r3, [HL]", 0xB3), ("  STM [HL], r0", 0xB4),
          ("  STM [HL], r3", 0xB7), ("  LDMW DE, [HL]", 0xB8), ("  LDMW HL, [DE]", 0xB9),
          ("  STMW [HL], DE", 0xBA), ("  STMW [DE], HL", 0xBB), ("  MOV MB, HL", 0xBC),
          ("  MOV HL, MB", 0xBD)]

LONE = ["  LDI r0, 7\n  ADD r0, r0\n  OUT r0\n  HALT",
        "  LDI HL, 8\n  LDI r0, 3\n  MOV [HL], r0\n  MOV r1, [HL]\n  OUT r1\n  HALT",
        "  LDI r0, 1\nloop:\n  PUSH r0\n  DJNZ r0, loop\n  HALT",
        "  LDI HL, 0x2000\n  MOV r0, [HL]\n  HALT"]

SELF_WRITE = ("  LDI HL, {bank}\n  MOV MB, HL\n  LDI HL, 40\n  LDI r0, {val}\n"
              "  STM [HL], r0\n  LDI r1, 0\n  LDM r2, [HL]\n  OUT r2\n  HALT")
TO_NEIGHBOUR = ("  LDI HL, {target}\n  MOV MB, HL\n  LDI HL, 64\n  LDI r0, 0x77\n"
                "  STM [HL], r0\n  HALT")
LOOP = "loop:\n  ADDI r0, 1\n  JMP loop"
SLEEP = "  HALT"
HALT_AFTER = "  ADDI r0, 1\n  ADDI r0, 1\n  HALT"

def check(name, condition, detail=""):
    if not condition:
        FAILS.append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + ("" if condition else f"  {detail}"))

def diffs(a, b):
    return sorted(k for k in a if k not in b or a[k] != b[k])

def refuse(fn, *args, **kw):

    try:
        fn(*args, **kw)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None

def step_until(group, machine, tick, limit=16):

    for _ in range(limit):
        if machine.tick >= tick or machine.status != STATUS_RUNNING:
            return
        group.step(strict=False)

def padded_page():

    return bytes(NCP8(asm("  HALT"), data=DATA).data)

def group_of_four():
    codes = [asm(SELF_WRITE.format(bank=i, val=0x11 * (i + 1))) for i in range(4)]
    return BankGroup.from_programs(codes, data=DATA, tick_budget=64)

def owning_pair(neighbour_src):
    return BankGroup.from_programs([asm(TO_NEIGHBOUR.format(target=1)), asm(neighbour_src)],
                                   tick_budget=32)

def text_at(image, want_prefix):

    at = 0
    while at < len(image):
        row = disasm.decode(image, at)
        if row.text.startswith(want_prefix):
            return at
        at += row.size or 1
    return None

def c1_encodings():
    print("C1 the family's encodings round-trip through asm and disasm")
    bad = []
    for text, sub in FAMILY:
        code = asm(text)
        if code[0] != 0x70 or code[1] != sub:
            bad.append(f"{text!r} emitted {code.hex()}, wanted 70{sub:02x}")
            continue
        row = disasm.decode(code, 0)
        if not row.assigned:
            bad.append(f"{text!r} -> {code.hex()} decodes as unassigned")
        elif asm(row.text) != code:
            bad.append(f"{text!r} disassembles to {row.text!r}, which re-assembles as "
                       f"{asm(row.text).hex()}")
    check("C1 every assigned bank subcode round-trips", not bad, "; ".join(bad[:3]))
    check("C1 the family stops at 0xBD", not [s for s in (0xBE, 0xBF) if s in ISA.ESCAPE],
          "a subcode past the family is assigned")
    refused = None
    try:
        asm("  LDM r0, [DE]")
    except Exception as exc:
        refused = str(exc)
    check("C1 a spelling outside the table is refused by name", refused is not None
          and "LDM" in refused, repr(refused))

def c2_one_bank():
    print("C2 a one-bank group is the lone machine")
    drift = []
    for i, text in enumerate(LONE):
        code = asm(text)
        lone = NCP8(code, data=DATA, inputs=b"\x01\x02", tick_budget=64)
        grp = BankGroup.from_programs([code], data=DATA, inputs=b"\x01\x02", tick_budget=64)
        for _ in range(70):
            if lone.status != STATUS_RUNNING:
                break
            try:
                lone.step()
            except MachineError:
                pass
            grp.step(strict=False)
            a, b = lone.record_state(), grp.machine(0).record_state()
            if a != b:
                drift.append(f"program {i} diverged on {diffs(a, b)}")
                break
    check("C2 every pinned program agrees tick by tick", not drift, "; ".join(drift[:2]))

def c3_own_page():
    print("C3 the page at a machine's own index is the DATA it executes from")
    g = group_of_four()
    wrong = [i for i in range(4)
             if g.machine(i).banks[g.machine(i).bank_own] is not g.machine(i).data]
    check("C3 own bank is the machine's own DATA object", not wrong, str(wrong))

def c4_isolation():
    print("C4 four machines write and read their own pages")
    g = group_of_four()
    g.run()
    outs = [bytes(g.out(i)) for i in range(4)]
    check("C4 each machine emits the byte it wrote to its own page",
          outs == [bytes([0x11 * (i + 1)]) for i in range(4)], str(outs))
    base = padded_page()
    wrong = []
    for j in range(4):
        want = bytearray(base)
        want[40] = 0x11 * (j + 1)
        if bytes(g.data(j)) != bytes(want):
            at = next(i for i in range(len(want)) if g.data(j)[i] != want[i])
            wrong.append((j, at, g.data(j)[at], want[at]))
    check("C4 no page holds a byte written by anyone else", not wrong, str(wrong[:4]))

def c5_running_neighbour():
    print("C5 writing a RUNNING neighbour's page faults and commits nothing")
    g = owning_pair(LOOP)
    writer, owner = g.machine(0), g.machine(1)
    stm_at = text_at(bytes(writer.code), "STM")
    check("C5 the probe program contains an STM", stm_at is not None,
          bytes(writer.code).hex())
    step_until(g, writer, 4)
    before, owner_before = writer.record_state(), owner.record_state()
    pages = (bytes(g.data(0)), bytes(g.data(1)))
    g.step(strict=False)
    committed = diffs(writer.record_state(), before)
    check("C5 the refusing tick commits only the three fault fields",
          writer.status == STATUS_ERROR and committed == ["fault_addr", "fault_reason",
                                                          "status"],
          f"status {writer.status}, committed {committed}")
    check("C5 the cause is BANK_BUSY at the address of the STM",
          writer.fault_reason == CAUSE["BANK_BUSY"] and stm_at is not None
          and writer.fault_addr == stm_at,
          f"cause {writer.fault_reason} at {writer.fault_addr:#x}, the STM is at "
          + ("unknown" if stm_at is None else f"{stm_at:#x}"))
    check("C5 neither page moved", (bytes(g.data(0)), bytes(g.data(1))) == pages,
          f"{pages[0][64]:#x} {pages[1][64]:#x}")
    check("C5 the refused tick did not advance the writer's counter",
          writer.record_state()["tick"] == before["tick"],
          f"tick {before['tick']} -> {writer.record_state()['tick']}")

    legal = BankGroup.from_programs([asm(TO_NEIGHBOUR.format(target=0)), asm(LOOP)],
                                    tick_budget=32)
    for _ in range(5):
        g.step(strict=False)
        while legal.tick < g.tick and legal.machine(1).status == STATUS_RUNNING:
            legal.step(strict=False)
    check("C5 the two groups stepped their neighbours the same number of times",
          legal.machine(1).tick == owner.tick, f"{legal.machine(1).tick} vs {owner.tick}")
    check("C5 the neighbour executes the same whether the refused store happened or not",
          diffs(owner.record_state(), legal.machine(1).record_state()) == [],
          str(diffs(owner.record_state(), legal.machine(1).record_state())))

def c6_quiescent_neighbour():
    print("C6 writing a QUIESCENT neighbour's page lands there")
    g = owning_pair(SLEEP)
    writer, owner = g.machine(0), g.machine(1)
    g.step(strict=False)
    owner_before = owner.record_state()
    g.run()
    after = owner.record_state()
    check("C6 the byte arrived in the neighbour's page", g.data(1)[64] == 0x77,
          hex(g.data(1)[64]))
    check("C6 the writer halted normally", writer.status == STATUS_HALT, writer.status)
    check("C6 only the written page differs on the owner",
          diffs(after, owner_before) == ["DATA"], str(diffs(after, owner_before)))

def c7_boundary():
    print("C7 quiescence is decided at the group's tick boundary")
    g = BankGroup.from_programs([asm(TO_NEIGHBOUR.format(target=1)), asm(HALT_AFTER)],
                                tick_budget=16)
    seen = []
    for _ in range(6):
        if g.machine(0).status != STATUS_RUNNING:
            break
        g.step(strict=False)
        seen.append((g.tick, g.machine(1).status, g.data(1)[64]))
    first = next((row for row in seen if row[2] == 0x77), None)
    check("C7 the write lands on the first boundary whose owner is already quiescent",
          first is not None and first[1] != STATUS_RUNNING, f"boundaries {seen}")
    check("C7 the boundary vector is one status per bank",
          len(g.machine(0).bank_owner_status) == 2, str(g.machine(0).bank_owner_status))

    owner_busy = "  ADDI r0, 1\n  ADDI r0, 1\n  ADDI r0, 1\n  HALT"
    writer_last = "  LDI HL, 0\n  MOV MB, HL\n  LDI r0, 0x33\n  STM [HL], r0\n  HALT"
    rev = BankGroup.from_programs([asm(owner_busy), asm(writer_last)], tick_budget=16)
    for _ in range(4):
        rev.step(strict=False)
    check("C7 a writer that steps after its owner still sees the boundary's own tick",
          rev.machine(1).status == STATUS_ERROR
          and rev.machine(1).fault_reason == CAUSE["BANK_BUSY"]
          and rev.data(0)[0] == 0,
          f"status {rev.machine(1).status} cause {rev.machine(1).fault_reason} byte "
          f"{rev.data(0)[0]:#x}")

    probe = asm("  LDI HL, 1\n  MOV MB, HL\n  LDI HL, 64\n  LDI r0, 0x77\n"
                "  STM [HL], r0\n  HALT")
    halted = STATUS_CODE[STATUS_HALT]
    for shown, want in ((None, "refused"), ((halted,), "refused"),
                        ((halted, halted), "landed")):
        m = NCP8(probe, data=DATA, tick_budget=16, config=ISA.MachineConfig(nbanks=2))
        m.banks = (m.data, bytearray(DATA))
        m.bank_own, m.bank_owner_status = 0, shown
        try:
            m.run()
        except MachineError:
            pass
        named = "none" if shown is None else f"{len(shown)} of 2"
        refused = (m.status == STATUS_ERROR and m.fault_reason == CAUSE["BANK_BUSY"]
                   and m.banks[1][64] == DATA[64])
        check(f"C7 an owner vector naming {named} bank statuses is {want}",
              refused == (want == "refused"),
              f"status {m.status} cause {m.fault_reason} byte {m.banks[1][64]:#x}")

def c8_reachability():
    print("C8 only a neighbour's DATA page is reachable")
    codes = [asm(SELF_WRITE.format(bank=i, val=0x11 * (i + 1))) for i in range(4)]
    g = BankGroup.from_programs(codes, data=DATA,
                                inputs=[bytes([20 + i]) for i in range(4)], tick_budget=64)
    m = g.machine(0)
    held = list(vars(m).values())
    leaked = []
    for other in g.machines[1:]:
        if any(v is other for v in held):
            leaked.append("the machine object itself")
        for attr in ("code", "out", "inputs"):
            if any(v is getattr(other, attr) for v in held):
                leaked.append(f"another machine's {attr}")
    check("C8 no neighbour's object, CODE, input stream or output stream is held",
          not leaked, ", ".join(sorted(set(leaked))))
    check("C8 the only foreign objects are DATA pages, one per bank",
          len(m.banks) == 4 and all(isinstance(b, bytearray) for b in m.banks),
          str([type(b).__name__ for b in m.banks]))

def c9_declaration():
    print("C9 NBANKS is declared and obeyed")
    code = asm("  HALT")
    for declared, size, want in ((5, 1, "refused"), (1, 1, "built"), (None, 3, "built")):
        cfg = ISA.MachineConfig(nbanks=declared)
        try:
            g = BankGroup.from_programs([code] * size, config=cfg)
            got, nb = "built", g.machine(0).nbanks
        except ISA.ConfigError:
            got, nb = "refused", None
        check(f"C9 a block declaring NBANKS={declared} in a group of {size} is {want}",
              got == want and (want != "built" or nb == size), f"{got} nbanks={nb}")
    check("C9 the group's bounds are stated", GROUP_MIN == 1 and GROUP_MAX >= 2,
          f"{GROUP_MIN}..{GROUP_MAX}")

def c10_causes():
    print("C10 the bank causes are numbered, ranked and singular")
    for cause in ("BANK_OOB", "BANK_BUSY"):
        check(f"C10 {cause} is a numbered cause the table can state",
              cause in CAUSE and CAUSE[cause] in ISA.CAUSE_NAME, str(max(CAUSE.values())))
    stacked = list(ISA.full_fault_site_order())
    bank_ranks = [i for i, (_s, c) in enumerate(stacked) if c in ("BANK_OOB", "BANK_BUSY")]
    data_rank = next(i for i, (_s, c) in enumerate(stacked) if c == "DATA_OOB")
    named = [stacked[i][1] for i in bank_ranks]
    check("C10 both bank sites are stacked, ahead of the address bound they outrank",
          named == ["BANK_OOB", "BANK_BUSY"] and bank_ranks[-1] < data_rank,
          f"{len(stacked)} sites, bank sites {named} at {bank_ranks}, DATA_OOB at "
          f"{data_rank}")
    code = asm("  LDI HL, 9\n  MOV MB, HL\n  LDI HL, 0xFFFF\n  STM [HL], r0\n  HALT")
    m = NCP8(code, data=DATA, tick_budget=16, config=ISA.MachineConfig(nbanks=2))
    m.banks = (m.data, bytearray(DATA))
    m.bank_own, m.bank_owner_status = 0, (STATUS_CODE[STATUS_HALT], STATUS_CODE[STATUS_HALT])
    try:
        m.run()
    except MachineError:
        pass
    check("C10 one tick that breaks two bounds records one cause",
          m.status == STATUS_ERROR and m.fault_reason == CAUSE["BANK_OOB"],
          f"cause {m.fault_reason} at {m.fault_addr:#x}")

def c11_sticky():
    print("C11 a bank fault is terminal, and a record is one machine's own state")
    g = owning_pair(LOOP)
    writer = g.machine(0)
    try:
        g.run()
    except MachineError:
        pass
    first = writer.record_state()
    for _ in range(3):
        g.step(strict=False)
    check("C11 stepping past the fault changes nothing", writer.record_state() == first,
          str(diffs(writer.record_state(), first)))
    g.machine(1).data[7] = 0xC3
    check("C11 a record holds the machine's own page, not a window onto a neighbour's",
          first["DATA"] == bytes(g.data(0)) and first["DATA"][7] != 0xC3
          and g.data(0)[7] != 0xC3,
          f"record byte 7 is {first['DATA'][7]:#x}, own page {g.data(0)[7]:#x}")

SELECT_PROGRAMS = [
    ("sets the selector",
     "  LDI r0, 0xFF\n  ADDI r0, 1\n  LDI HL, 0x1234\n  MOV MB, HL\n  HALT", 0x1234),
    ("reads it back into HL",
     "  LDI r0, 0xFF\n  ADDI r0, 1\n  LDI HL, 0xABCD\n  MOV MB, HL\n  LDI HL, 9\n"
     "  MOV HL, MB\n  HALT", 0xABCD),
    ("selects twice and keeps the last",
     "  LDI HL, 1\n  MOV MB, HL\n  LDI HL, 2\n  MOV MB, HL\n  MOV HL, MB\n  HALT", 2),
    ("names a bank it cannot reach",
     "  LDI HL, 0x1234\n  MOV MB, HL\n  HALT", 0x1234),
    ("selects its own page",
     "  LDI HL, 7\n  MOV MB, HL\n  LDI HL, 0\n  MOV MB, HL\n  MOV HL, MB\n  HALT", 0),
    ("never selects at all",
     "  LDI HL, 0x55\n  LDI r0, 1\n  ADD r0, r1\n  HALT", 0),
]

SELECT_THEN_STORE = ("  LDI r0, 0x5A\n  LDI HL, {bank}\n  MOV MB, HL\n  LDI HL, 4\n"
                     "  STM [HL], r0\n  HALT")
STORE_CUT = 4
CIRCUIT_PATHS = ("torch", "triton", "batch")

class BatchRow:

    def __init__(self, batch, i=0):
        self.batch = batch
        self.i = i

    def step(self):
        self.batch.step(1)

    def record_state(self):
        return self.batch.record_state(self.i)

    def install_state(self, snap):
        self.batch.install_state(self.i, snap)

    def snapshot(self):
        return self.batch.snapshot(self.i)

def machine(path, code, *, data=DATA, budget=32, config=None, rows=1, row=0):

    if path == "reference":
        return NCP8(code, data=data, tick_budget=budget, config=config)
    if path == "torch":
        return TorchCircuit(code, data=data, tick_budget=budget, config=config,
                            device="cpu")
    if path == "batch":
        b = TritonBatch(rows, tick_budget=budget, config=config)
        for i in range(rows):
            b.set_program(i, code, data)
        return BatchRow(b, row)
    return TritonCircuit(code, data=data, tick_budget=budget, config=config)

def ticks(handle, limit=12):

    seen = []
    while len(seen) < limit:
        got = tick_once(handle)
        seen.append(got)
        if got["status"] != STATUS_CODE[STATUS_RUNNING]:
            break
    return seen

def run_to_a_stop(handle, limit=12):

    for _ in range(limit):
        if handle.record_state()["status"] != STATUS_CODE[STATUS_RUNNING]:
            break
        handle.step()
    return handle.record_state()

def field_diffs(got, want):

    return [(f, got[f], want[f]) for f in TICK_FIELDS if got[f] != want[f]]

def c13_selector_row():
    print("C13 the bank selector is a state row on every path")
    row = next((f for f in ISA.STATE_FIELDS if f.name == "MB"), None)
    check("C13 the state table names MB as one 16-bit register",
          row is not None and (row.lo, row.hi, row.cells) == (0, 65535, None), str(row))
    check("C13 a record carries the selector", "MB" in ISA.RECORD_COMPONENTS,
          str(tuple(ISA.RECORD_COMPONENTS)))
    left_out = [n for n in ISA.STATE_FIELD_NAMES if n not in TICK_FIELDS]
    check("C13 the per-tick comparison reads every state row, the selector included",
          not left_out, f"TICK_FIELDS leaves out {left_out}")
    lone = NCP8(asm("  HALT"))
    check("C13 a machine that never selects starts on the bank it can reach",
          lone.MB == 0 and lone.MB < lone.nbanks and lone.snapshot()["MB"] == 0,
          f"MB {lone.MB}, NBANKS {lone.nbanks}")

    for what, text, want_mb in SELECT_PROGRAMS:
        code = asm(text)
        ref = ticks(machine("reference", code))
        check(f"C13 the reference ends on the selector the program names: {what}",
              ref[-1]["MB"] == want_mb,
              f"the reference record says {ref[-1]['MB']}, the program names {want_mb:#x}")
        for path in CIRCUIT_PATHS:
            got = ticks(machine(path, code))
            bad = ([] if len(got) == len(ref)
                   else [("tick count", len(got), len(ref))]) + [
                       (i, f) for i, (g, w) in enumerate(zip(got, ref))
                       for f in field_diffs(g, w)]
            check(f"C13 {path} matches both selector moves tick by tick: {what}",
                  not bad, str(bad[:3]))

    for what, text, want_mb in SELECT_PROGRAMS:
        code = asm(text)
        for src in ("reference",) + CIRCUIT_PATHS:
            rec = run_to_a_stop(machine(src, code))
            check(f"C13 a {src} record carries the selector after {what}",
                  rec["MB"] == want_mb, f"record {rec['MB']}, program names {want_mb:#x}")
            for dst in ("reference",) + CIRCUIT_PATHS:
                fresh = machine(dst, code)
                fresh.install_state(rec)
                back = fresh.record_state()
                check(f"C13 {src}->{dst} restores the selector after {what}",
                      back["MB"] == want_mb and back == rec,
                      f"restored {back['MB']}, components that differ "
                      f"{sorted(n for n in back if back[n] != rec[n])}")

    codes = [asm(SELECT_PROGRAMS[0][1]), asm(SELECT_PROGRAMS[1][1])]
    solo = [ticks(machine("reference", c)) for c in codes]
    batch = TritonBatch(2, tick_budget=32)
    for i, c in enumerate(codes):
        batch.set_program(i, c, DATA)
    handles = [BatchRow(batch, i) for i in range(2)]
    for _ in range(12):
        if all(h.record_state()["status"] != STATUS_CODE[STATUS_RUNNING] for h in handles):
            break
        batch.step(1)
    for i, want in enumerate(solo):
        got = handles[i].record_state()
        bad = ([] if got["tick"] == want[-1]["tick"]
               else [("tick", got["tick"], want[-1]["tick"])]) + field_diffs(got, want[-1])
        check(f"C13 row {i} holds the selector of its own program",
              got["MB"] == SELECT_PROGRAMS[i][2],
              f"row {i} says {got['MB']}, want {SELECT_PROGRAMS[i][2]:#x}")
        check(f"C13 row {i} agrees with the reference running its program alone",
              not bad, str(bad[:3]))

    pair_codes = [asm("  LDI HL, 0x1234\n  MOV MB, HL\n  HALT"),
                  asm("  LDI HL, 0xABCD\n  MOV MB, HL\n  HALT")]
    pair_mb = [0x1234, 0xABCD]
    pair = TritonBatch(2, tick_budget=32)
    for i, c in enumerate(pair_codes):
        pair.set_program(i, c, DATA)
    for i, mb in enumerate(pair_mb):
        pair.set_state(i, MB=mb)
    check("C13 a state set on one row leaves the other row's selector alone",
          [pair.record_state(i)["MB"] for i in range(2)] == pair_mb,
          f"{[pair.record_state(i)['MB'] for i in range(2)]}")
    keep = pair.record_state(1)
    pair.install_state(0, keep)
    check("C13 installing one row leaves the other row's record alone",
          pair.record_state(1) == keep and pair.record_state(0) == keep,
          f"row 1 {pair.record_state(1)['MB']}, row 0 {pair.record_state(0)['MB']}")

    two = with_nbanks(None, 2)
    halted = STATUS_CODE[STATUS_HALT]
    for bank, reaches in ((1, True), (9, False)):
        code = asm(SELECT_THEN_STORE.format(bank=bank))
        src = machine("torch", code, data=bytes(DATA_SIZE), config=two)
        for _ in range(STORE_CUT):
            src.step()
        rec = src.record_state()
        check(f"C13 a tensor-path record cut before the store carries selector {bank}",
              rec["MB"] == bank and rec["tick"] == STORE_CUT,
              f"record selector {rec['MB']} at tick {rec['tick']}")
        tgt = NCP8(code, data=bytes(DATA_SIZE), tick_budget=32, config=two)
        refused = refuse(tgt.install_state, rec)
        check(f"C13 selector {bank} installs into a machine loaded with "
              f"{two.nbanks} banks", refused is None and tgt.MB == bank, repr(refused))
        pages = (tgt.data, bytearray(DATA_SIZE))
        tgt.banks, tgt.bank_own, tgt.bank_owner_status = pages, 0, (halted, halted)
        try:
            tgt.step()
        except MachineError:
            pass
        moved = pages[1][4] == 0x5A and pages[0][4] == 0
        check(f"C13 the store the restored selector reaches is "
              + ("the named page" if reaches else "refused with the bound it broke"),
              (moved and tgt.status == STATUS_RUNNING) if reaches else
              (not moved and pages[1][4] == 0 and tgt.status == STATUS_ERROR
               and tgt.fault_reason == CAUSE["BANK_OOB"]),
              f"status {tgt.status} cause {tgt.fault_reason} foreign byte {pages[1][4]:#x}")

PAGE_PROGRAM = ("  LDI r0, 0x5A\n  LDI r1, 0xA5\n  LDI r3, 0x3C\n"
                "  LDI HL, 0x1122\n  LDI DE, 0x3344\n"
                "  LDI r2, 0xFF\n  ADDI r2, 1\n"
                "  LDI HL, {bank}\n  MOV MB, HL\n"
                "  LDI HL, {addr}\n  LDI DE, {addr2}\n"
                "  {access}\n"
                "  OUT r0\n  OUT r3\n  HALT")
PAGE_ACCESSES = ("LDM r0, [HL]", "STM [HL], r1", "LDMW DE, [HL]", "LDMW HL, [DE]",
                 "STMW [HL], DE", "STMW [DE], HL")

PAGE_SELECTORS = (0, 1, 255)

PAGE_DATA = bytes(range(1, 256)) * 16

BOUND_PROBE = ("  LDI HL, 9\n  MOV MB, HL\n  LDI HL, 0xFFFF\n  STM [HL], r0\n  HALT")
ADDR_PROBE = ("  LDI HL, 0xFFFF\n  STM [HL], r0\n  HALT")

FAULT_ONLY = ["fault_addr", "fault_reason", "status"]

def _row_records(batch, rows):

    for _ in range(40):
        batch.step(1)
        if all(batch.record_state(i)["status"] != STATUS_CODE[STATUS_RUNNING]
               for i in range(rows)):
            break
    return [batch.record_state(i) for i in range(rows)]

def c14_page_accesses():

    print("C14 the page accesses execute on the tensor, kernel and batch paths")
    runs = ticks_compared = refusals = oob_named = busy_named = 0
    for bank in PAGE_SELECTORS:
        for access in PAGE_ACCESSES:
            code = asm(PAGE_PROGRAM.format(bank=bank, access=access, addr=20,
                                           addr2=24))
            ref = ticks(machine("reference", code, data=PAGE_DATA), limit=16)
            runs += 1
            check(f"C14 the reference answers {access} at MB={bank} as the rule states",
                  ref[-1]["status"] == (STATUS_CODE[STATUS_ERROR] if bank else
                                        STATUS_CODE[STATUS_HALT])
                  and (not bank or ref[-1]["fault_reason"] == CAUSE["BANK_OOB"]),
                  f"status {ref[-1]['status']} cause {ref[-1]['fault_reason']}")
            for path in CIRCUIT_PATHS:
                got = ticks(machine(path, code, data=PAGE_DATA), limit=16)
                bad = ([] if len(got) == len(ref)
                       else [("tick count", len(got), len(ref))]) + [
                           (i, f) for i, (g, w) in enumerate(zip(got, ref))
                           for f in field_diffs(g, w)]
                ticks_compared += len(ref) * len(TICK_FIELDS)
                check(f"C14 {path} matches the reference tick by tick: {access} at "
                      f"MB={bank}", not bad, str(bad[:3]))
                busy_named += sum(1 for g in got
                                  if g["fault_reason"] == CAUSE["BANK_BUSY"])
                if bank:
                    refused = next((i for i, g in enumerate(got)
                                    if g["status"] == STATUS_CODE[STATUS_ERROR]), None)

                    moved = []
                    if refused:
                        moved = sorted(k for k in TICK_FIELDS
                                       if got[refused][k] != got[refused - 1][k])
                    ok = bool(refused) and moved == FAULT_ONLY
                    if ok:
                        refusals += 1
                    check(f"C14 {path} commits only the fault fields on the refusing "
                          f"tick: {access} at MB={bank}", ok,
                          f"error tick at {refused}, this tick committed {moved}")
                    if refused and got[refused]["fault_reason"] == CAUSE["BANK_OOB"]:
                        oob_named += 1
    refused_total = (len(PAGE_SELECTORS) - 1) * len(PAGE_ACCESSES) * len(CIRCUIT_PATHS)
    check("C14 every refusing tick committed only the three fault fields",
          refusals == refused_total, f"{refusals} of {refused_total}")
    check("C14 every tick a circuit path refused named BANK_OOB, the bound it broke",
          oob_named == refused_total, f"{oob_named} of {refused_total}")

    holders = {name: hasattr(cls, "install_banks")
               for name, cls in (("NCP8", NCP8), ("TorchCircuit", TorchCircuit),
                                 ("TritonCircuit", TritonCircuit),
                                 ("TritonBatch", TritonBatch))}
    check("C14 no tick of the sweep names BANK_BUSY on a path that holds one page",
          busy_named == 0 and holders["NCP8"]
          and not any(v for k, v in holders.items() if k != "NCP8"),
          f"{busy_named} tick(s) named it, page installers {holders}")

    for text, want in ((BOUND_PROBE, "BANK_OOB"), (ADDR_PROBE, "DATA_OOB")):
        code = asm(text)
        for path in ("reference",) + CIRCUIT_PATHS:
            got = ticks(machine(path, code, data=PAGE_DATA), limit=8)
            check(f"C14 {path} names {want} for the tick that breaks both bounds"
                  if want == "BANK_OOB" else
                  f"C14 {path} names {want} for the tick that breaks only the address",
                  got[-1]["fault_reason"] == CAUSE[want]
                  and got[-1]["status"] == STATUS_CODE[STATUS_ERROR],
                  f"cause {got[-1]['fault_reason']} at status {got[-1]['status']}")

    pair = [asm(PAGE_PROGRAM.format(bank=0, access="STM [HL], r0", addr=40, addr2=44)),
            asm(PAGE_PROGRAM.format(bank=0, access="STM [HL], r1", addr=41, addr2=45))]
    solo = [ticks(machine("reference", c, data=PAGE_DATA), limit=16) for c in pair]
    batch = TritonBatch(2, tick_budget=32)
    for i, c in enumerate(pair):
        batch.set_program(i, c, PAGE_DATA)
    got = _row_records(batch, 2)
    for i in range(2):
        bad = field_diffs(got[i], solo[i][-1])
        check(f"C14 batch row {i} reaches its own page and no other row's", not bad,
              str(bad[:3]))
    foreign = [i for i in range(2)
               if got[i]["DATA"] != solo[i][-1]["DATA"]]
    check("C14 no page of the batch holds a byte another row wrote", not foreign,
          str(foreign))
    return {"runs": runs, "ticks": ticks_compared, "selectors": PAGE_SELECTORS,
            "accesses": len(PAGE_ACCESSES), "paths": len(CIRCUIT_PATHS)}

def main():
    print("bank acceptance: the ownership rules of the reference path, measured")
    for fn in (c1_encodings, c2_one_bank, c3_own_page, c4_isolation, c5_running_neighbour,
               c6_quiescent_neighbour, c7_boundary, c8_reachability, c9_declaration,
               c10_causes, c11_sticky, c13_selector_row, c14_page_accesses):
        fn()
    print("\nC12 what is NOT RUN here, and the condition that closes each row")
    for row, why in (
            ("a batch is not a group",
             "TritonBatch holds one DATA page per row, so a row's selector can name only "
             "the page it owns: it carries a per-row MB and no foreign page. Measured by "
             "C14, which runs two bank programs in one batch and finds each row's page "
             "identical to the reference running that row's program alone"),
            ("BANK_BUSY on a circuit path",
             "the ownership rule is answered on all four paths, but the three circuit "
             "paths hold one page each, so no bank access there can name this cause. "
             "Measured by C14, which names it zero times in "
             f"{len(PAGE_SELECTORS) * len(PAGE_ACCESSES) * len(CIRCUIT_PATHS)} runs over "
             "every selector a lone machine can name, and by C5, which produces it on "
             "the reference; test_fault_registers.py C4 re-derives the absence")):
        print(f"  NOT RUN {row}   ({why})")
    print("\nC12 measured by C14, not listed as NOT RUN: the six page accesses answer "
          "on TorchCircuit, TritonCircuit and TritonBatch at every selector a machine "
          "that holds one page can name, and the two selector moves with them.")
    print(f"\nbank acceptance: {len(FAILS)} failure(s)")
    for f in FAILS:
        print("  !!", f)
    return 1 if FAILS else 0

if __name__ == "__main__":
    sys.exit(main())