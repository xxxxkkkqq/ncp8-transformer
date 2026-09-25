"""Fault-register acceptance: cause, address, and one cause per tick.

Six checks over the two fault registers the machine records when a tick stops it.
Every cause the table can produce is provoked and compared field by field against
the reference on all four paths (reference, tensor, Triton, batched); the cause and
status fields are paired in both directions, since a cause latched while the machine
still runs and a machine stopped without naming why are two different bugs; where two
sites can fire on one instruction the table's order decides the single reported cause;
a cause whose feature has not landed yet is exempted by a check re-derived from the
machine, so it fails the moment that feature arrives rather than quietly staying
exempt; widths are asserted on every tick, not only on the faulting one; and the
faulting tick both raises and records, after which the machine is a sticky terminal
state.

Run: python3 test_fault_registers.py
"""
from __future__ import annotations

import subprocess
import sys

import torch

import circuit_triton
import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import TritonBatch, TritonCircuit
from golden_sim import (DATA_SIZE, MachineError, NCP8, STATUS_CODE, STATUS_RUNNING,
                        asm)
from test_state_contract import FAULT_WRITES, VIEW_FIELDS, assert_widths, ref_view

CODE_SIZE = 4096
OUT_CAP = 8192
VEC = 0x0F00
WLO, WHI = 0x0F20, 0x0F21
TICK0 = 5

PC_SITES = (0, 256)

CAUSE = ISA.CAUSE

def img(head, pc=0, *, vectors=None, window=None, tail_pad=0, length=None):

    n = max(4, pc + len(head) + 2 + tail_pad)
    if vectors or window:
        n = max(n, WHI + 2)
    if length is not None:
        if length < pc + len(head):
            raise ValueError("the instruction bytes would not fit the image it names")
        n = length
    b = bytearray(n)
    b[pc:pc + len(head)] = bytes(head)
    for k, addr in (vectors or {}).items():
        b[VEC + 2 * k:VEC + 2 * k + 2] = (addr & 0xFFFF).to_bytes(2, "little")
    if window:
        b[WLO], b[WHI] = window
    return bytes(b)

def state(**over):

    s = dict(r=[1, 2, 3, 4], HL=8, DE=8, PC=0, SP=2048, C=0, Z=0, ipos=0, oplen=0,
             tick=TICK0, status=0, fault_reason=0, fault_addr=0, MB=0)
    s.update(over)
    return s

def _c_ok(pc):
    return img(asm("NOP"), pc), state(PC=pc), pc, b"", b""

def _c_bad_opcode(pc):
    return img(bytes([0x71, 0x00]), pc), state(PC=pc), pc, b"", b""

def _c_bad_subcode(pc):
    return img(bytes([0x70, 0x63, 0x00]), pc), state(PC=pc), pc, b"", b""

def _c_fetch_operand(pc):

    return (img(bytes([0x09, 0x00]), pc, length=pc + 2), state(PC=pc), pc, b"", b"")

def _c_fetch_prefix(pc):

    return img(bytes([0x70]), pc, length=pc + 1), state(PC=pc), pc, b"", b""

def _c_data_oob(pc):
    return (img(asm("MOV r0, [HL]"), pc), state(PC=pc, HL=DATA_SIZE), pc, b"", b"")

def _c_stack_underflow(pc):
    return img(asm("POP r0"), pc), state(PC=pc, SP=DATA_SIZE), pc, b"", b""

def _c_stack_underflow_ret(pc):
    return img(asm("RET"), pc), state(PC=pc, SP=DATA_SIZE - 1), pc, b"", b""

def _c_stack_overflow(pc):
    return img(asm("PUSH r0"), pc), state(PC=pc, SP=0), pc, b"", b""

def _c_stack_overflow_call(pc):
    return img(asm("CALL 0x1F00"), pc), state(PC=pc, SP=1), pc, b"", b""

def _c_div_zero(pc):
    return (img(asm("DIV r0, r1"), pc), state(PC=pc, r=[1, 0, 3, 4]), pc, b"", b"")

def _c_mod_zero(pc):
    return (img(asm("MOD r0, r1"), pc), state(PC=pc, r=[1, 0, 3, 4]), pc, b"", b"")

def _c_code_oob(pc):
    image = img(asm("LDC r0, [HL]"), pc)
    return image, state(PC=pc, HL=len(image)), pc, b"", b""

def _c_window(pc):

    image = img(asm("STC [HL], r0"), pc, window=(0x10, 0x18), tail_pad=0x18)
    return image, state(PC=pc, HL=0x04), pc, b"", b""

def _c_trap_unreg(pc):
    return img(asm("EXT 0"), pc), state(PC=pc), pc, b"", b""

def _c_trap_k_oob(pc):

    return img(bytes([0x70, 0x70, 16]), pc), state(PC=pc), pc, b"", b""

def _c_sp_pair_oob(pc):
    return img(asm("MOVW SP, HL"), pc), state(PC=pc, HL=DATA_SIZE + 1), pc, b"", b""

def _c_sp_add_oob(pc):
    return img(asm("ADD SP, 1"), pc), state(PC=pc, SP=DATA_SIZE), pc, b"", b""

def _c_ldx_wrap(pc):
    return (img(asm("LDX r0, [HL-1]"), pc), state(PC=pc, HL=0), pc, b"", b"")

def _c_stw_second_byte(pc):
    return (img(asm("STW [HL], DE"), pc), state(PC=pc, HL=DATA_SIZE - 1), pc, b"", b"")

def _c_out_cap(pc):
    return (img(asm("OUT r0"), pc), state(PC=pc, oplen=OUT_CAP), pc, b"", b"")

def _c_bank_oob(pc, mb=9, access="STM [HL], r0"):
    return (img(asm(access), pc), state(PC=pc, MB=mb), pc, b"", b"")

def _c_bank_oob_ldm(pc):
    return _c_bank_oob(pc, mb=255, access="LDM r0, [HL]")

def _c_bank_addr_oob(pc):

    return (img(asm("STM [HL], r0"), pc), state(PC=pc, HL=DATA_SIZE), pc, b"", b"")

def _c_bank_pair_addr_oob(pc):
    return (img(asm("STMW [HL], DE"), pc), state(PC=pc, HL=DATA_SIZE - 1), pc, b"", b"")

def _p_fetch_before_trap(pc):

    return (img(bytes([0x70, 0x70]), pc, length=pc + 2), state(PC=pc), pc, b"", b"")

def _p_fetch_before_stack(pc):

    return (img(bytes([0x0E, 0x00]), pc, length=pc + 2), state(PC=pc, SP=0), pc,
            b"", b"")

def _p_trap_before_stack(pc):

    return img(asm("EXT 0"), pc), state(PC=pc, SP=1), pc, b"", b""

def _p_code_oob_before_window(pc):

    image = img(asm("STC [HL], r0"), pc, window=(0x10, 0x18), tail_pad=0x18)
    return image, state(PC=pc, HL=len(image)), pc, b"", b""

def _p_data_oob_before_out_cap(pc):

    return (img(asm("OUTM"), pc), state(PC=pc, HL=DATA_SIZE, oplen=OUT_CAP), pc,
            b"", b"")

def _bank_selector_case(mb):

    def build(pc):
        return (img(asm("STM [HL], r0"), pc), state(PC=pc, MB=mb), pc, b"", b"")
    return build

def _p_bank_oob_before_data_oob(pc):

    return (img(asm("STM [HL], r0"), pc), state(PC=pc, MB=9, HL=DATA_SIZE), pc,
            b"", b"")

BANK_SELECTORS = (0, 1, 2, 63, 255, 4096, 65535)

FOREIGN_ADDR = 8

def _foreign_owner_running_names_busy():

    code = asm("  LDI HL, 1\n  MOV MB, HL\n  LDI HL, 8\n  STM [HL], r0\n  HALT")
    g = NCP8(code, data=bytes(DATA_SIZE), tick_budget=8,
             config=ISA.MachineConfig(nbanks=2))
    pages = (g.data, bytearray(DATA_SIZE))
    running = STATUS_CODE[STATUS_RUNNING]
    g.banks, g.bank_own, g.bank_owner_status = pages, 0, (running, running)
    for _ in range(6):
        try:
            g.step()
        except MachineError:
            break
    return (g.fault_reason == CAUSE["BANK_BUSY"] and g.status == "ERROR"
            and pages[1][FOREIGN_ADDR] == 0)

def _bank_busy_absent():

    for runner in (run_torch, run_triton, run_batch):
        for mb in BANK_SELECTORS:
            _raised, view, _out = runner(_bank_selector_case(mb), 0)
            if view["fault_reason"] == CAUSE["BANK_BUSY"]:
                return False
            want = CAUSE["OK"] if mb == 0 else CAUSE["BANK_OOB"]
            if view["fault_reason"] != want:
                return False
    if not hasattr(NCP8, "install_banks"):
        return False
    if any(hasattr(cls, "install_banks")
           for cls in (TorchCircuit, TritonCircuit, TritonBatch)):
        return False
    return _foreign_owner_running_names_busy()

def _p_trap_k_before_stack(pc):

    return img(bytes([0x70, 0x70, 16]), pc), state(PC=pc, SP=0), pc, b"", b""

CASES = (
    ("OK", "NOP commits a tick and names no cause", _c_ok),
    ("BAD_OPCODE", "unassigned single-byte opcode 0x71", _c_bad_opcode),
    ("BAD_SUBCODE", "reserved escape subcode 0x70 0x63", _c_bad_subcode),
    ("FETCH_OOB", "JMP with its operand byte past the image end", _c_fetch_operand),
    ("FETCH_OOB", "escape prefix as the last byte of the image", _c_fetch_prefix),
    ("DATA_OOB", "MOV r0, [HL] with HL == DATA_SIZE", _c_data_oob),
    ("DATA_OOB", "STW whose second byte leaves DATA", _c_stw_second_byte),
    ("DATA_OOB", "LDX wrapping below DATA through the frame offset", _c_ldx_wrap),
    ("DATA_OOB", "MOVW SP, HL to a value outside DATA", _c_sp_pair_oob),
    ("DATA_OOB", "ADD SP, i8 to a value outside DATA", _c_sp_add_oob),
    ("STACK_UNDERFLOW", "POP with nothing stored", _c_stack_underflow),
    ("STACK_UNDERFLOW", "RET one slot short of a return address", _c_stack_underflow_ret),
    ("STACK_OVERFLOW", "PUSH at SP == 0", _c_stack_overflow),
    ("STACK_OVERFLOW", "CALL with one slot left", _c_stack_overflow_call),
    ("DIV_ZERO", "DIV r0, r1 with r1 == 0", _c_div_zero),
    ("DIV_ZERO", "MOD r0, r1 with r1 == 0", _c_mod_zero),
    ("CODE_OOB", "LDC at an address past the image", _c_code_oob),
    ("WINDOW", "STC inside the image but outside the declared window", _c_window),
    ("TRAP_UNREG", "EXT 0 with a zero vector entry", _c_trap_unreg),
    ("TRAP_UNREG", "EXT 16, past the 16 handler slots", _c_trap_k_oob),
    ("OUT_CAP", "OUT as byte number OUT_CAP+1", _c_out_cap),
    ("BANK_OOB", "STM [HL], r0 with the selector past this machine's page count",
     _c_bank_oob),
    ("BANK_OOB", "LDM r0, [HL] with the selector at the end of the register's span",
     _c_bank_oob_ldm),
    ("DATA_OOB", "STM [HL], r0 through the page this machine holds, past its last byte",
     _c_bank_addr_oob),
    ("DATA_OOB", "STMW [HL], DE whose second byte leaves the page it holds",
     _c_bank_pair_addr_oob),
)

PAIRS = (

    ("unfetchable immediate + unregistered handler", _p_fetch_before_trap,
     "FETCH_OOB", "TRAP_UNREG"),
    ("unfetchable immediate + no room to push", _p_fetch_before_stack,
     "FETCH_OOB", "STACK_OVERFLOW"),
    ("handler past the vector table + no room to push", _p_trap_k_before_stack,
     "TRAP_UNREG", "STACK_OVERFLOW"),
    ("unregistered handler + no room to push", _p_trap_before_stack,
     "TRAP_UNREG", "STACK_OVERFLOW"),
    ("address past the image + address outside the window", _p_code_oob_before_window,
     "CODE_OOB", "WINDOW"),
    ("DATA address out of range + output stream at capacity", _p_data_oob_before_out_cap,
     "DATA_OOB", "OUT_CAP"),
    ("selector past the page count + address outside the page", _p_bank_oob_before_data_oob,
     "BANK_OOB", "DATA_OOB"),
)

UNPROVOKABLE_PAIRS = (
    ("FETCH_OOB (code byte) before BAD_SUBCODE",
     "the padded code byte always decodes, so only one of the two can fire"),
)

ABSENT_PROOFS = {
    "TRAP_DEPTH": ("no TDEPTH state exists to compare against TDLIM",
                   lambda: not hasattr(NCP8(bytes(2)), "TDEPTH")),
    "TRAP_FRAME": ("TRAPRET is not an assigned code point",
                   lambda: 0xA8 not in ISA.ESCAPE),
    "TRAP_UNBALANCED": ("TRAPRET is not an assigned code point",
                        lambda: 0xA8 not in ISA.ESCAPE),
    "BANK_BUSY": ("a circuit path holds one page, its own, so its bank access has no "
                  "foreign owner to wait for; the reference names the cause as soon as a "
                  "group driver hands it a second page",
                  lambda: _bank_busy_absent()),
    "PC_ILLEGAL": ("a jump past the image faults on the *next* fetch, not the write",
                   lambda: _bad_target_faults_late()),
}

def _bad_target_faults_late():

    g = NCP8(asm("JMP 0x0F00\nHALT"))
    g.step()
    return g.PC == 0x0F00 and g.status == "RUNNING"

def _install(machine, st, code, data, inputs, budget):

    m = machine(code, data=data, inputs=inputs, tick_budget=budget)
    m.load_state(st["r"], st["HL"], st["DE"], st["SP"], st["C"], st["Z"], st["tick"],
                 PC=st["PC"])
    _set_ipos(m, st["ipos"])
    _set_mb(m, st["MB"])
    return m

def _set_mb(m, n):

    if not n:
        return
    if isinstance(m, NCP8):
        m.MB = n
    elif isinstance(m, TorchCircuit):
        m.MB = torch.tensor([n], dtype=torch.int32, device=m.dev)
    else:
        m.S[int(circuit_triton.S_MB)] = n

def _set_ipos(m, n):

    if not n:
        return
    if isinstance(m, NCP8):
        m.ipos = n
    elif isinstance(m, TorchCircuit):
        m.ipos = torch.tensor([n], dtype=torch.int32, device=m.dev)
    else:
        m.S[10] = n

def run_reference(case, pc):
    code, st, addr, data, inputs = case(pc)
    g = _install(NCP8, st, code, data, inputs, 200_000)
    if st["oplen"]:
        g.out = bytearray(st["oplen"])
    raised = False
    try:
        g.step()
    except MachineError:
        raised = True
    return raised, ref_view(g), bytes(g.out)

def run_torch(case, pc):
    return _run_circuit(TorchCircuit, case, pc)

def run_triton(case, pc):
    return _run_circuit(TritonCircuit, case, pc)

def _run_circuit(Machine, case, pc):
    code, st, addr, data, inputs = case(pc)
    c = _install(Machine, st, code, data, inputs, 200_000)
    if st["oplen"]:
        _set_oplen(c, st["oplen"])
    c.step()
    return c.snapshot()["status"] == 3, c.snapshot(), c.out()

def _set_oplen(c, n):
    if isinstance(c, TorchCircuit):
        c.oplen = torch.tensor([n], dtype=torch.int32, device=c.dev)
    else:
        c.S[11] = n

def run_batch(case, pc):

    code, st, addr, data, inputs = case(pc)
    b = TritonBatch(1, max_in=max(1, len(inputs)))
    b.set_program(0, code, data or b"", inputs or b"")
    b.set_state(0, r=st["r"], HL=st["HL"], DE=st["DE"], PC=st["PC"], SP=st["SP"],
                C=st["C"], Z=st["Z"], ipos=st["ipos"], oplen=st["oplen"],
                tick=st["tick"], MB=st["MB"])
    b.step(1)
    snap = b.snapshot(0)
    return snap["status"] == 3, snap, b.out(0)

IMPLS = (("reference", run_reference), ("torch", run_torch), ("triton", run_triton),
         ("batch", run_batch))

def _cause_text(code):

    name = ISA.CAUSE_NAME.get(code)
    return f"{code} ({name})" if name else f"{code}, which no cause in the table assigns"

def _first_divergence(ref, got):

    for k in VIEW_FIELDS:
        if ref[k] != got[k]:
            return (k, ref[k], got[k])
    return None

def exercise(builders, label, counts, divergences):

    for cause, how, build in builders:
        for pc in PC_SITES:
            rows = {}
            for who, runner in IMPLS:
                raised, view, out = runner(build, pc)
                assert_widths(view, (cause, how, pc, who))
                rows[who] = (raised, view, out)
            r_raised, r_view, r_out = rows["reference"]
            if cause == "OK":
                assert not r_raised and r_view["status"] == 0, (how, pc, r_view)
                assert r_view["fault_reason"] == 0 and r_view["fault_addr"] == 0, (
                    how, pc, "a tick that did not fault recorded a cause", r_view)
            else:
                assert r_raised and r_view["status"] == 3, (how, pc, r_view)
                assert r_view["fault_reason"] == CAUSE[cause], (
                    how, pc, "reference named the wrong cause",
                    _cause_text(r_view["fault_reason"]), "expected", cause)
                assert r_view["fault_addr"] == build(pc)[2], (
                    how, pc, "fault_addr is not the instruction address at tick entry",
                    r_view["fault_addr"], build(pc)[2])
            counts[cause] = counts.get(cause, 0) + 1
            for who, (raised, view, out) in rows.items():
                if who == "reference":
                    continue
                d = _first_divergence(r_view, view)
                assert d is None, (how, pc, who, "differs from the reference", d)
                assert out == r_out, (how, pc, who, "output stream differs", out, r_out)
                if d is not None and cause not in divergences:
                    divergences[cause] = (who, pc, d)

def build_coverage_cases():

    return [(cause, how, build) for cause, how, build in CASES]

def test_c1_cause_table_is_covered_exhaustively():
    counts, divergences = {}, {}
    exercised = set()
    for cause, how, build in build_coverage_cases():
        if cause == "OK":
            continue
        exercised.add(cause)
    reachable = [name for name, _d in ISA.FAULT_CAUSES if name != "OK"
                 and name not in ISA.CAUSES_AWAITING_FEATURE]
    missing = sorted(set(reachable) - exercised)
    assert not missing, ("reachable causes with no case", missing)

    exercise(build_coverage_cases(), "C1", counts, divergences)
    for name, desc in ISA.FAULT_CAUSES:
        awaiting = name in ISA.CAUSES_AWAITING_FEATURE
        got = counts.get(name, 0)
        assert got > 0 or awaiting, (
            f"cause {ISA.CAUSE[name]} {name} ({desc}) was produced by no case and is "
            f"not on the waiting list")
        if awaiting:
            assert got == 0, (f"cause {name} is on the waiting list but a case produced "
                              f"it: the list is stale")
    _print_coverage(counts, divergences)
    return counts

def _print_coverage(counts, divergences):
    print(f"  cause coverage (each case run at {len(PC_SITES)} addresses, every run "
          f"compared on all four paths):")
    print(f"    {'code':>4}  {'name':18s} {'runs':>4}  {'checks':>6}  first divergence")
    for code, (name, desc) in enumerate(ISA.FAULT_CAUSES):
        got = counts.get(name, 0)
        if name in ISA.CAUSES_AWAITING_FEATURE:
            note = f"none yet: {ISA.CAUSES_AWAITING_FEATURE[name]}"
        elif got == 0:
            note = "NO CASE AND NOT EXEMPT"
        else:
            note = "-"
        div = divergences.get(name)
        if div:
            note = f"{div[0]} @ PC {div[1]}: field {div[2][0]} {div[2][2]} vs {div[2][1]}"
        print(f"    {code:4d}  {name:18s} {got:5d}  {got * len(PC_SITES) * 3:6d}  {note}")
    total = sum(counts.values())
    covered = sum(1 for n, _d in ISA.FAULT_CAUSES if counts.get(n))
    print(f"    {covered} of {len(ISA.FAULT_CAUSES)} causes produced by {total} runs "
          f"({total * 3} circuit/batch comparisons against the reference); "
          f"{len(ISA.FAULT_CAUSES) - covered} cannot be reached until their feature lands")

def test_c2_bijection_holds_on_every_tick_of_every_case():

    seen = {"cause-without-status": 0, "status-without-cause": 0, "paired": 0}
    for cause, how, build in build_coverage_cases() + [(p[0], p[0], p[1]) for p in PAIRS]:
        for pc in PC_SITES:
            for who, runner in IMPLS:
                raised, view, _out = runner(build, pc)
                bad = ISA.fault_state_error(view["status"], view["fault_reason"],
                                            view["fault_addr"])
                if bad is None:
                    seen["paired"] += 1
                    continue
                if "recorded while status is" in bad:
                    key = "cause-without-status"
                elif "while fault_reason is 0" in bad:
                    key = "status-without-cause"
                else:
                    key = None
                assert key is not None, (cause, pc, who, bad)
                seen[key] += 1
                raise AssertionError(f"{cause} @ PC {pc} on {who}: {bad}")
    assert seen["cause-without-status"] == 0 and seen["status-without-cause"] == 0, seen
    print(f"  C2 bijection: {seen['paired']} snapshots pair fault_reason with status "
          f"in both directions (cause without status 3: {seen['cause-without-status']}, "
          f"status 3 without cause: {seen['status-without-cause']})")
    return seen

def test_c3_at_most_one_cause_per_tick():

    builders = [(f"{win} before {lose}", build, win, lose)
                for what, build, win, lose in PAIRS]
    for what, build, win, lose in builders:
        for pc in PC_SITES:
            raised, view, _out = run_reference(build, pc)
            assert raised, (what, pc, "the double violation did not fault")
            assert view["fault_reason"] == CAUSE[win], (
                what, pc, "the tick named", ISA.fault_name(view["fault_reason"]),
                "but", win, "outranks", lose, "in isa_table.FAULT_SITE_ORDER")
            for who, runner in IMPLS:
                if who == "reference":
                    continue
                r2, v2, out2 = runner(build, pc)
                assert v2["fault_reason"] == CAUSE[win], (
                    who, what, pc, "named a different cause than the reference",
                    _cause_text(v2["fault_reason"]), "expected", win)
                assert v2["fault_addr"] == view["fault_addr"], (who, what, pc, v2, view)
                assert v2["status"] == 3, (who, what, pc, v2)

    assert len(set(ISA.FAULT_SITE_NAMES)) == len(ISA.FAULT_SITE_NAMES)
    assert ISA.SITE_RANK["TRAP_UNREG"] < ISA.SITE_RANK["STACK_PUSH"], (
        "the trap-vector cause must outrank the stack bound, as golden_sim checks them")
    assert ISA.SITE_RANK["BAD_SUBCODE"] < ISA.SITE_RANK["FETCH_OPERAND"], (
        "decode refusal must outrank the operand bound that follows it")
    for what, why in UNPROVOKABLE_PAIRS:
        print(f"    not provokable on this machine: {what} -- {why}")
    print(f"  C3 precedence: {len(builders)} ordering pairs provoked at "
          f"{len(PC_SITES)} addresses each, one cause named per tick in all four paths")
    return len(builders)

def test_c4_absent_causes_are_absent_for_a_reason():

    for name, proof in ISA.CAUSES_AWAITING_FEATURE.items():
        assert name in CAUSE, (name, "not a cause the table assigns")
        what, check = ABSENT_PROOFS[name]
        assert check() is True, (
            f"cause {name} is exempted because {proof}, and {what} was assumed to hold, "
            f"but the machine says otherwise: it is reachable now and needs a case")
    assert set(ABSENT_PROOFS) == set(ISA.CAUSES_AWAITING_FEATURE), (
        set(ABSENT_PROOFS) ^ set(ISA.CAUSES_AWAITING_FEATURE))

    for cause, _how, _build in CASES:
        assert cause not in ISA.CAUSES_AWAITING_FEATURE, (
            cause, "is exercised by a case and should not be exempted")
    for name, (what, _check) in ABSENT_PROOFS.items():
        print(f"    {name}: no case, because {what}")
    print(f"  C4 waiting list: all {len(ABSENT_PROOFS)} exemptions re-derived from the "
          f"machine, each of which fails the moment the feature lands")
    return len(ISA.CAUSES_AWAITING_FEATURE)

def test_c5_the_new_fields_join_the_width_sweep():

    assert "fault_reason" in _width_names() and "fault_addr" in _width_names(), (
        "the width sweep does not carry the fault fields")
    for cause, how, build in build_coverage_cases():
        for pc in PC_SITES:
            for who, runner in IMPLS:
                _raised, view, _out = runner(build, pc)
                for k in ("fault_reason", "fault_addr"):
                    v = view[k]
                    assert isinstance(v, int) and not isinstance(v, bool), (
                        who, cause, pc, k, "is not an integer:", type(v).__name__)
                    assert 0 <= v < (256 if k == "fault_reason" else 1 << 16), (
                        who, cause, pc, k, v)

    c = TorchCircuit(bytes([0x01]))
    assert c.fault_reason.dtype == torch.int32 and c.fault_addr.dtype == torch.int32
    t = TritonCircuit(bytes([0x01]))
    assert t.S.dtype == torch.int32 and int(t.S.numel()) == circuit_triton.STATE_ROWS
    b = TritonBatch(2)
    assert b.STATE.dtype == torch.int32 and b.STATE.shape[1] == circuit_triton.STATE_ROWS
    g = NCP8(bytes([0x71, 0x00]))
    try:
        g.step()
    except MachineError:
        pass
    assert isinstance(g.fault_reason, int) and isinstance(g.fault_addr, int)
    print(f"  C5 widths: fault_reason/fault_addr checked on every tick, all integer "
          f"state on all four paths")

def _width_names():
    import test_state_contract as T
    return set(T.WIDTHS)

def test_c6_reference_records_the_error_status_it_raises_for():

    prog = bytes([0x01, 0x71, 0x00])
    g = NCP8(prog)
    g.step()
    try:
        g.step()
    except MachineError as e:
        assert g.status == "ERROR" and STATUS_CODE[g.status] == 3, g.status
        assert g.fault_reason == CAUSE["BAD_OPCODE"] and g.fault_addr == 1, g.snapshot()
        assert "0x71" in str(e), ("the raise must still name the code point", str(e))
    else:
        raise AssertionError("the undefined opcode tick stopped raising")

    g.step()
    g.step()
    assert g.snapshot()["tick"] == 1 and g.PC == 1, g.snapshot()
    print("  C6 the faulting tick raises *and* records status 3 with its cause; later "
          "steps are the sticky no-op")

CHECKS = (
    test_c1_cause_table_is_covered_exhaustively,
    test_c2_bijection_holds_on_every_tick_of_every_case,
    test_c3_at_most_one_cause_per_tick,
    test_c4_absent_causes_are_absent_for_a_reason,
    test_c5_the_new_fields_join_the_width_sweep,
    test_c6_reference_records_the_error_status_it_raises_for,
)

def run_all():
    print("fault-register acceptance (cause table x boundary sweeps, four paths):")
    for fn in CHECKS:
        fn()
    print(f"fault-register acceptance: all {len(CHECKS)} checks passed")

SABOTAGES = (

    ("a cause recorded without status becoming 3",
     "circuit_torch.py",
     "        new_status = torch.where(err > 0, 3 * torch.ones(1, dtype=i32, device=dev),",
     "        new_status = torch.where(err > 0, 0 * torch.ones(1, dtype=i32, device=dev),",
     "probe_pairing"),
    ("status 3 with the cause left at its reset value",
     "golden_sim.py",
     "            self.fault_reason = reason",
     "            self.fault_reason = CAUSE[\"OK\"]",
     "probe_pairing"),
    ("fault_addr taken after the PC advance",
     "golden_sim.py",
     "            self.fault_addr = pc0",
     "            self.fault_addr = (pc0 + 1) & 0xFFFF",
     "probe_addr"),
    ("two causes permitted to fire in one tick (their codes are summed)",
     "circuit_torch.py",
     "    first = fired * (prior == 0).to(torch.int32)",
     "    first = fired * 0 + fired",
     "probe_precedence"),

    ("one cause code produced by no case at all",
     "test_fault_registers.py",
     '    ("OUT_CAP", "OUT as byte number OUT_CAP+1", _c_out_cap),\n',
     "",
     "probe_coverage"),
)

PROBES = {
    "probe_pairing": "test_c2_bijection_holds_on_every_tick_of_every_case",
    "probe_addr": "test_c1_cause_table_is_covered_exhaustively",
    "probe_precedence": "test_c3_at_most_one_cause_per_tick",
    "probe_coverage": "test_c1_cause_table_is_covered_exhaustively",
}

_PROBE_SRC = """
import sys
sys.path.insert(0, {root!r})
import test_fault_registers as F
try:
    F.{fn}()
except AssertionError as e:
    print("SABOTAGE-CAUGHT:", e)
else:
    print("SABOTAGE-MISSED: the check passed")
"""

def sabotage(root=None):

    import os
    import shutil
    import tempfile
    root = os.path.dirname(os.path.abspath(__file__)) if root is None else root
    for label, fname, old, new, probe in SABOTAGES:
        work = tempfile.mkdtemp(prefix="fault-sabotage-")
        try:
            for keep in ("golden_sim.py", "circuit_torch.py", "circuit_triton.py",
                         "isa_table.py", "test_state_contract.py", "programs.py",
                         "test_fault_registers.py", "disasm.py"):
                shutil.copy(os.path.join(root, keep), work)
            src = os.path.join(work, fname)
            text = open(src, encoding="utf-8").read()
            if text.count(old) != 1:
                print(f"[{label}] the sabotage anchor is not unique in {fname}")
                continue
            open(src, "w", encoding="utf-8").write(text.replace(old, new))
            out = subprocess.run(
                [sys.executable, "-c", _PROBE_SRC.format(root=work, fn=PROBES[probe])],
                capture_output=True, text=True, cwd=work)
            printed = (out.stdout or out.stderr).strip().splitlines()
            print(f"\n[{label}]  ({fname}, {probe} -> {PROBES[probe]})")
            for line in printed[-6:]:
                print("   ", line[:400])
        finally:
            shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    if "--sabotage" in sys.argv:
        sabotage()
    else:
        run_all()