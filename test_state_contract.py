"""State-contract acceptance for the reference and both circuits.

One check per contract the three implementations must hold identically: the
register write port truncates to the 8-bit register width and every state field
stays inside its declared width after every tick; a terminal status is sticky, so
stepping a stopped machine changes nothing and is not an error; the tick budget is
machine state checked per tick inside the datapath, so step() honours it and not
only run(); an illegal machine state cannot be constructed, every store address
is structurally inside its own machine, and a violating tick is atomic for any
exception; the output stream has one shared declared capacity and exceeding it is
an atomic error tick; and input validation raises rather than asserts, so it
survives python -O, with assembly errors kept out of the machine-error channel.

Run: python3 test_state_contract.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

import golden_sim
import programs
from circuit_torch import OUT_CAP as TORCH_OUT_CAP
from circuit_torch import TorchCircuit
from circuit_triton import DATA_SIZE, OUT_CAP as TRITON_OUT_CAP
from circuit_triton import TritonBatch, TritonCircuit, run_batch
from golden_sim import CODE_SIZE, DATA_SIZE as REF_DATA_SIZE
from golden_sim import AssemblyError, MachineError, NCP8, asm

REF_OUT_CAP = getattr(golden_sim, "OUT_CAP", None)
CAP = REF_OUT_CAP if REF_OUT_CAP else TRITON_OUT_CAP

STATUS_NAMES = {0: "RUNNING", 1: "HALT", 2: "OVERRUN", 3: "ERR"}

WIDTHS = {
    "r": (0, 256),
    "HL": (0, 1 << 16),
    "DE": (0, 1 << 16),
    "PC": (0, 1 << 16),
    "SP": (0, DATA_SIZE + 1),
    "C": (0, 2),
    "Z": (0, 2),
}

def ref_view(g):

    return dict(r=list(g.r), HL=g.HL, DE=g.DE, SP=g.SP, PC=g.PC, C=g.C, Z=g.Z,
                ipos=g.ipos, oplen=len(g.out), tick=g.tick,
                status={"RUNNING": 0, "HALT": 1, "OVERRUN": 2, "ERR": 3}[g.status])

def width_violations(view):

    bad = []
    for k, (lo, hi) in WIDTHS.items():
        v = view.get(k)
        if v is None:
            continue
        cells = list(enumerate(v)) if isinstance(v, (list, tuple)) else [(-1, v)]
        for i, x in cells:
            if not lo <= x < hi:
                bad.append((k if i < 0 else f"{k}[{i}]", x))
    return bad

def assert_widths(view, where):

    bad = width_violations(view)
    assert not bad, (where, "state field outside its declared width", bad, view)

def step_reference(g, where):

    g.step()
    assert_widths(ref_view(g), ("reference", where, g.tick))
    return ref_view(g)

def refuse(fn, *args, **kw):

    try:
        fn(*args, **kw)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None

def state_args(**over):

    a = dict(R=[0, 0, 0, 0], HL=0, DE=0, SP=DATA_SIZE, C=0, Z=0, tick=0)
    a.update(over)
    return [a["R"], a["HL"], a["DE"], a["SP"], a["C"], a["Z"], a["tick"]]

def getpc_at(pc0):

    b = bytearray(pc0 + 3)
    b[0], b[1], b[2] = 0x09, pc0 & 0xFF, pc0 >> 8
    b[pc0], b[pc0 + 1], b[pc0 + 2] = 0x14, 0x81, 0x00
    return bytes(b)

def test_d1_register_write_port_masks():
    for Mach in (TorchCircuit, TritonCircuit):
        code = getpc_at(300)
        g = NCP8(code)
        step_reference(g, "JMP 300")
        mid = step_reference(g, "GETPC r0 @ 300")
        end = step_reference(g, "ADD r0, r1")
        assert mid["r"][0] == 44 and mid["tick"] == 2, mid
        assert end["C"] == 0, ("the ADD must not inherit a carry from a truncated PC", end)
        c = Mach(code)
        c.step()
        c.step()
        cmid = c.snapshot()
        assert_widths(cmid, (Mach.__name__, "GETPC r0 @ 300"))
        assert cmid == mid, (Mach.__name__, "GETPC state", mid, cmid)
        c.step()
        cend = c.snapshot()
        assert_widths(cend, (Mach.__name__, "ADD r0, r1"))
        assert cend == end, (Mach.__name__, "carry after a truncated source", end, cend)
    print("  D1 register write port: GETPC at PC=300 commits r0=44 and the next ADD keeps"
          " C=0 on both circuits, matching the reference")

def test_d1_width_conformance_on_the_bundled_programs():

    high_pc = bytes([0x09, 0x2C, 0x01]) + b"\x00" * 297 + bytes([0x14, 0x00])
    cases = [(programs.MUL, bytes([200, 30])), (programs.FIB, bytes([13])),
             (programs.SUMREC, bytes([40])), (programs.LONG_ADD, bytes([8] + [7] * 24)),
             (programs.FRAME_MUL, bytes([0x34, 0x12, 1, 2, 3, 4])),
             (high_pc, b"")]
    ticks = 0
    for code, data in cases:
        for impl in ("reference", "torch", "triton"):
            if impl == "reference":
                m = NCP8(code, data=data)
                view = lambda: ref_view(m)
            else:
                m = (TorchCircuit if impl == "torch" else TritonCircuit)(code, data=data)
                view = m.snapshot
            while view()["status"] == 0:
                assert_widths(view(), (impl, "pre-tick"))
                m.step()
                ticks += 1
                assert_widths(view(), (impl, "post-tick"))
            assert view()["status"] == 1, (code[:4], view())
    print(f"  D1 width conformance: {ticks} ticks of the bundled programs keep every state"
          " field inside its declared width in all three implementations")

def test_d2_halt_is_sticky():
    code = asm("HALT\nNOP\nNOP\nHALT")
    g = NCP8(code)
    step_reference(g, "HALT")
    frozen = ref_view(g)
    for _ in range(3):
        step_reference(g, "step after halt")
        assert ref_view(g) == frozen, ("a step past halt must change nothing", frozen, ref_view(g))
    for Mach in (TorchCircuit, TritonCircuit):
        c = Mach(code)
        c.step()
        frozen = c.snapshot()
        assert frozen["status"] == 1, frozen
        for _ in range(3):
            c.step()
            assert c.snapshot() == frozen, (Mach.__name__, "step after halt moved state",
                                            frozen, c.snapshot())
    b = TritonBatch(1)
    b.set_program(0, code)
    b.run()
    before = b.snapshot(0)
    b.step(4)
    assert b.snapshot(0) == before, ("batch step past halt moved state", before, b.snapshot(0))
    print("  D2 terminal status is sticky: HALT then three more steps changes nothing in"
          " all three implementations and raises nothing")

def test_d2_overrun_is_sticky():
    code = asm("loop:\nNOP\nJMP loop\n")
    for Mach in (TorchCircuit, TritonCircuit):
        c = Mach(code, tick_budget=3)
        for _ in range(4):
            c.step()
        frozen = c.snapshot()
        assert frozen["status"] == 2, ("the budget must latch OVERRUN", frozen)
        for _ in range(3):
            c.step()
            assert c.snapshot() == frozen, (Mach.__name__, "step past OVERRUN", frozen,
                                            c.snapshot())
    print("  D2 OVERRUN is sticky: further steps leave the marked machine untouched")

def test_d2_run_on_a_stopped_machine_returns():

    code = asm("NOP\nHALT")
    for Mach in (TorchCircuit, TritonCircuit):
        c = Mach(code)
        c.run()
        snap = c.snapshot()
        assert c.run() == c.out(), Mach.__name__
        assert c.snapshot() == snap, (Mach.__name__, "run() on a stopped machine ticked",
                                      snap, c.snapshot())
    g = NCP8(code)
    g.run()
    assert g.run() == bytes(g.out) and g.tick == 2, g.snapshot()
    print("  D2 run() on a stopped machine returns immediately without ticking")

def test_d3_budget_is_per_step_state():
    code = asm("loop:\nNOP\nJMP loop\n")
    want = [(1, "RUNNING"), (2, "RUNNING"), (3, "RUNNING"), (3, "OVERRUN"), (3, "OVERRUN")]
    g = NCP8(code, tick_budget=3)
    seen = []
    for _ in range(5):
        v = step_reference(g, "budget")
        seen.append((v["tick"], STATUS_NAMES[v["status"]]))
    assert seen == want, ("reference", seen)
    for Mach in (TorchCircuit, TritonCircuit):
        c = Mach(code, tick_budget=3)
        seen = []
        for _ in range(5):
            c.step()
            s = c.snapshot()
            assert_widths(s, Mach.__name__)
            seen.append((s["tick"], STATUS_NAMES[s["status"]]))
        assert seen == want, (Mach.__name__, seen)
    b = TritonBatch(1)
    b.set_program(0, code)
    b.set_budget(0, 3)
    res = b.step(10)
    assert (res.ticks[0], res.status[0]) == (3, 2), ("batch step past budget", res)
    print("  D3 tick budget: step() past the budget latches OVERRUN and freezes the tick in"
          " the reference, both circuits and the batch step")

def test_d3_budget_outranks_the_error_of_its_own_tick():

    code = asm("DIV r0, r1\nHALT")
    for Mach in (TorchCircuit, TritonCircuit):
        c = Mach(code, tick_budget=0)
        c.step()
        assert c.snapshot()["status"] == 2, (Mach.__name__, c.snapshot())
        assert c.snapshot()["tick"] == 0, (Mach.__name__, c.snapshot())
    g = NCP8(code, tick_budget=0)
    g.step()
    assert g.status == "OVERRUN", g.status
    print("  D3 precedence: the budget is checked before the instruction executes")

def test_d4_reference_reports_illegal_sp_as_machine_error():
    code = asm("PUSH r0\nHALT")
    for sp in (4112, 5000, 65535):
        g = NCP8(code)
        g.SP = sp
        msg = refuse(g.step)
        assert msg is not None and "MachineError" in msg, (
            f"an out-of-range SP must raise MachineError, got {msg}")
        assert (g.SP, g.PC, g.tick) == (sp, 0, 0), ("error tick moved state", g.snapshot())
    print("  D4 reference stack accessors bound-check: an illegal SP raises MachineError"
          " and the tick stays atomic")

def test_d4_state_constructors_reject_illegal_states():

    cases = [("r[3]", dict(R=[0, 0, 0, 256]), "r[3]"), ("r[0]", dict(R=[256, 0, 0, 0]), "r[0]"),
             ("SP", dict(SP=DATA_SIZE + 1), str(DATA_SIZE + 1)),
             ("C", dict(C=2), "state field C"), ("Z", dict(Z=3), "state field Z")]
    for what, over, needle in cases:
        a = state_args(**over)
        for Mach in (TorchCircuit, TritonCircuit):
            msg = refuse(Mach(b"\x00").load_state, *a)
            assert msg is not None, (Mach.__name__, what, "load_state accepted an illegal state")
            assert needle in msg, (Mach.__name__, what, "message lacks the field", msg)
        msg = refuse(NCP8(b"\x00").load_state, *a)
        assert msg is not None, ("reference", what, "load_state accepted an illegal state")
        assert needle in msg, ("reference", what, "message lacks the field", msg)
    wide = [(4, "HL"), (5, "DE"), (6, "PC")]
    for i, field in wide:
        b = TritonBatch(2)
        b.set_program(0, b"\x00")
        b.set_program(1, b"\x00")
        msg = refuse(b.set_state, 1, **{field: 1 << 16})
        assert msg is not None, ("batch set_state", field, "accepted an illegal state")
        assert "machine 1" in msg and str(1 << 16) in msg, (field, msg)
        row = [0, 0, 0, 0, 0, 0, 0, DATA_SIZE, 0, 0, 0, 0, 0, 0]
        row[i] = 1 << 16
        msg = refuse(run_batch, [b"\x01", b"\x01"], states=[row[:], row[:]])
        assert msg is not None and "machine 0" in msg, ("run_batch(states=)", field, msg)
    b = TritonBatch(1)
    b.set_program(0, b"\x00")
    assert refuse(b.set_state, 0, r=(1, 2, 3, 4), HL=5, DE=6, PC=7, SP=8, C=1, Z=0,
                  ipos=1, oplen=CAP, tick=9, status=2) is None, \
        "the state constructor must accept every legal state"
    assert refuse(TorchCircuit(b"\x00").load_state, *state_args()) is None
    assert refuse(NCP8(b"\x00").load_state, *state_args()) is None
    print("  D4 state constructors: r/SP/C/Z/HL/DE/PC violations are refused with the"
          " offending index and value on the reference and both circuits")

def test_d4_store_address_stays_inside_its_own_machine():

    code = asm("PUSH r0\nHALT")
    b = TritonBatch(2)
    b.set_program(0, code, bytes(8))
    b.set_program(1, code, bytes(8))
    b.STATE[0, 0] = 0x33
    b.STATE[0, 7] = 4112

    col = 4111 % DATA_SIZE
    b.DATA[1, col] = 0xEE
    b.step(1)
    assert b.DATA[1, col].item() == 0xEE, (
        "machine 0 wrote outside its own DATA row", b.DATA[1, col].item())
    assert b.DATA[0, col].item() == 0x33, (
        "the masked store did not land inside machine 0", b.DATA[0, col].item())
    c = TritonCircuit(code, data=bytes(8))
    c.S[0] = 0x33
    c.S[7] = 4112
    c.step()
    assert int(c.DATA[col].item()) == 0x33, (
        "the single-machine store left this machine's DATA")
    print("  D4 store masking: an out-of-range SP cannot write outside its own row in the"
          " batch, and cannot leave DATA in the single-machine kernel")

def test_d4_store_masks_require_power_of_two_sizes():

    import circuit_triton as CT
    for name, size in (("CODE_SIZE", CT.CODE_SIZE), ("DATA_SIZE", CT.DATA_SIZE),
                       ("OUT_CAP", CT.OUT_CAP)):
        assert size > 0 and not (size & (size - 1)), (name, size, "not a power of two")
    refuse = []
    for bad in (0, -4096, 4000, 4095, 4097, 8191):
        try:
            CT._power_of_two_or_die("probe", bad)
        except ValueError:
            refuse.append(bad)
    assert len(refuse) == 6, ("the guard accepted non-power-of-two sizes", refuse)

def test_d4_step_rolls_back_on_any_exception():

    g = NCP8(asm("NOP\nNOP\nHALT"))
    g.trace = None
    msg = refuse(g.step)
    assert msg is not None and "AttributeError" in msg, ("the injected failure must propagate", msg)
    assert (g.PC, g.tick) == (0, 0), ("a tick that raised left state behind", g.PC, g.tick)
    print("  D4 atomicity: step() rolls back PC on any exception, not only MachineError")

def test_d4_image_guards_are_raises():
    msg = refuse(NCP8, bytes(CODE_SIZE + 1))
    assert msg is not None and str(CODE_SIZE) in msg, ("over-long CODE image accepted", msg)
    msg = refuse(NCP8, b"\x00", data=bytes(REF_DATA_SIZE + 1))
    assert msg is not None and str(REF_DATA_SIZE) in msg, ("over-long DATA image accepted", msg)
    b = TritonBatch(1)
    assert refuse(b.set_program, 0, bytes(CODE_SIZE + 1)) is not None
    assert refuse(b.set_program, 0, b"\x00", data=bytes(REF_DATA_SIZE + 1)) is not None
    assert refuse(TorchCircuit, bytes(CODE_SIZE + 1)) is not None
    assert refuse(TritonCircuit, bytes(CODE_SIZE + 1)) is not None
    assert refuse(TorchCircuit, b"\x00", data=bytes(REF_DATA_SIZE + 1)) is not None
    assert refuse(TritonCircuit, b"\x00", data=bytes(REF_DATA_SIZE + 1)) is not None
    print("  D4 image guards: an over-capacity CODE/DATA image is refused by every path")

def emitter(outer, inner, extra=1):

    src = ["  LDI r0, 0x5A", f"  LDI r3, {outer}", "outer:", f"  LDI r2, {inner}",
           "inner:", "  OUT r0", "  DJNZ r2, inner", "  DJNZ r3, outer"]
    src += ["  OUT r0"] * extra + ["  HALT"]
    return asm("\n".join(src))

def test_d5_one_capacity_constant():
    assert REF_OUT_CAP == TORCH_OUT_CAP == TRITON_OUT_CAP, (
        REF_OUT_CAP, TORCH_OUT_CAP, TRITON_OUT_CAP)
    assert REF_DATA_SIZE == DATA_SIZE == CODE_SIZE == 4096
    print(f"  D5 one output capacity constant shared by all three implementations: {CAP} bytes")

def test_d5_reference_stream_is_bounded():

    code = emitter(CAP // 128, 128)
    g = NCP8(code)
    raised = False
    while g.status == "RUNNING" and not raised:
        try:
            g.step()
        except MachineError:
            raised = True
    assert raised, f"the reference accepted {len(g.out)} bytes, past its declared capacity {CAP}"
    assert len(g.out) == CAP, ("output past capacity", len(g.out))
    assert_widths(ref_view(g), "reference at capacity")
    before = (len(g.out), g.tick)
    refuse(g.step)
    assert (len(g.out), g.tick) == before, "the error tick was not atomic"
    print(f"  D5 reference output stream refuses the byte past capacity, atomically at"
          f" exactly {CAP} bytes")

def test_d5_circuits_error_instead_of_truncating():

    code = asm("OUT r0\nOUT r0\nHALT")
    for Mach in (TorchCircuit, TritonCircuit):
        c = Mach(code)
        c.load_state(*state_args(R=[7, 0, 0, 0]))
        if Mach is TorchCircuit:
            c.oplen = torch.tensor([CAP - 1], dtype=torch.int32, device=c.dev)
        else:
            c.S[11] = CAP - 1
        c.step()
        s = c.snapshot()
        assert s["status"] == 0 and s["oplen"] == CAP, ("filling to capacity failed", s)
        assert len(c.out()) == CAP and c.out()[-1] == 7, ("last byte missing", s)
        c.step()
        s2 = c.snapshot()
        assert s2["status"] == 3, (Mach.__name__, "output past capacity reported as", s2)
        assert s2["oplen"] == CAP and s2["tick"] == 1, (
            Mach.__name__, "output overflow was not atomic", s2)
        assert len(c.out()) == CAP, (Mach.__name__, "output stream past capacity")
    print(f"  D5 circuits: the byte past capacity is an atomic error tick and the stream"
          f" stays at {CAP} bytes (was: silent truncation with oplen past the buffer)")

def test_d5_batch_and_resident_report_the_overflow():
    code = emitter(CAP // 128, 128)
    res = run_batch([code], [bytes(8)])
    assert res.status[0] == 3, ("batch overflow status", res.status)
    assert res.oplens[0] == CAP and len(res.outs[0]) == CAP, ("batch overflow length", res.oplens)
    c = TritonCircuit(code, data=bytes(8))
    out = c.run_resident()
    assert len(out) == CAP and c.snapshot()["status"] == 3, c.snapshot()
    g = NCP8(code)
    refuse(g.run)
    assert bytes(g.out) == out == res.outs[0], "the three overflow streams differ"
    print(f"  D5 end to end: reference, resident batch and run_resident() agree on the"
          f" {CAP}-byte stream and the error status")

UNDER_O = r"""
import sys
sys.path.insert(0, sys.argv[1])
from golden_sim import AssemblyError, NCP8, asm
out = []
for src in ("MOV r5, r0", "LDI r7, 3"):
    try:
        asm(src)
    except AssemblyError:
        out.append("assembly-error")
try:
    NCP8(b"\x00" * 4097)
except ValueError:
    out.append("code-size")
print(",".join(out) or "none")
"""

def test_d6_validation_survives_python_O():
    root = os.path.dirname(os.path.abspath(__file__))
    opt = subprocess.run([sys.executable, "-O", "-c", UNDER_O, root],
                         capture_output=True, text=True)
    assert opt.returncode == 0, opt.stderr
    want = "assembly-error,assembly-error,code-size"
    assert opt.stdout.strip() == want, ("python -O lost a validation", opt.stdout.strip())
    plain = subprocess.run([sys.executable, "-c", UNDER_O, root], capture_output=True, text=True)
    assert plain.stdout.strip() == want, ("a normal run differs", plain.stdout.strip())
    print("  D6 validation is a raise: python -O refuses a bad register operand and an"
          " over-capacity image exactly as a normal run does")

def test_d6_assembly_error_is_not_a_machine_error():
    assert not issubclass(AssemblyError, MachineError), (
        "AssemblyError must not be caught by except MachineError around a run")
    try:
        asm("FOO r0, r1")
    except MachineError:
        raise AssertionError("an assembly failure was caught as a machine error")
    except AssemblyError as e:
        assert e.line == 1 and "FOO" in str(e), str(e)
    else:
        raise AssertionError("an unknown mnemonic assembled silently")
    print("  D6 AssemblyError is decoupled from MachineError: a run's error handler cannot"
          " swallow a bad program text")

CHECKS = (
    test_d1_register_write_port_masks,
    test_d1_width_conformance_on_the_bundled_programs,
    test_d2_halt_is_sticky,
    test_d2_overrun_is_sticky,
    test_d2_run_on_a_stopped_machine_returns,
    test_d3_budget_is_per_step_state,
    test_d3_budget_outranks_the_error_of_its_own_tick,
    test_d4_reference_reports_illegal_sp_as_machine_error,
    test_d4_state_constructors_reject_illegal_states,
    test_d4_store_address_stays_inside_its_own_machine,
    test_d4_store_masks_require_power_of_two_sizes,
    test_d4_step_rolls_back_on_any_exception,
    test_d4_image_guards_are_raises,
    test_d5_one_capacity_constant,
    test_d5_reference_stream_is_bounded,
    test_d5_circuits_error_instead_of_truncating,
    test_d5_batch_and_resident_report_the_overflow,
    test_d6_validation_survives_python_O,
    test_d6_assembly_error_is_not_a_machine_error,
)

def run_all():
    print("state-contract acceptance:")
    for fn in CHECKS:
        fn()
    print(f"state-contract acceptance: all {len(CHECKS)} checks passed")

if __name__ == "__main__":
    run_all()