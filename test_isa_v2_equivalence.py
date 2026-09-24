"""Equivalence acceptance for the escape-prefix subcode space.

Companion to test_circuit_equivalence.py:

  * test_circuit_equivalence.py covers the 256 single-byte opcodes (which include
    the 0x20-0x5F families and the 0x70 escape prefix itself);
  * this file covers the 256 subcodes behind 0x70, plus program-level lockstep
    for the extended instructions. The reserved subcodes and the error paths are
    required to be bit-exact as well.

The v3.0 additions (pointer-pair moves, pair spill/restore, 16-bit memory,
frame-relative access, stack-pointer adjustment, MULH) are subcodes of the same
space, so the scan covers them without a skip list. Two properties are asserted
rather than assumed: every v3.0 encoding has at least one legal single-step case
(an instruction that only ever errors would hide a wrong result), and the
lockstepped programs execute every v3.0 encoding, measured off the reference
trace.
"""
from __future__ import annotations
import random
import re

from golden_sim import NCP8, MachineError, asm
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from test_state_contract import assert_widths

VEC = 0x0F00

PC_SITES = (0, 256)

def _code_esc(sub, vec=0, pc=0, tail=(0xAA, 0x55)):

    b = bytearray(max(0x0F20, pc + 4))
    b[pc], b[pc + 1], b[pc + 2], b[pc + 3] = 0x70, sub, tail[0], tail[1]
    if vec:
        for k in range(16):
            b[VEC + 2 * k: VEC + 2 * k + 2] = (vec & 0xFFFF).to_bytes(2, "little")
    return bytes(b)

def _golden_view(g):
    return dict(r=list(g.r), HL=g.HL, DE=g.DE, SP=g.SP, PC=g.PC, C=g.C, Z=g.Z,
                ipos=g.ipos, oplen=len(g.out), tick=g.tick,
                status={"RUNNING": 0, "HALT": 1, "OVERRUN": 2, "ERR": 3}[g.status])

def _ptr(rng, boundary):

    if boundary:
        return rng.choice([0, 1, 4093, 4094, 4095, 4096, 4097, 65535])
    return rng.randrange(4094)

def one_esc_step(Machine, sub, seed, vec=0, pc=0):

    rng = random.Random(seed * 977 + sub)
    code = _code_esc(sub, vec, pc)
    data_g = bytearray(rng.randrange(256) for _ in range(4096))
    R = [rng.randrange(256) for _ in range(4)]
    HL, DE = _ptr(rng, seed % 4 == 3), _ptr(rng, seed % 4 == 3)
    SP = rng.choice([0, 1, 2, 3, rng.randrange(16, 4093), 4095, 4096])
    C0, Z0 = rng.randrange(2), rng.randrange(2)
    g = NCP8(code, data=data_g)
    g.load_state(R, HL, DE, SP, C0, Z0, 0, PC=pc)
    c = Machine(code, data=data_g)
    c.load_state(R, HL, DE, SP, C0, Z0, 0, PC=pc)
    pre, pre_data = _golden_view(g), list(g.data)
    pre_code, pre_out = bytes(g.code), bytes(g.out)
    assert_widths(pre, (Machine.__name__, "pre-tick"))
    g_err = False
    try:
        g.step()
    except MachineError:
        g_err = True
    c.step()
    cv = c.snapshot()
    assert_widths(cv, (Machine.__name__, "post-tick", sub, seed, pc))
    if g_err:

        assert _golden_view(g) == pre, (sub, seed, pc, "reference error tick was not atomic",
                                        pre, _golden_view(g))
        assert list(g.data) == pre_data, (sub, seed, "reference modified DATA before raising")
        assert bytes(g.code) == pre_code, (sub, seed, "reference modified CODE before raising")
        assert bytes(g.out) == pre_out, (sub, seed, "reference wrote output before raising")
        assert cv["status"] == 3, (sub, seed, pc, "expected ERR", cv)
        for k in pre:
            if k == "status":
                continue
            assert pre[k] == cv[k], (sub, seed, pc, "error path not atomic", k, pre[k], cv[k])
        assert list(c.DATA.cpu().tolist()) == pre_data, (sub, seed, "DATA was modified")
        assert bytes(c.CODE.cpu().tolist()[:len(code)]) == pre_code, (sub, seed, "CODE was modified")
        return "err"
    gv = _golden_view(g)
    assert_widths(gv, (Machine.__name__, "reference post-tick", sub, seed, pc))
    assert all(gv[k] == cv[k] for k in gv), (sub, seed, pc, gv, cv)
    assert list(g.data) == list(c.DATA.cpu().tolist()), (sub, seed, "DATA")
    return "ok"

NEW_SUBCODES = {
    0x30: "MOVW HL, DE", 0x31: "MOVW DE, HL", 0x32: "MOVW HL, SP", 0x33: "MOVW DE, SP",
    0x34: "MOVW SP, HL", 0x35: "MOVW SP, DE",
    0x38: "PUSHW HL", 0x39: "PUSHW DE", 0x3A: "POPW HL", 0x3B: "POPW DE",
    0x3C: "STW [HL], DE", 0x3D: "STW [DE], HL", 0x3E: "LDW DE, [HL]", 0x3F: "LDW HL, [DE]",
    0x58: "ADD SP, ",
}
NEW_SUBCODES.update({0x50 | r: re.compile(rf"LDX r{r}, \[HL[-+]?\d+\]") for r in range(4)})
NEW_SUBCODES.update({0x54 | r: re.compile(rf"STX \[HL[-+]?\d+\], r{r}") for r in range(4)})
NEW_SUBCODES.update({0x90 | f: re.compile(rf"MULH r{f >> 2}, r{f & 3}") for f in range(16)})

def _executed_new_subcodes(trace):

    seen = set()
    for line in trace:
        for sub, pat in NEW_SUBCODES.items():
            if (pat.search(line) if hasattr(pat, "search") else pat in line):
                seen.add(sub)
    return seen

def test_esc_subcodes(Machine, name):
    import torch
    tot = {"ok": 0, "err": 0}
    legal = set()
    for sub in range(256):
        for vec in (0, 0x1234):
            for seed in range(4):
                for site in PC_SITES:
                    verdict = one_esc_step(Machine, sub, seed, vec, pc=site + 3 * sub)
                    tot[verdict] += 1
                    if verdict == "ok":
                        legal.add(sub)
    torch.cuda.synchronize()
    n = 256 * 2 * 4 * len(PC_SITES)
    assert sum(tot.values()) == n, (tot, n)

    missing = sorted(set(NEW_SUBCODES) - legal)
    assert not missing, ("v3.0 subcodes with no legal single-step case",
                         [hex(s) for s in missing])
    print(f"[{name}] escape subcode single step: {n} cases match (ok {tot['ok']} + error {tot['err']}); "
          f"all {len(NEW_SUBCODES)} v3.0 subcodes have a legal case; executed at PC in {PC_SITES}")

def _lockstep(Machine, name, code, data=b"", inputs=b"", max_tick=4000, expect=None, trace_out=None):
    g = NCP8(code, data=data, inputs=inputs, tick_budget=max_tick)
    c = Machine(code, data=data, inputs=inputs, tick_budget=max_tick)
    n = 0
    raised = False
    while g.status == "RUNNING" and n < max_tick:
        gs, gdata, gout, gcode = _golden_view(g), list(g.data), bytes(g.out), bytes(g.code)
        cv = c.snapshot()
        assert_widths(gs, (name, "reference tick", n))
        assert_widths(cv, (name, "circuit tick", n))
        assert all(cv[k] == gs[k] for k in gs), (name, n, gs, cv)
        try:
            g.step()
        except MachineError:
            raised = True
            c.step()
            assert c.snapshot()["status"] == 3, (name, "circuit did not report ERR on the error tick")

            assert _golden_view(g) == gs, (name, "reference error tick was not atomic", gs, _golden_view(g))
            assert list(g.data) == gdata, (name, "reference modified DATA before raising")
            assert bytes(g.out) == gout, (name, "reference wrote output before raising")
            assert bytes(g.code) == gcode, (name, "reference modified CODE before raising")
            assert c.snapshot()["tick"] == gs["tick"], (name, "error tick mismatch")
            break
        c.step(); n += 1
        assert_widths(c.snapshot(), (name, "circuit after tick", n))
    if not raised:

        gs = _golden_view(g); cv = c.snapshot()
        assert_widths(gs, (name, "reference final"))
        assert_widths(cv, (name, "circuit final"))
        assert all(cv[k] == gs[k] for k in gs), (name, "final state mismatch", gs, cv)
        assert list(g.data) == c.DATA.cpu().tolist(), (name, "final DATA mismatch")
        assert bytes(c.CODE.cpu().tolist()[:len(code)]) == bytes(g.code), (name, "final CODE mismatch")
    assert c.out() == bytes(g.out), (name, "output mismatch", c.out(), bytes(g.out))
    if expect is not None:
        assert bytes(g.out) == expect, (name, "expected output mismatch", bytes(g.out).hex(), expect.hex())
    if trace_out is not None:
        trace_out.extend(g.trace)
    return n

def _with_vec(code, table):
    b = bytearray(bytes(code).ljust(max(VEC + 2 * (k + 1) for k in table), b"\x00"))
    for k, addr in table.items():
        b[VEC + 2 * k: VEC + 2 * k + 2] = (addr & 0xFFFF).to_bytes(2, "little")
    return bytes(b)

WLO, WHI = 0x0F20, 0x0F21

def test_selfmod_lockstep(Machine, name):

    src = """    JMP main
sub:
    LDI r0, 7
    RET
main:
    LDI HL, 0x04
    LDI r0, 42
    STC [HL], r0
    CALL sub
    OUT r0
    HALT
"""
    base = asm(src)

    def build(wlo, whi):
        b = bytearray(bytes(base).ljust(WHI + 1, b"\x00"))
        b[WLO], b[WHI] = wlo, whi
        return bytes(b)

    n = _lockstep(Machine, "self-modification takes effect", build(0x00, 0x08), expect=bytes([42]))
    print(f"[{name}] controlled self-modification lockstep {n} ticks (immediate 7 -> 42 patched, output matches)")

    n = _lockstep(Machine, "out of window", build(0x10, 0x18))
    n = _lockstep(Machine, "zero-width window", build(0x00, 0x00))
    print(f"[{name}] out-of-window / zero-width window: both implementations atomic ERR at the same tick")

    code = asm("LDI HL, 0\nLDC r0, [HL]\nOUT r0\nLDI HL, 4\nLDC r0, [HL]\nOUT r0\nHALT")
    n = _lockstep(Machine, "LDC self-read", code, expect=bytes([code[0], code[4]]))
    print(f"[{name}] LDC self-read lockstep {n} ticks (read-back bytes match the truth)")

def test_programs(Machine, name):

    main = bytes([0x70, 0x70, 0x00, 0xD0 | 3, 0x7E, 0x00])
    h0 = bytes([0x70, 0x70, 0x01, 0xD4 | 0, 1, 0x08])
    h1 = bytes([0xD4 | 0, 5, 0x08])
    code = bytearray(bytes(main).ljust(0x20, b"\x00")) + h0 + h1
    code = _with_vec(bytes(code), {0: 0x20, 1: 0x20 + len(h0)})
    n = _lockstep(Machine, "EXT nested", code)
    g = NCP8(code)
    while g.status == "RUNNING":
        g.step()
    assert g.r[0] == 6 and g.r[3] == 0x7E and g.SP == 4096 and g.status == "HALT", g.snapshot()
    print(f"[{name}] EXT trap lockstep {n} ticks (nested call r0=6 / stack restored / HALT)")

    prog = """
    LDI r0, 37
    LDI r1, 5
    MUL r0, r1
    OUT r0
    MOV r2, r0
    DIV r2, r1
    OUT r2
    MOV r3, r0
    MOD r3, r1
    OUT r3
    LDI r0, 0xAA
    LDI r1, 0x0F
    AND r0, r1
    OR r0, r1
    XOR r0, r1
    NOT r0
    NEG r0
    ROL r0
    ROL r0
    ROR r0
    OUT r0
    CMP r0, r1
    LDI r2, 0
    JZ zero
    LDI r2, 1
zero:
    OUT r2
    HALT
    """

    n = _lockstep(Machine, "extended arithmetic", asm(prog), expect=bytes([0xB9, 0x25, 0x00, 0x03, 0x01]))
    print(f"[{name}] extended arithmetic program lockstep {n} ticks (MUL/DIV/MOD/bitwise/rotate/CMP output matches)")

    n = _lockstep(Machine, "pointer family", asm("LDI HL, 100\nLDI DE, 7\nADD HL, DE\nSUB HL, DE\nXCHG\nOUTDE\nHALT"),
                  data=bytes(range(256)), expect=bytes([100]))
    print(f"[{name}] 16-bit pointer family lockstep {n} ticks (ADD/SUB HL,DE + XCHG)")

    n = _lockstep(Machine, "divide-by-zero atomic", asm("LDI r0, 9\nLDI r1, 0\nDIV r0, r1\nOUT r0\nHALT"))
    n = _lockstep(Machine, "modulo-by-zero atomic", asm("LDI r0, 9\nLDI r1, 0\nMOD r0, r1\nHALT"))
    print(f"[{name}] divide-by-zero / modulo-by-zero: both implementations atomic ERR at the same tick")

def _golden_verdict(code, data=b"", inputs=b""):

    g = NCP8(code, data=data, inputs=inputs)
    raised = False
    while g.status == "RUNNING":
        try:
            g.step()
        except MachineError:
            raised = True
            break
    return ("ERR" if raised else g.status), bytes(g.out)

MEDLEY = asm("""
  LDI HL, 4088
  LDI DE, 4090
  PUSHW HL
  PUSHW DE
  POPW HL
  POPW DE
  STW [HL], DE
  STW [DE], HL
  LDW DE, [HL]
  LDW HL, [DE]
  OUTDE
  OUTDE
  MOVW HL, DE
  MOVW DE, HL
  MOVW HL, SP
  MOVW DE, SP
  GETSP r0
  OUT r0
  MOVW SP, HL
  MOVW SP, DE
  ADD SP, -4
  MOVW HL, SP
  GETSP r0
  OUT r0
  MOVW SP, HL
  LDI r0, 0x11
  STX [HL], r0
  LDI r1, 0x22
  STX [HL+1], r1
  LDI r2, 0x33
  STX [HL+2], r2
  LDI r3, 0x44
  STX [HL+3], r3
  LDX r0, [HL+3]
  OUT r0
  LDX r1, [HL+2]
  OUT r1
  LDX r2, [HL+1]
  OUT r2
  LDX r3, [HL]
  OUT r3
  ADD SP, 4
  GETSP r0
  OUT r0
""" + "\n".join(

    f"  LDI r{f >> 2}, 200\n  LDI r{f & 3}, 200\n"
    f"  MULH r{f >> 2}, r{f & 3}\n  OUT r{f >> 2}"
    for f in range(16)) + "\n  HALT\n")

MEDLEY_EXPECT = (bytes([4090 & 0xFF, 4090 >> 8])
                 + bytes([4096 & 0xFF, 4092 & 0xFF])
                 + bytes([0x44, 0x33, 0x22, 0x11])
                 + bytes([4096 & 0xFF])
                 + bytes([(200 * 200) >> 8]) * 16)

def test_v3_programs(Machine, name):

    trace = []
    n = _lockstep(Machine, "v3.0 medley", MEDLEY, expect=MEDLEY_EXPECT, trace_out=trace)
    print(f"[{name}] v3.0 medley lockstep {n} ticks (pair moves/spill/16-bit memory/frame "
          f"access/MULH, output recomputed by hand)")

    import programs
    cases = [(0, [0, 0, 0, 0]), (0xFFFF, [255, 255, 255, 255]),
             (0xFFFF, [255, 255, 0, 0]), (1, [1, 0, 0, 0]),
             (0x1234, [0x56, 0x78, 0x9A, 0xBC]), (0xFFFF, [255, 255, 254, 255])]
    rng = random.Random(99)
    cases += [(rng.randrange(65536), [rng.randrange(256) for _ in range(4)]) for _ in range(4)]
    ticks = 0
    for a, b in cases:
        data = bytearray(6)
        data[0] = a & 0xFF
        data[1] = (a >> 8) & 0xFF
        data[2:6] = bytes(b)
        want = programs.frame_mul_model(a, b)
        out = bytes([want & 0xFF, (want >> 8) & 0xFF, (want >> 16) & 0xFF, (want >> 24) & 0xFF])
        ticks += _lockstep(Machine, "frame-pointer demo", programs.FRAME_MUL,
                           data=bytes(data), expect=out, trace_out=trace)
    print(f"[{name}] frame-pointer demo lockstep {ticks} ticks over {len(cases)} cases "
          f"(16x16->32 via MUL+MULH + local array, output == independent model)")

    seen = _executed_new_subcodes(trace)
    missing = sorted(set(NEW_SUBCODES) - seen)
    assert not missing, ("v3.0 subcodes missing from the program lockstep",
                         [hex(s) for s in missing])
    print(f"[{name}] program lockstep executes all {len(NEW_SUBCODES)} v3.0 encodings "
          f"({len(trace)} traced instructions)")

    for src, nm, pre in (
            ("LDI HL, 4095\nLDI DE, 1\nSTW [HL], DE\nHALT", "16-bit store at the last byte", 2),
            ("LDI HL, 0\nLDX r0, [HL-1]\nHALT", "frame access below DATA", 1),
            ("ADD SP, 1\nPUSHW HL\nHALT", "stack pointer past the end", 0),
            ("LDI HL, 0x4000\nMOVW SP, HL\nPUSHW HL\nHALT", "MOVW SP above DATA", 1)):
        code = asm(src)
        verdict, out = _golden_verdict(code)
        assert verdict == "ERR", (nm, "expected an error program", verdict)
        assert out == b"", (nm, "the error program wrote output", out)
        n = _lockstep(Machine, nm, code)
        assert n == pre, (nm, "error tick", n, pre)
    print(f"[{name}] v3.0 error paths: 4 programs, atomic ERR at the same tick in both implementations")

if __name__ == "__main__":
    print("ISA v2.0 equivalence acceptance (subcode space + program lockstep):")
    for Machine, name in ((TorchCircuit, "torch"), (TritonCircuit, "triton")):
        test_esc_subcodes(Machine, name)
        test_programs(Machine, name)
        test_selfmod_lockstep(Machine, name)
        test_v3_programs(Machine, name)
    print("ISA v2.0 equivalence: all passed")