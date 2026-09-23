"""Equivalence acceptance for the escape-prefix subcode space.

Companion to test_circuit_equivalence.py:

  * test_circuit_equivalence.py covers the 256 single-byte opcodes (which include
    the 0x20-0x5F families and the 0x70 escape prefix itself);
  * this file covers the 256 subcodes behind 0x70, plus program-level lockstep
    for the extended instructions. The reserved subcodes and the error paths are
    required to be bit-exact as well.
"""
from __future__ import annotations
import random

from golden_sim import NCP8, MachineError, asm
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit

VEC = 0x0F00


def _code_esc(sub, vec=0, tail=(0xAA, 0x55)):

    b = bytearray(0x0F20)
    b[0], b[1], b[2], b[3] = 0x70, sub, tail[0], tail[1]
    if vec:
        for k in range(16):
            b[VEC + 2 * k: VEC + 2 * k + 2] = (vec & 0xFFFF).to_bytes(2, "little")
    return bytes(b)


def _golden_view(g):
    return dict(r=list(g.r), HL=g.HL, DE=g.DE, SP=g.SP, PC=g.PC, C=g.C, Z=g.Z,
                ipos=g.ipos, tick=g.tick,
                status={"RUNNING": 0, "HALT": 1, "OVERRUN": 2, "ERR": 3}[g.status])


def one_esc_step(Machine, sub, seed, vec=0):
    rng = random.Random(seed * 977 + sub)
    code = _code_esc(sub, vec)
    data_g = bytearray(rng.randrange(256) for _ in range(4096))
    R = [rng.randrange(256) for _ in range(4)]
    HL, DE = rng.randrange(4096), rng.randrange(4096)
    SP = rng.choice([0, 1, 2, 3, rng.randrange(16, 4093), 4095, 4096])
    C0, Z0 = rng.randrange(2), rng.randrange(2)
    g = NCP8(code, data=data_g)
    g.r, g.HL, g.DE, g.SP, g.C, g.Z = list(R), HL, DE, SP, C0, Z0
    c = Machine(code, data=data_g)
    c.load_state(R, HL, DE, SP, C0, Z0, 0)
    pre, pre_data = _golden_view(g), list(g.data)
    g_err = False
    try:
        g.step()
    except MachineError:
        g_err = True
    c.step()
    cv = c.snapshot()
    if g_err:
        assert cv["status"] == 3, (sub, seed, "expected ERR", cv)
        for k in pre:
            if k == "status":
                continue
            assert pre[k] == cv[k], (sub, seed, "error path not atomic", k, pre[k], cv[k])
        assert list(c.DATA.cpu().tolist()) == pre_data, (sub, seed, "DATA was modified")
        return "err"
    gv = _golden_view(g)
    assert all(gv[k] == cv[k] for k in gv), (sub, seed, gv, cv)
    assert list(g.data) == list(c.DATA.cpu().tolist()), (sub, seed, "DATA")
    return "ok"


def test_esc_subcodes(Machine, name):
    import torch
    tot = {"ok": 0, "err": 0}
    for sub in range(256):
        for vec in (0, 0x1234):
            for seed in range(4):
                tot[one_esc_step(Machine, sub, seed, vec)] += 1
    torch.cuda.synchronize()
    print(f"[{name}] escape subcode single step: 2048 cases match (ok {tot['ok']} + error {tot['err']})")


def _lockstep(Machine, name, code, data=b"", inputs=b"", max_tick=4000, expect=None):
    g = NCP8(code, data=data, inputs=inputs, tick_budget=max_tick)
    c = Machine(code, data=data, inputs=inputs, tick_budget=max_tick)
    n = 0
    while g.status == "RUNNING" and n < max_tick:
        gs = _golden_view(g); cv = c.snapshot()
        assert all(cv[k] == gs[k] for k in ("r", "HL", "DE", "SP", "PC", "C", "Z", "ipos", "tick")), (name, n, gs, cv)
        try:
            g.step()
        except MachineError:
            c.step()
            assert c.snapshot()["status"] == 3 and c.snapshot()["tick"] == gs["tick"], (name, "error tick mismatch")
            break
        c.step(); n += 1
    assert c.out() == bytes(g.out), (name, "output mismatch", c.out(), bytes(g.out))
    if expect is not None:
        assert bytes(g.out) == expect, (name, "expected output mismatch", bytes(g.out).hex(), expect.hex())
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


if __name__ == "__main__":
    print("ISA v2.0 equivalence acceptance (subcode space + program lockstep):")
    for Machine, name in ((TorchCircuit, "torch"), (TritonCircuit, "triton")):
        test_esc_subcodes(Machine, name)
        test_programs(Machine, name)
        test_selfmod_lockstep(Machine, name)
    print("ISA v2.0 equivalence: all passed")