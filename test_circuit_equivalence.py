"""Equivalence acceptance: both circuit implementations against the reference.

Test 1 (full opcode enumeration): all 256 opcodes x 6 random/boundary states,
single step, comparing the complete snapshot, the whole DATA array and the
output stream; error paths (undefined opcodes, out-of-range accesses, stack
bounds) are included.

Test 2 (program lockstep): the bundled programs are stepped tick by tick in all
three implementations.
"""
import random

from golden_sim import NCP8, MachineError
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from test_state_contract import FAULT_WRITES, assert_widths, circuit_view, ref_view

PC_SITES = (0, 256)

def golden_view(g):

    return ref_view(g)

def one_step_agreement(Machine, op, seed, pc=0):

    rng = random.Random(seed)
    code = bytes(pc) + bytes([op, rng.randrange(256), rng.randrange(256)])
    data_g = bytearray(rng.randrange(256) for _ in range(4096))
    inputs = bytes(rng.randrange(256) for _ in range(3))
    R = [rng.randrange(256) for _ in range(4)]
    HL = rng.choice([rng.randrange(4096), rng.randrange(4096, 4400)])
    DE = rng.choice([rng.randrange(4096), rng.randrange(4096, 4400)])
    SP = rng.choice([0, 1, 2, 3, rng.randrange(16, 4093), 4095, 4096])
    C0, Z0 = rng.randrange(2), rng.randrange(2)
    tick0 = rng.randrange(100)

    g = NCP8(code, data=data_g, inputs=inputs)
    g.load_state(R, HL, DE, SP, C0, Z0, tick0, PC=pc)
    c = Machine(code, data=data_g, inputs=inputs)
    c.load_state(R, HL, DE, SP, C0, Z0, tick0, PC=pc)

    pre_g = golden_view(g); pre_data = list(g.data)
    pre_code, pre_out = bytes(g.code), bytes(g.out)
    assert_widths(pre_g, (Machine.__name__, "pre-tick"))
    g_err = False
    try:
        g.step()
    except MachineError:
        g_err = True
    c.step()
    assert_widths(c.snapshot(), (Machine.__name__, "post-tick", op, seed, pc))

    if g_err:

        gv = golden_view(g)
        assert gv["status"] == 3, (op, seed, "the reference left no error status", gv)
        assert gv["fault_reason"] != 0, (op, seed, "the reference stopped without a cause", gv)
        assert gv["fault_addr"] == pre_g["PC"], (
            op, seed, "fault_addr is not the faulting instruction", pre_g["PC"], gv)
        for k in pre_g:
            if k in FAULT_WRITES:
                continue
            assert gv[k] == pre_g[k], (op, seed, "reference error tick was not atomic",
                                       k, pre_g[k], gv[k])
        assert list(g.data) == pre_data, (op, seed, "reference modified DATA before raising")
        assert bytes(g.code) == pre_code, (op, seed, "reference modified CODE before raising")
        assert bytes(g.out) == pre_out, (op, seed, "reference wrote output before raising")
        cv = circuit_view(c)
        assert cv == gv, (op, seed, pc, "the circuit's fault tick differs from the reference",
                          gv, cv)
        assert list(c.DATA.cpu().tolist()) == pre_data, (op, seed, "DATA was modified")
        assert bytes(c.CODE.cpu().tolist()[:len(code)]) == pre_code, (op, seed, "CODE")
        assert c.out() == bytes(g.out), (op, seed, "out")
        return "err"
    else:
        cv = circuit_view(c)
        assert_widths(golden_view(g), (Machine.__name__, "reference post-tick", op, seed, pc))
        assert golden_view(g) == cv, (op, seed, pc, golden_view(g), cv)
        assert list(g.data) == list(c.DATA.cpu().tolist()), (op, seed, "DATA")
        assert c.out() == bytes(g.out), (op, seed, "out")
        return "ok"

def test_all_opcodes(Machine, name):

    import torch
    total = {"ok": 0, "err": 0}
    ops = list(range(256))
    for op in ops:
        for seed in range(6):
            for site in PC_SITES:
                pc = site + op if site else 0
                total[one_step_agreement(Machine, op, seed, pc)] += 1
    torch.cuda.synchronize()
    n = len(ops) * 6 * len(PC_SITES)
    assert n == 3072, n
    assert min(PC_SITES) == 0 and max(PC_SITES) >= 256, PC_SITES
    print(f"[{name}] all-opcode single step: {n} cases match (ok {total['ok']} + error {total['err']})  ")
    print(f"        single step executed at PC in {PC_SITES} (wider than 8-bit sources"
          f" are only visible away from address 0): {n // len(PC_SITES)} cases per site")

def lockstep(g, c):
    n = 0
    while g.status == "RUNNING" and g.tick < g.tb and int(c.status.item()) == 0:
        g.step(); c.step(); n += 1
        gv = golden_view(g)
        assert_widths(gv, ("reference lockstep tick", n))
        assert_widths(c.snapshot(), (type(c).__name__, "lockstep tick", n))
        assert gv == c.snapshot(), n
        assert list(g.data) == list(c.DATA.cpu().tolist()), n
        assert c.out() == bytes(g.out), n
    assert g.status == "HALT", g.status
    assert int(c.status.item()) == 1
    assert c.out() == bytes(g.out)
    return n

def lockstep(g, c):
    n = 0
    while g.status == "RUNNING" and g.tick < g.tb and int(c.status.item()) == 0:
        g.step(); c.step(); n += 1
        assert golden_view(g) == c.snapshot(), n
        assert list(g.data) == list(c.DATA.cpu().tolist()), n
        assert c.out() == bytes(g.out), n
    assert g.status == "HALT", g.status
    assert int(c.status.item()) == 1
    assert c.out() == bytes(g.out)
    return n

def test_programs(Machine, name):
    import programs
    rng = random.Random(7)
    t = 0
    for _ in range(30):
        nbits = rng.randint(1, 12) * 8
        a, b = rng.getrandbits(nbits), rng.getrandbits(nbits)
        n = max((a.bit_length() + 7) // 8, (b.bit_length() + 7) // 8, 1)
        data = bytearray(1 + 3 * n); data[0] = n
        for i in range(n):
            data[1 + i] = (a >> (8 * i)) & 0xFF
            data[1 + n + i] = (b >> (8 * i)) & 0xFF
        t += lockstep(NCP8(programs.LONG_ADD, data=data), Machine(programs.LONG_ADD, data=data))
    print(f"[{name}] long_add lockstep: 30 cases {t} tick ")
    t = 0
    for k in range(14):
        data = bytearray(13); data[0] = k
        t += lockstep(NCP8(programs.FIB, data=data), Machine(programs.FIB, data=data))
    print(f"[{name}] fibonacci lockstep: F(0..13) {t} tick ")
    t = 0
    for n in [0, 1, 5, 30, 128]:
        data = bytearray(n + 16); data[0] = n
        t += lockstep(NCP8(programs.SUMREC, data=data), Machine(programs.SUMREC, data=data))
    print(f"[{name}] sumrec lockstep: 5 cases {t} tick ")

if __name__ == "__main__":
    test_all_opcodes(TorchCircuit, "torch")
    test_programs(TorchCircuit, "torch")
    test_all_opcodes(TritonCircuit, "triton")
    test_programs(TritonCircuit, "triton")
    print("\ndual-circuit equivalence: all passed")