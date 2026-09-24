"""Recursion stress acceptance.

Covers multiply (loop inside a subroutine), three-level nested CALL/RET
(return-address stack discipline) and unbounded recursion, where all
implementations must report the same error at the same tick with identical
state up to that tick.
"""
import random

import programs
from golden_sim import NCP8
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from test_circuit_equivalence import golden_view, lockstep

def _both(code, data, inputs=b""):
    return NCP8(code, data=data, inputs=inputs), TorchCircuit(code, data=data, inputs=inputs)

def test_mul():
    rng = random.Random(3)
    for _ in range(200):
        a, b = rng.randrange(256), rng.randrange(256)
        data = bytearray(2); data[0] = a; data[1] = b
        for Mach in (TorchCircuit, TritonCircuit):
            g = NCP8(programs.MUL, data=data)
            c = Mach(programs.MUL, data=data)
            lockstep(g, c)
        g, want = programs.run_mul(a, b)
        assert g == want, (a, b, g, want)
    print("multiply: 200 random pairs, reference == Python truth, both circuits lockstep")

def test_nested():
    for n in range(0, 125):
        data = bytearray(1); data[0] = n
        for Mach in (TorchCircuit, TritonCircuit):
            g = NCP8(programs.NESTED, data=data)
            c = Mach(programs.NESTED, data=data)
            lockstep(g, c)
        got, want = programs.run_nested(n)
        assert got == want, (n, got, want)
    print("three-level nested CALL/RET: stack depth up to 124, all bit-exact")

def test_overflow_atomic():

    for Mach in (TorchCircuit, TritonCircuit):
        g = NCP8(programs.OVERFLOW, data=bytearray(4))
        c = Mach(programs.OVERFLOW, data=bytearray(4))

        n = 0
        raised = False
        while True:
            g_running = g.status == "RUNNING" and g.tick < g.tb
            c_running = int(c.status.item()) == 0
            assert g_running == c_running, (n, "run/halt verdict diverged", g.status, int(c.status.item()))
            if not g_running:
                break
            pre, pre_data, pre_out = golden_view(g), list(g.data), bytes(g.out)

            try:
                g.step(); g_err = False
            except Exception:
                g_err = True
            c.step()
            n += 1
            if g_err:
                raised = True
                assert int(c.status.item()) == 3, (n, "reference overflowed but the circuit did not report ERR")

                assert golden_view(g) == pre, (n, "reference error tick was not atomic", pre, golden_view(g))
                assert list(g.data) == pre_data, (n, "reference modified DATA before raising")
                assert bytes(g.out) == pre_out, (n, "reference wrote output before raising")
                assert g.SP == 0, (n, "the failure must happen exactly on the last free slot", g.SP)
                break
            assert golden_view(g) == c.snapshot(), (n, "state diverged before the overflow")
        assert raised, "the reference never raised the stack error"
        print(f"  stack overflow captured at tick {n}: violating tick atomic in both implementations")
    print("stack overflow capture: violating tick atomic and identical (no silent wraparound)")

if __name__ == "__main__":
    test_mul()
    test_nested()
    test_overflow_atomic()
    print("\nrecursion stress: all passed")