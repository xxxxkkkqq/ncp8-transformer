"""Programs that read the machine's own state through ordinary instructions.

GETPC/GETSP/GETF return the executing instruction's own address, the stack
pointer and the packed flags. The programs sample these values while running and
write them to the output stream; acceptance compares that output against the
reference simulator's internal trace. The trace is used for comparison only and
never participates in computation.
"""
from golden_sim import NCP8, asm
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit


def golden_trace_field(g, mnemonic_prefix, field):


    return None


def run_and_capture(code, data=b""):

    g = NCP8(code, data=data)
    steps = []
    while g.status == "RUNNING":
        pre = (g.PC, list(g.r), g.SP, g.C, g.Z)
        g.step()
        if len(g.trace) > len(steps):
            steps.append(pre)
    return bytes(g.out), steps



PCPROOF = asm("""
  GETPC r0
  OUT r0
  GETPC r1
  OUT r1
  NOP
  GETPC r2
  OUT r2
  HALT
""")

def test_pc_proof():
    out, steps = run_and_capture(PCPROOF)


    getpc_addrs = []
    for i, line in enumerate(_trace_of(PCPROOF)):
        if "GETPC" in line:
            getpc_addrs.append(int(line.split()[1], 16) & 255)
    assert list(out) == getpc_addrs, (out.hex(), [hex(a) for a in getpc_addrs])

    assert len(set(getpc_addrs)) == 3
    assert getpc_addrs[0] < getpc_addrs[1] < getpc_addrs[2]
    print(f"L4 PC self-proof: machine reads its own PC {[hex(a) for a in getpc_addrs]} = trace bit-exact ")


def _trace_of(code):
    g = NCP8(code, data=b"")
    g.run()
    return g.trace



SPPROOF = asm("""
  LDI r0, 17
  PUSH r0
  GETSP r1
  OUT r1
  PUSH r0
  GETSP r1
  OUT r1
  POP r2
  POP r3
  GETSP r1
  OUT r1
  HALT
""")

def test_sp_proof():
    tr = _trace_of(SPPROOF)


    g = NCP8(SPPROOF)
    got = []
    while g.status == "RUNNING":
        pc0 = g.PC
        (op,) = g.code[pc0:pc0 + 1] if pc0 < len(g.code) else (0xff,)
        is_getsp = (op & 0xFC) == 0x18
        pre_sp = g.SP & 255
        g.step()
        if is_getsp:
            got.append(pre_sp)
    out = bytes(g.out)
    assert list(out) == got, (out.hex(), [hex(x) for x in got])

    assert got == [255, 254, 0], got
    print(f"L4 SP self-proof: PUSH/POP stack displacement read back {got} (255=SP4095,254=SP4094,0=back to 4096) = internal SP bit-exact ")



FPROOF = asm("""
  CLC
  LDI r0, 0
  TST r0
  GETF r1
  OUT r1
  LDI r0, 255
  ADDI r0, 1
  GETF r2
  OUT r2
  LDI r3, 5
  SUBI r3, 9
  GETF r3
  OUT r3
  HALT
""")

def test_flag_proof():
    g = NCP8(FPROOF)
    got = []
    while g.status == "RUNNING":
        pc0 = g.PC
        (op,) = g.code[pc0:pc0 + 1]
        is_getf = (op & 0xFC) == 0x1C
        pre = g.Z | (g.C << 1)
        g.step()
        if is_getf:
            got.append(pre)
    out = bytes(g.out)
    assert list(out) == got

    assert got == [1, 3, 2], got
    print(f"L4 flags self-proof: GETF read back {got} (Z/C combination) = internal flags bit-exact ")



def test_circuits_proof():
    from test_circuit_equivalence import lockstep
    for code, nm in [(PCPROOF, "PCPROOF"), (SPPROOF, "SPPROOF"), (FPROOF, "FPROOF")]:
        for Mach in (TorchCircuit, TritonCircuit):
            g = NCP8(code); c = Mach(code)
            lockstep(g, c)
    print("self-read programs: both circuits lockstep against the reference")


if __name__ == "__main__":
    test_pc_proof()
    test_sp_proof()
    test_flag_proof()
    test_circuits_proof()
    print("\nself-read traces: all passed")