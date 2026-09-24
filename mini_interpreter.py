"""A 16-opcode mini ISA interpreter implemented in NCP-8 assembly.

The interpreter keeps its own machine state (program counter, accumulator, index
register, flags) in fixed DATA slots, dispatches over 16 handlers through a
comparison chain, and is verified against an independent Python model of the
mini ISA, plus tick-by-tick lockstep across implementations.
"""
import random

from golden_sim import NCP8, asm
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from test_circuit_equivalence import golden_view
from test_state_contract import assert_widths

S_MPC, S_A, S_F, S_C = 0, 1, 2, 3
PROG_BASE = 8
MEM_BASE = 128
HALFNAMES = {1, 2, 3, 4, 10, 11, 12}

INTERP = asm("""
fetch:
  LDI HL, 0
  MOV r0, [HL]
  LDI HL, 8
  ADDI HL, r0
  MOV r2, [HL]
  INC HL
  MOV r3, [HL]
  LDI HL, 0
  MOV r1, r0
  ADDI r1, 1
  MOV [HL], r1
  MOV r0, r2
  SUBI r0, 0
  JZ h_halt
  MOV r0, r2
  SUBI r0, 1
  JZ h_ldi
  MOV r0, r2
  SUBI r0, 2
  JZ h_ldci
  MOV r0, r2
  SUBI r0, 3
  JZ h_lda
  MOV r0, r2
  SUBI r0, 4
  JZ h_sta
  MOV r0, r2
  SUBI r0, 5
  JZ h_ldic
  MOV r0, r2
  SUBI r0, 6
  JZ h_stic
  MOV r0, r2
  SUBI r0, 7
  JZ h_addc
  MOV r0, r2
  SUBI r0, 8
  JZ h_subc
  MOV r0, r2
  SUBI r0, 9
  JZ h_cmpc
  MOV r0, r2
  SUBI r0, 10
  JZ h_jz
  MOV r0, r2
  SUBI r0, 11
  JZ h_jnz
  MOV r0, r2
  SUBI r0, 12
  JZ h_jc
  MOV r0, r2
  SUBI r0, 13
  JZ h_out
  MOV r0, r2
  SUBI r0, 14
  JZ h_incc
  MOV r0, r2
  SUBI r0, 15
  JZ h_swap
  LDI HL, 0
  LDI r0, 99
  MOV [HL], r0
  JMP fetch

h_advop:
  LDI HL, 0
  MOV r0, [HL]
  ADDI r0, 1
  MOV [HL], r0
  RET

h_halt:
  HALT

h_ldi:
  CALL h_advop
  LDI HL, 1
  MOV [HL], r3
  JMP fetch

h_ldci:
  CALL h_advop
  LDI HL, 3
  MOV [HL], r3
  JMP fetch

h_lda:
  CALL h_advop
  LDI HL, 128
  ADDI HL, r3
  MOV r0, [HL]
  LDI HL, 1
  MOV [HL], r0
  JMP fetch

h_sta:
  CALL h_advop
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 128
  ADDI HL, r3
  MOV [HL], r0
  JMP fetch

h_ldic:
  LDI HL, 3
  MOV r0, [HL]
  LDI HL, 128
  ADDI HL, r0
  MOV r1, [HL]
  LDI HL, 1
  MOV [HL], r1
  JMP fetch

h_stic:
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 3
  MOV r1, [HL]
  LDI HL, 128
  ADDI HL, r1
  MOV [HL], r0
  JMP fetch

h_addc:
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 3
  MOV r1, [HL]
  ADD r0, r1
  LDI HL, 1
  MOV [HL], r0
  GETF r2
  LDI HL, 2
  MOV [HL], r2
  JMP fetch

h_subc:
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 3
  MOV r1, [HL]
  SUB r0, r1
  LDI HL, 1
  MOV [HL], r0
  GETF r2
  LDI HL, 2
  MOV [HL], r2
  JMP fetch

h_cmpc:
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 3
  MOV r1, [HL]
  SUB r0, r1
  GETF r2
  LDI HL, 2
  MOV [HL], r2
  JMP fetch

h_jz:
  CALL h_advop
  LDI HL, 2
  MOV r0, [HL]
  SHR r0
  JNC h_jz_no
  LDI HL, 0
  MOV [HL], r3
h_jz_no:
  JMP fetch

h_jnz:
  CALL h_advop
  LDI HL, 2
  MOV r0, [HL]
  SHR r0
  JC h_jnz_no
  LDI HL, 0
  MOV [HL], r3
h_jnz_no:
  JMP fetch

h_jc:
  CALL h_advop
  LDI HL, 2
  MOV r0, [HL]
  SHR r0
  SHR r0
  JNC h_jc_no
  LDI HL, 0
  MOV [HL], r3
h_jc_no:
  JMP fetch

h_out:
  LDI HL, 1
  MOV r0, [HL]
  OUT r0
  JMP fetch

h_incc:
  LDI HL, 3
  MOV r0, [HL]
  ADDI r0, 1
  MOV [HL], r0
  JMP fetch

h_swap:
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 3
  MOV r1, [HL]
  MOV [HL], r0
  LDI HL, 1
  MOV [HL], r1
  JMP fetch
""")



def mini_sim(prog, mem_init, max_steps=20000):
    A = C = 0
    Fz = Fc = 0
    MPC = 0
    M = dict(mem_init)
    out = bytearray()
    for _ in range(max_steps):
        op = prog[MPC]
        arg = prog[MPC + 1] if op in HALFNAMES else 0
        nxt = MPC + (2 if op in HALFNAMES else 1)
        if op == 0:
            return bytes(out)
        if op == 1: A = arg
        elif op == 2: C = arg
        elif op == 3: A = M.get(arg, 0)
        elif op == 4: M[arg] = A
        elif op == 5: A = M.get(C, 0)
        elif op == 6: M[C] = A
        elif op == 7:
            t = A + C; A = t & 255; Fc = t >> 8; Fz = int(A == 0)
        elif op == 8:
            Fc = int(A < C); A = (A - C) & 255; Fz = int(A == 0)
        elif op == 9:
            Fc = int(A < C); Fz = int(((A - C) & 255) == 0)
        elif op == 10:
            if Fz: nxt = arg
        elif op == 11:
            if not Fz: nxt = arg
        elif op == 12:
            if Fc: nxt = arg
        elif op == 13: out.append(A)
        elif op == 14: C = (C + 1) & 255
        elif op == 15: A, C = C, A
        MPC = nxt
    raise ValueError("mini step limit exceeded")


def build_state(prog, mem_init):
    data = bytearray(4096)
    for i, b in enumerate(prog):
        data[PROG_BASE + i] = b
    for m, v in mem_init.items():
        data[MEM_BASE + m] = v
    return data




def mini_asm(lines):

    labels, addr = {}, 0
    items = []
    for ln in lines:
        if isinstance(ln, str):
            labels[ln[:-1]] = addr
            continue
        if len(ln) == 1:
            items.append((ln[0], None, 1)); addr += 1
        else:
            items.append((ln[0], ln[1], 2)); addr += 2
    out = bytearray()
    for op, arg, size in items:
        out.append(op)
        if arg is not None:
            out.append(labels[arg] if isinstance(arg, str) else arg)
    return bytes(out)


def prog_sum(vals):
    n = len(vals)
    """total = sum(values); 8-bit version wraps on overflow, matching the reference model"""
    lines = [
        "L:", (5,), (15,), (4, 101), (3, 100), (7,), (4, 100), (3, 101), (15,), (14,),
        (1, n), (9,), (10, "D"), (11, "L"), "D:", (3, 100), (13,), (0,),
    ]
    mem = {102: n}
    return list(lines), mem


def run_test():

    rng = random.Random(7)
    for trial in range(5):
        n = rng.randrange(3, 6)
        vals = [rng.randrange(1, 60) for _ in range(n)]
        lines, mem = prog_sum(vals)
        prog = mini_asm(lines)
        mem[0], mem[100] = 0, 0
        for i, v in enumerate(vals):
            mem[i] = v
        want = mini_sim(prog, mem)
        g = NCP8(INTERP, data=build_state(prog, mem))
        got = g.run()
        assert_widths(g.snapshot(), "mini interpreter sum")
        assert bytes(got) == want, (trial, vals, bytes(got), want)
    print("interpreter sum: 5 random arrays, interpreter output == Python mini reference byte-exact")


    carry_prog = mini_asm([
        (1, 200), (2, 100), (7,), (13,), (12, "E"), (1, 99), (13,), "E:", (0,)])
    want = mini_sim(carry_prog, {})
    assert want == bytes([44]), want
    g = NCP8(INTERP, data=build_state(carry_prog, {}))
    got = bytes(g.run())
    assert_widths(g.snapshot(), "mini interpreter carry")
    assert got == want, (got, want)
    nocarry = mini_asm([(1, 10), (2, 20), (7,), (13,), (12, "E"), (1, 99), (13,), "E:", (0,)])
    want2 = mini_sim(nocarry, {})
    g = NCP8(INTERP, data=build_state(nocarry, {}))
    assert bytes(g.run()) == want2, (want2,)
    assert_widths(g.snapshot(), "mini interpreter no-carry")
    print("interpreter carry branch: JC depends on mini flags, both paths match the reference")


    from test_circuit_equivalence import lockstep
    vals = [10, 20, 7, 33, 12]
    lines, mem = prog_sum(vals)
    prog = mini_asm(lines)
    mem = dict(mem); mem[0], mem[100] = 0, 0
    for i, v in enumerate(vals):
        mem[i] = v
    state = build_state(prog, mem)
    for Mach, nm in ((TorchCircuit, "torch"), (TritonCircuit, "triton")):
        g = NCP8(INTERP, data=bytes(state))
        c = Mach(INTERP, data=bytes(state))
        ticks = lockstep(g, c)
        print(f"L3 meta-circular-{nm} lockstep: {ticks} tick bit-exact ")
    print("\nmini interpreter: all passed")


if __name__ == "__main__":
    run_test()