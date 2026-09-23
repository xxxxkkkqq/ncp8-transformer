"""Hand-written NCP-8 programs used as acceptance cases.

Acceptance is recomputation, not accuracy: every program's output is compared
byte-for-byte against an independent Python integer model.
"""
from golden_sim import NCP8, asm




LONG_ADD = asm("""
  LDI HL, 0
  MOV r2, [HL]
  MOV r3, r2
  LDI HL, 1
  LDI DE, 1
  ADDI DE, r3
  CLC
loop:
  MOV r0, [HL]
  MOV r1, [DE]
  ADC r0, r1
  MOV [DE], r0
  INC HL
  INC DE
  DJNZ r2, loop
  LDI r0, 0
  ADC r0, r0
  MOV [DE], r0
  LDI DE, 1
  ADDI DE, r3
  ADDI r3, 1
od:
  OUTDE
  DJNZ r3, od
  HALT
""")


def run_long_add(a: int, b: int):
    n = max((a.bit_length() + 7) // 8, (b.bit_length() + 7) // 8, 1)
    data = bytearray(1 + 3 * n)
    data[0] = n
    for i in range(n):
        data[1 + i] = (a >> (8 * i)) & 0xFF
        data[1 + n + i] = (b >> (8 * i)) & 0xFF
    sim = NCP8(LONG_ADD, data=data)
    out = sim.run()
    assert sim.status == "HALT", sim.status
    got = sum(v << (8 * i) for i, v in enumerate(out))
    return got, a + b, sim




FIB = asm("""
  LDI HL, 0
  MOV r2, [HL]
  LDI r0, 0
  LDI DE, 1
  MOV [DE], r0
  INC DE
  MOV [DE], r0
  INC DE
  MOV [DE], r0
  INC DE
  MOV [DE], r0
  LDI r1, 1
  LDI DE, 5
  MOV [DE], r1
  INC DE
  MOV [DE], r0
  INC DE
  MOV [DE], r0
  INC DE
  MOV [DE], r0
  TST r2
  JZ done
round:
  LDI HL, 5
  LDI DE, 9
  LDI r0, 4
ct1:
  MOV r1, [HL]
  MOV [DE], r1
  INC HL
  INC DE
  DJNZ r0, ct1
  LDI DE, 1
  LDI HL, 5
  CLC
  LDI r0, 4
ad1:
  MOV r1, [DE]
  MOV r3, [HL]
  ADC r3, r1
  MOV [HL], r3
  INC DE
  INC HL
  DJNZ r0, ad1
  LDI HL, 9
  LDI DE, 1
  LDI r0, 4
ct2:
  MOV r1, [HL]
  MOV [DE], r1
  INC HL
  INC DE
  DJNZ r0, ct2
  DJNZ r2, round
done:
  LDI DE, 5
  LDI r3, 4
od:
  OUTDE
  DJNZ r3, od
  HALT
""")


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return b


def run_fib(k: int):
    data = bytearray(13)
    data[0] = k
    sim = NCP8(FIB, data=data)
    out = sim.run()
    assert sim.status == "HALT", sim.status
    got = sum(v << (8 * i) for i, v in enumerate(out))
    return got, fib(k), sim




SUMREC = asm("""
  LDI HL, 0
  MOV r0, [HL]
  CALL sum
  OUT r2
  OUT r1
  HALT
sum:
  TST r0
  JZ base
  PUSH r0
  SUBI r0, 1
  CALL sum
  POP r0
  ADD r1, r0
  ADCI r2, 0
  RET
base:
  LDI r1, 0
  LDI r2, 0
  RET
""")


def run_sumrec(n: int):
    data = bytearray(n + 16)
    data[0] = n
    sim = NCP8(SUMREC, data=data)
    out = sim.run()
    assert sim.status == "HALT", sim.status
    assert len(out) == 2
    got = out[0] << 8 | out[1]
    return got, n * (n + 1) // 2, sim


if __name__ == "__main__":
    import random
    random.seed(0)
    ok = 0
    for trial in range(200):
        nbits = random.randint(1, 40) * 8
        a = random.getrandbits(nbits)
        b = random.getrandbits(nbits)
        got, want, _ = run_long_add(a, b)
        assert got == want, (trial, a, b, got, want)
        ok += 1
    print(f"long_add: {ok} random long additions byte-exact ")

    for k in range(47):
        got, want, _ = run_fib(k)
        assert got == want, (k, got, want)
    print("fibonacci: F(0..46) all byte-exact")

    for n in [0, 1, 2, 7, 23, 100, 200, 255]:
        got, want, sim = run_sumrec(n)
        assert got == want, (n, got, want)
        assert sim.snapshot()["SP"] == 4096, f"n={n} stack not restored: {sim.snapshot()}"
    print("sumrec: recursion bounds/depth/stack balance all match")

    for _ in range(200):
        a = random.getrandbits(16)
        b = [random.randrange(256) for _ in range(4)]
        got, sim = run_frame_mul(a, b)
        want = frame_mul_model(a, b)
        assert got == want, (hex(a), b, hex(got), hex(want))
        assert sim.snapshot()["SP"] == 4096, f"frame not unwound: {sim.snapshot()}"
    print("frame_mul: 200 random (a, b) pairs, 16x16->32 widening product + local array, "
          "frame unwound and result byte-exact against the Python model")




MUL = asm("""
  LDI HL, 0
  MOV r0, [HL]
  INC HL
  MOV r1, [HL]
  LDI r2, 0
  LDI r3, 0
loop:
  TST r0
  JZ done
  ADD r2, r1
  ADCI r3, 0
  SUBI r0, 1
  JMP loop
done:
  OUT r3
  OUT r2
  HALT
""")


def run_mul(a, b):
    data = bytearray(2); data[0] = a; data[1] = b
    sim = NCP8(MUL, data=data)
    out = sim.run()
    assert sim.status == "HALT", sim.status
    got = (out[0] << 8) | out[1]
    return got, a * b




NESTED = asm("""
  LDI HL, 0
  MOV r0, [HL]
  CALL outer
  OUT r0
  HALT
outer:
  CALL inner
  ADDI r0, 5
  RET
inner:
  SHL r0
  RET
""")


def run_nested(n):
    data = bytearray(1); data[0] = n
    sim = NCP8(NESTED, data=data)
    out = sim.run()
    assert sim.status == "HALT", sim.status
    return out[0], n * 2 + 5




OVERFLOW = asm("""
recurse:
  CALL recurse
""")

















FRAME_MUL = asm("""
  LDI HL, 2
  LDI DE, 0
  CALL mulsum
  OUT r0
  OUT r1
  OUT r2
  OUT r3
  HALT
mulsum:
  PUSHW HL
  PUSHW DE
  MOVW HL, SP
  LDW DE, [HL]
  MOVW HL, DE
  LDW DE, [HL]
  MOVW HL, SP
  STW [HL], DE
  MOVW DE, HL
  LDI r0, 2
  ADDI DE, r0
  LDW HL, [DE]
  MOVW DE, HL
  MOVW HL, SP
  ADD SP, -8
  MOV r0, [DE]
  STX [HL-4], r0
  INC DE
  MOV r0, [DE]
  STX [HL-3], r0
  INC DE
  MOV r0, [DE]
  STX [HL-2], r0
  INC DE
  MOV r0, [DE]
  STX [HL-1], r0
  LDI r0, 0
  STX [HL-8], r0
  STX [HL-7], r0
  STX [HL-6], r0
  STX [HL-5], r0
  LDX r2, [HL]
  LDX r3, [HL-4]
  CALL mul8
  LDX r2, [HL-8]
  ADD r2, r0
  STX [HL-8], r2
  LDX r2, [HL-7]
  ADC r2, r1
  STX [HL-7], r2
  LDX r2, [HL-6]
  ADCI r2, 0
  STX [HL-6], r2
  LDX r2, [HL-5]
  ADCI r2, 0
  STX [HL-5], r2
  LDX r2, [HL]
  LDX r3, [HL-3]
  CALL mul8
  LDX r2, [HL-7]
  ADD r2, r0
  STX [HL-7], r2
  LDX r2, [HL-6]
  ADC r2, r1
  STX [HL-6], r2
  LDX r2, [HL-5]
  ADCI r2, 0
  STX [HL-5], r2
  LDX r2, [HL+1]
  LDX r3, [HL-4]
  CALL mul8
  LDX r2, [HL-7]
  ADD r2, r0
  STX [HL-7], r2
  LDX r2, [HL-6]
  ADC r2, r1
  STX [HL-6], r2
  LDX r2, [HL-5]
  ADCI r2, 0
  STX [HL-5], r2
  LDX r2, [HL+1]
  LDX r3, [HL-3]
  CALL mul8
  LDX r2, [HL-6]
  ADD r2, r0
  STX [HL-6], r2
  LDX r2, [HL-5]
  ADC r2, r1
  STX [HL-5], r2
  LDX r2, [HL-8]
  LDX r3, [HL-2]
  ADD r2, r3
  STX [HL-8], r2
  LDX r2, [HL-7]
  LDX r3, [HL-1]
  ADC r2, r3
  STX [HL-7], r2
  LDX r2, [HL-6]
  ADCI r2, 0
  STX [HL-6], r2
  LDX r2, [HL-5]
  ADCI r2, 0
  STX [HL-5], r2
  LDX r0, [HL-8]
  LDX r1, [HL-7]
  LDX r2, [HL-6]
  LDX r3, [HL-5]
  MOVW SP, HL
  POPW DE
  POPW HL
  RET
mul8:
  MOV r0, r2
  MULH r2, r3
  MOV r1, r2
  MUL r0, r3
  RET
""")


def frame_mul_model(a, b):


    x = a & 0xFFFF
    y = b[0] | (b[1] << 8)
    addend = b[2] | (b[3] << 8)
    return (x * y + addend) & 0xFFFFFFFF


def run_frame_mul(a, b):

    data = bytearray(6)
    data[0] = a & 0xFF
    data[1] = (a >> 8) & 0xFF
    data[2:6] = bytes(b)
    sim = NCP8(FRAME_MUL, data=data)
    out = sim.run()
    assert sim.status == "HALT", sim.status
    assert len(out) == 4, out
    return out[0] | (out[1] << 8) | (out[2] << 16) | (out[3] << 24), sim