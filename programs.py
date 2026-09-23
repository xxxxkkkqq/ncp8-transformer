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