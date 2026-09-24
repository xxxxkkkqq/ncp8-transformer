"""Acceptance for the extended opcode set.

Covers the bitwise/multiply/divide families, comparison, rotates, 16-bit pointer
arithmetic, the escape prefix, the user-defined-instruction trap and the
controlled self-modification window. Every case is compared against Python
integer semantics byte for byte; no case is accepted on inspection.

Three areas that are easy to get wrong and are therefore covered explicitly:
flag conventions, error paths (which must be atomic: status only, nothing else
changes) and instruction-length/PC bookkeeping around the escape prefix.
"""
import random

from golden_sim import NCP8, MachineError, asm
from test_state_contract import assert_widths

CODE_SIZE = 4096
HANDLER_VEC = 0x0F00


def run_code(code, data=None, inputs=b"", r0=None, budget=5000):
    g = NCP8(code, data=data, inputs=inputs, tick_budget=budget)
    if r0 is not None:
        g.r[0] = r0 & 0xFF
    drive(g)
    return g


def expect_err(code, data=None, r0=None, r1=None):

    g = NCP8(code, data=data, tick_budget=5000)
    if r0 is not None:
        g.r[0] = r0 & 0xFF
    if r1 is not None:
        g.r[1] = r1 & 0xFF
    try:
        g.step()
        return False, g.snapshot()
    except MachineError:
        return True, g.snapshot()


def drive(g):

    while g.status == "RUNNING":
        g.step()
        assert_widths(g.snapshot(), ("isa_v2", g.tick))
    return g


def place_vector(code, k, addr):

    b = bytearray(code.ljust(HANDLER_VEC + 2 * (k + 1), b"\x00"))
    b[HANDLER_VEC + 2 * k: HANDLER_VEC + 2 * k + 2] = (addr & 0xFFFF).to_bytes(2, "little")
    return bytes(b)


def one_op(src, r0=None, r1=None):

    g = NCP8(asm(src))
    if r0 is not None: g.r[0] = r0 & 0xFF
    if r1 is not None: g.r[1] = r1 & 0xFF
    drive(g)
    return g






MUL32 = asm("""
  LDI HL, 4
  LDI r0, 0
  MOV [HL], r0
  INC HL
  MOV [HL], r0
  INC HL
  MOV [HL], r0
  INC HL
  MOV [HL], r0
  INC HL
  MOV [HL], r0
  INC HL
  MOV [HL], r0
  LDI HL, 0
  MOV r0, [HL]
  LDI HL, 20
  MOV [HL], r0
  LDI HL, 2
  MOV r0, [HL]
  LDI HL, 21
  MOV [HL], r0
  CALL mk_prod
  LDI r0, 0
  LDI HL, 27
  MOV [HL], r0
  CALL acc_prod
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 20
  MOV [HL], r0
  LDI HL, 2
  MOV r0, [HL]
  LDI HL, 21
  MOV [HL], r0
  CALL mk_prod
  LDI r0, 1
  LDI HL, 27
  MOV [HL], r0
  CALL acc_prod
  LDI HL, 0
  MOV r0, [HL]
  LDI HL, 20
  MOV [HL], r0
  LDI HL, 3
  MOV r0, [HL]
  LDI HL, 21
  MOV [HL], r0
  CALL mk_prod
  LDI r0, 1
  LDI HL, 27
  MOV [HL], r0
  CALL acc_prod
  LDI HL, 1
  MOV r0, [HL]
  LDI HL, 20
  MOV [HL], r0
  LDI HL, 3
  MOV r0, [HL]
  LDI HL, 21
  MOV [HL], r0
  CALL mk_prod
  LDI r0, 2
  LDI HL, 27
  MOV [HL], r0
  CALL acc_prod
  HALT
mk_prod:
  LDI HL, 22
  LDI r0, 0
  MOV [HL], r0
  INC HL
  MOV [HL], r0
  LDI HL, 24
  LDI DE, 21
  MOV r0, [DE]
  MOV [HL], r0
  INC HL
  LDI r0, 0
  MOV [HL], r0
  LDI HL, 20
  MOV r0, [HL]
  LDI r1, 8
bits:
  MOV r2, r0
  SHR r2
  JNC noadd
  CLC
  LDI HL, 22
  LDI DE, 24
  MOV r2, [HL]
  MOV r3, [DE]
  ADC r2, r3
  MOV [HL], r2
  INC HL
  INC DE
  MOV r2, [HL]
  MOV r3, [DE]
  ADC r2, r3
  MOV [HL], r2
noadd:
  SHR r0
  LDI HL, 24
  MOV r2, [HL]
  SHL r2
  MOV [HL], r2
  INC HL
  MOV r2, [HL]
  ROL r2
  MOV [HL], r2
  DJNZ r1, bits
  RET
acc_prod:
  LDI HL, 4
  LDI DE, 27
  MOV r0, [DE]
  ADDI HL, r0
  LDI DE, 22
  CLC
  MOV r0, [HL]
  MOV r1, [DE]
  ADC r0, r1
  MOV [HL], r0
  INC HL
  INC DE
  MOV r0, [HL]
  MOV r1, [DE]
  ADC r0, r1
  MOV [HL], r0
  INC HL
  MOV r0, [HL]
  ADCI r0, 0
  MOV [HL], r0
  INC HL
  MOV r0, [HL]
  ADCI r0, 0
  MOV [HL], r0
  RET
""")


def test_bitwise():
    rng = random.Random(7)
    for _ in range(400):
        a, b = rng.randrange(256), rng.randrange(256)
        g = one_op("AND r0, r1\nHALT", a, b)
        assert g.r[0] == a & b and g.Z == int((a & b) == 0), ("AND", a, b)
        g = one_op("OR r0, r1\nHALT", a, b)
        assert g.r[0] == a | b and g.Z == int((a | b) == 0), ("OR", a, b)
        g = one_op("XOR r0, r1\nHALT", a, b)
        assert g.r[0] == a ^ b and g.Z == int((a ^ b) == 0), ("XOR", a, b)

        g = one_op("AND r0, r1\nMOV r2, r0\nNOT r2\nNOT r0\nNOT r1\nOR r0, r1\nHALT", a, b)
        assert g.r[2] == g.r[0], ("DeMorgan", a, b)

    g = NCP8(asm("CLC\nLDI r0, 1\nSUBI r0, 5\nAND r0, r0\nHALT"))
    drive(g)
    assert g.C == 1, "AND must not clear C"
    print("  bitwise (AND/OR/XOR/NOT + De Morgan + C untouched)")


def test_mul():
    rng = random.Random(8)
    for _ in range(400):
        a, b = rng.randrange(256), rng.randrange(256)
        g = one_op("MUL r0, r1\nHALT", a, b)
        assert g.r[0] == (a * b) & 0xFF, ("MUL lo", a, b)
        assert g.C == int(a * b > 255), ("MUL C must be the high byte != 0", a, b)
        assert g.Z == int((a * b) & 0xFF == 0), ("MUL Z", a, b)




    for _ in range(60):
        A, B = rng.randrange(65536), rng.randrange(65536)
        g = NCP8(MUL32)
        g.data[0], g.data[1] = A & 255, A >> 8
        g.data[2], g.data[3] = B & 255, B >> 8
        drive(g)
        assert g.status == "HALT", g.status
        assert bytes(g.data[4:10]) == (A * B).to_bytes(6, "little"), \
            (A, B, bytes(g.data[4:10]).hex(), hex(A * B))
    print("  MUL contract (low byte + C as the high-byte-nonzero flag + Z), "
          "16x16 -> 32 shift/ADC chain byte-exact")


def test_div_mod():
    rng = random.Random(9)
    for _ in range(500):
        a = rng.randrange(256); b = rng.randrange(1, 256)
        g = one_op("DIV r0, r1\nHALT", a, b)
        assert g.r[0] == a // b and g.Z == int(a // b == 0), ("DIV", a, b)
        g = one_op("MOD r0, r1\nHALT", a, b)
        assert g.r[0] == a % b, ("MOD", a, b)

        g = one_op("MOV r2, r0\nDIV r2, r1\nMUL r2, r1\nMOV r3, r0\nMOD r3, r1\nADD r2, r3\nHALT", a, b)
        assert g.r[2] == a, ("identity", a, b)

    for src in ("DIV r0, r1\nHALT", "MOD r0, r1\nHALT"):
        code = asm(src)
        err, snap = expect_err(code, r0=10, r1=0)
        assert err, f"divide by zerodid not raise: {src}"
        assert snap["r"] == [10, 0, 0, 0] and snap["tick"] == 0, ("divide by zero was not atomic", snap)
    print("  DIV/MOD (500 pairs + identity + divide-by-zero atomic ERR)")


def test_cmp_rot_neg():
    rng = random.Random(10)
    for _ in range(600):
        a, b = rng.randrange(256), rng.randrange(256)
        g = one_op("CMP r0, r1\nHALT", a, b)
        assert (g.Z, g.C) == (int(a == b), int(a < b)), ("CMP", a, b)
        assert g.r[0] == a and g.r[1] == b, "CMP must not write back"
    for _ in range(300):
        a = rng.randrange(256); c = rng.randrange(2)

        code = asm("\n".join(["ROL r0"] * 9 + ["HALT"]))
        g = NCP8(code); g.r[0] = a; g.C = c
        drive(g)
        assert g.r[0] == a and g.C == c, ("ROL x9", a, c)
        code = asm("\n".join(["ROR r0"] * 9 + ["HALT"]))
        g = NCP8(code); g.r[0] = a; g.C = c
        drive(g)
        assert g.r[0] == a and g.C == c, ("ROR x9", a, c)

        x = (c << 8) | a
        xl = ((x << 1) | (x >> 8)) & 0x1FF
        g = NCP8(asm("ROL r0\nHALT")); g.r[0] = a; g.C = c
        drive(g)
        assert ((g.C << 8) | g.r[0]) == xl, ("ROL 9bit", a, c)
        xr = ((x >> 1) | ((x & 1) << 8)) & 0x1FF
        g = NCP8(asm("ROR r0\nHALT")); g.r[0] = a; g.C = c
        drive(g)
        assert ((g.C << 8) | g.r[0]) == xr, ("ROR 9bit", a, c)

        g = NCP8(asm("ROL r0\nROR r0\nHALT")); g.r[0] = a; g.C = c
        drive(g)
        assert g.r[0] == a and g.C == c, ("ROL/ROR are inverse", a, c)

        g = one_op("NEG r0\nNEG r0\nHALT", a)
        assert g.r[0] == a, ("NEG is self-inverse", a)
        g = one_op("NEG r0\nHALT", a)
        assert g.r[0] == (-a) & 0xFF and g.C == int(a != 0), ("NEG semantics", a)
        g = one_op("NOT r0\nHALT", a)
        assert g.r[0] == (~a) & 0xFF, ("NOT", a)
    print("  CMP (no writeback) / ROL,ROR 9-bit rotate semantics and x9 identity / NEG self-inverse")


def test_ptr16():
    rng = random.Random(11)
    for _ in range(400):
        hl, de = rng.randrange(65536), rng.randrange(65536)
        g = NCP8(asm("ADD HL, DE\nHALT")); g.HL, g.DE = hl, de
        drive(g)
        assert g.HL == (hl + de) & 0xFFFF and g.C == int(hl + de > 0xFFFF), ("ADD HL,DE", hl, de)
        g = NCP8(asm("SUB HL, DE\nHALT")); g.HL, g.DE = hl, de
        drive(g)
        assert g.HL == (hl - de) & 0xFFFF and g.C == int(hl < de), ("SUB HL,DE", hl, de)
        g = NCP8(asm("XCHG\nHALT")); g.HL, g.DE = hl, de
        drive(g)
        assert (g.HL, g.DE) == (de, hl), "XCHG"
    print("  16-bit pointers (ADD/SUB HL,DE with carry + XCHG)")


def test_esc_and_trap():



    for sub in (0x36, 0x37, 0x59, 0x63, 0x6F, 0x71, 0x7F, 0x88, 0x8F, 0xA0, 0xFF):
        err, snap = expect_err(bytes([0x70, sub]), r0=7)
        assert err, f"reserved subcode {sub:#04x} did not raise"
        assert snap["r"][0] == 7 and snap["PC"] == 0, ("reserved subcode was not atomic", sub)

    for op in (0x71, 0x75, 0x7F):
        err, _ = expect_err(bytes([op]))
        assert err, f"reserved opcode {op:#04x} did not raise"

    err, _ = expect_err(bytes([0x70, 0x70, 0x10]))
    assert err, "EXT with k=16 did not raise"
    err, _ = expect_err(place_vector(bytes([0x70, 0x70, 0x00]), 0, 0))
    assert err, "EXT with an unregistered vector did not raise"

    main = bytes([0x70, 0x70, 0x00, 0x00])
    handler_addr = 0x10
    handler = bytes([0xD4 | 0, 1, 0x08])
    code = bytearray(main.ljust(handler_addr, b"\x00")) + handler
    code = place_vector(bytes(code), 0, handler_addr)
    g = NCP8(code); g.r[0] = 41
    drive(g)
    assert g.r[0] == 42 and g.status == "HALT", ("EXT call failed", g.snapshot(), g.trace)
    assert g.SP == 4096, "stack not restored after EXT (return address bookkeeping)"

    h0 = bytes([0x70, 0x70, 0x01, 0xD4 | 0, 1, 0x08])
    h1 = bytes([0xD4 | 0, 1, 0x08])
    code2 = bytearray(bytearray(main).ljust(handler_addr, b"\x00")) + h0 + h1
    code2 = place_vector(bytes(code2), 0, handler_addr)
    code2 = place_vector(bytes(code2), 1, handler_addr + len(h0))
    g = NCP8(code2); g.r[0] = 0
    drive(g)
    assert g.r[0] == 2 and g.SP == 4096, ("EXT nested call", g.r[0], g.SP)
    print("  escape prefix + user-instruction trap (reserved subcode atomic ERR / unregistered ERR / call-return-nested bookkeeping)")


def test_pc_bookkeeping():

    code = bytearray(bytes([0x70, 0x70, 0x00]) + bytes([0xD0 | 3, 0xAB, 0x00]))
    h = bytes([0xD4 | 0, 1, 0x08])
    code = bytearray(bytes(code).ljust(0x10, b"\x00")) + h
    code = place_vector(bytes(code), 0, 0x10)
    g = NCP8(bytes(code))
    drive(g)
    assert g.r[3] == 0xAB and g.r[0] == 1, ("PC bookkeeping wrong after EXT", g.snapshot())
    print("  escape PC bookkeeping (2-byte prefix, 3-byte trap; return lands correctly)")


def test_selfmod():


    WLO, WHI = 0x0F20, 0x0F21

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
    stc_pc = base.index(bytes([0x70, 0x80]))

    def build(wlo, whi):
        b = bytearray(bytes(base).ljust(WLO + 2, b"\x00"))
        b[WLO], b[WHI] = wlo, whi
        return bytes(b)


    g = NCP8(build(0x00, 0x08))
    drive(g)
    assert bytes(g.out) == bytes([42]), ("self-modification had no effect", bytes(g.out))

    for wlo, whi in ((0x10, 0x18), (0x00, 0x00)):
        g2 = NCP8(build(wlo, whi))
        err = False
        try:
            drive(g2)
        except MachineError:
            err = True
        assert err, f"window[{wlo:#x},{whi:#x}) did not raise"

    code_bad = build(0x10, 0x18)
    g3 = NCP8(code_bad)
    err, snap = False, None
    try:
        drive(g3)
    except MachineError:
        err, snap = True, g3.snapshot()
    assert err and snap["PC"] == stc_pc and snap["r"][0] == 42, ("out-of-window write was not atomic", stc_pc, snap)

    code = asm("LDI HL, 0\nLDC r0, [HL]\nOUT r0\nLDI HL, 1\nLDC r0, [HL]\nOUT r0\nHALT")
    g4 = NCP8(code)
    drive(g4)
    assert bytes(g4.out) == bytes([code[0], code[1]]), ("LDC code read mismatch", bytes(g4.out), code[:2])

    for src2 in ("LDI HL, 4000\nLDC r0, [HL]\nHALT", "LDI HL, 4000\nSTC [HL], r0\nHALT"):
        g5 = NCP8(asm(src2))
        err = False
        try:
            drive(g5)
        except MachineError:
            err = True
        assert err, f"out of rangedid not raise: {src2}"
    print("  controlled self-modification (write takes effect / out-of-window and zero-width window atomic ERR / self-read byte-exact / out-of-range ERR)")


if __name__ == "__main__":
    print("ISA v2.0 acceptance:")
    test_bitwise(); test_mul(); test_div_mod(); test_cmp_rot_neg()
    test_ptr16(); test_esc_and_trap(); test_pc_bookkeeping(); test_selfmod()
    print("ISA v2.0: all passed")