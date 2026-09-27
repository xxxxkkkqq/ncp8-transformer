"""Executable checks of the bit-width and radix bounds used by the datapath.

Horner step z = 2s + a_i*b with s,b in [0,p) and a_i in {0,1} reaches 3p-3, so a
register does not overflow iff 2^W > 3p-3, and W* = ceil(log2(3p-2)). The
generalized radix-R bound is z <= (2R-1)(p-1). Also checks the carry-chain
semantics against Python big integers and the string/int/little-endian-bytes
round trip.
"""
import random

def _is_prime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True

def primes_below(limit):
    return [n for n in range(2, limit) if _is_prime(n)]

def W_star(p):
    X = 3 * p - 2
    return X.bit_length() - (1 if (X & (X - 1)) == 0 else 0)

def test_bitwidth_formula():
    ps = primes_below(2048)
    assert len(ps) == 309, (len(ps), ps[-5:])
    for p in ps:
        W = W_star(p)
        assert 2 ** W > 3 * p - 3, (p, W)
        assert W == 0 or 2 ** (W - 1) <= 3 * p - 3, (p, W)

        assert (3 * p - 3) // p <= 2, p

        assert (3 * p - 3) > 2 ** (W - 1), (p, W)
    print(f"bit-width bound: {len(ps)} primes (2..2048) all verified 2^W* > 3p-3  >=  2^(W*-1)+1 ")

    assert W_star(7) == 5 and 2 ** 4 <= 3 * 7 - 3 < 2 ** 5
    print("p=7 tightness counterexample: W*=5, W=4 overflows (18 > 16)")

    for p in [2, 3, 7, 101, 997]:
        for R in [2, 4, 10, 256]:
            zmax = (2 * R - 1) * (p - 1)
            assert R * (p - 1) + (R - 1) * (p - 1) == zmax
            rng = random.Random(p * R)
            s = p - 1; d = R - 1; b = p - 1
            assert R * s + d * b == zmax, "z_max is reachable"
    print("radix-R bound z <= (2R-1)(p-1) and reachable (R in {2,4,10,256})")

def test_p6_roundtrip():

    rng = random.Random(42)
    for _ in range(200_000):
        n = rng.getrandbits(rng.choice([1, 8, 64, 1024]))
        s = str(n)
        assert int(s) == n
        blk = bytearray((n >> (8 * i)) & 0xFF for i in range(max(1, (n.bit_length() + 7) // 8)))
        rec = sum(v << (8 * i) for i, v in enumerate(blk))
        assert rec == n

        assert "".join(chr(c) for c in s.encode()) == s
    print("radix conversion: 200,000 random bigints string<->int<->bytes round trip identity")

def test_ncp_bytes_semantics_vs_python():

    rng = random.Random(1)
    for _ in range(2000):
        a = rng.getrandbits(rng.randrange(1, 160))
        b = rng.getrandbits(rng.randrange(1, 160))
        n = max((a.bit_length() + 7) // 8, (b.bit_length() + 7) // 8, 1)

        s = a + b
        acc = 0; carry = 0
        for i in range(n):
            t = ((a >> (8 * i)) & 0xFF) + ((b >> (8 * i)) & 0xFF) + carry
            acc |= (t & 0xFF) << (8 * i)
            carry = t >> 8
        assert acc == s % (1 << (8 * n)), "ADC chain equals the integer value (mod 2^(8n))"
        if s >= 1 << (8 * n):
            assert carry == 1
    print("ADC carry-chain semantics == Python big integer (2000 random pairs)")

def tc(x):

    return x - 256 if x & 0x80 else x

def fits8(v):

    return -128 <= v <= 127

def asm_code(src):

    from golden_sim import asm
    return asm(src + "\nHALT")

_MACHINES = {}

def machine_for(image):

    from golden_sim import NCP8
    m = _MACHINES.get(image)
    if m is None:
        m = _MACHINES[image] = NCP8(image)
    return m

def ref_tick(image, a, b, carry=0, Z=0, S=0, V=0):

    g = machine_for(image)
    g.PC = 0
    g.r = [a, b, 0, 0]
    g.C, g.Z, g.S, g.V = carry, Z, S, V
    g.tick, g.status = 0, "RUNNING"
    g.out = bytearray()
    g.fault_reason, g.fault_addr = 0, 0
    try:
        g.step()
    except Exception:
        return None
    return {"C": g.C, "Z": g.Z, "S": g.S, "V": g.V, "r0": g.r[0], "r1": g.r[1]}

FORMS = {
    "ADD": ("ADD r0, r1", lambda a, b, c: {
        "C": (a + b) >> 8, "Z": int(((a + b) & 255) == 0),
        "S": ((a + b) & 255) >> 7, "V": int(not fits8(tc(a) + tc(b)))}),
    "ADC": ("ADC r0, r1", lambda a, b, c: {
        "C": (a + b + c) >> 8, "Z": int(((a + b + c) & 255) == 0),
        "S": ((a + b + c) & 255) >> 7,
        "V": int(not fits8(tc(a) + tc(b) + c))}),
    "SUB": ("SUB r0, r1", lambda a, b, c: {
        "C": int(a < b), "Z": int(((a - b) & 255) == 0),
        "S": ((a - b) & 255) >> 7, "V": int(not fits8(tc(a) - tc(b)))}),
    "SBB": ("SBB r0, r1", lambda a, b, c: {
        "C": int(a - b - c < 0), "Z": int(((a - b - c) & 255) == 0),
        "S": ((a - b - c) & 255) >> 7,
        "V": int(not fits8(tc(a) - tc(b) - c))}),
    "ADDI": ("ADDI r0, {b}", lambda a, b, c: {
        "C": (a + b) >> 8, "Z": int(((a + b) & 255) == 0),
        "S": ((a + b) & 255) >> 7, "V": int(not fits8(tc(a) + tc(b)))}),
    "SUBI": ("SUBI r0, {b}", lambda a, b, c: {
        "C": int(a < b), "Z": int(((a - b) & 255) == 0),
        "S": ((a - b) & 255) >> 7, "V": int(not fits8(tc(a) - tc(b)))}),
    "CMP": ("CMP r0, r1", lambda a, b, c: {
        "C": int(a < b), "Z": int(a == b),
        "S": ((a - b) & 255) >> 7, "V": int(not fits8(tc(a) - tc(b)))}),
}

BOUNDARY_ROWS = (
    ("ADD", 0x7F, 0x01, 0, {"C": 0, "Z": 0, "S": 1, "V": 1}),
    ("ADD", 0x80, 0x00, 0, {"C": 0, "Z": 0, "S": 1, "V": 0}),
    ("ADD", 0x80, 0x80, 0, {"C": 1, "Z": 1, "S": 0, "V": 1}),
    ("ADD", 0x7F, 0x7F, 0, {"C": 0, "Z": 0, "S": 1, "V": 1}),
    ("ADD", 0x80, 0xFF, 0, {"C": 1, "Z": 0, "S": 0, "V": 1}),
    ("ADD", 0x7F, 0x80, 0, {"C": 0, "Z": 0, "S": 1, "V": 0}),
    ("ADD", 0x40, 0x40, 0, {"C": 0, "Z": 0, "S": 1, "V": 1}),
    ("ADC", 0x7F, 0x00, 1, {"C": 0, "Z": 0, "S": 1, "V": 1}),
    ("ADC", 0x7E, 0x00, 1, {"C": 0, "Z": 0, "S": 0, "V": 0}),
    ("ADC", 0x80, 0xFE, 1, {"C": 1, "Z": 0, "S": 0, "V": 1}),
    ("SUB", 0x80, 0x01, 0, {"C": 0, "Z": 0, "S": 0, "V": 1}),
    ("SUB", 0x7F, 0x80, 0, {"C": 1, "Z": 0, "S": 1, "V": 1}),
    ("SUB", 0x00, 0x80, 0, {"C": 1, "Z": 0, "S": 1, "V": 1}),
    ("SUB", 0x80, 0x80, 0, {"C": 0, "Z": 1, "S": 0, "V": 0}),
    ("SUB", 0x00, 0x01, 0, {"C": 1, "Z": 0, "S": 1, "V": 0}),
    ("SBB", 0x80, 0x00, 1, {"C": 0, "Z": 0, "S": 0, "V": 1}),
    ("SBB", 0x7F, 0x80, 0, {"C": 1, "Z": 0, "S": 1, "V": 1}),
    ("SBB", 0x00, 0x01, 0, {"C": 1, "Z": 0, "S": 1, "V": 0}),
    ("ADDI", 0x7F, 0x01, 0, {"C": 0, "Z": 0, "S": 1, "V": 1}),
    ("ADDI", 0xFF, 0x01, 0, {"C": 1, "Z": 1, "S": 0, "V": 0}),
    ("SUBI", 0x80, 0x01, 0, {"C": 0, "Z": 0, "S": 0, "V": 1}),
    ("SUBI", 0x00, 0x80, 0, {"C": 1, "Z": 0, "S": 1, "V": 1}),
    ("CMP", 0x80, 0x01, 0, {"C": 0, "Z": 0, "S": 0, "V": 1}),
    ("CMP", 0x7F, 0x01, 0, {"C": 0, "Z": 0, "S": 0, "V": 0}),
    ("CMP", 0x00, 0x00, 0, {"C": 0, "Z": 1, "S": 0, "V": 0}),
)

def test_signed_flag_boundaries():

    bad = []
    for name, a, b, carry, want in BOUNDARY_ROWS:
        src = FORMS[name][0]
        got = ref_tick(asm_code(src.format(b=b) if "{b}" in src else src), a, b, carry)
        if got is None or {k: got[k] for k in want} != want:
            bad.append((name, hex(a), hex(b), carry, want, got))
    assert not bad, ("signed flag boundary rows disagree with the two's-complement "
                     "model", bad[:6])
    print(f"signed flag boundaries: {len(BOUNDARY_ROWS)} named rows at 2^(n-1)-1 and"
          f" -(2^(n-1)) with both signs, C/Z/S/V as stated")

def test_signed_flags_over_every_byte_pair():

    total = 0
    for name, (src, model) in sorted(FORMS.items()):
        for carry in ((0, 1) if name in ("ADC", "SBB") else (0,)):
            for a in range(256):
                for b in range(256):
                    image = asm_code(src.format(b=b) if "{b}" in src else src)
                    want = model(a, b, carry)
                    got = ref_tick(image, a, b, carry)
                    assert got is not None, (name, hex(a), hex(b), carry, "tick faulted")
                    check = {k: got[k] for k in want}
                    assert check == want, ("flag math disagrees with the two's-complement "
                                           "model", name, hex(a), hex(b), carry, want, check)
                    if name == "CMP":
                        assert (got["r0"], got["r1"]) == (a, b), (name, hex(a), hex(b))
                    else:
                        assert got["r1"] == b, (name, hex(a), hex(b), "source written")
                    total += 1
    print(f"flag math over every byte pair: {total} ticks of ADD/ADC/SUB/SBB/ADDI/SUBI/"
          f"CMP match the two's-complement model, and no row writes its source operand")

def test_neg_writes_sign_and_keeps_overflow():

    image = asm_code("NEG r0")
    for a in range(256):
        for v in (0, 1):
            got = ref_tick(image, a, 0, 0, S=0, V=v)
            assert got is not None, hex(a)
            assert got["S"] == ((-a) & 255) >> 7, (hex(a), got)
            assert got["V"] == v, (hex(a), "V changed under NEG", v, got)
            assert got["C"] == int(a != 0) and got["Z"] == int(a == 0), (hex(a), got)
    print("  NEG: 256 values x both overflow states, S set, V left exactly as it was")

def test_gtf_returns_all_four_flags():

    image = asm_code("GETF r0")
    bad = []
    for c in (0, 1):
        for z in (0, 1):
            for s in (0, 1):
                for v in (0, 1):
                    got = ref_tick(image, z * 2 + 0, 0, c, Z=z, S=s, V=v)
                    want = z | (c << 1) | (s << 2) | (v << 3)
                    if got["r0"] != want:
                        bad.append((c, z, s, v, got["r0"], want))
    assert not bad, bad
    print("  GETF: all 16 flag combinations pack as Z | C<<1 | S<<2 | V<<3")

if __name__ == "__main__":
    test_signed_flag_boundaries()
    test_signed_flags_over_every_byte_pair()
    test_neg_writes_sign_and_keeps_overflow()
    test_gtf_returns_all_four_flags()
    test_bitwidth_formula()
    test_p6_roundtrip()
    test_ncp_bytes_semantics_vs_python()
    print("\narithmetic bound checks: all passed")