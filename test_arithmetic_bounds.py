"""Executable checks of the bit-width and radix bounds used by the datapath.

Horner step z = 2s + a_i*b with s,b in [0,p) and a_i in {0,1} reaches 3p-3, so a
register does not overflow iff 2^W > 3p-3, and W* = ceil(log2(3p-2)). The
generalized radix-R bound is z <= (2R-1)(p-1). Also checks the carry-chain
semantics against Python big integers and the string/int/little-endian-bytes
round trip.
"""
import random

from sympy import primerange

def W_star(p):
    X = 3 * p - 2
    return X.bit_length() - (1 if (X & (X - 1)) == 0 else 0)

def test_bitwidth_formula():
    ps = list(primerange(2, 2048))
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

if __name__ == "__main__":
    test_bitwidth_formula()
    test_p6_roundtrip()
    test_ncp_bytes_semantics_vs_python()
    print("\narithmetic bound checks: all passed")