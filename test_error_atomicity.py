"""Error-tick atomicity acceptance for the reference and both circuits.

The ISA error contract is: a tick that violates a bound writes status = 3 and
changes nothing else. This suite sweeps the boundary states where a partial
write used to be possible: every stack-touching instruction (PUSH, POP, CALL,
RET, EXT) crossed with SP in {0, 1, 2, 3, 4094, 4095, 4096}, plus the memory
instructions at HL / DE in {0, 4095, 4096}. For a violating case it asserts that
the tick reports an error and that the whole visible state, the whole DATA and
CODE images, the output stream and the input position are unchanged, on the
reference simulator itself as well as on both circuits. Legal boundary cases
must not report an error, and their commit is compared against a hand-written
Python model of the ISA.
"""
from __future__ import annotations

from golden_sim import NCP8, MachineError
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit

DATA_SIZE = 4096
VEC = 0x0F00
WLO, WHI = 0x0F20, 0x0F21
VEC_TGT = 0x0040
CALL_TGT = 0x1F00
STATUS = {"RUNNING": 0, "HALT": 1, "OVERRUN": 2, "ERR": 3}

STACK_SP = (0, 1, 2, 3, 4094, 4095, 4096)
LIMIT_ADDRS = (0, 4095, 4096)

INIT_R = [0x11, 0x22, 0x33, 0x44]
INIT_C, INIT_Z = 1, 0
TICK0 = 7
INPUTS = b"\xAB\xCD"

DATA_IMAGE = bytes((i * 7 + 13) & 0xFF for i in range(DATA_SIZE))


def build_code(head, vec0=None):





    b = bytearray(bytes(head).ljust(WHI + 2, b"\x00"))
    if vec0 is not None:
        b[VEC:VEC + 2] = (vec0 & 0xFFFF).to_bytes(2, "little")
    return bytes(b)


def ref_view(g):
    return dict(r=list(g.r), HL=g.HL, DE=g.DE, SP=g.SP, PC=g.PC, C=g.C, Z=g.Z,
                ipos=g.ipos, oplen=len(g.out), tick=g.tick, status=STATUS[g.status])


def run_reference(code, sp, hl, de):

    g = NCP8(code, data=DATA_IMAGE, inputs=INPUTS)
    g.r = list(INIT_R)
    g.HL, g.DE, g.SP = hl, de, sp
    g.C, g.Z, g.tick = INIT_C, INIT_Z, TICK0
    try:
        g.step()
        raised = False
    except MachineError:
        raised = True
    return raised, ref_view(g), list(g.data), bytes(g.code), bytes(g.out)


def run_circuit(Machine, code, sp, hl, de):

    c = Machine(code, data=DATA_IMAGE, inputs=INPUTS)
    c.load_state(INIT_R, hl, de, sp, INIT_C, INIT_Z, TICK0)
    try:
        c.step()
        raised = False
    except MachineError:
        raised = True
    snap = c.snapshot()
    if snap["status"] == 3:
        raised = True
    data = c.DATA.cpu().tolist()
    code_img = bytes(c.CODE.cpu().tolist()[:len(code)])
    return raised, snap, data, code_img, c.out()


def check_case(name, code, sp, hl, de, expect_err, expect_commit=None):







    runs = [("reference", run_reference(code, sp, hl, de)),
            ("torch", run_circuit(TorchCircuit, code, sp, hl, de)),
            ("triton", run_circuit(TritonCircuit, code, sp, hl, de))]
    ref_pre = dict(r=list(INIT_R), HL=hl, DE=de, SP=sp, PC=0, C=INIT_C, Z=INIT_Z,
                   ipos=0, oplen=0, tick=TICK0, status=0)
    for label, (raised, post, data, code_img, out) in runs:
        if expect_err:
            assert raised, (name, label, "the violating tick did not report an error")
            if label == "reference":

                assert post == ref_pre, (name, label, "reference error tick was not atomic", ref_pre, post)
            else:
                assert post["status"] == 3, (name, label, "expected status 3", post)
                for k, v in ref_pre.items():
                    if k != "status":
                        assert post[k] == v, (name, label, "error tick changed a state field", k, v, post[k])
            assert data == list(DATA_IMAGE), (name, label, "error tick changed DATA")
            assert code_img == code, (name, label, "error tick changed CODE")
            assert out == b"", (name, label, "error tick changed the output stream", out)
        else:
            exp_state, exp_data, exp_out = expect_commit
            assert not raised, (name, label, "a legal boundary tick reported an error")
            assert post == exp_state, (name, label, "legal tick state mismatch", exp_state, post)
            assert data == list(exp_data), (name, label, "legal tick DATA mismatch")
            assert out == exp_out, (name, label, "legal tick output mismatch", exp_out, out)


    r_raised, r_post, r_data, r_code, r_out = runs[0][1]
    keys = [k for k in ref_pre if k != "status"] if expect_err else list(ref_pre)
    for label, (raised, post, data, code_img, out) in runs[1:]:
        assert raised == r_raised, (name, label, "error verdict differs from the reference")
        assert all(post[k] == r_post[k] for k in keys), \
            (name, label, "state differs from the reference", post, r_post)
        assert (data, code_img, out) == (r_data, r_code, r_out), \
            (name, label, "memory or output differs from the reference")
    return "err" if expect_err else "ok"


def legal_expect(kind, sp, hl, de):


    r = list(INIT_R)
    d = bytearray(DATA_IMAGE)
    out = b""
    st = dict(r=r, HL=hl, DE=de, SP=sp, PC=1, C=INIT_C, Z=INIT_Z, ipos=0, oplen=0,
              tick=TICK0 + 1, status=0)
    ret_lo, ret_hi = (0 + 3) & 0xFF, (0 + 3) >> 8
    if kind == "PUSH r0":
        st["SP"] = sp - 1
        d[sp - 1] = r[0]
    elif kind == "POP r0":
        st["r"] = [d[sp], r[1], r[2], r[3]]
        st["SP"] = sp + 1
    elif kind == "CALL a16":
        st["SP"] = sp - 2
        d[sp - 1], d[sp - 2] = ret_lo, ret_hi
        st["PC"] = CALL_TGT
    elif kind == "RET":

        st["PC"] = (d[sp] << 8) | d[sp + 1]
        st["SP"] = sp + 2
    elif kind == "EXT 0":
        st["SP"] = sp - 2
        d[sp - 1], d[sp - 2] = ret_lo, ret_hi
        st["PC"] = VEC_TGT
    elif kind == "MOV r0, [HL]":
        st["r"] = [d[hl], r[1], r[2], r[3]]
    elif kind == "MOV [HL], r0":
        d[hl] = r[0]
    elif kind == "OUTM":
        out = bytes([d[hl]])
        st["HL"] = hl + 1
        st["oplen"] = 1
    elif kind == "MOV r0, [DE]":
        st["r"] = [d[de], r[1], r[2], r[3]]
    elif kind == "MOV [DE], r0":
        d[de] = r[0]
    elif kind == "OUTDE":
        out = bytes([d[de]])
        st["DE"] = de + 1
        st["oplen"] = 1
    else:
        raise AssertionError(f"unknown case kind {kind!r}")
    return st, bytes(d), out



STACK_CASES = (
    ("PUSH r0", bytes([0xF0 | 0])),
    ("POP r0", bytes([0xF4 | 0])),
    ("CALL a16", bytes([0x0E, CALL_TGT & 0xFF, CALL_TGT >> 8])),
    ("RET", bytes([0x08])),
    ("EXT 0", bytes([0x70, 0x70, 0x00])),
)

STACK_ERR = {
    "PUSH r0": lambda sp: sp < 1,
    "POP r0": lambda sp: sp >= DATA_SIZE,
    "CALL a16": lambda sp: sp < 2,
    "RET": lambda sp: sp + 2 > DATA_SIZE,
    "EXT 0": lambda sp: sp < 2,
}
MEM_CASES = (
    ("MOV r0, [HL]", bytes([0xE0 | 0]), "HL"),
    ("MOV [HL], r0", bytes([0xE4 | 0]), "HL"),
    ("OUTM", bytes([0x06]), "HL"),
    ("MOV r0, [DE]", bytes([0xE8 | 0]), "DE"),
    ("MOV [DE], r0", bytes([0xEC | 0]), "DE"),
    ("OUTDE", bytes([0x07]), "DE"),
)


def test_stack_boundaries():
    tot = {"ok": 0, "err": 0}
    for kind, head in STACK_CASES:
        code = build_code(head, vec0=VEC_TGT if kind.startswith("EXT") else None)
        for sp in STACK_SP:
            expect_err = STACK_ERR[kind](sp)
            commit = None if expect_err else legal_expect(kind, sp, 0, 0)
            tot[check_case(f"{kind} @ SP={sp}", code, sp, 0, 0, expect_err, commit)] += 1
        print(f"  {kind:9s}: SP {STACK_SP} -> "
              f"{sum(1 for sp in STACK_SP if STACK_ERR[kind](sp))} error / "
              f"{sum(1 for sp in STACK_SP if not STACK_ERR[kind](sp))} legal")
    assert tot == {"ok": 27, "err": 8}, tot
    print(f"  stack instructions x SP boundary: {sum(tot.values())} cases "
          f"(error {tot['err']} + legal {tot['ok']})")
    return tot


def test_memory_boundaries():
    tot = {"ok": 0, "err": 0}
    for kind, head, ptr in MEM_CASES:
        code = build_code(head)
        for addr in LIMIT_ADDRS:
            hl, de = (addr, 0) if ptr == "HL" else (0, addr)
            expect_err = addr >= DATA_SIZE
            commit = None if expect_err else legal_expect(kind, DATA_SIZE, hl, de)
            tot[check_case(f"{kind} @ {ptr}={addr}", code, DATA_SIZE, hl, de,
                           expect_err, commit)] += 1
    assert tot == {"ok": 12, "err": 6}, tot
    print(f"  memory instructions x HL/DE in {LIMIT_ADDRS}: {sum(tot.values())} cases "
          f"(error {tot['err']} + legal {tot['ok']})")
    return tot


if __name__ == "__main__":
    print("error-tick atomicity (reference + both circuits):")
    a = test_stack_boundaries()
    b = test_memory_boundaries()
    cases = sum(a.values()) + sum(b.values())
    print(f"  {cases} cases x 3 implementations = {cases * 3} probes, "
          f"error {a['err'] + b['err']} + legal {a['ok'] + b['ok']}")
    print("error-tick atomicity: all passed")