"""Error-tick atomicity acceptance for the reference and both circuits.

The ISA error contract is: a tick that violates a bound writes status = 3 and
changes nothing else. This suite sweeps the boundary states where a partial
write used to be possible: every stack-touching instruction (PUSH, POP, CALL,
RET, EXT and the two-byte PUSHW/POPW) crossed with SP in
{0, 1, 2, 3, 4094, 4095, 4096}; the 8-bit memory instructions at HL / DE in
{0, 4095, 4096}; the 16-bit memory instructions at the last addresses that still
have room for a second byte; the frame-relative accesses at the wrap around 0 and
at the end of DATA; and the stack-pointer writes (ADD SP, i8 and MOVW SP, HL/DE),
whose legal range is [0, DATA_SIZE].

For a violating case it asserts that the tick reports an error and that the whole
visible state, the whole DATA and CODE images, the output stream and the input
position are unchanged, on the reference simulator itself as well as on both
circuits. Legal boundary cases must not report an error, and their commit is
compared against a hand-written Python model of the ISA. The instructions with no
access to check (MULH, the non-SP MOVW forms) are covered by the subcode
enumeration instead.
"""
from __future__ import annotations

from golden_sim import MachineError, NCP8, asm
import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from test_state_contract import FAULT_WRITES, VIEW_FIELDS, assert_widths, ref_view

DATA_SIZE = 4096
VEC = 0x0F00
WLO, WHI = 0x0F20, 0x0F21
VEC_TGT = 0x0040
CALL_TGT = 0x1F00

PC_SITES = (0, 256)

STACK_SP = (0, 1, 2, 3, 4094, 4095, 4096)
LIMIT_ADDRS = (0, 4095, 4096)
WIDE_ADDRS = (0, 1, 4093, 4094, 4095, 4096)

INIT_R = [0x11, 0x22, 0x33, 0x44]
INIT_C, INIT_Z = 1, 0
TICK0 = 7
INPUTS = b"\xAB\xCD"

DATA_IMAGE = bytes((i * 7 + 13) & 0xFF for i in range(DATA_SIZE))

def build_code(head, pc=0):

    b = bytearray(bytes(head).ljust(max(VEC + 2, pc + len(head) + 2), b"\x00"))
    b[pc:pc + len(head)] = bytes(head)
    return bytes(b)

def config_for(vec0):

    return None if vec0 is None else ISA.MachineConfig(vec={0: vec0})

def run_reference(code, sp, hl, de, pc=0, vec0=None):

    g = NCP8(code, data=DATA_IMAGE, inputs=INPUTS, config=config_for(vec0))
    g.load_state(INIT_R, hl, de, sp, INIT_C, INIT_Z, TICK0, PC=pc)
    try:
        g.step()
        raised = False
    except MachineError:
        raised = True
    return raised, ref_view(g), list(g.data), bytes(g.code)[:len(code)], bytes(g.out)

def run_circuit(Machine, code, sp, hl, de, pc=0, vec0=None):

    c = Machine(code, data=DATA_IMAGE, inputs=INPUTS, config=config_for(vec0))
    c.load_state(INIT_R, hl, de, sp, INIT_C, INIT_Z, TICK0, PC=pc)
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

def check_case(name, code, sp, hl, de, expect_err, expect_commit=None, pc=0,
            vec0=None):

    runs = [("reference", run_reference(code, sp, hl, de, pc, vec0)),
            ("torch", run_circuit(TorchCircuit, code, sp, hl, de, pc, vec0)),
            ("triton", run_circuit(TritonCircuit, code, sp, hl, de, pc, vec0))]
    ref_pre = dict(r=list(INIT_R), HL=hl, DE=de, MB=0, SP=sp, PC=pc, C=INIT_C, Z=INIT_Z,
                   ipos=0, oplen=0, tick=TICK0, status=0, fault_reason=0, fault_addr=0)
    assert set(ref_pre) == set(VIEW_FIELDS), (
        "this suite's field list and the comparison set have drifted apart")
    for label, (raised, post, data, code_img, out) in runs:
        assert_widths(post, (name, label, "post-state"))
        if expect_err:
            assert raised, (name, label, "the violating tick did not report an error")
            assert post["status"] == 3, (name, label, "expected status 3", post)
            for k, v in ref_pre.items():
                if k in FAULT_WRITES:
                    continue
                assert post[k] == v, (name, label, "error tick changed a state field", k, v, post[k])
            assert post["fault_reason"] != 0, (
                name, label, "an error tick stopped without naming a cause", post)
            assert post["fault_addr"] == pc, (
                name, label, "fault_addr must be the faulting instruction's address",
                pc, post)
            assert data == list(DATA_IMAGE), (name, label, "error tick changed DATA")
            assert code_img == code, (name, label, "error tick changed CODE")
            assert out == b"", (name, label, "error tick changed the output stream", out)
        else:
            exp_state, exp_data, exp_out = expect_commit
            assert not raised, (name, label, "a legal boundary tick reported an error")
            assert post == exp_state, (name, label, "legal tick state mismatch", exp_state, post)
            assert data == list(exp_data), (name, label, "legal tick DATA mismatch")
            assert out == exp_out, (name, label, "legal tick output mismatch", exp_out, post)

    r_raised, r_post, r_data, r_code, r_out = runs[0][1]
    for label, (raised, post, data, code_img, out) in runs[1:]:
        assert raised == r_raised, (name, label, "error verdict differs from the reference")
        assert all(post[k] == r_post[k] for k in VIEW_FIELDS), \
            (name, label, "state differs from the reference", post, r_post)
        assert (data, code_img, out) == (r_data, r_code, r_out), \
            (name, label, "memory or output differs from the reference")
    return "err" if expect_err else "ok"

def legal_expect(kind, sp, hl, de, imm=None, pc=0):

    r = list(INIT_R)
    d = bytearray(DATA_IMAGE)
    out = b""
    st = dict(r=r, HL=hl, DE=de, MB=0, SP=sp, PC=pc + 1, C=INIT_C, Z=INIT_Z, ipos=0,
              oplen=0, tick=TICK0 + 1, status=0, fault_reason=0, fault_addr=0)
    ret_lo, ret_hi = (pc + 3) & 0xFF, (pc + 3) >> 8
    if kind in ("PUSHW HL", "PUSHW DE", "POPW HL", "POPW DE", "STW [HL], DE",
                "STW [DE], HL", "LDW DE, [HL]", "LDW HL, [DE]",
                "MOVW SP, HL", "MOVW SP, DE"):
        st["PC"] = pc + 2
    if kind in ("LDX r0, [HL+i]", "STX [HL+i], r0", "ADD SP, i8"):
        st["PC"] = pc + 3
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

    elif kind in ("PUSHW HL", "PUSHW DE"):
        v = hl if kind == "PUSHW HL" else de
        st["SP"] = sp - 2
        d[sp - 2] = v & 0xFF
        d[sp - 1] = (v >> 8) & 0xFF
    elif kind in ("POPW HL", "POPW DE"):
        v = d[sp] | (d[sp + 1] << 8)
        if kind == "POPW HL":
            st["HL"] = v
        else:
            st["DE"] = v
        st["SP"] = sp + 2
    elif kind == "STW [HL], DE":
        d[hl] = de & 0xFF
        d[hl + 1] = (de >> 8) & 0xFF
    elif kind == "STW [DE], HL":
        d[de] = hl & 0xFF
        d[de + 1] = (hl >> 8) & 0xFF
    elif kind == "LDW DE, [HL]":
        st["DE"] = d[hl] | (d[hl + 1] << 8)
    elif kind == "LDW HL, [DE]":
        st["HL"] = d[de] | (d[de + 1] << 8)
    elif kind == "LDX r0, [HL+i]":
        st["r"] = [d[(hl + imm) & 0xFFFF], r[1], r[2], r[3]]
    elif kind == "STX [HL+i], r0":
        d[(hl + imm) & 0xFFFF] = r[0]
    elif kind == "ADD SP, i8":
        st["SP"] = (sp + imm) & 0xFFFF
    elif kind == "MOVW SP, HL":
        st["SP"] = hl
    elif kind == "MOVW SP, DE":
        st["SP"] = de
    else:
        raise AssertionError(f"unknown case kind {kind!r}")
    return st, bytes(d), out

STACK_CASES = (
    ("PUSH r0", bytes([0xF0 | 0])),
    ("POP r0", bytes([0xF4 | 0])),
    ("CALL a16", bytes([0x0E, CALL_TGT & 0xFF, CALL_TGT >> 8])),
    ("RET", bytes([0x08])),
    ("EXT 0", bytes([0x70, 0x70, 0x00])),

    ("PUSHW HL", bytes([0x70, 0x38])),
    ("PUSHW DE", bytes([0x70, 0x39])),
    ("POPW HL", bytes([0x70, 0x3A])),
    ("POPW DE", bytes([0x70, 0x3B])),
)

STACK_ERR = {
    "PUSH r0": lambda sp: sp < 1,
    "POP r0": lambda sp: sp >= DATA_SIZE,
    "CALL a16": lambda sp: sp < 2,
    "RET": lambda sp: sp + 2 > DATA_SIZE,
    "EXT 0": lambda sp: sp < 2,
    "PUSHW HL": lambda sp: sp < 2,
    "PUSHW DE": lambda sp: sp < 2,
    "POPW HL": lambda sp: sp + 2 > DATA_SIZE,
    "POPW DE": lambda sp: sp + 2 > DATA_SIZE,
}
MEM_CASES = (
    ("MOV r0, [HL]", bytes([0xE0 | 0]), "HL"),
    ("MOV [HL], r0", bytes([0xE4 | 0]), "HL"),
    ("OUTM", bytes([0x06]), "HL"),
    ("MOV r0, [DE]", bytes([0xE8 | 0]), "DE"),
    ("MOV [DE], r0", bytes([0xEC | 0]), "DE"),
    ("OUTDE", bytes([0x07]), "DE"),
)

WIDE_CASES = (
    ("STW [HL], DE", bytes([0x70, 0x3C]), "HL"),
    ("STW [DE], HL", bytes([0x70, 0x3D]), "DE"),
    ("LDW DE, [HL]", bytes([0x70, 0x3E]), "HL"),
    ("LDW HL, [DE]", bytes([0x70, 0x3F]), "DE"),
)

FRAME_CASES = (
    ("LDX r0, [HL+i]", "LDX r0, [HL-1]", 1, -1, False),
    ("LDX r0, [HL+i]", "LDX r0, [HL-1]", 0, -1, True),
    ("LDX r0, [HL+i]", "LDX r0, [HL+1]", 4094, 1, False),
    ("LDX r0, [HL+i]", "LDX r0, [HL+1]", 4095, 1, True),
    ("LDX r0, [HL+i]", "LDX r0, [HL+127]", 3968, 127, False),
    ("STX [HL+i], r0", "STX [HL-1], r0", 1, -1, False),
    ("STX [HL+i], r0", "STX [HL-1], r0", 0, -1, True),
    ("STX [HL+i], r0", "STX [HL+1], r0", 4095, 1, True),
    ("STX [HL+i], r0", "STX [HL-128], r0", 128, -128, False),
    ("STX [HL+i], r0", "STX [HL], r0", 4095, 0, False),
)

ADD_SP_CASES = (
    (4096, 1, True), (4096, 0, False), (4096, -128, False), (4095, 1, False),
    (0, -1, True), (0, 0, False), (1, -1, False), (3, -3, False), (127, -128, True),
)

MOVW_SP_CASES = (
    ("MOVW SP, HL", bytes([0x70, 0x34]), "HL"),
    ("MOVW SP, DE", bytes([0x70, 0x35]), "DE"),
)
SP_PAIR_VALUES = (0, 4095, 4096, 4097, 65535)

def _seed_last_byte(addr):

    return DATA_IMAGE[addr] | (DATA_IMAGE[addr + 1] << 8)

def test_stack_boundaries():
    tot = {"ok": 0, "err": 0}
    for kind, head in STACK_CASES:
        for pc in PC_SITES:
            code = build_code(head, pc=pc)
            vec0 = VEC_TGT if kind.startswith("EXT") else None
            for sp in STACK_SP:
                expect_err = STACK_ERR[kind](sp)
                commit = None if expect_err else legal_expect(kind, sp, 0, 0, pc=pc)
                tot[check_case(f"{kind} @ SP={sp} PC={pc}", code, sp, 0, 0,
                               expect_err, commit, pc, vec0)] += 1
        print(f"  {kind:9s}: SP {STACK_SP} at PC {PC_SITES} -> "
              f"{sum(1 for sp in STACK_SP if STACK_ERR[kind](sp)) * len(PC_SITES)} error / "
              f"{sum(1 for sp in STACK_SP if not STACK_ERR[kind](sp)) * len(PC_SITES)} legal")
    assert tot == {"ok": 94, "err": 32}, tot
    print(f"  stack instructions x SP boundary x {len(PC_SITES)} addresses: "
          f"{sum(tot.values())} cases (error {tot['err']} + legal {tot['ok']})")
    return tot

def test_memory_boundaries():
    tot = {"ok": 0, "err": 0}
    for kind, head, ptr in MEM_CASES:
        for pc in PC_SITES:
            code = build_code(head, pc=pc)
            for addr in LIMIT_ADDRS:
                hl, de = (addr, 0) if ptr == "HL" else (0, addr)
                expect_err = addr >= DATA_SIZE
                commit = None if expect_err else legal_expect(kind, DATA_SIZE, hl, de,
                                                              pc=pc)
                tot[check_case(f"{kind} @ {ptr}={addr} PC={pc}", code, DATA_SIZE, hl, de,
                               expect_err, commit, pc)] += 1
    assert tot == {"ok": 24, "err": 12}, tot
    print(f"  memory instructions x HL/DE in {LIMIT_ADDRS} x {len(PC_SITES)} addresses: "
          f"{sum(tot.values())} cases (error {tot['err']} + legal {tot['ok']})")
    return tot

def test_wide_memory_boundaries():

    tot = {"ok": 0, "err": 0}
    for kind, head, ptr in WIDE_CASES:
        for pc in PC_SITES:
            code = build_code(head, pc=pc)
            for addr in WIDE_ADDRS:
                hl, de = (addr, 0) if ptr == "HL" else (0, addr)
                expect_err = addr + 1 >= DATA_SIZE
                commit = None if expect_err else legal_expect(kind, DATA_SIZE, hl, de,
                                                              pc=pc)
                tot[check_case(f"{kind} @ {ptr}={addr} PC={pc}", code, DATA_SIZE, hl, de,
                               expect_err, commit, pc)] += 1
        print(f"  {kind:13s}: {ptr} {WIDE_ADDRS} at PC {PC_SITES} -> "
              f"{sum(1 for a in WIDE_ADDRS if a + 1 >= DATA_SIZE) * len(PC_SITES)} error / "
              f"{sum(1 for a in WIDE_ADDRS if a + 1 < DATA_SIZE) * len(PC_SITES)} legal")
    assert tot == {"ok": 32, "err": 16}, tot
    print(f"  16-bit memory instructions x {ptr} in {WIDE_ADDRS} x {len(PC_SITES)} "
          f"addresses: {sum(tot.values())} cases (error {tot['err']} + legal {tot['ok']})")
    return tot

def test_frame_boundaries():

    tot = {"ok": 0, "err": 0}
    for kind, src, hl, imm, expect_err in FRAME_CASES:
        for pc in PC_SITES:
            code = build_code(asm(src), pc=pc)
            commit = None if expect_err else legal_expect(kind, DATA_SIZE, hl, 0, imm,
                                                          pc=pc)
            tot[check_case(f"{src} @ HL={hl} PC={pc}", code, DATA_SIZE, hl, 0,
                           expect_err, commit, pc)] += 1
    assert tot == {"ok": 12, "err": 8}, tot
    print(f"  frame-relative accesses x {len(PC_SITES)} addresses: {sum(tot.values())} "
          f"cases (error {tot['err']} + legal {tot['ok']})")
    return tot

def test_sp_boundaries():

    tot = {"ok": 0, "err": 0}
    for sp, imm, expect_err in ADD_SP_CASES:
        for pc in PC_SITES:
            code = build_code(asm(f"ADD SP, {imm}"), pc=pc)
            commit = None if expect_err else legal_expect("ADD SP, i8", sp, 0, 0, imm,
                                                          pc=pc)
            tot[check_case(f"ADD SP, {imm} @ SP={sp} PC={pc}", code, sp, 0, 0,
                           expect_err, commit, pc)] += 1
    for kind, head, ptr in MOVW_SP_CASES:
        for pc in PC_SITES:
            code = build_code(head, pc=pc)
            for val in SP_PAIR_VALUES:
                hl, de = (val, 0) if ptr == "HL" else (0, val)
                expect_err = val > DATA_SIZE
                commit = None if expect_err else legal_expect(kind, DATA_SIZE, hl, de,
                                                              pc=pc)
                tot[check_case(f"{kind} @ {ptr}={val} PC={pc}", code, DATA_SIZE, hl, de,
                               expect_err, commit, pc)] += 1
    assert tot == {"ok": 24, "err": 14}, tot
    print(f"  stack-pointer writes x {len(PC_SITES)} addresses: {sum(tot.values())} "
          f"cases (error {tot['err']} + legal {tot['ok']})")
    return tot

if __name__ == "__main__":
    print("error-tick atomicity (reference + both circuits):")
    a = test_stack_boundaries()
    b = test_memory_boundaries()
    c = test_wide_memory_boundaries()
    d = test_frame_boundaries()
    e = test_sp_boundaries()
    cases = sum(a.values()) + sum(b.values()) + sum(c.values()) + sum(d.values()) + sum(e.values())
    err = a["err"] + b["err"] + c["err"] + d["err"] + e["err"]
    ok = a["ok"] + b["ok"] + c["ok"] + d["ok"] + e["ok"]
    print(f"  {cases} cases x 3 implementations = {cases * 3} probes, "
          f"error {err} + legal {ok}")
    print("error-tick atomicity: all passed")