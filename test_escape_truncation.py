"""Escape-operand truncation acceptance for the reference and both circuits.

The escape space (prefix 0x70) grows over time, and every row that declares an
operand byte (row["l"] >= 1) hands the fetcher a byte that can sit past the end
of the loaded image. The contract for that state is FETCH_OOB on the tick that
wanted the byte, and both circuits carry the same decision. The suite derives
its code-point list from the decode table at run time -- no opcode is written
down here -- so a future escape instruction with an operand byte joins the
sweep the day it lands. Each code point is walked in three shapes: the image
cut off mid-instruction, the instruction exactly filling the image, and a
one-byte NOP shift that moves the truncation away from address 0. Legal
boundary cases must not report an error, and every case is compared field by
field across the three implementations, with the error ticks asserted atomic.
"""
from __future__ import annotations

import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import TritonCircuit
from test_state_contract import FAULT_WRITES, VIEW_FIELDS, assert_widths
from test_error_atomicity import (DATA_IMAGE, INIT_C, INIT_R, INIT_S, INIT_V,
                                  INIT_Z, TICK0, run_circuit, run_reference)

FETCH_OOB = ISA.CAUSE["FETCH_OOB"]

CODEPOINTS = tuple(sorted(sub for sub, row in ISA.ESCAPE.items() if row["l"] >= 1))

WITHOUT_OPERAND = tuple(sorted(sub for sub, row in ISA.ESCAPE.items()
                               if row["l"] == 0))
assert WITHOUT_OPERAND, \
    "every escape row now declares an operand byte; re-read this derivation"

IMMEDIATES = (0x00, 0x01, 0x80, 0xFF)

HL0 = DE0 = 0
SP0 = 0x0100

def _nop():

    cands = [op for op, row in ISA.SINGLE.items()
             if row["l"] == 0 and row["alu"] == "NOP"]
    assert len(cands) == 1, ("the filler byte is not one single-byte code point",
                             cands)
    return cands[0]

NOP = _nop()

def families():

    fams = {}
    for sub in CODEPOINTS:
        fams.setdefault(ISA.ESCAPE[sub]["alu"], []).append(sub)
    return fams

def shapes(sub):

    row = ISA.ESCAPE[sub]
    head = bytes([ISA.ESCAPE_PREFIX, sub])
    yield ("truncated at image end", head, 0, True)
    for v in IMMEDIATES:
        yield (f"exact fit, operand {v:#04x}",
               head + bytes([v]) * row["l"], 0, False)
    yield (f"{NOP:#04x} in front, truncated at PC 1",
           bytes([NOP]) + head, 1, True)

def probe(name, code, pc, operand_crosses):

    runs = [("reference", run_reference(code, SP0, HL0, DE0, pc)),
            ("torch", run_circuit(TorchCircuit, code, SP0, HL0, DE0, pc)),
            ("triton", run_circuit(TritonCircuit, code, SP0, HL0, DE0, pc))]
    pre = dict(r=list(INIT_R), HL=HL0, DE=DE0, MB=0, SP=SP0, PC=pc, C=INIT_C,
               Z=INIT_Z, S=INIT_S, V=INIT_V, TDEPTH=0, STC_COUNT=0, STC_FIRST=0,
               ipos=0, oplen=0, tick=TICK0, status=0, fault_reason=0, fault_addr=0)
    assert set(pre) == set(VIEW_FIELDS), \
        "this suite's pre-state and the shared field list have drifted apart"
    for label, (raised, post, data, code_img, out) in runs:
        assert_widths(post, (name, label, "post-state"))
        assert raised == (post["status"] == 3), \
            (name, label, "the raised verdict and the latched status disagree",
             raised, post)
        if post["status"] == 3:
            for k, v in pre.items():
                if k in FAULT_WRITES:
                    continue
                assert post[k] == v, \
                    (name, label, "error tick changed a state field", k, v, post[k])
            assert post["fault_reason"] != 0, \
                (name, label, "an error tick stopped without naming a cause", post)
            assert post["fault_addr"] == pc, \
                (name, label, "fault_addr must be the faulting instruction's "
                              "address", pc, post)
            assert data == list(DATA_IMAGE), (name, label, "error tick changed DATA")
            assert code_img == code, (name, label, "error tick changed CODE")
            assert out == b"", (name, label, "error tick changed the output stream",
                                out)
        else:
            assert post["fault_reason"] == 0, \
                (name, label, "status and fault_reason disagree", post)
        if operand_crosses:
            assert post["status"] == 3 and post["fault_reason"] == FETCH_OOB, \
                (name, label,
                 "the operand byte past CODELEN must fault FETCH_OOB, got "
                 f"status={post['status']} reason="
                 f"{ISA.CAUSE_NAME.get(post['fault_reason'])}", post)
        else:
            assert post["fault_reason"] != FETCH_OOB, \
                (name, label, "an exactly fitting image was rejected as "
                              "FETCH_OOB", post)
    ref = runs[0][1]
    for label, run in runs[1:]:
        assert run[0] == ref[0], \
            (name, label, "error verdict differs from the reference")
        assert all(run[1][k] == ref[1][k] for k in VIEW_FIELDS), \
            (name, label, "state differs from the reference", run[1], ref[1])
        assert run[2:] == ref[2:], \
            (name, label, "memory, code or output differs from the reference")
    return runs

def test_escape_truncation():
    fams = families()
    per_cp = 2 + len(IMMEDIATES)
    total = len(CODEPOINTS) * per_cp
    print(f"census: {len(ISA.ESCAPE)} escape rows, {len(CODEPOINTS)} declare an "
          f"operand byte (l >= 1: "
          + ", ".join(f"{alu} {len(subs)}" for alu, subs in fams.items())
          + f"); {len(WITHOUT_OPERAND)} rows declare none and stay out of scope -- "
            "an l == 0 row takes no operand byte, so no fetch on its behalf can "
            "cross CODELEN (its prefix+subcode pair is one shared path for the "
            "whole space, walked to its bound by every truncated shape below)")
    print(f"census: {len(fams)} families x {per_cp} shapes "
          f"({len(IMMEDIATES)} exact-fit immediates {IMMEDIATES}) = {total} cases "
          f"x 3 implementations = {total * 3} one-step probes")
    errs = {lab: 0 for lab in ("reference", "torch", "triton")}
    for alu, subs in fams.items():
        fam_err = 0
        for sub in subs:
            mnem = ISA.ESCAPE[sub]["mnem"]
            for shape, code, pc, crosses in shapes(sub):
                name = f"escape {sub:#04x} {mnem} -- {shape}"
                runs = probe(name, code, pc, crosses)
                if runs[0][1][1]["status"] == 3:
                    fam_err += 1
                    for lab, _run in runs:
                        errs[lab] += 1
        print(f"  {alu}: {len(subs)} codepoint(s) "
              f"{', '.join(f'{s:#04x}' for s in subs)} -> {len(subs) * per_cp} cases "
              f"(faulting {fam_err}, clean {len(subs) * per_cp - fam_err})")
    assert all(errs[lab] == errs["reference"] for lab in errs), \
        ("the implementations did not see the same number of faulting ticks", errs)
    print("summary: " + ", ".join(f"{lab} {errs[lab]} faulting / "
                                  f"{total - errs[lab]} clean" for lab in errs))
    print(f"escape truncation: {total} cases x 3 implementations, all agreed")
    return total

if __name__ == "__main__":
    test_escape_truncation()
    print("escape truncation: all passed")