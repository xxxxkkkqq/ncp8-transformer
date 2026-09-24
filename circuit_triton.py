"""Fused Triton kernel implementation of the NCP-8 datapath.

Third implementation of the same ISA semantics: the whole fetch-decode-execute
cycle runs inside a single kernel. Public interface is kept compatible with the
tensor implementation so that both can be driven by the same equivalence tests.

Two execution paths share the same per-tick body:
  * TritonCircuit.step(): one launch per tick, driven by the host;
  * TritonBatch: a resident batched executor that keeps B machines in device
    buffers (CODE [B,4096], DATA [B,4096], INPUTS [B,max_in], OUTBUF [B,OUT_CAP],
    state [B,14]) and runs the tick loop inside the kernel, so a whole program run
    costs one launch for the whole batch. The per-machine status drives the loop:
    a machine that has halted, errored or used up its tick budget stops advancing
    while the rest of the batch keeps running. run_batch() is the one-shot form and
    TritonCircuit.run_resident() is the B = 1 form.

Error contract: a violating tick writes status = 3 only; all other state and the
tick counter stay unchanged. The same body also latches OVERRUN (status = 2) when
the machine's tick reaches its budget and refuses the output byte past OUT_CAP,
both as atomic ticks that commit nothing else. Every store address is masked into
its own machine's buffer, so a write cannot leave the row it was given, and
check_state()/set_state()/load_state() refuse a state outside the declared widths.

The escape-prefix dispatch chain assigns the carry explicitly in the operations
that leave it alone, written as C & 1 rather than a plain copy. With this many
branches the Triton frontend can otherwise yield a branch's computed value into
the carry the branch is supposed to leave untouched - a wrong flag with a correct
data path - and an explicit assignment is what keeps it out. C is one bit wide,
so C & 1 is C. The subcode enumeration in test_isa_v2_equivalence.py reports
exactly that failure mode, which is how the DIV/MOD, NOT and MULH sites were
found.

State tensor layout: [r0,r1,r2,r3, HL, DE, PC, SP, C, Z, ipos, oplen, tick, status]
"""
import torch
import triton
import triton.language as tl
from typing import NamedTuple

DATA_SIZE = 4096
CODE_SIZE = 4096
OUT_CAP = 8192




ESC_DIV = tl.constexpr(0x100)
ESC_MOD = tl.constexpr(0x101)
ESC_CMP = tl.constexpr(0x102)
ESC_NOT = tl.constexpr(0x103)
ESC_NEG = tl.constexpr(0x104)
ESC_ROL = tl.constexpr(0x105)
ESC_ROR = tl.constexpr(0x106)
ESC_ADD_HLDE = tl.constexpr(0x107)
ESC_SUB_HLDE = tl.constexpr(0x108)
ESC_XCHG = tl.constexpr(0x109)
ESC_EXT = tl.constexpr(0x10A)
ESC_STC = tl.constexpr(0x10B)
ESC_LDC = tl.constexpr(0x10C)
ESC_BAD = tl.constexpr(0x10D)

ESC_MOVW_HL_DE = tl.constexpr(0x10E)
ESC_MOVW_DE_HL = tl.constexpr(0x10F)
ESC_MOVW_HL_SP = tl.constexpr(0x110)
ESC_MOVW_DE_SP = tl.constexpr(0x111)
ESC_MOVW_SP_HL = tl.constexpr(0x112)
ESC_MOVW_SP_DE = tl.constexpr(0x113)
ESC_PUSHW_HL = tl.constexpr(0x114)
ESC_PUSHW_DE = tl.constexpr(0x115)
ESC_POPW_HL = tl.constexpr(0x116)
ESC_POPW_DE = tl.constexpr(0x117)
ESC_STW_HLDE = tl.constexpr(0x118)
ESC_STW_DEHL = tl.constexpr(0x119)
ESC_LDW_DEHL = tl.constexpr(0x11A)
ESC_LDW_HLDE = tl.constexpr(0x11B)
ESC_LDX = tl.constexpr(0x11C)
ESC_STX = tl.constexpr(0x11D)
ESC_ADD_SP = tl.constexpr(0x11E)
ESC_MULH = tl.constexpr(0x11F)

def _power_of_two_or_die(name, size):


    if size <= 0 or size & (size - 1):
        raise ValueError(f"{name}={size} is not a positive power of two; the store "
                         f"masks would let a write leave its machine's own buffer")



for _name, _size in (("CODE_SIZE", CODE_SIZE), ("DATA_SIZE", DATA_SIZE),
                     ("OUT_CAP", OUT_CAP)):
    _power_of_two_or_die(_name, _size)
CODE_MASK = tl.constexpr(CODE_SIZE - 1)


def check_state(R, HL, DE, SP, C, Z, tick=0, PC=0, ipos=0, oplen=0, status=0, where=""):






    def outside(field, value, lo, hi):
        raise ValueError(f"{where}state field {field} is {value}, outside [{lo}, {hi}]")

    for i, v in enumerate(list(R)):
        if not 0 <= v < 256:
            outside(f"r[{i}]", v, 0, 255)
    for field, v in (("HL", HL), ("DE", DE), ("PC", PC)):
        if not 0 <= v < 65536:
            outside(field, v, 0, 65535)
    if not 0 <= SP <= DATA_SIZE:
        outside("SP", SP, 0, DATA_SIZE)
    for field, v in (("C", C), ("Z", Z)):
        if v not in (0, 1):
            outside(field, v, 0, 1)
    if not 0 <= oplen <= OUT_CAP:
        outside("oplen", oplen, 0, OUT_CAP)
    for field, v in (("ipos", ipos), ("tick", tick)):
        if v < 0:
            outside(field, v, 0, "unbounded")
    if status not in (0, 1, 2, 3):
        outside("status", status, 0, 3)


@triton.jit
def _get4(v0, v1, v2, v3, idx):
    return tl.where(idx == 0, v0, tl.where(idx == 1, v1, tl.where(idx == 2, v2, v3)))


@triton.jit
def _wr(r0, r1, r2, r3, d, v):





    w = v & 255
    return (tl.where(d == 0, w, r0), tl.where(d == 1, w, r1),
            tl.where(d == 2, w, r2), tl.where(d == 3, w, r3))



@triton.jit
def _dec_esc(sub):













    if sub <= 0x2F:
        eop = ESC_DIV + (sub >> 4); d = (sub >> 2) & 3; s = sub & 3; lx = 0
    elif sub >= 0x30 and sub <= 0x35:
        eop = ESC_MOVW_HL_DE + (sub - 0x30); d = 0; s = 0; lx = 0
    elif sub >= 0x38 and sub <= 0x3F:
        eop = ESC_PUSHW_HL + (sub - 0x38); d = 0; s = 0; lx = 0
    elif sub >= 0x40 and sub <= 0x4F:
        eop = ESC_NOT + ((sub >> 2) & 3); d = sub & 3; s = sub & 3; lx = 0
    elif sub >= 0x50 and sub <= 0x53:
        eop = ESC_LDX; d = sub & 3; s = sub & 3; lx = 1
    elif sub >= 0x54 and sub <= 0x57:
        eop = ESC_STX; d = sub & 3; s = sub & 3; lx = 1
    elif sub == 0x58:
        eop = ESC_ADD_SP; d = 0; s = 0; lx = 1
    elif sub >= 0x60 and sub <= 0x62:
        eop = ESC_ADD_HLDE + (sub - 0x60); d = 0; s = 0; lx = 0
    elif sub == 0x70:
        eop = ESC_EXT; d = 0; s = 0; lx = 1
    elif sub >= 0x80 and sub <= 0x87:
        eop = ESC_STC + ((sub >> 2) & 1); d = sub & 3; s = sub & 3; lx = 0
    elif sub >= 0x90 and sub <= 0x9F:
        eop = ESC_MULH; d = (sub >> 2) & 3; s = sub & 3; lx = 0
    else:
        eop = ESC_BAD; d = 0; s = 0; lx = 0
    return eop, d, s, lx


@triton.jit
def _tick(CODE, DATA, INPUTS, OUTBUF, S, CODELEN, INLEN, BD, DS, OC):

















    r0 = tl.load(S + 0); r1 = tl.load(S + 1); r2 = tl.load(S + 2); r3 = tl.load(S + 3)
    HL = tl.load(S + 4); DE = tl.load(S + 5); PC = tl.load(S + 6); SP = tl.load(S + 7)
    C = tl.load(S + 8); Z = tl.load(S + 9); IPO = tl.load(S + 10); OL = tl.load(S + 11)

    nR0, nR1, nR2, nR3 = r0, r1, r2, r3
    nHL, nDE, nSP = HL, DE, SP
    nC, nZ, nIPO = C, Z, IPO
    nOL = OL
    nPC = PC + 1
    NST = 0
    OVAL = 0; OEN = 0
    A1 = 0; V1 = 0; E1 = 0
    A2 = 0; V2 = 0; E2 = 0
    A3 = 0; V3 = 0; E3 = 0
    err = 0



    eop = 0xFFFF
    ed = 0; es = 0; elen = 1

    if PC < CODELEN:
        op = tl.load(CODE + PC)
        eop = op
        ed = op & 3
        es = op & 3
        if op == 0x70:


            if PC + 2 > CODELEN:
                err = 1
            else:
                sub = tl.load(CODE + PC + 1)
                eop, ed, es, elx = _dec_esc(sub)
                elen = 2 + elx
                nPC = PC + elen
                if PC + elen > CODELEN:


                    eop = 0xFFFF
                    err = 1

        if eop <= 0x1F:
            if eop == 0x00:
                NST = 1
            elif eop == 0x01:
                pass
            elif eop == 0x02:
                nHL = (HL + 1) & 0xFFFF
            elif eop == 0x03:
                nHL = (HL - 1) & 0xFFFF
            elif eop == 0x04:
                nDE = (DE + 1) & 0xFFFF
            elif eop == 0x05:
                nC = 0
            elif eop == 0x06:
                if HL >= DS:
                    err = 1
                else:
                    OVAL = tl.load(DATA + HL); OEN = 1; nHL = (HL + 1) & 0xFFFF
            elif eop == 0x07:
                if DE >= DS:
                    err = 1
                else:
                    OVAL = tl.load(DATA + DE); OEN = 1; nDE = (DE + 1) & 0xFFFF
            elif eop == 0x08:
                if SP + 1 >= DS:
                    err = 1
                else:
                    nPC = (tl.load(DATA + SP) << 8) | tl.load(DATA + SP + 1)
                    nSP = SP + 2
            elif eop >= 0x09 and eop <= 0x0D:
                if PC + 3 > CODELEN:
                    err = 1
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    if eop == 0x09:
                        nPC = t
                    elif eop == 0x0A:
                        nPC = t if Z == 1 else PC + 3
                    elif eop == 0x0B:
                        nPC = t if Z == 0 else PC + 3
                    elif eop == 0x0C:
                        nPC = t if C == 1 else PC + 3
                    else:
                        nPC = t if C == 0 else PC + 3
            elif eop == 0x0E:
                if SP < 2 or PC + 3 > CODELEN:
                    err = 1
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    ret = PC + 3
                    A1 = SP - 1; V1 = ret & 0xFF; E1 = 1
                    A2 = SP - 2; V2 = (ret >> 8) & 0xFF; E2 = 1
                    nSP = SP - 2; nPC = t
            elif eop == 0x0F or eop == 0x10:
                if PC + 3 > CODELEN:
                    err = 1
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    nPC = PC + 3
                    if eop == 0x0F:
                        nHL = t
                    else:
                        nDE = t
            elif eop == 0x11 or eop == 0x12:
                if PC + 2 > CODELEN:
                    err = 1
                else:
                    rs = _get4(r0, r1, r2, r3, tl.load(CODE + PC + 1) & 3)
                    nPC = PC + 2
                    if eop == 0x11:
                        nHL = (HL + rs) & 0xFFFF
                    else:
                        nDE = (DE + rs) & 0xFFFF
            elif eop == 0x13:
                nPC = HL & 0xFFFF
            elif eop >= 0x14 and eop <= 0x1F:
                d = eop & 3
                if eop < 0x18:
                    v = PC
                elif eop < 0x1C:
                    v = SP & 255
                else:
                    v = Z + 2 * C
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, d, v)
            else:
                err = 1
        elif eop >= 0x20 and eop <= 0x5F:

            f = eop & 0xF
            d = (f >> 2) & 3
            a = _get4(r0, r1, r2, r3, d)
            b = _get4(r0, r1, r2, r3, f & 3)
            k = eop >> 4
            v = a
            if k == 2:
                v = a & b; nZ = (v == 0).to(tl.int32)
            elif k == 3:
                v = a | b; nZ = (v == 0).to(tl.int32)
            elif k == 4:
                v = a ^ b; nZ = (v == 0).to(tl.int32)
            elif k == 5:
                t = a * b; v = t & 255
                nC = (t > 255).to(tl.int32); nZ = (v == 0).to(tl.int32)
            else:
                err = 1
            nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, d, v)
        elif eop >= 0x80 and eop <= 0xCF:
            f = eop & 0xF
            d = (f >> 2) & 3
            a = _get4(r0, r1, r2, r3, d)
            b = _get4(r0, r1, r2, r3, f & 3)
            k = eop >> 4
            v = a
            if k == 8:
                t = a + b; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
            elif k == 9:
                v = (a - b) & 255; nC = (a < b).to(tl.int32); nZ = (v == 0).to(tl.int32)
            elif k == 10:
                t = a + b + C; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
            elif k == 11:
                t = a - b - C; v = t & 255; nC = (t < 0).to(tl.int32); nZ = (v == 0).to(tl.int32)
            elif k == 12:
                v = b
            else:
                err = 1
            nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, d, v)
        elif eop >= 0xD0 and eop <= 0xDF:
            if PC + 2 > CODELEN:
                err = 1
            else:
                i8 = tl.load(CODE + PC + 1)
                rr = _get4(r0, r1, r2, r3, eop & 3)
                nPC = PC + 2
                if eop <= 0xD3:
                    v = i8
                elif eop <= 0xD7:
                    t = rr + i8; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
                elif eop <= 0xDB:
                    v = (rr - i8) & 255; nC = (rr < i8).to(tl.int32); nZ = (v == 0).to(tl.int32)
                else:
                    t = rr + i8 + C; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, eop & 3, v)
        elif eop >= 0x60 and eop <= 0x6F:
            d = eop & 3
            rr = _get4(r0, r1, r2, r3, d)
            v = rr
            if eop < 0x64:
                nC = rr >> 7; v = (rr << 1) & 255; nZ = (v == 0).to(tl.int32)
            elif eop < 0x68:
                nC = rr & 1; v = rr >> 1; nZ = (v == 0).to(tl.int32)
            elif eop < 0x6C:
                nZ = (rr == 0).to(tl.int32)
                v = rr
            else:
                if PC + 3 > CODELEN:
                    err = 1
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    v = (rr - 1) & 255
                    nPC = PC + 3
                    if v != 0:
                        nPC = t
            nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, d, v)
        elif eop >= 0xE0 and eop <= 0xFF:
            d = eop & 3
            rr = _get4(r0, r1, r2, r3, d)
            v = rr
            if eop < 0xE4:
                if HL >= DS:
                    err = 1
                else:
                    v = tl.load(DATA + HL)
            elif eop < 0xE8:
                if HL >= DS:
                    err = 1
                else:
                    A1 = HL; V1 = rr; E1 = 1
            elif eop < 0xEC:
                if DE >= DS:
                    err = 1
                else:
                    v = tl.load(DATA + DE)
            elif eop < 0xF0:
                if DE >= DS:
                    err = 1
                else:
                    A1 = DE; V1 = rr; E1 = 1
            elif eop < 0xF4:
                if SP <= 0:
                    err = 1
                else:
                    A1 = SP - 1; V1 = rr; E1 = 1; nSP = SP - 1
            elif eop < 0xF8:
                if SP >= DS:
                    err = 1
                else:
                    v = tl.load(DATA + SP); nSP = SP + 1
            elif eop < 0xFC:
                OVAL = rr; OEN = 1
            else:
                if IPO < INLEN:
                    v = tl.load(INPUTS + IPO); nIPO = IPO + 1
                else:
                    v = 0; nC = 1
            nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, d, v)
        elif eop >= 0x100:








            if eop == ESC_DIV or eop == ESC_MOD:
                a = _get4(r0, r1, r2, r3, ed)
                b = _get4(r0, r1, r2, r3, es)
                if b == 0:
                    err = 1
                else:
                    if eop == ESC_DIV:
                        v = a // b
                    else:
                        v = a % b
                    nZ = (v == 0).to(tl.int32)
                    nC = C & 1
                    nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            elif eop == ESC_CMP:

                a = _get4(r0, r1, r2, r3, ed)
                b = _get4(r0, r1, r2, r3, es)
                nZ = (a == b).to(tl.int32)
                nC = (a < b).to(tl.int32)
            elif eop == ESC_NOT:
                rr = _get4(r0, r1, r2, r3, ed)
                v = (~rr) & 255
                nZ = (v == 0).to(tl.int32)
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
                nC = C & 1
            elif eop == ESC_NEG:
                rr = _get4(r0, r1, r2, r3, ed)
                v = (-rr) & 255
                nC = (rr != 0).to(tl.int32)
                nZ = (v == 0).to(tl.int32)
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            elif eop == ESC_ROL:

                rr = _get4(r0, r1, r2, r3, ed)
                nC = rr >> 7
                v = ((rr << 1) | C) & 255
                nZ = (v == 0).to(tl.int32)
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            elif eop == ESC_ROR:
                rr = _get4(r0, r1, r2, r3, ed)
                nC = rr & 1
                v = (rr >> 1) | (C << 7)
                nZ = (v == 0).to(tl.int32)
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            elif eop == ESC_ADD_HLDE:
                t = HL + DE
                nC = t >> 16
                nHL = t & 0xFFFF
            elif eop == ESC_SUB_HLDE:
                nC = (HL < DE).to(tl.int32)
                nHL = (HL - DE) & 0xFFFF
            elif eop == ESC_XCHG:
                nHL = DE
                nDE = HL
            elif eop == ESC_EXT:



                if PC + 3 > CODELEN:
                    err = 1
                else:
                    k = tl.load(CODE + PC + 2)
                    if k >= 16:
                        err = 1
                    else:
                        tgt = tl.load(CODE + 0x0F00 + 2 * k) \
                            | (tl.load(CODE + 0x0F00 + 2 * k + 1) << 8)
                        if tgt == 0 or SP < 2:
                            err = 1
                        else:
                            ret = PC + 3
                            A1 = SP - 1; V1 = ret & 0xFF; E1 = 1
                            A2 = SP - 2; V2 = (ret >> 8) & 0xFF; E2 = 1
                            nSP = SP - 2; nPC = tgt
            elif eop == ESC_STC:



                if HL >= CODELEN:
                    err = 1
                else:
                    wlo = tl.load(CODE + 0x0F20)
                    whi = tl.load(CODE + 0x0F21)
                    if HL < wlo or HL >= whi:
                        err = 1
                    else:
                        A3 = HL; V3 = _get4(r0, r1, r2, r3, ed); E3 = 1
            elif eop == ESC_LDC:

                if HL >= CODELEN:
                    err = 1
                else:
                    v = tl.load(CODE + HL)
                    nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            elif eop == ESC_MOVW_HL_DE:
                nHL = DE
            elif eop == ESC_MOVW_DE_HL:
                nDE = HL
            elif eop == ESC_MOVW_HL_SP:
                nHL = SP
            elif eop == ESC_MOVW_DE_SP:
                nDE = SP
            elif eop == ESC_MOVW_SP_HL:


                if HL > DS:
                    err = 1
                else:
                    nSP = HL
            elif eop == ESC_MOVW_SP_DE:
                if DE > DS:
                    err = 1
                else:
                    nSP = DE
            elif eop == ESC_PUSHW_HL or eop == ESC_PUSHW_DE:


                if SP < 2:
                    err = 1
                else:
                    if eop == ESC_PUSHW_HL:
                        v = HL
                    else:
                        v = DE
                    A1 = SP - 1; V1 = (v >> 8) & 0xFF; E1 = 1
                    A2 = SP - 2; V2 = v & 0xFF; E2 = 1
                    nSP = SP - 2
            elif eop == ESC_POPW_HL or eop == ESC_POPW_DE:
                if SP + 2 > DS:
                    err = 1
                else:
                    v = tl.load(DATA + SP) | (tl.load(DATA + SP + 1) << 8)
                    if eop == ESC_POPW_HL:
                        nHL = v
                    else:
                        nDE = v
                    nSP = SP + 2
            elif eop == ESC_STW_HLDE or eop == ESC_STW_DEHL:

                if eop == ESC_STW_HLDE:
                    adr = HL
                    v = DE
                else:
                    adr = DE
                    v = HL
                if adr + 1 >= DS:
                    err = 1
                else:
                    A1 = adr; V1 = v & 0xFF; E1 = 1
                    A2 = adr + 1; V2 = (v >> 8) & 0xFF; E2 = 1
            elif eop == ESC_LDW_DEHL or eop == ESC_LDW_HLDE:

                if eop == ESC_LDW_DEHL:
                    adr = HL
                else:
                    adr = DE
                if adr + 1 >= DS:
                    err = 1
                else:
                    v = tl.load(DATA + adr) | (tl.load(DATA + adr + 1) << 8)
                    if eop == ESC_LDW_DEHL:
                        nDE = v
                    else:
                        nHL = v
            elif eop == ESC_LDX or eop == ESC_STX:


                sx = tl.load(CODE + PC + 2)
                sx = sx - 256 * (sx >> 7)
                adr = (HL + sx) & 0xFFFF
                if adr >= DS:
                    err = 1
                elif eop == ESC_LDX:
                    v = tl.load(DATA + adr)
                    nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
                else:
                    A1 = adr; V1 = _get4(r0, r1, r2, r3, ed); E1 = 1
            elif eop == ESC_ADD_SP:

                sx = tl.load(CODE + PC + 2)
                sx = sx - 256 * (sx >> 7)
                v = (SP + sx) & 0xFFFF
                if v > DS:
                    err = 1
                else:
                    nSP = v
            elif eop == ESC_MULH:




                v = ((_get4(r0, r1, r2, r3, ed) * _get4(r0, r1, r2, r3, es)) >> 8) & 255
                nZ = (v == 0).to(tl.int32)
                nC = C & 1
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            else:
                err = 1
        else:
            err = 1
    else:
        err = 1

    OT = tl.load(S + 12)
    ST = tl.load(S + 13)
    NT = OT + 1
    if OEN == 1 and OL >= OC:

        err = 1
    if ST != 0:


        RST = ST
        RTK = OT
    elif OT >= BD:

        tl.store(S + 13, 2)
        RST = 2
        RTK = OT
    elif err == 1:

        tl.store(S + 13, 3)
        RST = 3
        RTK = OT
    else:
        nOL = OL + OEN
        tl.store(S + 0, nR0); tl.store(S + 1, nR1); tl.store(S + 2, nR2); tl.store(S + 3, nR3)
        tl.store(S + 4, nHL); tl.store(S + 5, nDE); tl.store(S + 6, nPC); tl.store(S + 7, nSP)
        tl.store(S + 8, nC); tl.store(S + 9, nZ); tl.store(S + 10, nIPO); tl.store(S + 11, nOL)
        tl.store(S + 12, NT)
        tl.store(S + 13, NST)


        if OEN == 1:
            tl.store(OUTBUF + (OL & (OC - 1)), OVAL)
        if E1 == 1:
            tl.store(DATA + (A1 & (DS - 1)), V1)
        if E2 == 1:
            tl.store(DATA + (A2 & (DS - 1)), V2)
        if E3 == 1:
            tl.store(CODE + (A3 & CODE_MASK), V3)
        RST = NST
        RTK = NT
    return RST, RTK


@triton.jit
def ncp_step_kernel(CODE, DATA, INPUTS, OUTBUF, S, BUDGET, CODELEN, INLEN, DS, OC):





    BD = tl.load(BUDGET + 0)
    _tick(CODE, DATA, INPUTS, OUTBUF, S, CODELEN, INLEN, BD, DS, OC)


@triton.jit
def ncp_resident_kernel(CODE, DATA, INPUTS, OUTBUF, STATES, CODELENS, INLENS, BUDGETS,
                        STEP_LIMIT,
                        CS: tl.constexpr, DS: tl.constexpr, INS: tl.constexpr,
                        OCS: tl.constexpr):














    pid = tl.program_id(0)
    CP = CODE + pid * CS
    DP = DATA + pid * DS
    IP = INPUTS + pid * INS
    OP = OUTBUF + pid * OCS
    ST = STATES + pid * 14
    CL = tl.load(CODELENS + pid)
    IL = tl.load(INLENS + pid)
    BD = tl.load(BUDGETS + pid)
    st = tl.load(ST + 13)
    tk = tl.load(ST + 12)
    n = 0
    while (st == 0) & ((STEP_LIMIT <= 0) | (n < STEP_LIMIT)):
        st, tk = _tick(CP, DP, IP, OP, ST, CL, IL, BD, DS, OCS)
        n += 1


class BatchResult(NamedTuple):







    outs: list
    status: list
    ticks: list
    oplens: list


class TritonBatch:















    def __init__(self, n, device="cuda", max_in=1, tick_budget=200_000, num_warps=1):
        if n < 1:
            raise ValueError("batch size must be >= 1")
        self.dev = torch.device(device)
        self.n = n
        self.out_cap = OUT_CAP
        self.max_in = max(1, int(max_in))
        self.num_warps = num_warps
        i32 = torch.int32
        self.CODE = torch.zeros((n, CODE_SIZE), dtype=i32, device=self.dev)
        self.DATA = torch.zeros((n, DATA_SIZE), dtype=i32, device=self.dev)
        self.INPUTS = torch.zeros((n, self.max_in), dtype=i32, device=self.dev)
        self.OUTBUF = torch.zeros((n, self.out_cap), dtype=i32, device=self.dev)
        self.STATE = torch.zeros((n, 14), dtype=i32, device=self.dev)
        self.STATE[:, 7] = DATA_SIZE
        self.CODELENS = torch.zeros(n, dtype=i32, device=self.dev)
        self.INLENS = torch.zeros(n, dtype=i32, device=self.dev)
        self.BUDGETS = torch.full((n,), int(tick_budget), dtype=i32, device=self.dev)

    def _row(self, i):

        if not isinstance(i, int) or not 0 <= i < self.n:
            raise ValueError(f"machine index {i} is outside a batch of {self.n} machines")

    def set_program(self, i, code, data=None, inputs=None):





        self._row(i)
        cb = bytes(code)
        if len(cb) > CODE_SIZE:
            raise ValueError(f"machine {i}: code is {len(cb)} bytes, above CODE_SIZE {CODE_SIZE}")
        self.CODE[i] = 0
        if cb:
            self.CODE[i, :len(cb)] = torch.tensor(list(cb), dtype=torch.int32, device=self.dev)
        self.CODELENS[i] = len(cb)
        db = bytes(data) if data else b""
        if len(db) > DATA_SIZE:
            raise ValueError(f"machine {i}: data image is {len(db)} bytes, "
                             f"above DATA_SIZE {DATA_SIZE}")
        self.DATA[i] = 0
        if db:
            self.DATA[i, :len(db)] = torch.tensor(list(db), dtype=torch.int32, device=self.dev)
        ib = bytes(inputs) if inputs else b""
        if len(ib) > self.max_in:
            raise ValueError(f"machine {i}: input stream of {len(ib)} bytes exceeds the "
                             f"batch input capacity {self.max_in}")
        self.INPUTS[i] = 0
        if ib:
            self.INPUTS[i, :len(ib)] = torch.tensor(list(ib), dtype=torch.int32, device=self.dev)
        self.INLENS[i] = len(ib)
        self.STATE[i] = 0
        self.STATE[i, 7] = DATA_SIZE

    def set_state(self, i, r=(0, 0, 0, 0), HL=0, DE=0, PC=0, SP=DATA_SIZE,
                  C=0, Z=0, ipos=0, oplen=0, tick=0, status=0):






        self._row(i)
        check_state(r, HL, DE, SP, C, Z, tick=tick, PC=PC, ipos=ipos, oplen=oplen,
                    status=status, where=f"machine {i}: ")
        self.STATE[i, 0:4] = torch.tensor(list(r), dtype=torch.int32, device=self.dev)
        self.STATE[i, 4] = HL
        self.STATE[i, 5] = DE
        self.STATE[i, 6] = PC
        self.STATE[i, 7] = SP
        self.STATE[i, 8] = C
        self.STATE[i, 9] = Z
        self.STATE[i, 10] = ipos
        self.STATE[i, 11] = oplen
        self.STATE[i, 12] = tick
        self.STATE[i, 13] = status

    def set_budget(self, i, budget):

        self._row(i)
        self.BUDGETS[i] = int(budget)

    def run(self):





        self._launch(0)
        return self.results()

    def step(self, steps=1):





        self._launch(int(steps))
        return self.results()

    def _launch(self, step_limit):
        ncp_resident_kernel[(self.n,)](
            self.CODE, self.DATA, self.INPUTS, self.OUTBUF, self.STATE,
            self.CODELENS, self.INLENS, self.BUDGETS, step_limit,
            CS=CODE_SIZE, DS=DATA_SIZE, INS=self.max_in, OCS=self.out_cap,
            num_warps=self.num_warps)

    def results(self):

        status = [int(v) for v in self.STATE[:, 13].cpu().tolist()]
        ticks = [int(v) for v in self.STATE[:, 12].cpu().tolist()]
        oplens = [int(v) for v in self.STATE[:, 11].cpu().tolist()]
        for i, o in enumerate(oplens):
            if o > self.out_cap:
                raise RuntimeError(
                    f"machine {i} emitted {o} bytes, above the {self.out_cap}-byte "
                    "per-machine output capacity of this batch")
        flat = []
        if self.n:
            flat = torch.cat([self.OUTBUF[i, :o] for i, o in enumerate(oplens)]).cpu().tolist()
        outs = []
        off = 0
        for o in oplens:
            outs.append(bytes(flat[off:off + o]))
            off += o
        return BatchResult(outs, status, ticks, oplens)

    def out(self, i):

        self._row(i)
        n = min(int(self.STATE[i, 11].item()), self.out_cap)
        return bytes(self.OUTBUF[i, :n].cpu().tolist())

    def snapshot(self, i):

        self._row(i)
        s = self.STATE[i].cpu().tolist()
        return dict(r=s[0:4], HL=s[4], DE=s[5], SP=s[7], PC=s[6], C=s[8], Z=s[9],
                    ipos=s[10], oplen=s[11], tick=s[12], status=s[13])

    def data(self, i):

        self._row(i)
        return self.DATA[i].cpu().tolist()

    def code(self, i):

        self._row(i)
        return self.CODE[i].cpu().tolist()


def run_batch(codes, datas=None, inputs=None, budgets=None, states=None,
              tick_budget=200_000, device="cuda", num_warps=1):











    n = len(codes)
    max_in = 1
    if inputs:
        max_in = max(1, max((len(v) for v in inputs), default=1))
    b = TritonBatch(n, device=device, max_in=max_in,
                    tick_budget=tick_budget, num_warps=num_warps)
    for i in range(n):
        b.set_program(i, codes[i],
                      datas[i] if datas else None,
                      inputs[i] if inputs else None)
        if budgets is not None:
            b.set_budget(i, budgets[i])
    if states is not None:
        if len(states) != n:
            raise ValueError(f"states must be {n} rows, got {len(states)}")
        for i, s in enumerate(states):
            row = list(s)
            if len(row) != 14:
                raise ValueError(f"machine {i}: state row has {len(row)} values, need 14")
            b.set_state(i, r=row[0:4], HL=row[4], DE=row[5], PC=row[6], SP=row[7],
                        C=row[8], Z=row[9], ipos=row[10], oplen=row[11], tick=row[12],
                        status=row[13])
    return b.run()


class TritonCircuit:


    def __init__(self, code, data=None, inputs=b"", tick_budget=200_000, device="cuda"):
        if len(code) > CODE_SIZE:
            raise ValueError(f"code image is {len(code)} bytes, above CODE_SIZE {CODE_SIZE}")
        if data and len(data) > DATA_SIZE:
            raise ValueError(f"data image is {len(data)} bytes, above DATA_SIZE {DATA_SIZE}")
        dev = torch.device(device)
        self.dev = dev
        i32 = torch.int32
        self.codelen = len(code)
        self.CODE = torch.zeros(max(1, CODE_SIZE), dtype=i32, device=dev)
        self.CODE[: len(code)] = torch.tensor(list(code), dtype=i32, device=dev)
        self.DATA = torch.zeros(DATA_SIZE, dtype=i32, device=dev)
        if data:
            self.DATA[: len(data)] = torch.tensor(list(data), dtype=i32, device=dev)
        self.INP = torch.tensor(list(inputs) if inputs else [0], dtype=i32, device=dev)
        self.inlen = len(inputs)
        self.OUTBUF = torch.zeros(OUT_CAP, dtype=i32, device=dev)
        self.S = torch.zeros(14, dtype=i32, device=dev)
        self.S[7] = DATA_SIZE
        self.BUDGET = torch.tensor([int(tick_budget)], dtype=i32, device=dev)
        self.tb = int(tick_budget)

    def load_state(self, R, HL, DE, SP, C, Z, tick, PC=0):

        check_state(R, HL, DE, SP, C, Z, tick=tick, PC=PC)
        self.S[0:4] = torch.tensor(list(R), dtype=torch.int32, device=self.dev)
        self.S[4] = HL; self.S[5] = DE; self.S[6] = PC
        self.S[7] = SP; self.S[8] = C; self.S[9] = Z
        self.S[12] = tick

    def step(self):
        ncp_step_kernel[(1,)](self.CODE, self.DATA, self.INP, self.OUTBUF, self.S,
                              self.BUDGET, self.codelen, self.inlen, DATA_SIZE, OUT_CAP)

    @property
    def status(self):
        return self.S[13:14]

    @property
    def tick(self):
        return self.S[12:13]

    @property
    def oplen(self):
        return self.S[11:12]

    def snapshot(self):
        s = self.S.cpu().tolist()
        return dict(r=s[0:4], HL=s[4], DE=s[5], SP=s[7], PC=s[6], C=s[8], Z=s[9],
                    ipos=s[10], oplen=s[11], tick=s[12], status=s[13])

    def out(self):

        n = min(int(self.S[11].item()), OUT_CAP)
        return bytes(self.OUTBUF[:n].cpu().tolist())

    def run(self):





        while int(self.status.item()) == 0:
            self.step()
        return self.out()

    def run_resident(self):







        b = TritonBatch(1, device=self.dev, max_in=max(1, int(self.INP.numel())))
        b.CODE[0].copy_(self.CODE)
        b.CODELENS[0] = self.codelen
        b.DATA[0].copy_(self.DATA)
        b.INPUTS[0, :self.INP.numel()].copy_(self.INP)
        b.INLENS[0] = self.inlen
        b.BUDGETS[0] = self.tb
        b.STATE[0].copy_(self.S)
        b.run()
        self.CODE.copy_(b.CODE[0])
        self.DATA.copy_(b.DATA[0])
        self.OUTBUF.copy_(b.OUTBUF[0])
        self.S.copy_(b.STATE[0])
        return self.out()