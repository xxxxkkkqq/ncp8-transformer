"""Fused Triton kernel implementation of the NCP-8 datapath.

Third implementation of the same ISA semantics: the whole fetch-decode-execute
cycle runs inside a single kernel. Public interface is kept compatible with the
tensor implementation so that both can be driven by the same equivalence tests.

Error contract: a violating tick writes status = 3 only; all other state and the
tick counter stay unchanged.

State tensor layout: [r0,r1,r2,r3, HL, DE, PC, SP, C, Z, ipos, oplen, tick, status]
"""
import torch
import triton
import triton.language as tl

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


@triton.jit
def _get4(v0, v1, v2, v3, idx):
    return tl.where(idx == 0, v0, tl.where(idx == 1, v1, tl.where(idx == 2, v2, v3)))


@triton.jit
def _wr(r0, r1, r2, r3, d, v):

    return (tl.where(d == 0, v, r0), tl.where(d == 1, v, r1),
            tl.where(d == 2, v, r2), tl.where(d == 3, v, r3))


@triton.jit
def _dec_esc(sub):









    if sub <= 0x2F:
        eop = ESC_DIV + (sub >> 4); d = (sub >> 2) & 3; s = sub & 3; lx = 0
    elif sub >= 0x40 and sub <= 0x4F:
        eop = ESC_NOT + ((sub >> 2) & 3); d = sub & 3; s = sub & 3; lx = 0
    elif sub >= 0x60 and sub <= 0x62:
        eop = ESC_ADD_HLDE + (sub - 0x60); d = 0; s = 0; lx = 0
    elif sub == 0x70:
        eop = ESC_EXT; d = 0; s = 0; lx = 1
    elif sub >= 0x80 and sub <= 0x87:
        eop = ESC_STC + ((sub >> 2) & 1); d = sub & 3; s = sub & 3; lx = 0
    else:
        eop = ESC_BAD; d = 0; s = 0; lx = 0
    return eop, d, s, lx


@triton.jit
def ncp_step_kernel(CODE, DATA, INPUTS, OUTBUF, S, CODELEN, INLEN, DS):
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
            else:
                err = 1
        else:
            err = 1
    else:
        err = 1

    if err == 1:
        tl.store(S + 13, 3)
    else:
        nOL = OL + OEN
        tl.store(S + 0, nR0); tl.store(S + 1, nR1); tl.store(S + 2, nR2); tl.store(S + 3, nR3)
        tl.store(S + 4, nHL); tl.store(S + 5, nDE); tl.store(S + 6, nPC); tl.store(S + 7, nSP)
        tl.store(S + 8, nC); tl.store(S + 9, nZ); tl.store(S + 10, nIPO); tl.store(S + 11, nOL)
        tl.store(S + 12, tl.load(S + 12) + 1)
        tl.store(S + 13, NST)
        if OEN == 1:
            tl.store(OUTBUF + OL, OVAL)
        if E1 == 1:
            tl.store(DATA + A1, V1)
        if E2 == 1:
            tl.store(DATA + A2, V2)
        if E3 == 1:
            tl.store(CODE + A3, V3)


class TritonCircuit:


    def __init__(self, code, data=None, inputs=b"", tick_budget=200_000, device="cuda"):
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
        self.tb = tick_budget

    def load_state(self, R, HL, DE, SP, C, Z, tick):
        self.S[0:4] = torch.tensor(R, dtype=torch.int32, device=self.dev)
        self.S[4] = HL; self.S[5] = DE; self.S[6] = 0
        self.S[7] = SP; self.S[8] = C; self.S[9] = Z
        self.S[12] = tick

    def step(self):
        ncp_step_kernel[(1,)](self.CODE, self.DATA, self.INP, self.OUTBUF, self.S,
                              self.codelen, self.inlen, DATA_SIZE)

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
        n = int(self.S[11].item())
        return bytes(self.OUTBUF[:n].cpu().tolist())

    def run(self):
        while int(self.status.item()) == 0 and int(self.tick.item()) < self.tb:
            self.step()
        if int(self.status.item()) == 0:
            self.S[13] = 2
        return self.out()