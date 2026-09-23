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


@triton.jit
def _get4(v0, v1, v2, v3, idx):
    return tl.where(idx == 0, v0, tl.where(idx == 1, v1, tl.where(idx == 2, v2, v3)))


@triton.jit
def _set4(s, d, v):
    return tl.where(s == d, v, s)


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
    err = 0

    if PC < CODELEN:
        op = tl.load(CODE + PC)
        if op <= 0x1F:
            if op == 0x00:
                NST = 1
            elif op == 0x01:
                pass
            elif op == 0x02:
                nHL = (HL + 1) & 0xFFFF
            elif op == 0x03:
                nHL = (HL - 1) & 0xFFFF
            elif op == 0x04:
                nDE = (DE + 1) & 0xFFFF
            elif op == 0x05:
                nC = 0
            elif op == 0x06:
                if HL >= DS:
                    err = 1
                else:
                    OVAL = tl.load(DATA + HL); OEN = 1; nHL = (HL + 1) & 0xFFFF
            elif op == 0x07:
                if DE >= DS:
                    err = 1
                else:
                    OVAL = tl.load(DATA + DE); OEN = 1; nDE = (DE + 1) & 0xFFFF
            elif op == 0x08:
                if SP + 1 >= DS:
                    err = 1
                else:
                    nPC = (tl.load(DATA + SP) << 8) | tl.load(DATA + SP + 1)
                    nSP = SP + 2
            elif op >= 0x09 and op <= 0x0D:
                if PC + 3 > CODELEN:
                    err = 1
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    if op == 0x09:
                        nPC = t
                    elif op == 0x0A:
                        nPC = t if Z == 1 else PC + 3
                    elif op == 0x0B:
                        nPC = t if Z == 0 else PC + 3
                    elif op == 0x0C:
                        nPC = t if C == 1 else PC + 3
                    else:
                        nPC = t if C == 0 else PC + 3
            elif op == 0x0E:
                if SP < 2 or PC + 3 > CODELEN:
                    err = 1
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    ret = PC + 3
                    A1 = SP - 1; V1 = ret & 0xFF; E1 = 1
                    A2 = SP - 2; V2 = (ret >> 8) & 0xFF; E2 = 1
                    nSP = SP - 2; nPC = t
            elif op == 0x0F or op == 0x10:
                if PC + 3 > CODELEN:
                    err = 1
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    nPC = PC + 3
                    if op == 0x0F:
                        nHL = t
                    else:
                        nDE = t
            elif op == 0x11 or op == 0x12:
                if PC + 2 > CODELEN:
                    err = 1
                else:
                    rs = _get4(r0, r1, r2, r3, tl.load(CODE + PC + 1) & 3)
                    nPC = PC + 2
                    if op == 0x11:
                        nHL = (HL + rs) & 0xFFFF
                    else:
                        nDE = (DE + rs) & 0xFFFF
            elif op == 0x13:
                nPC = HL & 0xFFFF
            elif op >= 0x14 and op <= 0x1F:
                d = op & 3
                if op < 0x18:
                    v = PC
                elif op < 0x1C:
                    v = SP & 255
                else:
                    v = Z + 2 * C
                if d == 0:
                    nR0 = v
                elif d == 1:
                    nR1 = v
                elif d == 2:
                    nR2 = v
                else:
                    nR3 = v
            else:
                err = 1
        elif op >= 0x80 and op <= 0xCF:
            f = op & 0xF
            d = (f >> 2) & 3
            a = _get4(r0, r1, r2, r3, d)
            b = _get4(r0, r1, r2, r3, f & 3)
            k = op >> 4
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
            if d == 0:
                nR0 = v
            elif d == 1:
                nR1 = v
            elif d == 2:
                nR2 = v
            elif d == 3:
                nR3 = v
        elif op >= 0xD0 and op <= 0xDF:
            if PC + 2 > CODELEN:
                err = 1
            else:
                i8 = tl.load(CODE + PC + 1)
                rr = _get4(r0, r1, r2, r3, op & 3)
                nPC = PC + 2
                if op <= 0xD3:
                    v = i8
                elif op <= 0xD7:
                    t = rr + i8; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
                elif op <= 0xDB:
                    v = (rr - i8) & 255; nC = (rr < i8).to(tl.int32); nZ = (v == 0).to(tl.int32)
                else:
                    t = rr + i8 + C; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
                d = op & 3
                if d == 0:
                    nR0 = v
                elif d == 1:
                    nR1 = v
                elif d == 2:
                    nR2 = v
                else:
                    nR3 = v
        elif op >= 0x60 and op <= 0x6F:
            d = op & 3
            rr = _get4(r0, r1, r2, r3, d)
            v = rr
            if op < 0x64:
                nC = rr >> 7; v = (rr << 1) & 255; nZ = (v == 0).to(tl.int32)
            elif op < 0x68:
                nC = rr & 1; v = rr >> 1; nZ = (v == 0).to(tl.int32)
            elif op < 0x6C:
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
            if d == 0:
                nR0 = v
            elif d == 1:
                nR1 = v
            elif d == 2:
                nR2 = v
            else:
                nR3 = v
        elif op >= 0xE0 and op <= 0xFF:
            d = op & 3
            rr = _get4(r0, r1, r2, r3, d)
            v = rr
            touched = 1
            if op < 0xE4:
                if HL >= DS:
                    err = 1
                else:
                    v = tl.load(DATA + HL)
            elif op < 0xE8:
                if HL >= DS:
                    err = 1
                else:
                    A1 = HL; V1 = rr; E1 = 1
            elif op < 0xEC:
                if DE >= DS:
                    err = 1
                else:
                    v = tl.load(DATA + DE)
            elif op < 0xF0:
                if DE >= DS:
                    err = 1
                else:
                    A1 = DE; V1 = rr; E1 = 1
            elif op < 0xF4:
                if SP <= 0:
                    err = 1
                else:
                    A1 = SP - 1; V1 = rr; E1 = 1; nSP = SP - 1
            elif op < 0xF8:
                if SP >= DS:
                    err = 1
                else:
                    v = tl.load(DATA + SP); nSP = SP + 1
            elif op < 0xFC:
                OVAL = rr; OEN = 1
                touched = 0
            else:
                if IPO < INLEN:
                    v = tl.load(INPUTS + IPO); nIPO = IPO + 1
                else:
                    v = 0; nC = 1
            if d == 0:
                nR0 = v
            elif d == 1:
                nR1 = v
            elif d == 2:
                nR2 = v
            else:
                nR3 = v
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