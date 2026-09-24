"""Tensor implementation of the NCP-8 datapath.

Second implementation of the same ISA semantics. The host Python layer never
reads machine state and never branches on it: every control signal comes from a
decode ROM (opcode -> ALU selector, operand selectors, instruction length) and
the datapath is built from one-hot gated rows (adders, shifters, multiplexers,
two RAMs).

Conventions:
  * all per-opcode information is a [256] row vector; each state field builds a
    "next value" row first and is then reduced to a scalar by the one-hot of the
    current instruction;
  * a bounds/illegal-access violation only writes status = 3 for that tick; every
    other field is left untouched (same contract as the reference simulator
    raising an error).
"""
from __future__ import annotations
import torch

DATA_SIZE = 4096
CODE_SIZE = 4096
OUT_CAP = 8192

NOP, ADD, SUB, ADC, SBB, MOV, TST, SHL, SHR, DJNZ, CLC = range(11)
LDI, ADDI, SUBI, ADCI = 11, 12, 13, 14
MOV_R_HL, MOV_HL_R, MOV_R_DE, MOV_DE_R = 15, 16, 17, 18
PUSH, POP, OUT, IN = 19, 20, 21, 22
INC_HL, DEC_HL, INC_DE = 23, 24, 25
OUTM, OUTDE = 26, 27
JMP, JZ, JNZ, JC, JNC = 28, 29, 30, 31, 32
CALL, RET = 33, 34
LDI_HL, LDI_DE, ADDI_HL, ADDI_DE = 35, 36, 37, 38
HALT, BAD = 39, 40
JPHL, GETPC, GETSP, GETF = 41, 42, 43, 44

AND, OR, XOR, MUL = 45, 46, 47, 48
DIV, MOD, CMP = 49, 50, 51
NOT, NEG, ROL, ROR = 52, 53, 54, 55
ADD_HLDE, SUB_HLDE, XCHG, EXT = 56, 57, 58, 59
STC, LDC = 60, 61

MOVW_HL_DE, MOVW_DE_HL, MOVW_HL_SP, MOVW_DE_SP, MOVW_SP_HL, MOVW_SP_DE = 62, 63, 64, 65, 66, 67
PUSHW_HL, PUSHW_DE, POPW_HL, POPW_DE = 68, 69, 70, 71
STW_HLDE, STW_DEHL, LDW_DEHL, LDW_HLDE = 72, 73, 74, 75
LDX, STX, ADD_SP, MULH = 76, 77, 78, 79
K = 80

def _rom2():

    alu = [BAD] * 256; s0 = [0] * 256; s1 = [0] * 256; lx = [0] * 256

    def put(sub, a, r0=0, r1=0, l=0):
        alu[sub], s0[sub], s1[sub], lx[sub] = a, r0, r1, l
    for f in range(16):
        put(0x00 + f, DIV, (f >> 2) & 3, f & 3)
        put(0x10 + f, MOD, (f >> 2) & 3, f & 3)
        put(0x20 + f, CMP, (f >> 2) & 3, f & 3)
    for r in range(4):
        put(0x40 | r, NOT, r, r); put(0x44 | r, NEG, r, r)
        put(0x48 | r, ROL, r, r); put(0x4C | r, ROR, r, r)
    put(0x60, ADD_HLDE); put(0x61, SUB_HLDE); put(0x62, XCHG)
    put(0x70, EXT, 0, 0, l=1)
    for r in range(4):
        put(0x80 | r, STC, r, r)
        put(0x84 | r, LDC, r, r)

    for k, sub in enumerate((0x30, 0x31, 0x32, 0x33, 0x34, 0x35)):
        put(sub, MOVW_HL_DE + k)
    for k, sub in enumerate((0x38, 0x39, 0x3A, 0x3B)):
        put(sub, PUSHW_HL + k)
    for k, sub in enumerate((0x3C, 0x3D, 0x3E, 0x3F)):
        put(sub, STW_HLDE + k)
    for r in range(4):
        put(0x50 | r, LDX, r, r, l=1)
        put(0x54 | r, STX, r, r, l=1)
    put(0x58, ADD_SP, 0, 0, l=1)
    for f in range(16):
        put(0x90 + f, MULH, (f >> 2) & 3, f & 3)
    t = lambda xs: torch.tensor(xs, dtype=torch.int32)
    return t(alu), t(s0), t(s1), t(lx)

_ALU2, _S02, _S12, _LX2 = _rom2()

def _rom():
    alu = [BAD] * 256; s0 = [0] * 256; s1 = [0] * 256; ln = [1] * 256

    def put(op, a, r0=0, r1=0, l=1):
        alu[op], s0[op], s1[op], ln[op] = a, r0, r1, l

    put(0x00, HALT); put(0x01, NOP)
    put(0x02, INC_HL); put(0x03, DEC_HL); put(0x04, INC_DE); put(0x05, CLC)
    put(0x06, OUTM); put(0x07, OUTDE); put(0x08, RET)
    put(0x09, JMP, l=3); put(0x0A, JZ, l=3); put(0x0B, JNZ, l=3)
    put(0x0C, JC, l=3); put(0x0D, JNC, l=3); put(0x0E, CALL, l=3)
    put(0x0F, LDI_HL, l=3); put(0x10, LDI_DE, l=3)
    put(0x11, ADDI_HL, l=2); put(0x12, ADDI_DE, l=2)
    put(0x13, JPHL)
    for r in range(4):
        put(0x14 | r, GETPC, r, r)
        put(0x18 | r, GETSP, r, r)
        put(0x1C | r, GETF, r, r)
    for f in range(16):
        put(0x20 + f, AND, (f >> 2) & 3, f & 3)
        put(0x30 + f, OR, (f >> 2) & 3, f & 3)
        put(0x40 + f, XOR, (f >> 2) & 3, f & 3)
        put(0x50 + f, MUL, (f >> 2) & 3, f & 3)
    put(0x70, BAD, l=2)
    for f in range(16):
        put(0x80 + f, ADD, (f >> 2) & 3, f & 3)
        put(0x90 + f, SUB, (f >> 2) & 3, f & 3)
        put(0xA0 + f, ADC, (f >> 2) & 3, f & 3)
        put(0xB0 + f, SBB, (f >> 2) & 3, f & 3)
        put(0xC0 + f, MOV, (f >> 2) & 3, f & 3)
    for r in range(4):
        put(0xD0 | r, LDI, r, r, 2)
        put(0xD4 | r, ADDI, r, r, 2)
        put(0xD8 | r, SUBI, r, r, 2)
        put(0xDC | r, ADCI, r, r, 2)
        put(0x60 | r, SHL, r, r)
        put(0x64 | r, SHR, r, r)
        put(0x68 | r, TST, r, r)
        put(0x6C | r, DJNZ, r, r, 3)
        put(0xE0 | r, MOV_R_HL, r, r)
        put(0xE4 | r, MOV_HL_R, r, r)
        put(0xE8 | r, MOV_R_DE, r, r)
        put(0xEC | r, MOV_DE_R, r, r)
        put(0xF0 | r, PUSH, r, r)
        put(0xF4 | r, POP, r, r)
        put(0xF8 | r, OUT, r, r)
        put(0xFC | r, IN, r, r)
    t = lambda xs: torch.tensor(xs, dtype=torch.int32)
    return t(alu), t(s0), t(s1), t(ln)

_ALU, _S0, _S1, _LEN = _rom()

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

class TorchCircuit:
    def __init__(self, code, data=None, inputs=b"", tick_budget=200_000, device="cuda", rom_override=None):

        if len(code) > CODE_SIZE:
            raise ValueError(f"code image is {len(code)} bytes, above CODE_SIZE {CODE_SIZE}")
        if data and len(data) > DATA_SIZE:
            raise ValueError(f"data image is {len(data)} bytes, above DATA_SIZE {DATA_SIZE}")
        dev = torch.device(device)
        self.dev = dev
        i32 = torch.int32
        self.codelen = len(code)
        self.CODE = torch.zeros(CODE_SIZE, dtype=i32, device=dev)
        self.CODE[: len(code)] = torch.tensor(list(code), dtype=i32, device=dev)
        self.DATA = torch.zeros(DATA_SIZE, dtype=i32, device=dev)
        if data:
            self.DATA[: len(data)] = torch.tensor(list(data), dtype=i32, device=dev)
        self.INPUTS = torch.zeros(1, dtype=i32, device=dev)
        self.inlen = torch.zeros(1, dtype=i32, device=dev)
        self.set_inputs(inputs)
        self.OUTBUF = torch.zeros(OUT_CAP, dtype=i32, device=dev)
        self.R = torch.zeros(4, dtype=i32, device=dev)
        for n in ("HL", "DE", "PC", "SP", "C", "Z", "ipos", "oplen", "tick"):
            setattr(self, n, torch.zeros(1, dtype=i32, device=dev))
        self.SP += DATA_SIZE
        self.status = torch.zeros(1, dtype=i32, device=dev)
        self.TB = torch.tensor([int(tick_budget)], dtype=i32, device=dev)
        self.tb = int(tick_budget)
        if rom_override is not None:
            alu_v, s0_v, s1_v, ln_v = [v.cpu() for v in rom_override]
        else:
            alu_v, s0_v, s1_v, ln_v = _ALU, _S0, _S1, _LEN
        self.alu_t = alu_v.to(dev); self.ln_t = ln_v.to(dev); self.s0_t = s0_v.to(dev)
        self.OH_S0 = (s0_v[:, None] == torch.arange(4, dtype=i32)[None, :]).to(i32).to(dev)
        self.OH_S1 = (s1_v[:, None] == torch.arange(4, dtype=i32)[None, :]).to(i32).to(dev)
        self.OH_R = ((torch.arange(256, dtype=i32) & 3)[:, None] == torch.arange(4, dtype=i32)[None, :]).to(i32).to(dev)
        if rom_override is not None:
            self.alu2_t = torch.full((256,), BAD, dtype=i32, device=dev)
            self.ohs0e = torch.zeros((256, 4), dtype=i32, device=dev)
            self.ohs1e = torch.zeros((256, 4), dtype=i32, device=dev)
            self.lx2_t = torch.zeros(256, dtype=i32, device=dev)
        else:
            self.alu2_t = _ALU2.to(dev); self.lx2_t = _LX2.to(dev)
            ar4 = torch.arange(4, dtype=i32, device=dev)
            self.ohs0e = (_S02.to(dev)[:, None] == ar4[None, :]).to(i32)
            self.ohs1e = (_S12.to(dev)[:, None] == ar4[None, :]).to(i32)
        self.ROW = torch.arange(256, dtype=i32, device=dev)
        self.KR = torch.arange(K, dtype=i32, device=dev)
        self.RD = torch.arange(DATA_SIZE, dtype=i32, device=dev)
        self.RC = torch.arange(CODE_SIZE, dtype=i32, device=dev)
        self.IW0 = torch.tensor(0x0F20, dtype=i32, device=dev)
        self.IW1 = torch.tensor(0x0F21, dtype=i32, device=dev)
        self.RO = torch.arange(OUT_CAP, dtype=i32, device=dev)
        self._acts = None

    def set_inputs(self, b):
        self.INPUTS = torch.tensor(list(b) if b else [0], dtype=torch.int32, device=self.dev)
        self.inlen = torch.tensor(len(b), dtype=torch.int32, device=self.dev)

    def _oh(self, x):

        return (self.alu_t == x).to(torch.int32)

    def step(self, ext=None):

        dev, i32 = self.dev, torch.int32
        w = lambda m, a, b: m * a + (1 - m) * b

        zero = torch.zeros((), dtype=i32, device=dev)
        fetch_ok = (self.PC < self.codelen).to(i32)
        op = self.CODE.index_select(0, self.PC.clamp(0, CODE_SIZE - 1).reshape(1)).reshape(())
        sub = self._g(self.CODE, self.PC + 1)
        if ext is not None:

            op = ext[0].reshape(()).to(i32)
            imm0 = ext[1].reshape(()).to(i32)
            imm1 = ext[2].reshape(()).to(i32)
            esc = zero
            fetch_ok = torch.ones(1, dtype=i32, device=dev)
        else:
            esc = (op == 0x70).to(i32)
            imm0 = torch.where(esc > 0, self._g(self.CODE, self.PC + 2), self._g(self.CODE, self.PC + 1))
            imm1 = torch.where(esc > 0, self._g(self.CODE, self.PC + 3), self._g(self.CODE, self.PC + 2))
        ind = (self.ROW == op).to(i32) * (1 - esc) + ((self.ROW == sub).to(i32)) * esc

        _oh1 = self._oh
        oh = lambda x: _oh1(x) * (1 - esc) + ((self.alu2_t == x).to(i32)) * esc
        _s0t, _s1t, _lnt = self.OH_S0, self.OH_S1, self.ln_t
        oh_s0 = _s0t * (1 - esc) + self.ohs0e * esc
        oh_s1 = _s1t * (1 - esc) + self.ohs1e * esc
        ln_e = _lnt * (1 - esc) + (2 + self.lx2_t) * esc
        t16 = imm0 | (imm1 << 8)

        a = (self.R * (ind[:, None] * oh_s0).sum(0)).sum()
        b = (self.R * (ind[:, None] * oh_s1).sum(0)).sum()
        rr = (self.R * (ind[:, None] * self.OH_R).sum(0)).sum()

        a_e = (self.R * self.OH_R.index_select(0, ((sub >> 2) & 3).reshape(1))[0]).sum()
        b_e = (self.R * self.OH_R.index_select(0, (sub & 3).reshape(1))[0]).sum()
        a = a * (1 - esc) + a_e * esc
        b = b * (1 - esc) + b_e * esc
        rr = rr * (1 - esc) + b_e * esc
        rvi = (imm0 & 3).reshape(1)
        rv = (self.R * self.OH_R.index_select(0, rvi)[0]).sum()

        dl = self._g(self.DATA, self.HL)
        der = self._g(self.DATA, self.DE)
        dsp = self._g(self.DATA, self.SP)
        dsp1 = self._g(self.DATA, self.SP + 1)
        inb = self._g(self.INPUTS, self.ipos)
        eof = (self.ipos >= self.inlen).to(i32)

        sx = imm0 - ((imm0 >> 7) & 1) * 256
        fr = (self.HL + sx) & 0xFFFF
        dfr = self._g(self.DATA, fr)
        dhl1 = self._g(self.DATA, self.HL + 1)
        dde1 = self._g(self.DATA, self.DE + 1)

        t_add = a + b; v_add = t_add & 255; c_add = t_add >> 8
        t_adc = a + b + self.C; v_adc = t_adc & 255; c_adc = t_adc >> 8
        v_sub = (a - b) & 255; c_sub = (a < b).to(i32)
        t_sbb = a - b - self.C; v_sbb = t_sbb & 255; c_sbb = (t_sbb < 0).to(i32)
        t_addi = rr + imm0; v_addi = t_addi & 255; c_addi = t_addi >> 8
        v_subi = (rr - imm0) & 255; c_subi = (rr < imm0).to(i32)
        t_adci = rr + imm0 + self.C; v_adci = t_adci & 255; c_adci = t_adci >> 8
        v_shl = (rr << 1) & 255; c_shl = rr >> 7
        v_shr = rr >> 1; c_shr = rr & 1
        v_dj = (rr - 1) & 255

        v_and = a & b; v_or = a | b; v_xor = a ^ b
        t_mul = a * b; v_mul = t_mul & 255; c_mul = (t_mul > 255).to(i32)
        b_safe = torch.where(b == 0, torch.ones((), dtype=i32, device=dev), b)
        v_div = a // b_safe; v_mod = a % b_safe
        cmp_c = (a < b).to(i32)
        v_not = (~rr) & 255
        v_neg = (-rr) & 255; c_neg = (rr != 0).to(i32)
        v_rol = ((rr << 1) | self.C) & 255; c_rol = rr >> 7
        v_ror = (rr >> 1) | (self.C << 7); c_ror = rr & 1
        t_hladd = self.HL + self.DE; v_hladd = t_hladd & 0xFFFF; c_hladd = t_hladd >> 16
        v_hlsub = (self.HL - self.DE) & 0xFFFF; c_hlsub = (self.HL < self.DE).to(i32)

        veclo = self._g(self.CODE, 0x0F00 + 2 * imm0)
        vec = veclo | (self._g(self.CODE, 0x0F00 + 2 * imm0 + 1) << 8)
        ext_ok = (imm0 < 16).to(i32) * (vec != 0).to(i32)

        wlo = self._g(self.CODE, self.IW0); whi = self._g(self.CODE, self.IW1)
        code_ok = (self.HL < self.codelen).to(i32)
        sc_ok = code_ok * (self.HL >= wlo).to(i32) * (self.HL < whi).to(i32)
        ldc_v = self._g(self.CODE, self.HL)

        v_mulh = ((a * b) >> 8) & 255
        v_popw = dsp | (dsp1 << 8)
        v_ldw_de = dl | (dhl1 << 8)
        v_ldw_hl = der | (dde1 << 8)
        v_sp_add = (self.SP + sx) & 0xFFFF

        RHL_EN = oh(MOV_R_HL) + oh(MOV_HL_R)
        rows_R_val = (
            oh(ADD) * v_add + oh(SUB) * v_sub + oh(ADC) * v_adc
            + oh(SBB) * v_sbb + oh(MOV) * b + oh(LDI) * imm0
            + oh(ADDI) * v_addi + oh(SUBI) * v_subi + oh(ADCI) * v_adci
            + oh(SHL) * v_shl + oh(SHR) * v_shr + oh(DJNZ) * v_dj
            + oh(POP) * dsp + oh(IN) * (1 - eof) * inb
            + oh(MOV_R_HL) * dl + oh(MOV_R_DE) * der
            + oh(GETPC) * self.PC + oh(GETSP) * (self.SP & 255)
            + oh(GETF) * (self.Z + 2 * self.C)
            + oh(AND) * v_and + oh(OR) * v_or + oh(XOR) * v_xor + oh(MUL) * v_mul
            + oh(DIV) * v_div + oh(MOD) * v_mod
            + oh(NOT) * v_not + oh(NEG) * v_neg + oh(ROL) * v_rol + oh(ROR) * v_ror
            + oh(LDC) * ldc_v
            + oh(LDX) * dfr + oh(MULH) * v_mulh
        )
        rows_R_en = (
            oh(ADD) + oh(SUB) + oh(ADC) + oh(SBB) + oh(MOV)
            + oh(LDI) + oh(ADDI) + oh(SUBI) + oh(ADCI)
            + oh(SHL) + oh(SHR) + oh(DJNZ) + oh(POP) + oh(IN)
            + oh(MOV_R_HL) + oh(MOV_R_DE)
            + oh(GETPC) + oh(GETSP) + oh(GETF)
            + oh(AND) + oh(OR) + oh(XOR) + oh(MUL) + oh(DIV) + oh(MOD)
            + oh(NOT) + oh(NEG) + oh(ROL) + oh(ROR) + oh(LDC)
            + oh(LDX) + oh(MULH)
        )

        rows_HL = self.HL + oh(INC_HL) - oh(DEC_HL) + oh(OUTM)
        rows_HL = w(oh(LDI_HL), t16, rows_HL)
        rows_HL = w(oh(ADDI_HL), self.HL + rv, rows_HL)
        rows_HL = w(oh(ADD_HLDE), v_hladd, rows_HL)
        rows_HL = w(oh(SUB_HLDE), v_hlsub, rows_HL)
        rows_HL = w(oh(XCHG), self.DE, rows_HL)
        rows_HL = w(oh(MOVW_HL_DE), self.DE, rows_HL)
        rows_HL = w(oh(MOVW_HL_SP), self.SP, rows_HL)
        rows_HL = w(oh(POPW_HL), v_popw, rows_HL)
        rows_HL = w(oh(LDW_HLDE), v_ldw_hl, rows_HL)

        rows_DE = self.DE + oh(INC_DE) + oh(OUTDE)
        rows_DE = w(oh(LDI_DE), t16, rows_DE)
        rows_DE = w(oh(ADDI_DE), self.DE + rv, rows_DE)
        rows_DE = w(oh(XCHG), self.HL, rows_DE)
        rows_DE = w(oh(MOVW_DE_HL), self.HL, rows_DE)
        rows_DE = w(oh(MOVW_DE_SP), self.SP, rows_DE)
        rows_DE = w(oh(POPW_DE), v_popw, rows_DE)
        rows_DE = w(oh(LDW_DEHL), v_ldw_de, rows_DE)

        rows_SP = w(oh(PUSH), self.SP - 1,
                   w(oh(POP), self.SP + 1,
                     w(oh(CALL), self.SP - 2,
                       w(oh(EXT), self.SP - 2,
                         w(oh(RET), self.SP + 2, self.SP)))))
        rows_SP = w(oh(PUSHW_HL) + oh(PUSHW_DE), self.SP - 2, rows_SP)
        rows_SP = w(oh(POPW_HL) + oh(POPW_DE), self.SP + 2, rows_SP)
        rows_SP = w(oh(MOVW_SP_HL), self.HL, rows_SP)
        rows_SP = w(oh(MOVW_SP_DE), self.DE, rows_SP)
        rows_SP = w(oh(ADD_SP), v_sp_add, rows_SP)

        rows_C = self.C + oh(ADD) * (c_add - self.C) + oh(SUB) * (c_sub - self.C) \
            + oh(ADC) * (c_adc - self.C) + oh(SBB) * (c_sbb - self.C) \
            + oh(ADDI) * (c_addi - self.C) + oh(SUBI) * (c_subi - self.C) \
            + oh(ADCI) * (c_adci - self.C) + oh(SHL) * (c_shl - self.C) \
            + oh(SHR) * (c_shr - self.C) + oh(CLC) * (0 - self.C) \
            + oh(IN) * eof * (1 - self.C) \
            + oh(MUL) * (c_mul - self.C) + oh(NEG) * (c_neg - self.C) \
            + oh(ROL) * (c_rol - self.C) + oh(ROR) * (c_ror - self.C) \
            + oh(CMP) * (cmp_c - self.C) \
            + oh(ADD_HLDE) * (c_hladd - self.C) + oh(SUB_HLDE) * (c_hlsub - self.C)

        rows_Z = self.Z + oh(ADD) * ((v_add == 0).to(i32) - self.Z) \
            + oh(SUB) * ((v_sub == 0).to(i32) - self.Z) \
            + oh(ADC) * ((v_adc == 0).to(i32) - self.Z) \
            + oh(SBB) * ((v_sbb == 0).to(i32) - self.Z) \
            + oh(ADDI) * ((v_addi == 0).to(i32) - self.Z) \
            + oh(SUBI) * ((v_subi == 0).to(i32) - self.Z) \
            + oh(ADCI) * ((v_adci == 0).to(i32) - self.Z) \
            + oh(TST) * ((rr == 0).to(i32) - self.Z) \
            + oh(SHL) * ((v_shl == 0).to(i32) - self.Z) \
            + oh(SHR) * ((v_shr == 0).to(i32) - self.Z) \
            + oh(AND) * ((v_and == 0).to(i32) - self.Z) + oh(OR) * ((v_or == 0).to(i32) - self.Z) \
            + oh(XOR) * ((v_xor == 0).to(i32) - self.Z) + oh(MUL) * ((v_mul == 0).to(i32) - self.Z) \
            + oh(DIV) * ((v_div == 0).to(i32) - self.Z) + oh(MOD) * ((v_mod == 0).to(i32) - self.Z) \
            + oh(NOT) * ((v_not == 0).to(i32) - self.Z) + oh(NEG) * ((v_neg == 0).to(i32) - self.Z) \
            + oh(ROL) * ((v_rol == 0).to(i32) - self.Z) + oh(ROR) * ((v_ror == 0).to(i32) - self.Z) \
            + oh(CMP) * ((a == b).to(i32) - self.Z) \
            + oh(MULH) * ((v_mulh == 0).to(i32) - self.Z)

        fall = self.PC + ln_e
        retv = (dsp << 8) | dsp1
        jz_t = w(self.Z.reshape(1), t16, fall); jnz_t = w((1 - self.Z).reshape(1), t16, fall)
        jc_t = w(self.C.reshape(1), t16, fall); jnc_t = w((1 - self.C).reshape(1), t16, fall)
        dj_t = w((v_dj != 0).to(i32).reshape(1), t16, fall)
        rows_PC = fall
        rows_PC = w(oh(JMP), t16, rows_PC)
        rows_PC = w(oh(CALL), t16, rows_PC)
        rows_PC = w(oh(RET), retv, rows_PC)
        rows_PC = w(oh(JZ), jz_t, rows_PC)
        rows_PC = w(oh(JNZ), jnz_t, rows_PC)
        rows_PC = w(oh(JC), jc_t, rows_PC)
        rows_PC = w(oh(JNC), jnc_t, rows_PC)
        rows_PC = w(oh(DJNZ), dj_t, rows_PC)
        rows_PC = w(oh(JPHL), self.HL, rows_PC)
        rows_PC = w(oh(EXT), vec, rows_PC)

        rows_ipos = w(oh(IN) * (1 - eof), self.ipos + 1, self.ipos)

        rows_out_val = oh(OUT) * rr + oh(OUTM) * dl + oh(OUTDE) * der
        rows_out_en = oh(OUT) + oh(OUTM) + oh(OUTDE)

        fetch_ok_rows = (self.PC + ln_e <= self.codelen).to(i32)
        hl_ok = (self.HL < DATA_SIZE).to(i32); de_ok = (self.DE < DATA_SIZE).to(i32)
        sp_lo = (self.SP > 0).to(i32); sp_lo2 = (self.SP >= 2).to(i32); sp_hi = (self.SP < DATA_SIZE).to(i32)
        sp_hi1 = ((self.SP + 1) < DATA_SIZE).to(i32)

        hl_ok2 = ((self.HL + 1) < DATA_SIZE).to(i32)
        de_ok2 = ((self.DE + 1) < DATA_SIZE).to(i32)
        fr_ok = (fr < DATA_SIZE).to(i32)
        sp_hi2 = ((self.SP + 2) <= DATA_SIZE).to(i32)
        hl_sp_ok = (self.HL <= DATA_SIZE).to(i32)
        de_sp_ok = (self.DE <= DATA_SIZE).to(i32)
        sp_add_ok = (v_sp_add <= DATA_SIZE).to(i32)
        rd_rows = (oh(MOV_R_HL) + oh(MOV_HL_R)) * (1 - hl_ok) \
            + (oh(MOV_R_DE) + oh(MOV_DE_R)) * (1 - de_ok) \
            + oh(OUTM) * (1 - hl_ok) + oh(OUTDE) * (1 - de_ok) \
            + (oh(STW_HLDE) + oh(LDW_DEHL)) * (1 - hl_ok2) \
            + (oh(STW_DEHL) + oh(LDW_HLDE)) * (1 - de_ok2) \
            + (oh(LDX) + oh(STX)) * (1 - fr_ok)
        st_rows = (oh(POP) * (1 - sp_hi) + oh(RET) * (1 - sp_hi1)
                   + oh(PUSH) * (1 - sp_lo) + oh(CALL) * (1 - sp_lo2)
                   + oh(EXT) * (1 - sp_lo2)
                   + (oh(PUSHW_HL) + oh(PUSHW_DE)) * (1 - sp_lo2)
                   + (oh(POPW_HL) + oh(POPW_DE)) * (1 - sp_hi2))
        v2_err = (oh(DIV) * (b == 0).to(i32) + oh(MOD) * (b == 0).to(i32) + oh(EXT) * (1 - ext_ok)
                  + oh(LDC) * (1 - code_ok) + oh(STC) * (1 - sc_ok))
        v3_err = (oh(MOVW_SP_HL) * (1 - hl_sp_ok) + oh(MOVW_SP_DE) * (1 - de_sp_ok)
                  + oh(ADD_SP) * (1 - sp_add_ok))

        out_ovf = (ind * rows_out_en * (self.oplen >= OUT_CAP).to(i32)).sum()
        err = (ind * (oh(BAD) + (1 - fetch_ok_rows) + rd_rows + st_rows + v2_err + v3_err)).sum() \
            + out_ovf + (1 - fetch_ok)

        sel = lambda rows: (ind * rows).sum()
        R_w = (ind[:, None] * oh_s0 * rows_R_en[:, None]).sum(0)
        R_v = (ind[:, None] * oh_s0 * (rows_R_en * rows_R_val)[:, None]).sum(0)
        ok = (err == 0).to(i32)

        running = (self.status == 0).to(i32)
        over = running * (self.tick >= self.TB).to(i32)
        m = ok * running * (1 - over)

        a1 = sel(oh(MOV_HL_R) * self.HL + oh(MOV_DE_R) * self.DE
                 + oh(PUSH) * (self.SP - 1) + oh(CALL) * (self.SP - 1)
                 + oh(EXT) * (self.SP - 1)
                 + (oh(PUSHW_HL) + oh(PUSHW_DE)) * (self.SP - 1)
                 + oh(STW_HLDE) * self.HL + oh(STW_DEHL) * self.DE
                 + oh(STX) * fr)
        v1 = sel(oh(MOV_HL_R) * rr + oh(MOV_DE_R) * rr + oh(PUSH) * rr
                 + oh(CALL) * ((self.PC + 3) & 255) + oh(EXT) * ((self.PC + 3) & 255)
                 + oh(PUSHW_HL) * ((self.HL >> 8) & 255) + oh(PUSHW_DE) * ((self.DE >> 8) & 255)
                 + oh(STW_HLDE) * (self.DE & 255) + oh(STW_DEHL) * (self.HL & 255)
                 + oh(STX) * rr)
        e1 = sel(oh(MOV_HL_R) + oh(MOV_DE_R) + oh(PUSH) + oh(CALL) + oh(EXT)
                 + oh(PUSHW_HL) + oh(PUSHW_DE) + oh(STW_HLDE) + oh(STW_DEHL) + oh(STX)) * m
        oh1 = ((self.RD == a1).to(i32)) * e1
        self.DATA = oh1 * v1 + (1 - oh1) * self.DATA
        a2 = sel(oh(CALL) * (self.SP - 2) + oh(EXT) * (self.SP - 2)
                 + (oh(PUSHW_HL) + oh(PUSHW_DE)) * (self.SP - 2)
                 + oh(STW_HLDE) * (self.HL + 1) + oh(STW_DEHL) * (self.DE + 1))
        v2 = sel(oh(CALL) * ((self.PC + 3) >> 8) + oh(EXT) * ((self.PC + 3) >> 8)
                 + oh(PUSHW_HL) * (self.HL & 255) + oh(PUSHW_DE) * (self.DE & 255)
                 + oh(STW_HLDE) * ((self.DE >> 8) & 255) + oh(STW_DEHL) * ((self.HL >> 8) & 255))
        e2 = sel(oh(CALL) + oh(EXT) + oh(PUSHW_HL) + oh(PUSHW_DE)
                 + oh(STW_HLDE) + oh(STW_DEHL)) * m
        oh2 = ((self.RD == a2).to(i32)) * e2
        self.DATA = oh2 * v2 + (1 - oh2) * self.DATA

        a3 = sel(oh(STC) * self.HL); v3 = sel(oh(STC) * rr); e3 = sel(oh(STC)) * m
        oh3 = ((self.RC == a3).to(i32)) * e3
        self.CODE = oh3 * v3 + (1 - oh3) * self.CODE

        ov = sel(rows_out_val) * m
        oe = sel(rows_out_en) * m
        oh_out = (self.RO == self.oplen).to(i32) * oe
        self.OUTBUF = oh_out * ov + (1 - oh_out) * self.OUTBUF
        self.oplen = self.oplen + oe

        halt_en = sel(oh(HALT))
        new_status = torch.where(err > 0, 3 * torch.ones(1, dtype=i32, device=dev),
                                 torch.where(halt_en > 0, torch.ones(1, dtype=i32, device=dev),
                                             torch.zeros(1, dtype=i32, device=dev)))
        new_status = torch.where(over > 0, 2 * torch.ones(1, dtype=i32, device=dev), new_status)
        self.status = running * new_status + (1 - running) * self.status

        self.R = m * ((R_w * R_v + (1 - R_w) * self.R) & 255) + (1 - m) * self.R
        self.HL = (m * sel(rows_HL) + (1 - m) * self.HL) & 0xFFFF
        self.DE = (m * sel(rows_DE) + (1 - m) * self.DE) & 0xFFFF
        self.PC = m * sel(rows_PC) + (1 - m) * self.PC
        self.SP = m * sel(rows_SP) + (1 - m) * self.SP
        self.C = m * sel(rows_C) + (1 - m) * self.C
        self.Z = m * sel(rows_Z) + (1 - m) * self.Z
        self.ipos = m * sel(rows_ipos) + (1 - m) * self.ipos
        self.tick = self.tick + m

    def _g(self, buf, idx):
        return buf.index_select(0, idx.clamp(0, buf.numel() - 1).reshape(1)).reshape(()).to(torch.int32)

    def load_state(self, R, HL, DE, SP, C, Z, tick, PC=0):

        check_state(R, HL, DE, SP, C, Z, tick=tick, PC=PC)
        t = torch.tensor
        self.R = t(R, dtype=torch.int32, device=self.dev)
        self.HL = t([HL], dtype=torch.int32, device=self.dev)
        self.DE = t([DE], dtype=torch.int32, device=self.dev)
        self.SP = t([SP], dtype=torch.int32, device=self.dev)
        self.C = t([C], dtype=torch.int32, device=self.dev)
        self.Z = t([Z], dtype=torch.int32, device=self.dev)
        self.tick = t([tick], dtype=torch.int32, device=self.dev)
        self.PC = t([PC], dtype=torch.int32, device=self.dev)

    def snapshot(self):
        return dict(r=self.R.tolist(), HL=self.HL.item(), DE=self.DE.item(),
                    SP=self.SP.item(), PC=self.PC.item(), C=self.C.item(), Z=self.Z.item(),
                    ipos=self.ipos.item(), oplen=self.oplen.item(), tick=self.tick.item(),
                    status=int(self.status.item()))

    def out(self):

        n = min(int(self.oplen.item()), OUT_CAP)
        return bytes(self.OUTBUF[:n].cpu().tolist())

    def run(self):

        while int(self.status.item()) == 0:
            self.step()
        return self.out()