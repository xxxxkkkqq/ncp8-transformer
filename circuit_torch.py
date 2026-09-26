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

import isa_table as ISA

DATA_SIZE = 4096
CODE_SIZE = 4096
OUT_CAP = 8192

globals().update(ISA.ALU_ID)
K = ISA.K

_FAULT_SITES = ISA.FAULT_SITE_NAMES
_FAULT_CODES = [ISA.FAULT_SITE_CAUSE[s] for s in _FAULT_SITES]

BANK_PAGES = 1
BANK_OWN = 0

def fault_cause(fired, codes):

    fired = (fired > 0).to(torch.int32)
    codes = torch.as_tensor(codes, dtype=torch.int32, device=fired.device)
    if fired.ndim != 1 or codes.ndim != 1 or fired.shape[0] != codes.shape[0]:
        raise ValueError(
            f"fault_cause takes one signal per site: fired is {tuple(fired.shape)}, "
            f"codes is {tuple(codes.shape)}. Two ranks would broadcast into an outer "
            f"product whose sum is a number the cause table never assigned")
    if fired.shape[0] != len(ISA.FAULT_SITE_ORDER):
        raise ValueError(f"fault_cause was given {fired.shape[0]} site signals, but "
                         f"isa_table.FAULT_SITE_ORDER names {len(ISA.FAULT_SITE_ORDER)}")
    prior = torch.cumsum(fired, 0) - fired
    first = fired * (prior == 0).to(torch.int32)
    return (first * codes).sum()

def _rom(rows):

    return [torch.tensor(list(col), dtype=torch.int32) for col in rows]

_ALU, _S0, _S1, _LEN = _rom(ISA.single_rom())

_ALU2, _S02, _S12, _LX2 = _rom(ISA.escape_rom())

def check_state(R, HL, DE, SP, C, Z, tick=0, PC=0, ipos=0, oplen=0, status=0,
                fault_reason=0, fault_addr=0, mb=0, s=0, v=0, td=0, where=""):

    if not 0 <= oplen <= OUT_CAP:
        raise ValueError(f"{where}state field oplen is {oplen}, outside [0, {OUT_CAP}]")
    bad = ISA.state_error({"r": list(R), "HL": HL, "DE": DE, "MB": mb, "PC": PC,
                           "SP": SP, "C": C, "Z": Z, "S": s, "V": v, "ipos": ipos, "tick": tick,
                           "status": status, "fault_reason": fault_reason,
                           "fault_addr": fault_addr, "TDEPTH": td},
                          where)
    if bad is not None:
        raise ValueError(bad)

class TorchCircuit:
    def __init__(self, code, data=None, inputs=b"",
                 tick_budget=ISA.TICK_BUDGET_DEFAULT, device="cuda",
                 rom_override=None, out_cap=OUT_CAP, config=None):

        if len(code) > CODE_SIZE:
            raise ValueError(f"code image is {len(code)} bytes, above CODE_SIZE {CODE_SIZE}")
        if data and len(data) > DATA_SIZE:
            raise ValueError(f"data image is {len(data)} bytes, above DATA_SIZE {DATA_SIZE}")
        cfg = ISA.MachineConfig() if config is None else config
        if config is not None and not isinstance(config, ISA.MachineConfig):
            raise ISA.ConfigError(
                f"config must be an isa_table.MachineConfig, got a "
                f"{type(config).__name__}: configuration is validated at load, and a "
                f"mapping or tuple would arrive unchecked")
        self.config = cfg
        dev = torch.device(device)
        self.dev = dev
        i32 = torch.int32
        self.codelen = len(code) if cfg.codelen is None else cfg.codelen
        if self.codelen > CODE_SIZE:
            raise ISA.ConfigError(
                f"CODELEN={self.codelen} is past the {CODE_SIZE}-byte CODE buffer this "
                f"machine allocates: the bound has to name storage that exists")
        if self.codelen > len(code):
            raise ISA.ConfigError(
                f"CODELEN={self.codelen} is past the end of the {len(code)}-byte "
                f"image: the machine would fetch bytes that were never loaded")

        bad = cfg.check_for_program(self.codelen, "TorchCircuit: ")
        if bad is not None:
            raise ISA.ConfigError(bad)

        self.entry = ISA.DEFAULT_ENTRY if cfg.entry is None else cfg.entry
        bad = ISA.entry_error(self.entry, self.codelen, "TorchCircuit: ")
        if bad is not None:
            raise ISA.ConfigError(bad)
        self.out_cap = ISA.resolve_constraint("OUTCAP", "out_cap", out_cap, OUT_CAP,
                                              cfg.outcap)
        ISA.check_capacity(self.out_cap, OUT_CAP)

        self.nbanks = 1 if cfg.nbanks is None else cfg.nbanks
        self.tdlim = ISA.DEFAULT_TDLIM if cfg.tdlim is None else cfg.tdlim

        self.splim = 0 if cfg.splim is None else cfg.splim
        self.SPLIM = torch.tensor([self.splim], dtype=i32, device=dev)
        self.CODE = torch.zeros(CODE_SIZE, dtype=i32, device=dev)
        self.CODE[: len(code)] = torch.tensor(list(code), dtype=i32, device=dev)
        self.DATA = torch.zeros(DATA_SIZE, dtype=i32, device=dev)
        if data:
            self.DATA[: len(data)] = torch.tensor(list(data), dtype=i32, device=dev)
        self.INPUTS = torch.zeros(1, dtype=i32, device=dev)
        self.inlen = torch.zeros(1, dtype=i32, device=dev)
        self.set_inputs(inputs)
        self.OUTBUF = torch.zeros(self.out_cap, dtype=i32, device=dev)
        self.R = torch.zeros(4, dtype=i32, device=dev)
        for n in ("HL", "DE", "MB", "PC", "SP", "C", "Z", "S", "V", "TDEPTH", "ipos",
                  "oplen", "tick"):
            setattr(self, n, torch.zeros(1, dtype=i32, device=dev))
        self.SP += DATA_SIZE
        self.PC += self.entry
        self.status = torch.zeros(1, dtype=i32, device=dev)

        self.fault_reason = torch.zeros(1, dtype=i32, device=dev)
        self.fault_addr = torch.zeros(1, dtype=i32, device=dev)
        self.TB = torch.tensor(
            [ISA.resolve_constraint("TICKBUDGET", "tick_budget", int(tick_budget),
                                    ISA.TICK_BUDGET_DEFAULT, cfg.tickbudget)],
            dtype=i32, device=dev)
        self.tb = int(self.TB.item())
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
        self.RO = torch.arange(self.out_cap, dtype=i32, device=dev)

        self.cfg_win = torch.tensor(list(cfg.window()), dtype=i32, device=dev)
        self.cfg_vec = torch.tensor(list(cfg.vectors()), dtype=i32, device=dev)

        self.fault_codes = torch.tensor(_FAULT_CODES, dtype=i32, device=dev)
        self._acts = None

    def set_inputs(self, b):
        self.INPUTS = torch.tensor(list(b) if b else [0], dtype=torch.int32, device=self.dev)
        self.inlen = torch.tensor(len(b), dtype=torch.int32, device=self.dev)

    def _oh(self, x):

        return (self.alu_t == x).to(torch.int32)

    def step(self, ext=None):

        dev, i32 = self.dev, torch.int32
        w = lambda m, a, b: m * a + (1 - m) * b

        pc_entry = self.PC.clone()

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
            esc = (op == ISA.ESCAPE_PREFIX).to(i32)
            imm0 = torch.where(esc > 0, self._g(self.CODE, self.PC + ISA.PREFIX_BYTES),
                               self._g(self.CODE, self.PC + 1))
            imm1 = torch.where(esc > 0, self._g(self.CODE, self.PC + ISA.PREFIX_BYTES + 1),
                               self._g(self.CODE, self.PC + 2))
        ind = (self.ROW == op).to(i32) * (1 - esc) + ((self.ROW == sub).to(i32)) * esc

        _oh1 = self._oh
        oh = lambda x: _oh1(x) * (1 - esc) + ((self.alu2_t == x).to(i32)) * esc
        _s0t, _s1t, _lnt = self.OH_S0, self.OH_S1, self.ln_t
        oh_s0 = _s0t * (1 - esc) + self.ohs0e * esc
        oh_s1 = _s1t * (1 - esc) + self.ohs1e * esc
        ln_e = _lnt * (1 - esc) + (ISA.PREFIX_BYTES + self.lx2_t) * esc
        t16 = imm0 | (imm1 << 8)

        a = (self.R * (ind[:, None] * oh_s0).sum(0)).sum()
        b = (self.R * (ind[:, None] * oh_s1).sum(0)).sum()
        rr = (self.R * (ind[:, None] * self.OH_R).sum(0)).sum()

        a_s0 = a

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

        dsp2 = self._g(self.DATA, self.SP + 2)
        dsp3 = self._g(self.DATA, self.SP + 3)
        inb = self._g(self.INPUTS, self.ipos)
        eof = (self.ipos >= self.inlen).to(i32)

        sx = imm0 - ((imm0 >> 7) & 1) * 256
        fr = (self.HL + sx) & 0xFFFF
        dfr = self._g(self.DATA, fr)
        dhl1 = self._g(self.DATA, self.HL + 1)
        dde1 = self._g(self.DATA, self.DE + 1)

        t_add = a + b; v_add = t_add & 255; c_add = t_add >> 8

        s_add = (v_add >> 7) & 1; v_ovf_add = ((a ^ v_add) & (b ^ v_add)) >> 7
        t_adc = a + b + self.C; v_adc = t_adc & 255; c_adc = t_adc >> 8
        s_adc = (v_adc >> 7) & 1; v_ovf_adc = ((a ^ v_adc) & (b ^ v_adc)) >> 7
        v_sub = (a - b) & 255; c_sub = (a < b).to(i32)
        s_sub = (v_sub >> 7) & 1; v_ovf_sub = ((a ^ b) & (v_sub ^ a)) >> 7
        t_sbb = a - b - self.C; v_sbb = t_sbb & 255; c_sbb = (t_sbb < 0).to(i32)
        s_sbb = (v_sbb >> 7) & 1; v_ovf_sbb = ((a ^ b) & (v_sbb ^ a)) >> 7
        t_addi = rr + imm0; v_addi = t_addi & 255; c_addi = t_addi >> 8
        s_addi = (v_addi >> 7) & 1
        v_ovf_addi = ((rr ^ v_addi) & (imm0 ^ v_addi)) >> 7
        v_subi = (rr - imm0) & 255; c_subi = (rr < imm0).to(i32)
        s_subi = (v_subi >> 7) & 1
        v_ovf_subi = ((rr ^ imm0) & (v_subi ^ rr)) >> 7
        t_adci = rr + imm0 + self.C; v_adci = t_adci & 255; c_adci = t_adci >> 8
        s_adci = (v_adci >> 7) & 1
        v_ovf_adci = ((rr ^ v_adci) & (imm0 ^ v_adci)) >> 7
        v_shl = (rr << 1) & 255; c_shl = rr >> 7
        v_shr = rr >> 1; c_shr = rr & 1
        v_dj = (rr - 1) & 255

        v_and = a & b; v_or = a | b; v_xor = a ^ b
        t_mul = a * b; v_mul = t_mul & 255; c_mul = (t_mul > 255).to(i32)
        b_safe = torch.where(b == 0, torch.ones((), dtype=i32, device=dev), b)
        v_div = a // b_safe; v_mod = a % b_safe
        cmp_c = (a < b).to(i32)
        v_cmp = (a - b) & 255
        cmp_s = (v_cmp >> 7) & 1
        cmp_v = ((a ^ b) & (v_cmp ^ a)) >> 7
        v_not = (~rr) & 255
        v_neg = (-rr) & 255; c_neg = (rr != 0).to(i32)
        s_neg = (v_neg >> 7) & 1
        v_rol = ((rr << 1) | self.C) & 255; c_rol = rr >> 7
        v_ror = (rr >> 1) | (self.C << 7); c_ror = rr & 1
        t_hladd = self.HL + self.DE; v_hladd = t_hladd & 0xFFFF; c_hladd = t_hladd >> 16
        v_hlsub = (self.HL - self.DE) & 0xFFFF; c_hlsub = (self.HL < self.DE).to(i32)

        vec = self._g(self.cfg_vec, imm0)
        ext_ok = (imm0 < ISA.VEC_COUNT).to(i32) * (vec != 0).to(i32)

        fpack = (self.Z * ISA.FLAG_BITS_PACKED["Z"] + self.C * ISA.FLAG_BITS_PACKED["C"]
                 + self.S * ISA.FLAG_BITS_PACKED["S"] + self.V * ISA.FLAG_BITS_PACKED["V"])
        depth_full = (self.TDEPTH == self.tdlim).to(i32)
        trap_tag = (dsp3 == ISA.TRAP_TAG).to(i32)

        wlo, whi = self.cfg_win[0], self.cfg_win[1]
        code_ok = (self.HL < self.codelen).to(i32)

        win_ok = (self.HL >= wlo).to(i32) * (self.HL < whi).to(i32)
        ldc_v = self._g(self.CODE, self.HL)

        v_mulh = ((a * b) >> 8) & 255
        v_popw = dsp | (dsp1 << 8)
        v_ldw_de = dl | (dhl1 << 8)
        v_ldw_hl = der | (dde1 << 8)
        v_sp_add = (self.SP + sx) & 0xFFFF
        v_ldm = dl

        bank_oh = (oh(LDM) + oh(STM) + oh(LDMW_DE_HL) + oh(LDMW_HL_DE)
                   + oh(STMW_HL_DE) + oh(STMW_DE_HL))
        bank_in = (self.MB < self.nbanks).to(i32) * (self.MB < BANK_PAGES).to(i32)
        bank_oob = bank_oh * (1 - bank_in)
        bank_busy = bank_oh * bank_in * (1 - (self.MB == BANK_OWN).to(i32))

        RHL_EN = oh(MOV_R_HL) + oh(MOV_HL_R)
        rows_R_val = (
            oh(ADD) * v_add + oh(SUB) * v_sub + oh(ADC) * v_adc
            + oh(SBB) * v_sbb + oh(MOV) * b + oh(LDI) * imm0
            + oh(ADDI) * v_addi + oh(SUBI) * v_subi + oh(ADCI) * v_adci
            + oh(SHL) * v_shl + oh(SHR) * v_shr + oh(DJNZ) * v_dj
            + oh(POP) * dsp + oh(IN) * (1 - eof) * inb
            + oh(MOV_R_HL) * dl + oh(MOV_R_DE) * der
            + oh(GETPC) * self.PC + oh(GETSP) * (self.SP & 255)
            + oh(GETF) * fpack
            + oh(AND) * v_and + oh(OR) * v_or + oh(XOR) * v_xor + oh(MUL) * v_mul
            + oh(DIV) * v_div + oh(MOD) * v_mod
            + oh(NOT) * v_not + oh(NEG) * v_neg + oh(ROL) * v_rol + oh(ROR) * v_ror
            + oh(LDC) * ldc_v
            + oh(LDX) * dfr + oh(MULH) * v_mulh
            + oh(LDM) * dl
        )
        rows_R_en = (
            oh(ADD) + oh(SUB) + oh(ADC) + oh(SBB) + oh(MOV)
            + oh(LDI) + oh(ADDI) + oh(SUBI) + oh(ADCI)
            + oh(SHL) + oh(SHR) + oh(DJNZ) + oh(POP) + oh(IN)
            + oh(MOV_R_HL) + oh(MOV_R_DE)
            + oh(GETPC) + oh(GETSP) + oh(GETF)
            + oh(AND) + oh(OR) + oh(XOR) + oh(MUL) + oh(DIV) + oh(MOD)
            + oh(NOT) + oh(NEG) + oh(ROL) + oh(ROR) + oh(LDC)
            + oh(LDX) + oh(MULH) + oh(LDM)
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
        rows_HL = w(oh(MOV_HL_MB), self.MB, rows_HL)

        rows_HL = w(oh(LDMW_HL_DE), v_ldw_hl, rows_HL)

        rows_DE = self.DE + oh(INC_DE) + oh(OUTDE)
        rows_DE = w(oh(LDI_DE), t16, rows_DE)
        rows_DE = w(oh(ADDI_DE), self.DE + rv, rows_DE)
        rows_DE = w(oh(XCHG), self.HL, rows_DE)
        rows_DE = w(oh(MOVW_DE_HL), self.HL, rows_DE)
        rows_DE = w(oh(MOVW_DE_SP), self.SP, rows_DE)
        rows_DE = w(oh(POPW_DE), v_popw, rows_DE)
        rows_DE = w(oh(LDW_DEHL), v_ldw_de, rows_DE)
        rows_DE = w(oh(LDMW_DE_HL), v_ldw_de, rows_DE)

        rows_MB = self.MB + oh(MOV_MB_HL) * (self.HL - self.MB)

        rows_SP = w(oh(PUSH), self.SP - 1,
                   w(oh(POP), self.SP + 1,
                     w(oh(CALL), self.SP - 2,
                       w(oh(CALL_HL), self.SP - 2,
                         w(oh(EXT), self.SP - 4,
                           w(oh(RET), self.SP + 2,
                             w(oh(TRAPRET), self.SP + 4, self.SP)))))))
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
            + oh(ADD_HLDE) * (c_hladd - self.C) + oh(SUB_HLDE) * (c_hlsub - self.C) \
            + oh(TRAPRET) * (((dsp2 >> 1) & 1) - self.C)

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
            + oh(MULH) * ((v_mulh == 0).to(i32) - self.Z) \
            + oh(TRAPRET) * ((dsp2 & 1) - self.Z)

        rows_S = self.S + oh(ADD) * (s_add - self.S) \
            + oh(ADC) * (s_adc - self.S) \
            + oh(SUB) * (s_sub - self.S) \
            + oh(SBB) * (s_sbb - self.S) \
            + oh(ADDI) * (s_addi - self.S) \
            + oh(SUBI) * (s_subi - self.S) \
            + oh(ADCI) * (s_adci - self.S) \
            + oh(CMP) * (cmp_s - self.S) \
            + oh(NEG) * (s_neg - self.S) \
            + oh(TRAPRET) * (((dsp2 >> 2) & 1) - self.S)

        rows_V = self.V + oh(ADD) * (v_ovf_add - self.V) \
            + oh(ADC) * (v_ovf_adc - self.V) \
            + oh(SUB) * (v_ovf_sub - self.V) \
            + oh(SBB) * (v_ovf_sbb - self.V) \
            + oh(ADDI) * (v_ovf_addi - self.V) \
            + oh(SUBI) * (v_ovf_subi - self.V) \
            + oh(ADCI) * (v_ovf_adci - self.V) \
            + oh(CMP) * (cmp_v - self.V) \
            + oh(TRAPRET) * (((dsp2 >> 3) & 1) - self.V)

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
        rel = (fall + sx) & 0xFFFF
        js_t = w(self.S.reshape(1), rel, fall)
        jns_t = w((1 - self.S).reshape(1), rel, fall)
        vs_t = w(self.V.reshape(1), rel, fall)
        vc_t = w((1 - self.V).reshape(1), rel, fall)
        rows_PC = w(oh(JS), js_t, rows_PC)
        rows_PC = w(oh(JNS), jns_t, rows_PC)
        rows_PC = w(oh(VS), vs_t, rows_PC)
        rows_PC = w(oh(VC), vc_t, rows_PC)
        rows_PC = w(oh(JPHL), self.HL, rows_PC)

        rows_PC = w(oh(TRAPRET), retv, rows_PC)
        rows_PC = w(oh(CALL_HL), self.HL, rows_PC)
        rows_PC = w(oh(EXT), vec, rows_PC)

        rows_ipos = w(oh(IN) * (1 - eof), self.ipos + 1, self.ipos)

        rows_TDEPTH = w(oh(EXT), self.TDEPTH + 1,
                        w(oh(TRAPRET), self.TDEPTH - 1, self.TDEPTH))

        rows_out_val = oh(OUT) * rr + oh(OUTM) * dl + oh(OUTDE) * der
        rows_out_en = oh(OUT) + oh(OUTM) + oh(OUTDE)

        fetch_ok_rows = (self.PC + ln_e <= self.codelen).to(i32)
        hl_ok = (self.HL < DATA_SIZE).to(i32); de_ok = (self.DE < DATA_SIZE).to(i32)

        sp_lo = (((self.SP - self.SPLIM) > 0) & (self.SP <= DATA_SIZE)).to(i32)
        sp_lo2 = (((self.SP - self.SPLIM) >= 2) & (self.SP <= DATA_SIZE)).to(i32)
        sp_hi = (self.SP < DATA_SIZE).to(i32)
        sp_hi1 = ((self.SP + 1) < DATA_SIZE).to(i32)

        hl_ok2 = ((self.HL + 1) < DATA_SIZE).to(i32)
        de_ok2 = ((self.DE + 1) < DATA_SIZE).to(i32)
        fr_ok = (fr < DATA_SIZE).to(i32)
        sp_hi2 = ((self.SP + 2) <= DATA_SIZE).to(i32)

        sp_lo4 = (((self.SP - self.SPLIM) >= 4) & (self.SP <= DATA_SIZE)).to(i32)
        sp_hi4 = ((self.SP + 4) <= DATA_SIZE).to(i32)
        hl_sp_ok = (self.HL <= DATA_SIZE).to(i32)
        de_sp_ok = (self.DE <= DATA_SIZE).to(i32)
        sp_add_ok = (v_sp_add <= DATA_SIZE).to(i32)

        rd_rows = (oh(MOV_R_HL) + oh(MOV_HL_R)) * (1 - hl_ok) \
            + (oh(MOV_R_DE) + oh(MOV_DE_R)) * (1 - de_ok) \
            + oh(OUTM) * (1 - hl_ok) + oh(OUTDE) * (1 - de_ok) \
            + (oh(STW_HLDE) + oh(LDW_DEHL)) * (1 - hl_ok2) \
            + (oh(STW_DEHL) + oh(LDW_HLDE)) * (1 - de_ok2) \
            + (oh(LDX) + oh(STX)) * (1 - fr_ok) \
            + (oh(LDM) + oh(STM)) * (1 - hl_ok) \
            + (oh(LDMW_DE_HL) + oh(STMW_HL_DE)) * (1 - hl_ok2) \
            + (oh(LDMW_HL_DE) + oh(STMW_DE_HL)) * (1 - de_ok2)

        push_rows = (oh(PUSH) * (1 - sp_lo) + oh(CALL) * (1 - sp_lo2)
                     + oh(CALL_HL) * (1 - sp_lo2)
                     + oh(EXT) * (1 - sp_lo4)
                     + (oh(PUSHW_HL) + oh(PUSHW_DE)) * (1 - sp_lo2))
        pop_rows = (oh(POP) * (1 - sp_hi) + oh(RET) * (1 - sp_hi1)
                    + oh(TRAPRET) * (1 - sp_hi4)
                    + (oh(POPW_HL) + oh(POPW_DE)) * (1 - sp_hi2))
        v3_err = (oh(MOVW_SP_HL) * (1 - hl_sp_ok) + oh(MOVW_SP_DE) * (1 - de_sp_ok)
                  + oh(ADD_SP) * (1 - sp_add_ok))

        out_ovf = (ind * rows_out_en * (self.oplen >= self.out_cap).to(i32)).sum()

        cl = self.codelen

        trapret_live = (self.TDEPTH > 0).to(i32) * sp_hi4 * trap_tag
        pc_hit = (oh(JMP) + oh(CALL)) * (t16 >= cl).to(i32) \
            + oh(RET) * (retv >= cl).to(i32) \
            + oh(TRAPRET) * trapret_live * (retv >= cl).to(i32) \
            + oh(JZ) * self.Z * (t16 >= cl).to(i32) \
            + oh(JNZ) * (1 - self.Z) * (t16 >= cl).to(i32) \
            + oh(JC) * self.C * (t16 >= cl).to(i32) \
            + oh(JNC) * (1 - self.C) * (t16 >= cl).to(i32) \
            + oh(DJNZ) * (v_dj != 0).to(i32) * (t16 >= cl).to(i32) \
            + (oh(JS) * self.S + oh(JNS) * (1 - self.S)
               + oh(VS) * self.V + oh(VC) * (1 - self.V)) * (rel >= cl).to(i32) \
            + oh(JPHL) * (self.HL >= cl).to(i32) \
            + oh(CALL_HL) * (self.HL >= cl).to(i32) \
            + oh(EXT) * ext_ok * (1 - depth_full) * (vec >= cl).to(i32)
        pcw_clean = fetch_ok_rows * (1 - push_rows) * (1 - pop_rows)

        bad_rows = (ind * oh(BAD)).sum()
        fetch_code = (((1 - fetch_ok)
                       + esc * (self.PC + ISA.PREFIX_BYTES > self.codelen).to(i32)).sum())

        fired = torch.stack(tuple(v.reshape(()) for v in (
            (ind * pc_hit * pcw_clean).sum(),
            fetch_code,
            bad_rows * (1 - esc),
            bad_rows * esc,
            (ind * (1 - fetch_ok_rows)).sum(),

            (ind * bank_oob).sum(),
            (ind * bank_busy).sum(),
            (ind * oh(EXT) * (1 - ext_ok)).sum(),
            (ind * oh(EXT) * ext_ok * depth_full).sum(),
            (ind * oh(TRAPRET) * (self.TDEPTH == 0).to(i32)).sum(),
            (ind * rd_rows).sum(),
            (ind * push_rows).sum(),
            (ind * pop_rows).sum(),
            (ind * oh(TRAPRET) * (self.TDEPTH > 0).to(i32) * sp_hi4
             * (1 - trap_tag)).sum(),
            (ind * (oh(DIV) + oh(MOD)) * (b == 0).to(i32)).sum(),
            (ind * (oh(LDC) + oh(STC)) * (1 - code_ok)).sum(),
            (ind * oh(STC) * code_ok * (1 - win_ok)).sum(),
            (ind * v3_err).sum(),
            out_ovf,
        )))
        cause = fault_cause(fired, self.fault_codes)

        err = (cause != 0).to(i32)

        sel = lambda rows: (ind * rows).sum()
        R_w = (ind[:, None] * oh_s0 * rows_R_en[:, None]).sum(0)
        R_v = (ind[:, None] * oh_s0 * (rows_R_en * rows_R_val)[:, None]).sum(0)
        ok = (err == 0).to(i32)

        running = (self.status == 0).to(i32)
        over = running * (self.tick >= self.TB).to(i32)
        m = ok * running * (1 - over)

        a1 = sel(oh(MOV_HL_R) * self.HL + oh(MOV_DE_R) * self.DE
                 + oh(PUSH) * (self.SP - 1) + oh(CALL) * (self.SP - 1)
                 + oh(CALL_HL) * (self.SP - 1)
                 + oh(EXT) * (self.SP - 1)
                 + (oh(PUSHW_HL) + oh(PUSHW_DE)) * (self.SP - 1)
                 + oh(STW_HLDE) * self.HL + oh(STW_DEHL) * self.DE
                 + oh(STX) * fr
                 + oh(STM) * self.HL + oh(STMW_HL_DE) * self.HL
                 + oh(STMW_DE_HL) * self.DE)
        v1 = sel(oh(MOV_HL_R) * rr + oh(MOV_DE_R) * rr + oh(PUSH) * rr
                 + oh(CALL) * ((self.PC + 3) & 255) + oh(CALL_HL) * ((self.PC + 2) & 255)
                 + oh(EXT) * ISA.TRAP_TAG
                 + oh(PUSHW_HL) * ((self.HL >> 8) & 255) + oh(PUSHW_DE) * ((self.DE >> 8) & 255)
                 + oh(STW_HLDE) * (self.DE & 255) + oh(STW_DEHL) * (self.HL & 255)
                 + oh(STX) * rr
                 + oh(STM) * a_s0 + oh(STMW_HL_DE) * (self.DE & 255)
                 + oh(STMW_DE_HL) * (self.HL & 255))
        e1 = sel(oh(MOV_HL_R) + oh(MOV_DE_R) + oh(PUSH) + oh(CALL) + oh(CALL_HL)
                 + oh(EXT)
                 + oh(PUSHW_HL) + oh(PUSHW_DE) + oh(STW_HLDE) + oh(STW_DEHL) + oh(STX)
                 + oh(STM) + oh(STMW_HL_DE) + oh(STMW_DE_HL)) * m
        oh1 = ((self.RD == a1).to(i32)) * e1
        self.DATA = oh1 * v1 + (1 - oh1) * self.DATA
        a2 = sel(oh(CALL) * (self.SP - 2) + oh(CALL_HL) * (self.SP - 2)
                 + oh(EXT) * (self.SP - 2)
                 + (oh(PUSHW_HL) + oh(PUSHW_DE)) * (self.SP - 2)
                 + oh(STW_HLDE) * (self.HL + 1) + oh(STW_DEHL) * (self.DE + 1)
                 + oh(STMW_HL_DE) * (self.HL + 1) + oh(STMW_DE_HL) * (self.DE + 1))
        v2 = sel(oh(CALL) * ((self.PC + 3) >> 8) + oh(CALL_HL) * ((self.PC + 2) >> 8)
                 + oh(EXT) * fpack
                 + oh(PUSHW_HL) * (self.HL & 255) + oh(PUSHW_DE) * (self.DE & 255)
                 + oh(STW_HLDE) * ((self.DE >> 8) & 255) + oh(STW_DEHL) * ((self.HL >> 8) & 255)
                 + oh(STMW_HL_DE) * ((self.DE >> 8) & 255)
                 + oh(STMW_DE_HL) * ((self.HL >> 8) & 255))
        e2 = sel(oh(CALL) + oh(CALL_HL) + oh(EXT) + oh(PUSHW_HL) + oh(PUSHW_DE)
                 + oh(STW_HLDE) + oh(STW_DEHL) + oh(STMW_HL_DE) + oh(STMW_DE_HL)) * m
        oh2 = ((self.RD == a2).to(i32)) * e2
        self.DATA = oh2 * v2 + (1 - oh2) * self.DATA

        a3 = sel(oh(EXT) * (self.SP - 3))
        v3 = sel(oh(EXT) * ((self.PC + 3) & 255))
        e3 = sel(oh(EXT)) * m
        oh3 = ((self.RD == a3).to(i32)) * e3
        self.DATA = oh3 * v3 + (1 - oh3) * self.DATA
        a4 = sel(oh(EXT) * (self.SP - 4))
        v4 = sel(oh(EXT) * ((self.PC + 3) >> 8))
        e4 = sel(oh(EXT)) * m
        oh4 = ((self.RD == a4).to(i32)) * e4
        self.DATA = oh4 * v4 + (1 - oh4) * self.DATA

        a5 = sel(oh(STC) * self.HL); v5 = sel(oh(STC) * rr); e5 = sel(oh(STC)) * m
        oh5 = ((self.RC == a5).to(i32)) * e5
        self.CODE = oh5 * v5 + (1 - oh5) * self.CODE

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

        fault_w = running * (1 - over) * (err > 0).to(i32)
        self.fault_reason = torch.where(fault_w > 0, cause.reshape(1), self.fault_reason)
        self.fault_addr = torch.where(fault_w > 0, pc_entry, self.fault_addr)

        self.R = m * ((R_w * R_v + (1 - R_w) * self.R) & 255) + (1 - m) * self.R
        self.HL = (m * sel(rows_HL) + (1 - m) * self.HL) & 0xFFFF
        self.DE = (m * sel(rows_DE) + (1 - m) * self.DE) & 0xFFFF
        self.MB = (m * sel(rows_MB) + (1 - m) * self.MB) & 0xFFFF
        self.PC = m * sel(rows_PC) + (1 - m) * self.PC
        self.SP = m * sel(rows_SP) + (1 - m) * self.SP
        self.C = m * sel(rows_C) + (1 - m) * self.C
        self.Z = m * sel(rows_Z) + (1 - m) * self.Z
        self.S = m * (sel(rows_S) & 1) + (1 - m) * self.S
        self.V = m * (sel(rows_V) & 1) + (1 - m) * self.V
        self.TDEPTH = m * sel(rows_TDEPTH) + (1 - m) * self.TDEPTH
        self.ipos = m * sel(rows_ipos) + (1 - m) * self.ipos
        self.tick = self.tick + m

    def _g(self, buf, idx):
        return buf.index_select(0, idx.clamp(0, buf.numel() - 1).reshape(1)).reshape(()).to(torch.int32)

    def load_state(self, R, HL, DE, SP, C, Z, tick, PC=0, fault_reason=None,
                   fault_addr=None, S=None, V=None):

        fr = int(self.fault_reason.item()) if fault_reason is None else fault_reason
        fa = int(self.fault_addr.item()) if fault_addr is None else fault_addr
        sv = int(self.S.item()) if S is None else S
        vv = int(self.V.item()) if V is None else V
        check_state(R, HL, DE, SP, C, Z, tick=tick, PC=PC, ipos=int(self.ipos.item()),
                    oplen=int(self.oplen.item()), status=int(self.status.item()),
                    fault_reason=fr, fault_addr=fa, mb=int(self.MB.item()), s=sv, v=vv,
                    td=int(self.TDEPTH.item()))
        t = torch.tensor
        self.fault_reason = t([fr], dtype=torch.int32, device=self.dev)
        self.fault_addr = t([fa], dtype=torch.int32, device=self.dev)
        self.S = t([sv], dtype=torch.int32, device=self.dev)
        self.V = t([vv], dtype=torch.int32, device=self.dev)
        self.R = t(R, dtype=torch.int32, device=self.dev)
        self.HL = t([HL], dtype=torch.int32, device=self.dev)
        self.DE = t([DE], dtype=torch.int32, device=self.dev)
        self.SP = t([SP], dtype=torch.int32, device=self.dev)
        self.C = t([C], dtype=torch.int32, device=self.dev)
        self.Z = t([Z], dtype=torch.int32, device=self.dev)
        self.tick = t([tick], dtype=torch.int32, device=self.dev)
        self.PC = t([PC], dtype=torch.int32, device=self.dev)

    def _record_inputs(self):

        return bytes(int(v) for v in self.INPUTS[: int(self.inlen.item())].cpu().tolist())

    def _record_block(self):

        lo, hi = self.config.window()
        return ISA.MachineConfig(codelen=self.codelen, entry=self.entry,
                                 winlo=lo, winhi=hi,
                                 vec=self.config.vectors(), nbanks=self.nbanks,
                                 tdlim=self.tdlim, splim=self.splim,
                                 tickbudget=self.tb, outcap=self.out_cap)

    _RECORD_SCALARS = ("HL", "DE", "MB", "PC", "SP", "C", "Z", "S", "V", "TDEPTH",
                       "ipos", "tick", "status", "fault_reason", "fault_addr")
    _RECORD_READERS = {
        **{n: (lambda m, n=n: int(getattr(m, n).item())) for n in _RECORD_SCALARS},
        "r": lambda m: [int(v) for v in m.R.tolist()],

        "CODE": lambda m: bytes(int(v) for v in m.CODE.cpu().tolist()),
        "DATA": lambda m: bytes(int(v) for v in m.DATA.cpu().tolist()),
        "out": lambda m: m.out(),
        "inputs": lambda m: m._record_inputs(),
        "block": lambda m: m._record_block().as_dict(),
    }

    def _record_bounds(self):

        return ISA.RecordBounds(where="TorchCircuit: ", out_cap=self.out_cap,
                                code_size=CODE_SIZE, data_size=DATA_SIZE,
                                inputs=self._record_inputs(),
                                block=self._record_block())

    def record_state(self):

        return ISA.publish_record(self, self._RECORD_READERS, self._record_bounds())

    def install_state(self, snap):

        got = ISA.check_record(snap, self._record_bounds())
        st = got.state
        check_state(st["r"], st["HL"], st["DE"], st["SP"], st["C"], st["Z"],
                    tick=st["tick"], PC=st["PC"], ipos=st["ipos"],
                    oplen=len(got.out), status=st["status"],
                    fault_reason=st["fault_reason"], fault_addr=st["fault_addr"],
                    mb=st["MB"], s=st["S"], v=st["V"], td=st["TDEPTH"],
                    where="TorchCircuit: ")
        t = torch.tensor
        i32 = torch.int32
        self.R = t(st["r"], dtype=i32, device=self.dev)
        for name in self._RECORD_SCALARS:
            setattr(self, name, t([st[name]], dtype=i32, device=self.dev))
        self.OUTBUF.zero_()
        if got.out:
            self.OUTBUF[: len(got.out)] = t(list(got.out), dtype=i32, device=self.dev)
        self.oplen = t([len(got.out)], dtype=i32, device=self.dev)
        self.CODE = t(list(got.code), dtype=i32, device=self.dev)
        self.DATA = t(list(got.data), dtype=i32, device=self.dev)

    def snapshot(self):

        return dict(r=self.R.tolist(), HL=self.HL.item(), DE=self.DE.item(),
                    MB=self.MB.item(),
                    SP=self.SP.item(), PC=self.PC.item(), C=self.C.item(), Z=self.Z.item(),
                    S=self.S.item(), V=self.V.item(),
                    TDEPTH=int(self.TDEPTH.item()),
                    ipos=self.ipos.item(), oplen=self.oplen.item(), tick=self.tick.item(),
                    status=int(self.status.item()),
                    fault_reason=int(self.fault_reason.item()),
                    fault_addr=int(self.fault_addr.item()))

    def out(self):

        n = min(int(self.oplen.item()), self.out_cap)
        return bytes(self.OUTBUF[:n].cpu().tolist())

    def run(self):

        while int(self.status.item()) == 0:
            self.step()
        return self.out()