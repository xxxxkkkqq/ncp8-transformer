"""NCP-8 reference simulator and two-pass assembler.

This module is the semantic reference for the ISA: pure Python integers, no
floating point, no randomness, fully deterministic. It defines the observable
state (registers r0-r3, pointers HL/DE, stack pointer, PC, carry/zero flags,
input/output streams, tick counter, status) and the exact per-tick transition.

Status codes: 0 = running, 1 = halted, 2 = tick budget exhausted, 3 = error.
An error tick writes only the three fault fields - status, fault_reason (the cause
code from isa_table's table) and fault_addr (the address of the instruction that
faulted, taken at tick entry) - and no register, memory, flag, PC or tick update;
step() rolls back on any exception, not only on a machine error. Stepping a
machine whose status is not running is a no-op that changes nothing at all and is
not an error. The tick budget and the output capacity (OUT_CAP bytes) are machine
state and are applied inside step(). check_state()/load_state() refuse a state
whose fields are outside their declared widths.

Opcode layout (fields do not overlap):
  0x00-0x1F  no-operand / a16 / i16 / self-read families
  0x20-0x5F  bitwise and multiply (base|(r<<2)|s: AND/OR/XOR/MUL)
  0x60-0x6F  single-register family (SHL/SHR/TST/DJNZ)
  0x70       escape prefix (two-byte opcode; subcode space reserved for extension)
  0x80-0xCF  register-to-register (ADD/SUB/ADC/SBB/MOV)
  0xD0-0xDF  register-to-immediate (LDI/ADDI/SUBI/ADCI)
  0xE0-0xFF  memory / stack / IO families
"""
from __future__ import annotations
import re

import isa_table as ISA

DATA_SIZE = 4096
CODE_SIZE = 4096
OUT_CAP = 8192

R_BITS = 8
PTR_BITS = 16
FAULT_BITS = ISA.FAULT_BITS

STATUS_RUNNING = "RUNNING"
STATUS_HALT = "HALT"
STATUS_OVERRUN = "OVERRUN"
STATUS_ERROR = "ERROR"
STATUS_CODE = {STATUS_RUNNING: 0, STATUS_HALT: 1, STATUS_OVERRUN: 2, STATUS_ERROR: 3}

CAUSE = ISA.CAUSE

class MachineError(Exception):

    def __init__(self, msg, reason=0):
        super().__init__(msg)
        self.reason = int(reason)

class FaultCauseMissing(MachineError):

    pass

def check_state(R, HL, DE, SP, C, Z, tick=0, PC=0, ipos=0, oplen=0, status=0,
                fault_reason=0, fault_addr=0, where=""):

    def outside(field, value, lo, hi):
        raise ValueError(f"{where}state field {field} is {value}, outside [{lo}, {hi}]")

    for i, v in enumerate(list(R)):
        if not 0 <= v < (1 << R_BITS):
            outside(f"r[{i}]", v, 0, (1 << R_BITS) - 1)
    for field, v in (("HL", HL), ("DE", DE), ("PC", PC)):
        if not 0 <= v < (1 << PTR_BITS):
            outside(field, v, 0, (1 << PTR_BITS) - 1)
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
    bad = ISA.fault_state_error(status, fault_reason, fault_addr, where)
    if bad is not None:
        raise ValueError(bad)

class NCP8:
    def __init__(self, code, data=None, inputs=b"", tick_budget=ISA.TICK_BUDGET_DEFAULT,
                 out_cap=OUT_CAP, config=None):

        if len(code) > CODE_SIZE:
            raise ValueError(f"code image is {len(code)} bytes, above CODE_SIZE {CODE_SIZE}")
        cfg = ISA.MachineConfig() if config is None else config
        if config is not None and not isinstance(config, ISA.MachineConfig):
            raise ISA.ConfigError(
                f"config must be an isa_table.MachineConfig, got a "
                f"{type(config).__name__}: configuration is validated at load, and a "
                f"mapping or tuple would arrive unchecked")
        self.config = cfg
        self.code = bytes(code)

        self.codelen = len(self.code) if cfg.codelen is None else cfg.codelen
        if self.codelen > len(self.code):
            raise ISA.ConfigError(
                f"CODELEN={self.codelen} is past the end of the {len(self.code)}-byte "
                f"image: the machine would fetch bytes that were never loaded")
        self.data = bytearray(DATA_SIZE)
        if data:
            if len(data) > DATA_SIZE:
                raise ValueError(f"data image is {len(data)} bytes, above DATA_SIZE {DATA_SIZE}")
            self.data[: len(data)] = data
        self.r = [0, 0, 0, 0]
        self.HL = 0
        self.DE = 0
        self.SP = DATA_SIZE
        self.PC = 0
        self.C = 0
        self.Z = 0
        self.inputs = bytes(inputs)
        self.ipos = 0
        self.out = bytearray()
        self.out_buf_width = OUT_CAP
        self.out_cap = ISA.resolve_constraint("OUTCAP", "out_cap", out_cap, OUT_CAP,
                                              cfg.outcap)
        ISA.check_capacity(self.out_cap, self.out_buf_width)
        self.tick = 0
        self.tb = ISA.resolve_constraint("TICKBUDGET", "tick_budget",
                                         int(tick_budget), ISA.TICK_BUDGET_DEFAULT,
                                         cfg.tickbudget)

        self.nbanks = 1 if cfg.nbanks is None else cfg.nbanks
        self.tdlim = 0 if cfg.tdlim is None else cfg.tdlim
        self.status = STATUS_RUNNING

        self.fault_reason = CAUSE["OK"]
        self.fault_addr = 0
        self.trace: list[str] = []

    def load_state(self, R, HL, DE, SP, C, Z, tick, PC=0, fault_reason=None,
                   fault_addr=None):

        fr = self.fault_reason if fault_reason is None else fault_reason
        fa = self.fault_addr if fault_addr is None else fault_addr
        check_state(R, HL, DE, SP, C, Z, tick=tick, PC=PC, ipos=self.ipos,
                    status=STATUS_CODE[self.status], fault_reason=fr, fault_addr=fa)
        self.r = list(R)
        self.HL, self.DE, self.SP, self.C, self.Z = HL, DE, SP, C, Z
        self.tick, self.PC = tick, PC
        self.fault_reason, self.fault_addr = fr, fa

    def _fault(self, cause, msg):

        raise MachineError(msg, reason=cause)

    def _mem(self, addr):
        if not 0 <= addr < DATA_SIZE:
            self._fault(CAUSE["DATA_OOB"], f"DATA out of range: {addr}")

    def _mem16(self, addr):

        if not (0 <= addr and addr + 1 < DATA_SIZE):
            self._fault(CAUSE["DATA_OOB"], f"DATA out of range: {addr}")
        return addr

    def _fetch(self, n):
        if self.PC + n > self.codelen:
            self._fault(CAUSE["FETCH_OOB"], f"PC out of range: {self.PC}")
        b = self.code[self.PC: self.PC + n]
        self.PC += n
        return b

    def _window(self):

        cfg = self.config
        if cfg.has_window:
            return cfg.winlo, cfg.winhi, True
        WLO, WHI = 0x0F20, 0x0F21
        wl = self.code[WLO] if WLO < len(self.code) else 0
        wh = self.code[WHI] if WHI < len(self.code) else 0
        return wl, wh, False

    def _vector(self, k):

        cfg = self.config
        if cfg.has_vec:
            return cfg.vector(k)
        a = 0x0F00 + 2 * k
        if a + 1 < len(self.code):
            return (self.code[a + 1] << 8) | self.code[a]
        return 0

    def _stack_room(self, n):

        if not (0 <= self.SP - n and self.SP <= DATA_SIZE):
            self._fault(CAUSE["STACK_OVERFLOW"], "stack overflow")

    def _stack_have(self, n):

        if not (0 <= self.SP and self.SP + n <= DATA_SIZE):
            self._fault(CAUSE["STACK_UNDERFLOW"], "stack underflow")

    def _emit(self, v):

        if len(self.out) >= self.out_cap:
            self._fault(CAUSE["OUT_CAP"], f"output capacity {self.out_cap} exhausted")
        self.out.append(v & 0xFF)

    def _push(self, byte):
        self._stack_room(1)
        self.SP -= 1
        self.data[self.SP] = byte & 0xFF

    def _pop(self):
        self._stack_have(1)
        v = self.data[self.SP]
        self.SP += 1
        return v

    def step(self):

        pc0 = self.PC
        try:
            self._step_inner()
        except MachineError as e:

            self.PC = pc0
            reason = e.reason
            if reason not in ISA.CAUSE_NAME or reason == CAUSE["OK"]:
                raise FaultCauseMissing(
                    f"the fault at PC {pc0:#04x} ({e}) carried no cause code from the "
                    f"cause table, so the error tick cannot record fault_reason") from e
            self.status = STATUS_ERROR
            self.fault_reason = reason
            self.fault_addr = pc0
            raise
        except BaseException:
            self.PC = pc0
            raise

    def _step_inner(self):

        if self.status != STATUS_RUNNING:
            return

        if self.tick >= self.tb:
            self.status = STATUS_OVERRUN
            return
        pc0 = self.PC

        (op,) = self._fetch(1)
        if op == ISA.ESCAPE_PREFIX:
            (sub,) = self._fetch(1)
            row = ISA.ESCAPE.get(sub)
            if row is None:
                self._fault(CAUSE["BAD_SUBCODE"], f"reserved subcode {sub:#04x} (ESC)  @ {pc0:#04x}")
        else:
            row = ISA.SINGLE.get(op)
            if row is None:
                self._fault(CAUSE["BAD_OPCODE"], f"undefined opcode {op:#04x} @ {pc0:#04x}")
        sel = row["alu"]
        s0, s1 = row["s0"], row["s1"]
        imm = self._fetch(row["l"])
        m = "???"

        if sel == "HALT":
            self.status = STATUS_HALT; m = "HALT"
        elif sel == "NOP": m = "NOP"
        elif sel == "INC_HL": self.HL = (self.HL + 1) & 0xFFFF; m = "INC HL"
        elif sel == "DEC_HL": self.HL = (self.HL - 1) & 0xFFFF; m = "DEC HL"
        elif sel == "INC_DE": self.DE = (self.DE + 1) & 0xFFFF; m = "INC DE"
        elif sel == "CLC": self.C = 0; m = "CLC"
        elif sel == "OUTM":
            self._mem(self.HL); self._emit(self.data[self.HL])
            self.HL = (self.HL + 1) & 0xFFFF; m = "OUTM"
        elif sel == "OUTDE":
            self._mem(self.DE); self._emit(self.data[self.DE])
            self.DE = (self.DE + 1) & 0xFFFF; m = "OUTDE"
        elif sel == "RET":
            self._stack_have(2)
            hi = self._pop(); lo = self._pop(); self.PC = hi << 8 | lo; m = "RET"
        elif sel in ("JMP", "JZ", "JNZ", "JC", "JNC"):
            t = imm[1] << 8 | imm[0]
            if sel == "JMP": self.PC = t; m = "JMP"
            elif sel == "JZ":   m = "JZ";   self.PC = t if self.Z else self.PC
            elif sel == "JNZ":  m = "JNZ";  self.PC = t if not self.Z else self.PC
            elif sel == "JC":   m = "JC";   self.PC = t if self.C else self.PC
            else:               m = "JNC";  self.PC = t if not self.C else self.PC
        elif sel == "CALL":
            t = imm[1] << 8 | imm[0]; ret = self.PC
            self._stack_room(2)
            self._push(ret & 0xFF); self._push(ret >> 8); self.PC = t; m = "CALL"
        elif sel == "LDI_HL":
            self.HL = imm[1] << 8 | imm[0]; m = f"LDI HL, {self.HL}"
        elif sel == "LDI_DE":
            self.DE = imm[1] << 8 | imm[0]; m = f"LDI DE, {self.DE}"
        elif sel == "ADDI_HL":
            s = imm[0] & 3
            self.HL = (self.HL + self.r[s]) & 0xFFFF; m = f"ADDI HL, r{s}"
        elif sel == "ADDI_DE":
            s = imm[0] & 3
            self.DE = (self.DE + self.r[s]) & 0xFFFF; m = f"ADDI DE, r{s}"
        elif sel == "JPHL":
            self.PC = self.HL; m = "JPHL"
        elif sel == "GETPC":
            self.r[s0] = pc0 & 255; m = f"GETPC r{s0}"
        elif sel == "GETSP":
            self.r[s0] = self.SP & 255; m = f"GETSP r{s0}"
        elif sel == "GETF":
            self.r[s0] = self.Z | (self.C << 1); m = f"GETF r{s0}"
        elif sel == "ADD":
            a, b = self.r[s0], self.r[s1]
            t = a + b; self.C = t >> 8; self.r[s0] = t & 0xFF
            self.Z = int((t & 0xFF) == 0); m = f"ADD r{s0}, r{s1}"
        elif sel == "SUB":
            a, b = self.r[s0], self.r[s1]
            self.C = int(a < b); v = (a - b) & 0xFF; self.r[s0] = v
            self.Z = int(v == 0); m = f"SUB r{s0}, r{s1}"
        elif sel == "ADC":
            a, b = self.r[s0], self.r[s1]
            t = a + b + self.C; self.C = t >> 8; self.r[s0] = t & 0xFF
            self.Z = int((t & 0xFF) == 0); m = f"ADC r{s0}, r{s1}"
        elif sel == "SBB":
            a, b = self.r[s0], self.r[s1]
            t = a - b - self.C; self.C = int(t < 0); v = t & 0xFF; self.r[s0] = v
            self.Z = int(v == 0); m = f"SBB r{s0}, r{s1}"
        elif sel == "MOV":
            b = self.r[s1]; self.r[s0] = b; m = f"MOV r{s0}, r{s1}"
        elif sel == "AND":
            a, b = self.r[s0], self.r[s1]
            v = a & b; self.r[s0] = v; self.Z = int(v == 0); m = f"AND r{s0}, r{s1}"
        elif sel == "OR":
            a, b = self.r[s0], self.r[s1]
            v = a | b; self.r[s0] = v; self.Z = int(v == 0); m = f"OR r{s0}, r{s1}"
        elif sel == "XOR":
            a, b = self.r[s0], self.r[s1]
            v = a ^ b; self.r[s0] = v; self.Z = int(v == 0); m = f"XOR r{s0}, r{s1}"
        elif sel == "MUL":
            a, b = self.r[s0], self.r[s1]
            t = a * b; self.C = int(t > 255); v = t & 0xFF
            self.r[s0] = v; self.Z = int(v == 0); m = f"MUL r{s0}, r{s1}"
        elif sel in ("DIV", "MOD"):
            a, b = self.r[s0], self.r[s1]
            if b == 0:
                self._fault(CAUSE["DIV_ZERO"], f"divide by zero {sel} r{s0}, r{s1} @ {pc0:#04x}")
            v = a // b if sel == "DIV" else a % b
            self.r[s0] = v; self.Z = int(v == 0); m = f"{sel} r{s0}, r{s1}"
        elif sel == "CMP":
            a, b = self.r[s0], self.r[s1]
            self.Z = int(a == b); self.C = int(a < b); m = f"CMP r{s0}, r{s1}"
        elif sel == "NOT":
            v = (~self.r[s0]) & 0xFF; self.r[s0] = v
            self.Z = int(v == 0); m = f"NOT r{s0}"
        elif sel == "NEG":
            t = (-self.r[s0]) & 0xFF; self.C = int(self.r[s0] != 0)
            self.r[s0] = t; self.Z = int(t == 0); m = f"NEG r{s0}"
        elif sel == "ROL":
            v = self.r[s0]; self.C, self.r[s0] = v >> 7, ((v << 1) | self.C) & 0xFF
            self.Z = int(self.r[s0] == 0); m = f"ROL r{s0}"
        elif sel == "ROR":
            v = self.r[s0]; self.C, self.r[s0] = v & 1, ((v >> 1) | (self.C << 7)) & 0xFF
            self.Z = int(self.r[s0] == 0); m = f"ROR r{s0}"
        elif sel == "MOVW_HL_DE":
            self.HL = self.DE; m = "MOVW HL, DE"
        elif sel == "MOVW_DE_HL":
            self.DE = self.HL; m = "MOVW DE, HL"
        elif sel == "MOVW_HL_SP":
            self.HL = self.SP; m = "MOVW HL, SP"
        elif sel == "MOVW_DE_SP":
            self.DE = self.SP; m = "MOVW DE, SP"
        elif sel == "MOVW_SP_HL":
            if self.HL > DATA_SIZE:
                self._fault(CAUSE["DATA_OOB"], f"MOVW SP, HL out of range {self.HL} @ {pc0:#04x}")
            self.SP = self.HL; m = "MOVW SP, HL"
        elif sel == "MOVW_SP_DE":
            if self.DE > DATA_SIZE:
                self._fault(CAUSE["DATA_OOB"], f"MOVW SP, DE out of range {self.DE} @ {pc0:#04x}")
            self.SP = self.DE; m = "MOVW SP, DE"
        elif sel == "PUSHW_HL" or sel == "PUSHW_DE":
            v = self.HL if sel == "PUSHW_HL" else self.DE
            self._stack_room(2); self.SP -= 2
            self.data[self.SP] = v & 0xFF
            self.data[self.SP + 1] = (v >> 8) & 0xFF
            m = "PUSHW HL" if sel == "PUSHW_HL" else "PUSHW DE"
        elif sel == "POPW_HL" or sel == "POPW_DE":
            self._stack_have(2)
            v = self.data[self.SP] | (self.data[self.SP + 1] << 8)
            if sel == "POPW_HL":
                self.HL = v; m = "POPW HL"
            else:
                self.DE = v; m = "POPW DE"
            self.SP += 2
        elif sel == "STW_HLDE":
            self._mem16(self.HL)
            self.data[self.HL] = self.DE & 0xFF
            self.data[self.HL + 1] = (self.DE >> 8) & 0xFF
            m = "STW [HL], DE"
        elif sel == "STW_DEHL":
            self._mem16(self.DE)
            self.data[self.DE] = self.HL & 0xFF
            self.data[self.DE + 1] = (self.HL >> 8) & 0xFF
            m = "STW [DE], HL"
        elif sel == "LDW_DEHL":
            self._mem16(self.HL)
            self.DE = self.data[self.HL] | (self.data[self.HL + 1] << 8)
            m = "LDW DE, [HL]"
        elif sel == "LDW_HLDE":
            self._mem16(self.DE)
            self.HL = self.data[self.DE] | (self.data[self.DE + 1] << 8)
            m = "LDW HL, [DE]"
        elif sel == "LDX":
            sx = (imm[0] ^ 0x80) - 0x80
            addr = (self.HL + sx) & 0xFFFF
            self._mem(addr)
            self.r[s0] = self.data[addr]; m = f"LDX r{s0}, [HL{sx:+d}]"
        elif sel == "STX":
            sx = (imm[0] ^ 0x80) - 0x80
            addr = (self.HL + sx) & 0xFFFF
            self._mem(addr)
            self.data[addr] = self.r[s0]; m = f"STX [HL{sx:+d}], r{s0}"
        elif sel == "ADD_SP":
            sx = (imm[0] ^ 0x80) - 0x80
            sp = (self.SP + sx) & 0xFFFF
            if sp > DATA_SIZE:
                self._fault(CAUSE["DATA_OOB"], f"ADD SP out of range {sp} @ {pc0:#04x}")
            self.SP = sp; m = f"ADD SP, {sx}"
        elif sel == "ADD_HLDE":
            t = self.HL + self.DE; self.C = t >> 16; self.HL = t & 0xFFFF; m = "ADD HL, DE"
        elif sel == "SUB_HLDE":
            self.C = int(self.HL < self.DE); self.HL = (self.HL - self.DE) & 0xFFFF; m = "SUB HL, DE"
        elif sel == "XCHG":
            self.HL, self.DE = self.DE, self.HL; m = "XCHG HL, DE"
        elif sel == "LDC":
            if not 0 <= self.HL < self.codelen:
                self._fault(CAUSE["CODE_OOB"], f"LDC out of range {self.HL} @ {pc0:#04x}")
            self.r[s0] = self.code[self.HL]; m = f"LDC r{s0}, [HL]"
        elif sel == "STC":

            wl, wh, from_cfg = self._window()
            if not 0 <= self.HL < self.codelen:
                self._fault(CAUSE["CODE_OOB"], f"STC out of range {self.HL} @ {pc0:#04x}")
            if self.config.has_vec and ISA.LEGACY_VEC_BASE <= self.HL < ISA.LEGACY_CONFIG_HI:

                self._fault(CAUSE["WINDOW"],
                    f"WINDOW: STC target {self.HL:#06x} is a legacy configuration cell this build treats as a constraint "
                    f"[{ISA.LEGACY_VEC_BASE:#06x},{ISA.LEGACY_CONFIG_HI:#06x}), "
                    f"unrelated to this machine's window declaration @ {pc0:#04x}")
            if not (wl <= self.HL < wh):
                self._fault(CAUSE["WINDOW"],
                    f"STC outside window {self.HL:#x} not in [{wl:#x},{wh:#x}) @ {pc0:#04x}")
            b = bytearray(self.code); b[self.HL] = self.r[s0]; self.code = bytes(b)
            m = f"STC [HL], r{s0}"
        elif sel == "MULH":
            v = ((self.r[s0] * self.r[s1]) >> 8) & 0xFF
            self.r[s0] = v; self.Z = int(v == 0); m = f"MULH r{s0}, r{s1}"
        elif sel == "EXT":

            k = imm[0]
            if k >= 16:
                self._fault(CAUSE["TRAP_UNREG"], f"EXT k out of range {k} @ {pc0:#04x}")
            tgt = self._vector(k)
            if tgt == 0:
                self._fault(CAUSE["TRAP_UNREG"], f"EXT handler {k} unregistered @ {pc0:#04x}")
            self._stack_room(2)
            self._push(self.PC & 0xFF); self._push((self.PC >> 8) & 0xFF)
            self.PC = tgt; m = f"EXT {k}"
        elif sel in ("LDI", "ADDI", "SUBI", "ADCI"):
            i = imm[0]
            if sel == "LDI":
                self.r[s0] = i; m = f"LDI r{s0}, {i}"
            elif sel == "ADDI":
                t = self.r[s0] + i
                self.C = t >> 8; self.r[s0] = t & 0xFF
                self.Z = int((t & 0xFF) == 0); m = f"ADDI r{s0}, {i}"
            elif sel == "SUBI":
                a = self.r[s0]; self.C = int(a < i); v = (a - i) & 0xFF
                self.r[s0] = v; self.Z = int(v == 0); m = f"SUBI r{s0}, {i}"
            else:
                t = self.r[s0] + i + self.C
                self.C = t >> 8; self.r[s0] = t & 0xFF
                self.Z = int((t & 0xFF) == 0); m = f"ADCI r{s0}, {i}"
        elif sel == "SHL":
            self.C = self.r[s0] >> 7; v = (self.r[s0] << 1) & 0xFF; self.r[s0] = v
            self.Z = int(v == 0); m = f"SHL r{s0}"
        elif sel == "SHR":
            self.C = self.r[s0] & 1; v = self.r[s0] >> 1; self.r[s0] = v
            self.Z = int(v == 0); m = f"SHR r{s0}"
        elif sel == "TST":
            self.Z = int(self.r[s0] == 0); m = f"TST r{s0}"
        elif sel == "DJNZ":
            t = imm[1] << 8 | imm[0]
            self.r[s0] = (self.r[s0] - 1) & 0xFF
            if self.r[s0] != 0: self.PC = t
            m = f"DJNZ r{s0}"
        elif sel == "MOV_R_HL":
            self._mem(self.HL); self.r[s0] = self.data[self.HL]; m = f"MOV r{s0}, [HL]"
        elif sel == "MOV_HL_R":
            self._mem(self.HL); self.data[self.HL] = self.r[s0]; m = f"MOV [HL], r{s0}"
        elif sel == "MOV_R_DE":
            self._mem(self.DE); self.r[s0] = self.data[self.DE]; m = f"MOV r{s0}, [DE]"
        elif sel == "MOV_DE_R":
            self._mem(self.DE); self.data[self.DE] = self.r[s0]; m = f"MOV [DE], r{s0}"
        elif sel == "PUSH": self._push(self.r[s0]); m = f"PUSH r{s0}"
        elif sel == "POP": self.r[s0] = self._pop(); m = f"POP r{s0}"
        elif sel == "OUT": self._emit(self.r[s0]); m = f"OUT r{s0}"
        elif sel == "IN":
            if self.ipos < len(self.inputs):
                self.r[s0] = self.inputs[self.ipos]; self.ipos += 1
            else:
                self.r[s0] = 0; self.C = 1
            m = f"IN r{s0}"
        else:

            self._fault(CAUSE["BAD_SUBCODE" if row["space"] == "escape" else "BAD_OPCODE"],
                        f"undefined opcode {op:#04x} @ {pc0:#04x}")

        self.trace.append(
            f"{self.tick:6d} {pc0:04X} {m:18s} | r=[{self.r[0]:3d},{self.r[1]:3d},{self.r[2]:3d},{self.r[3]:3d}]"
            f" DE={self.DE:04X} HL={self.HL:04X} SP={self.SP:04X} C={self.C} Z={self.Z}"
        )
        self.tick += 1

    def run(self):
        while self.status == STATUS_RUNNING:
            self.step()
        return self.out

    def snapshot(self):

        return dict(r=list(self.r), HL=self.HL, DE=self.DE, SP=self.SP, PC=self.PC,
                    C=self.C, Z=self.Z, ipos=self.ipos, tick=self.tick,
                    status=self.status, fault_reason=self.fault_reason,
                    fault_addr=self.fault_addr)

    def status_code(self):

        return STATUS_CODE[self.status]

class AssemblyError(Exception):

    def __init__(self, msg, line=None):
        self.msg = msg
        self.line = line
        super().__init__(f"line {line}: {msg}" if line is not None else msg)

def asm(src: str) -> bytes:
    pat = re.compile(r"^(\w+):$")
    items = []
    for lineno, raw in enumerate(src.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        mm = pat.match(line)
        if mm:
            items.append(("label", mm.group(1), lineno, line))
            continue
        name, _, args = line.partition(" ")
        args = [a.strip() for a in args.split(",")] if args.strip() else []
        items.append(("inst", name.strip(), args, lineno, line))

    def enc(name, args, labels, lineno, text, strict):
        def _bad(msg):
            raise AssemblyError(msg, lineno)

        def _reg(x):

            if not isinstance(x, str) or x not in ("r0", "r1", "r2", "r3"):
                _bad(f"invalid register operand {x!r} in {text!r}")
            return int(x[1])

        def _value(x):

            s = str(x)
            if s in labels:
                return labels[s]
            try:
                return int(s, 0)
            except ValueError:
                pass
            if re.fullmatch(r"[A-Za-z_]\w*", s):
                _bad(f"undefined symbol {s!r} in {text!r}")
            _bad(f"unsupported operand expression {s!r} in {text!r}")

        def _addr(x, mnemonic):

            if not strict:

                return 0
            v = _value(x)
            if not 0 <= v <= 0xFFFF:
                _bad(f"{mnemonic} address {v} is out of range 0..65535 in {text!r}")
            return v

        def _imm8(x, mnemonic):

            if not strict:
                return 0
            s = str(x)
            if s in labels:
                _bad(f"{mnemonic} needs a numeric 8-bit immediate, {s!r} is a label in {text!r}")
            try:
                v = int(s, 0)
            except ValueError:
                if re.fullmatch(r"[A-Za-z_]\w*", s):
                    _bad(f"undefined symbol {s!r} in {text!r}")
                _bad(f"unsupported operand expression {s!r} in {text!r}")
            if not 0 <= v <= 0xFF:
                _bad(f"{mnemonic} immediate {v} is out of range 0..255 in {text!r}")
            return v

        def _imm8s(x, mnemonic):

            if not strict:
                return 0
            s = str(x)
            if s in labels:
                _bad(f"{mnemonic} needs a numeric signed 8-bit immediate, {s!r} is a label in {text!r}")
            try:
                v = int(s, 0)
            except ValueError:
                if re.fullmatch(r"[A-Za-z_]\w*", s):
                    _bad(f"undefined symbol {s!r} in {text!r}")
                _bad(f"unsupported operand expression {s!r} in {text!r}")
            if not -128 <= v <= 127:
                _bad(f"{mnemonic} immediate {v} is out of range -128..127 in {text!r}")
            return v & 0xFF

        def _frame_off(x, mnemonic):

            if not strict:
                return 0
            s = str(x).replace(" ", "")
            mm = re.fullmatch(r"\[HL(?:([+-])([^+\-\[\]]+))?\]", s)
            if not mm:
                _bad(f"{mnemonic} needs an [HL+i8] address operand, {x!r} is not one in {text!r}")
            sign, num = mm.group(1), mm.group(2)
            if num is None:
                return 0
            if num in labels:
                _bad(f"{mnemonic} needs a numeric signed 8-bit offset, {num!r} is a label in {text!r}")
            try:
                v = int(num, 0)
            except ValueError:
                if re.fullmatch(r"[A-Za-z_]\w*", num):
                    _bad(f"undefined symbol {num!r} in {text!r}")
                _bad(f"unsupported operand expression {num!r} in {text!r}")
            if sign == "-":
                v = -v
            if not -128 <= v <= 127:
                _bad(f"{mnemonic} offset {v} is out of range -128..127 in {text!r}")
            return v & 0xFF

        simple = {"HALT": 0x00, "NOP": 0x01, "INC HL": 0x02, "DEC HL": 0x03,
                  "INC DE": 0x04, "CLC": 0x05, "OUTM": 0x06, "OUTDE": 0x07, "RET": 0x08,
                  "LDI HL": 0x0F, "LDI DE": 0x10, "ADDI HL": 0x11, "ADDI DE": 0x12}
        if args and f"{name} {args[0]}" in simple:
            base = simple[f"{name} {args[0]}"]
            rest = args[1:]
            if base in (0x0F, 0x10):

                v = _addr(rest[0], name)
                return bytes([base, v & 0xFF, v >> 8])
            if base in (0x11, 0x12):
                return bytes([base, _reg(rest[0])])
            return bytes([base])
        if name in simple:
            return bytes([simple[name]])
        if name == "JPHL":
            return bytes([0x13])
        if name in ("GETPC", "GETSP", "GETF"):
            base = {"GETPC": 0x14, "GETSP": 0x18, "GETF": 0x1C}[name]
            return bytes([base | _reg(args[0])])
        jump = {"JMP": 0x09, "JZ": 0x0A, "JNZ": 0x0B, "JC": 0x0C, "JNC": 0x0D, "CALL": 0x0E}
        if name in jump:

            a = _addr(args[0], name)
            return bytes([jump[name], a & 0xFF, a >> 8])
        if name == "DJNZ":
            a = _addr(args[1], name)
            return bytes([0x6C | _reg(args[0]), a & 0xFF, a >> 8])
        if name == "LDI":
            return bytes([0xD0 | _reg(args[0]), _imm8(args[1], name)])
        if name == "ADDI":
            return bytes([0xD4 | _reg(args[0]), _imm8(args[1], name)])
        if name == "SUBI":
            return bytes([0xD8 | _reg(args[0]), _imm8(args[1], name)])
        if name == "ADCI":
            return bytes([0xDC | _reg(args[0]), _imm8(args[1], name)])
        if name in ("ADD", "SUB") and args and args[0] == "HL":
            return bytes([0x70, {"ADD": 0x60, "SUB": 0x61}[name]])
        if name == "ADD" and args and args[0] == "SP":
            return bytes([0x70, 0x58, _imm8s(args[1], name)])
        if name == "XCHG":
            return bytes([0x70, 0x62])
        if name == "STC":
            return bytes([0x70, 0x80 | _reg(args[1])])
        if name == "LDC":
            return bytes([0x70, 0x84 | _reg(args[0])])
        if name == "EXT":
            return bytes([0x70, 0x70, _imm8(args[0], name)])
        rr2 = {"AND": 0x20, "OR": 0x30, "XOR": 0x40, "MUL": 0x50}
        if name in rr2:
            return bytes([rr2[name] | (_reg(args[0]) << 2) | _reg(args[1])])
        rr3 = {"DIV": 0x00, "MOD": 0x10, "CMP": 0x20}
        if name in rr3:
            return bytes([0x70, rr3[name] | (_reg(args[0]) << 2) | _reg(args[1])])
        un2 = {"NOT": 0x40, "NEG": 0x44, "ROL": 0x48, "ROR": 0x4C}
        if name in un2:
            return bytes([0x70, un2[name] | _reg(args[0])])
        rr = {"ADD": 0x80, "SUB": 0x90, "ADC": 0xA0, "SBB": 0xB0, "MOV": 0xC0}
        if name == "MOV":
            mem = {"[HL]": 0xE0, "[DE]": 0xE8}
            if args[1] in ("[HL]", "[DE]"):
                return bytes([mem[args[1]] | _reg(args[0])])
            if args[0] in ("[HL]", "[DE]"):
                return bytes([(mem[args[0]] + 4) | _reg(args[1])])
            return bytes([0xC0 | (_reg(args[0]) << 2) | _reg(args[1])])
        if name in rr:
            return bytes([rr[name] | (_reg(args[0]) << 2) | _reg(args[1])])
        unary = {"PUSH": 0xF0, "POP": 0xF4, "OUT": 0xF8, "IN": 0xFC,
                 "SHL": 0x60, "SHR": 0x64, "TST": 0x68}
        if name in unary:
            return bytes([unary[name] | _reg(args[0])])

        movw = {("HL", "DE"): 0x30, ("DE", "HL"): 0x31, ("HL", "SP"): 0x32,
                ("DE", "SP"): 0x33, ("SP", "HL"): 0x34, ("SP", "DE"): 0x35}
        if name == "MOVW":
            if tuple(args) not in movw:
                raise AssemblyError(f"unknown operand combination {args!r} in {text!r}", lineno)
            return bytes([0x70, movw[tuple(args)]])
        if name in ("PUSHW", "POPW"):
            if args not in (["HL"], ["DE"]):
                raise AssemblyError(f"{name} needs HL or DE in {text!r}", lineno)
            return bytes([0x70, {"PUSHW": 0x38, "POPW": 0x3A}[name] + (args == ["DE"])])
        if name in ("STW", "LDW"):
            wide = {("STW", "[HL]", "DE"): 0x3C, ("STW", "[DE]", "HL"): 0x3D,
                    ("LDW", "DE", "[HL]"): 0x3E, ("LDW", "HL", "[DE]"): 0x3F}
            key = tuple([name] + args)
            if key not in wide:
                raise AssemblyError(f"unknown operand combination {args!r} in {text!r}", lineno)
            return bytes([0x70, wide[key]])
        if name == "LDX":
            return bytes([0x70, 0x50 | _reg(args[0]), _frame_off(args[1], name)])
        if name == "STX":
            return bytes([0x70, 0x54 | _reg(args[1]), _frame_off(args[0], name)])
        if name == "MULH":
            return bytes([0x70, 0x90 | (_reg(args[0]) << 2) | _reg(args[1])])
        raise AssemblyError(f"unknown instruction {name!r} in {text!r}", lineno)

    labels, addr = {}, 0

    def enc_checked(it, labels, strict):

        b = enc(it[1], it[2], labels, it[3], it[4], strict)
        want = ISA.instruction_length(b)
        if want != len(b):
            raise AssemblyError(
                f"assembler emitted {b.hex()} for {it[4]!r}, which the decode table "
                f"reads as "
                + ("no assigned instruction" if want is None else f"{want} bytes"),
                it[3])
        return b

    for it in items:
        if it[0] == "label":
            labels[it[1]] = addr
        else:
            addr += len(enc_checked(it, labels, False))
    out = bytearray()
    for it in items:
        if it[0] != "label":
            out += enc_checked(it, labels, True)
    if len(out) > CODE_SIZE:
        raise AssemblyError(f"assembled code is {len(out)} bytes, above CODE_SIZE {CODE_SIZE}")
    return bytes(out)