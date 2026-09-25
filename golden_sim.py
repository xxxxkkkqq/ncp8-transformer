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
state and are applied inside step(). Every bound the machine obeys - the length of
the code, the self-modification window, the trap vectors, the tick budget and the
output capacity - is load-time configuration handed in beside the image, and no cell
of CODE holds any of them. check_state()/load_state() refuse a state whose fields are
outside their declared widths.

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

import isa_forms
import isa_table as ISA

DATA_SIZE = 4096
CODE_SIZE = 4096
OUT_CAP = 8192

STATUS_RUNNING = "RUNNING"
STATUS_HALT = "HALT"
STATUS_OVERRUN = "OVERRUN"
STATUS_ERROR = "ERROR"
STATUS_CODE = {STATUS_RUNNING: 0, STATUS_HALT: 1, STATUS_OVERRUN: 2, STATUS_ERROR: 3}

STATUS_NAME = {c: n for n, c in STATUS_CODE.items()}

CAUSE = ISA.CAUSE

class MachineError(Exception):

    def __init__(self, msg, reason=0):
        super().__init__(msg)
        self.reason = int(reason)

class FaultCauseMissing(MachineError):

    pass

def check_state(R, HL, DE, SP, C, Z, tick=0, PC=0, ipos=0, oplen=0, status=0,
                fault_reason=0, fault_addr=0, mb=0, where=""):

    if not 0 <= oplen <= OUT_CAP:
        raise ValueError(f"{where}state field oplen is {oplen}, outside [0, {OUT_CAP}]")
    bad = ISA.state_error({"r": list(R), "HL": HL, "DE": DE, "MB": mb, "PC": PC,
                           "SP": SP, "C": C, "Z": Z, "ipos": ipos, "tick": tick,
                           "status": status, "fault_reason": fault_reason,
                           "fault_addr": fault_addr},
                          where)
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

        self.MB = 0
        self.bank_own = 0
        self.banks = (self.data,)
        self.bank_owner_status = None
        self.status = STATUS_RUNNING

        self.fault_reason = CAUSE["OK"]
        self.fault_addr = 0
        self.trace: list[str] = []

    def load_state(self, R, HL, DE, SP, C, Z, tick, PC=0, fault_reason=None,
                   fault_addr=None):

        fr = self.fault_reason if fault_reason is None else fault_reason
        fa = self.fault_addr if fault_addr is None else fault_addr
        check_state(R, HL, DE, SP, C, Z, tick=tick, PC=PC, ipos=self.ipos,
                    status=STATUS_CODE[self.status], fault_reason=fr, fault_addr=fa,
                    mb=self.MB)
        self.r = list(R)
        self.HL, self.DE, self.SP, self.C, self.Z = HL, DE, SP, C, Z
        self.tick, self.PC = tick, PC
        self.fault_reason, self.fault_addr = fr, fa

    def _record_inputs(self):

        return self.inputs

    def _record_code(self):

        return bytes(self.code).ljust(CODE_SIZE, b"\x00")[:CODE_SIZE]

    def _record_block(self):

        lo, hi = self.config.window()
        return ISA.MachineConfig(codelen=self.codelen, winlo=lo, winhi=hi,
                                 vec=self.config.vectors(), nbanks=self.nbanks,
                                 tdlim=self.tdlim, tickbudget=self.tb,
                                 outcap=self.out_cap)

    _RECORD_READERS = {
        "r": lambda m: list(m.r),
        "HL": lambda m: m.HL,
        "DE": lambda m: m.DE,
        "MB": lambda m: m.MB,
        "PC": lambda m: m.PC,
        "SP": lambda m: m.SP,
        "C": lambda m: m.C,
        "Z": lambda m: m.Z,
        "ipos": lambda m: m.ipos,
        "tick": lambda m: m.tick,
        "status": lambda m: m.status_code(),
        "fault_reason": lambda m: m.fault_reason,
        "fault_addr": lambda m: m.fault_addr,
        "CODE": lambda m: m._record_code(),
        "DATA": lambda m: bytes(m.data),
        "out": lambda m: bytes(m.out),
        "inputs": lambda m: m._record_inputs(),
        "block": lambda m: m._record_block().as_dict(),
    }

    def _record_bounds(self):

        return ISA.RecordBounds(where="NCP8: ", out_cap=self.out_cap,
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
                    status=st["status"], fault_reason=st["fault_reason"],
                    fault_addr=st["fault_addr"], mb=st["MB"], where="NCP8: ")
        self.r = list(st["r"])
        self.HL, self.DE, self.PC, self.SP = st["HL"], st["DE"], st["PC"], st["SP"]
        self.MB = st["MB"]
        self.C, self.Z = st["C"], st["Z"]
        self.ipos, self.tick = st["ipos"], st["tick"]
        self.status = STATUS_NAME[st["status"]]
        self.fault_reason, self.fault_addr = st["fault_reason"], st["fault_addr"]
        self.code = got.code

        self.data[:] = got.data
        self.out = bytearray(got.out)

    def _fault(self, cause, msg):

        raise MachineError(msg, reason=cause)

    def _mem(self, addr):
        if not 0 <= addr < DATA_SIZE:
            self._fault(CAUSE["DATA_OOB"], f"DATA out of range: {addr}")

    def _mem16(self, addr):

        if not (0 <= addr and addr + 1 < DATA_SIZE):
            self._fault(CAUSE["DATA_OOB"], f"DATA out of range: {addr}")
        return addr

    def install_banks(self, pages, own, owner_status):

        pages = tuple(pages)
        if len(pages) != self.nbanks:
            raise ISA.ConfigError(
                f"this machine was loaded with NBANKS={self.nbanks} and a group of "
                f"{len(pages)} pages: a bank set is declared at load, not assembled by "
                f"the driver that steps it")
        if not 0 <= own < len(pages):
            raise ISA.ConfigError(f"bank index {own} is outside the {len(pages)} pages")
        if pages[own] is not self.data:
            raise ISA.ConfigError("the page this machine is told it owns is not its own "
                                  "DATA, so its reads and writes would leave its bank")
        if len(owner_status) != len(pages):
            raise ISA.ConfigError(f"{len(owner_status)} owner statuses for "
                                  f"{len(pages)} banks: every page's owner has to be "
                                  f"named for the quiescence rule to be decidable")
        self.banks = pages
        self.bank_own = own
        self.bank_owner_status = tuple(owner_status)

    def _bank_page(self, mb):

        if mb >= self.nbanks or mb >= len(self.banks):
            self._fault(CAUSE["BANK_OOB"],
                        f"MB={mb} is outside the {self.nbanks} banks this machine was "
                        f"loaded with")
        if mb == self.bank_own:
            return self.banks[mb]
        st = self.bank_owner_status
        if st is None or mb >= len(st) or st[mb] == STATUS_CODE[STATUS_RUNNING]:
            self._fault(CAUSE["BANK_BUSY"],
                        f"bank {mb} is being run by its own machine")
        return self.banks[mb]

    def _bank_mem(self, page, addr):

        if not 0 <= addr < len(page):
            self._fault(CAUSE["DATA_OOB"],
                        f"address {addr} is outside the {len(page)}-byte bank page")

    def _bank_mem16(self, page, addr):

        if not (0 <= addr and addr + 1 < len(page)):
            self._fault(CAUSE["DATA_OOB"],
                        f"address {addr} and {addr + 1} are not both inside the "
                        f"{len(page)}-byte bank page")
        return addr

    def _fetch(self, n):
        if self.PC + n > self.codelen:
            self._fault(CAUSE["FETCH_OOB"], f"PC out of range: {self.PC}")
        b = self.code[self.PC: self.PC + n]
        self.PC += n
        return b

    def _window(self):

        return self.config.window()

    def _vector(self, k):

        return self.config.vector(k)

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
            wl, wh = self._window()
            if not 0 <= self.HL < self.codelen:
                self._fault(CAUSE["CODE_OOB"], f"STC out of range {self.HL} @ {pc0:#04x}")
            if not (wl <= self.HL < wh):
                self._fault(CAUSE["WINDOW"],
                    f"WINDOW: STC target {self.HL:#06x} is outside the declared window "
                    f"(winlo {wl:#06x}, winhi {wh:#06x}, upper bound exclusive); both "
                    f"bounds are load-time configuration and no cell of CODE holds "
                    f"either @ {pc0:#04x}")
            b = bytearray(self.code); b[self.HL] = self.r[s0]; self.code = bytes(b)
            m = f"STC [HL], r{s0}"
        elif sel == "MULH":
            v = ((self.r[s0] * self.r[s1]) >> 8) & 0xFF
            self.r[s0] = v; self.Z = int(v == 0); m = f"MULH r{s0}, r{s1}"
        elif sel == "LDM":
            page = self._bank_page(self.MB)
            self._bank_mem(page, self.HL)
            self.r[s0] = page[self.HL]; m = f"LDM r{s0}, [HL]"
        elif sel == "STM":
            page = self._bank_page(self.MB)
            self._bank_mem(page, self.HL)
            page[self.HL] = self.r[s0]; m = f"STM [HL], r{s0}"
        elif sel == "LDMW_DE_HL":
            page = self._bank_page(self.MB)
            self._bank_mem16(page, self.HL)
            self.DE = page[self.HL] | (page[self.HL + 1] << 8); m = "LDMW DE, [HL]"
        elif sel == "LDMW_HL_DE":
            page = self._bank_page(self.MB)
            self._bank_mem16(page, self.DE)
            self.HL = page[self.DE] | (page[self.DE + 1] << 8); m = "LDMW HL, [DE]"
        elif sel == "STMW_HL_DE":
            page = self._bank_page(self.MB)
            self._bank_mem16(page, self.HL)
            page[self.HL] = self.DE & 0xFF
            page[self.HL + 1] = (self.DE >> 8) & 0xFF
            m = "STMW [HL], DE"
        elif sel == "STMW_DE_HL":
            page = self._bank_page(self.MB)
            self._bank_mem16(page, self.DE)
            page[self.DE] = self.HL & 0xFF
            page[self.DE + 1] = (self.HL >> 8) & 0xFF
            m = "STMW [DE], HL"
        elif sel == "MOV_MB_HL":
            self.MB = self.HL & 0xFFFF; m = "MOV MB, HL"
        elif sel == "MOV_HL_MB":
            self.HL = self.MB & 0xFFFF; m = "MOV HL, MB"
        elif sel == "EXT":
            k = imm[0]
            if k >= ISA.VEC_COUNT:
                self._fault(CAUSE["TRAP_UNREG"], f"EXT k out of range {k} @ {pc0:#04x}")
            tgt = self._vector(k)
            if tgt == 0:
                self._fault(CAUSE["TRAP_UNREG"],
                            f"EXT handler {k} is not registered: this machine keeps "
                            f"its trap entry points in load-time configuration and "
                            f"CODE holds none, so declaring one at load is the only "
                            f"way to install a handler @ {pc0:#04x}")
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

        return dict(r=list(self.r), HL=self.HL, DE=self.DE, MB=self.MB, SP=self.SP,
                    PC=self.PC,
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

class _Operand(Exception):

    pass

def _literal(text):

    try:
        return int(text.replace("_", ""), 0)
    except (ValueError, TypeError):
        return None

def _is_symbol(text):
    return bool(re.fullmatch(r"[A-Za-z_]\w*", text))

def _operand_value(kind, x, name, labels, strict):

    s = str(x).strip()
    if kind in ("HL", "DE", "SP", "MB", "[HL]", "[DE]"):
        return None
    if kind in ("r", "rcanon"):
        return int(s[1])
    if kind in ("a16", "i16"):
        v = labels[s] if s in labels else _literal(s)
        if v is None:
            if not strict:
                return 0
            if s in isa_forms.RESERVED:
                raise _Operand(f"{s!r} names a register or pointer, not a target")
            if _is_symbol(s):
                raise _Operand(f"undefined symbol {s!r}")
            raise _Operand(f"unsupported operand expression {s!r}")
        if not 0 <= v <= 0xFFFF:
            raise _Operand(f"{name} address {v} is out of range 0..65535")
        return v
    if kind in ("i8", "k"):
        word, hi = ("8-bit immediate", 0xFF) if kind == "i8" else ("trap number", 0xFF)
        if s in labels:
            raise _Operand(f"{name} needs a numeric {word}, {s!r} is a label")
        v = _literal(s)
        if v is None:
            if not strict and _is_symbol(s):
                return 0
            if _is_symbol(s):
                raise _Operand(f"undefined symbol {s!r}")
            raise _Operand(f"unsupported operand expression {s!r}")
        if not 0 <= v <= hi:
            if kind == "k":
                raise _Operand(f"{name} trap number {v} is out of range 0..{hi}")
            raise _Operand(f"{name} immediate {v} is out of range 0..{hi}")
        return v & 0xFF
    if kind == "soff":
        if s in labels:
            raise _Operand(f"{name} needs a numeric signed 8-bit immediate, "
                           f"{s!r} is a label")
        v = _literal(s)
        if v is None:
            if not strict and _is_symbol(s):
                return 0
            if _is_symbol(s):
                raise _Operand(f"undefined symbol {s!r}")
            raise _Operand(f"unsupported operand expression {s!r}")
        if not -128 <= v <= 127:
            raise _Operand(f"{name} immediate {v} is out of range -128..127")
        return v & 0xFF
    if kind == "[HL+-i8]":
        m = re.fullmatch(r"\[HL(?:([+-])([^+\-\[\]]+))?\]", s.replace(" ", ""))
        if not m:
            raise _Operand(f"{name} needs an [HL+i8] address operand, {s!r} is not one")
        sign, num = m.group(1), m.group(2)
        if num is None:
            return 0
        if num in labels:
            raise _Operand(f"{name} needs a numeric signed 8-bit offset, {num!r} is a "
                           "label")
        v = _literal(num)
        if v is None:
            if not strict and _is_symbol(num):
                return 0
            if _is_symbol(num):
                raise _Operand(f"undefined symbol {num!r}")
            raise _Operand(f"unsupported operand expression {num!r}")
        if sign == "-":
            v = -v
        if not -128 <= v <= 127:
            raise _Operand(f"{name} offset {v} is out of range -128..127")
        return v & 0xFF
    raise KeyError(f"golden_sim: undeclared operand kind {kind!r}")

def _diagnose(kind, x, name, labels):

    try:
        _operand_value(kind, x, name, labels, True)
    except _Operand as e:
        return str(e)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return None

def _code(byte_index, shift):
    return ("code", byte_index, shift)

PLACE_KIND = {"r": "code", "rcanon": "byte", "a16": "word", "i16": "word", "i8": "byte",
              "soff": "byte", "k": "byte", "[HL+-i8]": "byte", "HL": None, "DE": None,
              "SP": None, "MB": None, "[HL]": None, "[DE]": None}

ENC = {
    ("ADC", ("r", "r")): (b"\xa0", [_code(0, 2), _code(0, 0)]),
    ("ADCI", ("r", "i8")): (b"\xdc", [_code(0, 0), ("byte",)]),
    ("ADD", ("HL", "DE")): (b"\x70\x60", [None, None]),
    ("ADD", ("SP", "soff")): (b"\x70\x58", [None, ("byte",)]),
    ("ADD", ("r", "r")): (b"\x80", [_code(0, 2), _code(0, 0)]),
    ("ADDI", ("DE", "rcanon")): (b"\x12", [None, ("byte",)]),
    ("ADDI", ("HL", "rcanon")): (b"\x11", [None, ("byte",)]),
    ("ADDI", ("r", "i8")): (b"\xd4", [_code(0, 0), ("byte",)]),
    ("AND", ("r", "r")): (b"\x20", [_code(0, 2), _code(0, 0)]),
    ("CALL", ("a16",)): (b"\x0e", [("word",)]),
    ("CLC", ()): (b"\x05", []),
    ("CMP", ("r", "r")): (b"\x70\x20", [_code(1, 2), _code(1, 0)]),
    ("DEC", ("HL",)): (b"\x03", [None]),
    ("DJNZ", ("r", "a16")): (b"\x6c", [_code(0, 0), ("word",)]),
    ("DIV", ("r", "r")): (b"\x70\x00", [_code(1, 2), _code(1, 0)]),
    ("EXT", ("k",)): (b"\x70\x70", [("byte",)]),
    ("GETF", ("r",)): (b"\x1c", [_code(0, 0)]),
    ("GETPC", ("r",)): (b"\x14", [_code(0, 0)]),
    ("GETSP", ("r",)): (b"\x18", [_code(0, 0)]),
    ("HALT", ()): (b"\x00", []),
    ("IN", ("r",)): (b"\xfc", [_code(0, 0)]),
    ("INC", ("DE",)): (b"\x04", [None]),
    ("INC", ("HL",)): (b"\x02", [None]),
    ("JC", ("a16",)): (b"\x0c", [("word",)]),
    ("JMP", ("a16",)): (b"\x09", [("word",)]),
    ("JNC", ("a16",)): (b"\x0d", [("word",)]),
    ("JNZ", ("a16",)): (b"\x0b", [("word",)]),
    ("JPHL", ()): (b"\x13", []),
    ("JZ", ("a16",)): (b"\x0a", [("word",)]),
    ("LDC", ("r", "[HL]")): (b"\x70\x84", [_code(1, 0), None]),
    ("LDI", ("DE", "i16")): (b"\x10", [None, ("word",)]),
    ("LDI", ("HL", "i16")): (b"\x0f", [None, ("word",)]),
    ("LDI", ("r", "i8")): (b"\xd0", [_code(0, 0), ("byte",)]),
    ("LDM", ("r", "[HL]")): (b"\x70\xb0", [_code(1, 0), None]),
    ("LDMW", ("DE", "[HL]")): (b"\x70\xb8", [None, None]),
    ("LDMW", ("HL", "[DE]")): (b"\x70\xb9", [None, None]),
    ("LDW", ("DE", "[HL]")): (b"\x70\x3e", [None, None]),
    ("LDW", ("HL", "[DE]")): (b"\x70\x3f", [None, None]),
    ("LDX", ("r", "[HL+-i8]")): (b"\x70\x50", [_code(1, 0), ("byte",)]),
    ("MOD", ("r", "r")): (b"\x70\x10", [_code(1, 2), _code(1, 0)]),
    ("MOV", ("[DE]", "r")): (b"\xec", [None, _code(0, 0)]),
    ("MOV", ("[HL]", "r")): (b"\xe4", [None, _code(0, 0)]),
    ("MOV", ("HL", "MB")): (b"\x70\xbd", [None, None]),
    ("MOV", ("MB", "HL")): (b"\x70\xbc", [None, None]),
    ("MOV", ("r", "[DE]")): (b"\xe8", [_code(0, 0), None]),
    ("MOV", ("r", "[HL]")): (b"\xe0", [_code(0, 0), None]),
    ("MOV", ("r", "r")): (b"\xc0", [_code(0, 2), _code(0, 0)]),
    ("MOVW", ("DE", "HL")): (b"\x70\x31", [None, None]),
    ("MOVW", ("DE", "SP")): (b"\x70\x33", [None, None]),
    ("MOVW", ("HL", "DE")): (b"\x70\x30", [None, None]),
    ("MOVW", ("HL", "SP")): (b"\x70\x32", [None, None]),
    ("MOVW", ("SP", "DE")): (b"\x70\x35", [None, None]),
    ("MOVW", ("SP", "HL")): (b"\x70\x34", [None, None]),
    ("MUL", ("r", "r")): (b"\x50", [_code(0, 2), _code(0, 0)]),
    ("MULH", ("r", "r")): (b"\x70\x90", [_code(1, 2), _code(1, 0)]),
    ("NEG", ("r",)): (b"\x70\x44", [_code(1, 0)]),
    ("NOP", ()): (b"\x01", []),
    ("NOT", ("r",)): (b"\x70\x40", [_code(1, 0)]),
    ("OR", ("r", "r")): (b"\x30", [_code(0, 2), _code(0, 0)]),
    ("OUT", ("r",)): (b"\xf8", [_code(0, 0)]),
    ("OUTDE", ()): (b"\x07", []),
    ("OUTM", ()): (b"\x06", []),
    ("POP", ("r",)): (b"\xf4", [_code(0, 0)]),
    ("POPW", ("DE",)): (b"\x70\x3b", [None]),
    ("POPW", ("HL",)): (b"\x70\x3a", [None]),
    ("PUSH", ("r",)): (b"\xf0", [_code(0, 0)]),
    ("PUSHW", ("DE",)): (b"\x70\x39", [None]),
    ("PUSHW", ("HL",)): (b"\x70\x38", [None]),
    ("RET", ()): (b"\x08", []),
    ("ROL", ("r",)): (b"\x70\x48", [_code(1, 0)]),
    ("ROR", ("r",)): (b"\x70\x4c", [_code(1, 0)]),
    ("SBB", ("r", "r")): (b"\xb0", [_code(0, 2), _code(0, 0)]),
    ("SHL", ("r",)): (b"\x60", [_code(0, 0)]),
    ("SHR", ("r",)): (b"\x64", [_code(0, 0)]),
    ("STC", ("[HL]", "r")): (b"\x70\x80", [None, _code(1, 0)]),
    ("STM", ("[HL]", "r")): (b"\x70\xb4", [None, _code(1, 0)]),
    ("STMW", ("[DE]", "HL")): (b"\x70\xbb", [None, None]),
    ("STMW", ("[HL]", "DE")): (b"\x70\xba", [None, None]),
    ("STW", ("[DE]", "HL")): (b"\x70\x3d", [None, None]),
    ("STW", ("[HL]", "DE")): (b"\x70\x3c", [None, None]),
    ("STX", ("[HL+-i8]", "r")): (b"\x70\x54", [("byte",), _code(1, 0)]),
    ("SUB", ("HL", "DE")): (b"\x70\x61", [None, None]),
    ("SUB", ("r", "r")): (b"\x90", [_code(0, 2), _code(0, 0)]),
    ("SUBI", ("r", "i8")): (b"\xd8", [_code(0, 0), ("byte",)]),
    ("TST", ("r",)): (b"\x68", [_code(0, 0)]),
    ("XCHG", ("HL", "DE")): (b"\x70\x62", [None, None]),
    ("XOR", ("r", "r")): (b"\x40", [_code(0, 2), _code(0, 0)]),
}

def _encode(name, shape, values):

    codes, places = ENC[(name, shape)]
    out = bytearray(codes)
    tail = bytearray()
    for place, v in zip(places, values):
        if place is None or v is None:
            continue
        if place[0] == "code":
            out[place[1]] |= v << place[2]
        elif place[0] == "byte":
            tail.append(v & 0xFF)
        else:
            tail += (v & 0xFFFF).to_bytes(2, "little")
    return bytes(out) + bytes(tail)

def _codepoint_of(codes):

    return codes[0] if len(codes) == 1 else 0x7000 | codes[1]

def _check_encodings():

    declared = {(n, s) for n, shapes in isa_forms.FORMS.items() for s in shapes}
    problems = [f"no encoding for accepted shape {key}"
                for key in sorted(declared - set(ENC))]
    problems += [f"encoding for {key} names no shape FORMS accepts"
                 for key in sorted(set(ENC) - declared)]
    info = isa_forms.shape_info()
    for name, shape in sorted(declared & set(ENC)):
        codes, places = ENC[(name, shape)]
        size, cps = info[(name, shape)]
        if len(places) != len(shape):
            problems.append(f"{name} {shape}: {len(places)} placements for "
                            f"{len(shape)} operands")
            continue
        for kind, place in zip(shape, places):
            got = place[0] if place else None
            if got not in (None, "code", "byte", "word"):
                problems.append(f"{name} {shape}: unknown placement {place}")
            elif PLACE_KIND[kind] != got:
                problems.append(f"{name} {shape}: kind {kind!r} placed as {got}, "
                                "expected "
                                + repr(PLACE_KIND[kind]))
            elif got == "code" and (place[1] >= len(codes) or place[2] not in (0, 2)):
                problems.append(f"{name} {shape}: code placement {place} is outside "
                                "the code bytes")
        width = sum({"byte": 1, "word": 2}.get(p[0], 0) for p in places if p)
        if len(codes) + width != size:
            problems.append(f"{name} {shape}: emits {len(codes) + width} bytes, the "
                            f"decoder reads {size}")
        fields = [i for i, k in enumerate(shape) if k == "r"]
        emitted = set()
        for combo in range(4 ** len(fields)):
            rest, values = combo, []
            for i in range(len(shape)):
                if i in fields:
                    values.append(rest % 4)
                    rest //= 4
                else:
                    values.append(0)
            body = _encode(name, shape, values)
            emitted.add(_codepoint_of(body[:len(codes)]))
        if emitted != set(cps):
            problems.append(f"{name} {shape}: emits code points "
                            f"{sorted(hex(c) for c in emitted)} but isa_table assigns "
                            f"{sorted(hex(c) for c in cps)}")
    return problems

_ENCODING_PROBLEMS = _check_encodings()
if _ENCODING_PROBLEMS:
    raise RuntimeError("golden_sim's encoding table disagrees with the decode table: "
                       + "; ".join(_ENCODING_PROBLEMS[:6]))

def asm(src: str) -> bytes:

    if not isinstance(src, str):
        raise AssemblyError(f"asm() needs source text, got a {type(src).__name__}")
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

        shape = isa_forms.match(name, args)
        if shape is None:
            _reason, pos, kind = isa_forms.blame(name, args)
            detail = None
            if kind is not None:
                detail = _diagnose(kind, args[pos], name, labels)
            raise AssemblyError(isa_forms.refusal(name, args, text, detail), lineno)
        try:
            values = [_operand_value(k, a, name, labels, strict)
                      for k, a in zip(shape, args)]
        except _Operand as e:
            raise AssemblyError(isa_forms.refusal(name, args, text, str(e)),
                                lineno) from None
        return _encode(name, shape, values)

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
            if it[1] in labels:
                raise AssemblyError(f"symbol {it[1]!r} defined twice (already a label) "
                                    f"in {it[3]!r}", it[2])
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