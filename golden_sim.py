"""NCP-8 reference simulator and two-pass assembler.

This module is the semantic reference for the ISA: pure Python integers, no
floating point, no randomness, fully deterministic. It defines the observable
state (registers r0-r3, pointers HL/DE, stack pointer, PC, carry/zero flags,
input/output streams, tick counter, status) and the exact per-tick transition.

Status codes: 0 = running, 1 = halted, 2 = tick budget exhausted, 3 = error.
An error tick has no side effects: no register, memory, flag or PC update.

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

DATA_SIZE = 4096
CODE_SIZE = 4096


class MachineError(Exception):
    pass


class NCP8:
    def __init__(self, code, data=None, inputs=b"", tick_budget=200_000):
        assert len(code) <= CODE_SIZE
        self.code = bytes(code)
        self.data = bytearray(DATA_SIZE)
        if data:
            assert len(data) <= DATA_SIZE
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
        self.tick = 0
        self.tb = tick_budget
        self.status = "RUNNING"
        self.trace: list[str] = []

    def _mem(self, addr):
        if not 0 <= addr < DATA_SIZE:
            raise MachineError(f"DATA out of range: {addr}")

    def _fetch(self, n):
        if self.PC + n > len(self.code):
            raise MachineError(f"PC out of range: {self.PC}")
        b = self.code[self.PC: self.PC + n]
        self.PC += n
        return b

    def _push(self, byte):
        if self.SP <= 0:
            raise MachineError("stack underflow")
        self.SP -= 1
        self.data[self.SP] = byte & 0xFF

    def _pop(self):
        if self.SP >= DATA_SIZE:
            raise MachineError("stack overflow")
        v = self.data[self.SP]
        self.SP += 1
        return v

    def step(self):


        pc0 = self.PC
        try:
            self._step_inner()
        except MachineError:
            self.PC = pc0
            raise

    def _step_inner(self):
        if self.status != "RUNNING":
            raise MachineError("machine already stopped")
        if self.tick >= self.tb:
            self.status = "OVERRUN"
            return
        pc0 = self.PC
        (op,) = self._fetch(1)
        r = op & 3
        m = "???"

        if op <= 0x1F:
            if op == 0x00: m = "HALT"; self.status = "HALT"
            elif op == 0x01: m = "NOP"
            elif op == 0x02: self.HL = (self.HL + 1) & 0xFFFF; m = "INC HL"
            elif op == 0x03: self.HL = (self.HL - 1) & 0xFFFF; m = "DEC HL"
            elif op == 0x04: self.DE = (self.DE + 1) & 0xFFFF; m = "INC DE"
            elif op == 0x05: self.C = 0; m = "CLC"
            elif op == 0x06:
                self._mem(self.HL); self.out.append(self.data[self.HL])
                self.HL = (self.HL + 1) & 0xFFFF; m = "OUTM"
            elif op == 0x07:
                self._mem(self.DE); self.out.append(self.data[self.DE])
                self.DE = (self.DE + 1) & 0xFFFF; m = "OUTDE"
            elif op == 0x08:
                hi = self._pop(); lo = self._pop(); self.PC = hi << 8 | lo; m = "RET"
            elif 0x09 <= op <= 0x0D:
                lo, hi = self._fetch(2); t = hi << 8 | lo
                if op == 0x09: self.PC = t; m = "JMP"
                elif op == 0x0A: m = "JZ";   self.PC = t if self.Z else self.PC
                elif op == 0x0B: m = "JNZ";  self.PC = t if not self.Z else self.PC
                elif op == 0x0C: m = "JC";   self.PC = t if self.C else self.PC
                elif op == 0x0D: m = "JNC";  self.PC = t if not self.C else self.PC
            elif op == 0x0E:
                lo, hi = self._fetch(2); t = hi << 8 | lo; ret = self.PC
                self._push(ret & 0xFF); self._push(ret >> 8); self.PC = t; m = "CALL"
            elif op == 0x0F:
                lo, hi = self._fetch(2); self.HL = hi << 8 | lo; m = f"LDI HL, {self.HL}"
            elif op == 0x10:
                lo, hi = self._fetch(2); self.DE = hi << 8 | lo; m = f"LDI DE, {self.DE}"
            elif op == 0x11:
                (s,) = self._fetch(1); self.HL = (self.HL + self.r[s & 3]) & 0xFFFF; m = f"ADDI HL, r{s & 3}"
            elif op == 0x12:
                (s,) = self._fetch(1); self.DE = (self.DE + self.r[s & 3]) & 0xFFFF; m = f"ADDI DE, r{s & 3}"
            elif op == 0x13:
                self.PC = self.HL; m = "JPHL"
            elif op & 0xFC == 0x14:
                self.r[r] = pc0 & 255; m = f"GETPC r{r}"
            elif op & 0xFC == 0x18:
                self.r[r] = self.SP & 255; m = f"GETSP r{r}"
            elif op & 0xFC == 0x1C:
                self.r[r] = self.Z | (self.C << 1); m = f"GETF r{r}"
            else:
                raise MachineError(f"undefined opcode {op:#04x} @ {pc0:#04x}")
        elif 0x80 <= op <= 0xCF:
            s0, s1 = (op >> 2) & 3, op & 3
            base = op & 0xF0
            a, b = self.r[s0], self.r[s1]
            if base == 0x80:
                t = a + b; self.C = t >> 8; self.r[s0] = t & 0xFF; self.Z = int((t & 0xFF) == 0); m = f"ADD r{s0}, r{s1}"
            elif base == 0x90:
                self.C = int(a < b); v = (a - b) & 0xFF; self.r[s0] = v; self.Z = int(v == 0); m = f"SUB r{s0}, r{s1}"
            elif base == 0xA0:
                t = a + b + self.C; self.C = t >> 8; self.r[s0] = t & 0xFF; self.Z = int((t & 0xFF) == 0); m = f"ADC r{s0}, r{s1}"
            elif base == 0xB0:
                t = a - b - self.C; self.C = int(t < 0); v = t & 0xFF; self.r[s0] = v; self.Z = int(v == 0); m = f"SBB r{s0}, r{s1}"
            elif base == 0xC0:
                self.r[s0] = b; m = f"MOV r{s0}, r{s1}"
            else:
                raise MachineError(f"undefined opcode {op:#04x} @ {pc0:#04x}")
        elif 0x20 <= op <= 0x5F:
            s0, s1 = (op >> 2) & 3, op & 3
            base = op & 0xF0
            a, b = self.r[s0], self.r[s1]
            if base == 0x20:
                v = a & b; self.r[s0] = v; self.Z = int(v == 0); m = f"AND r{s0}, r{s1}"
            elif base == 0x30:
                v = a | b; self.r[s0] = v; self.Z = int(v == 0); m = f"OR r{s0}, r{s1}"
            elif base == 0x40:
                v = a ^ b; self.r[s0] = v; self.Z = int(v == 0); m = f"XOR r{s0}, r{s1}"
            elif base == 0x50:
                t = a * b; self.C = int(t > 255); v = t & 0xFF
                self.r[s0] = v; self.Z = int(v == 0); m = f"MUL r{s0}, r{s1}"
            else:
                raise MachineError(f"undefined opcode {op:#04x} @ {pc0:#04x}")
        elif op == 0x70:
            (sub,) = self._fetch(1)
            rs = sub & 3
            fam = sub & 0xF0
            if fam in (0x00, 0x10, 0x20):
                s0, s1 = (sub >> 2) & 3, sub & 3
                a, b = self.r[s0], self.r[s1]
                if fam == 0x00:
                    if b == 0:
                        raise MachineError(f"divide by zero DIV r{s0}, r{s1} @ {pc0:#04x}")
                    v = a // b; self.r[s0] = v; self.Z = int(v == 0); m = f"DIV r{s0}, r{s1}"
                elif fam == 0x10:
                    if b == 0:
                        raise MachineError(f"divide by zero MOD r{s0}, r{s1} @ {pc0:#04x}")
                    v = a % b; self.r[s0] = v; self.Z = int(v == 0); m = f"MOD r{s0}, r{s1}"
                else:
                    self.Z = int(a == b); self.C = int(a < b); m = f"CMP r{s0}, r{s1}"
            elif sub & 0xFC in (0x40, 0x44, 0x48, 0x4C):
                if sub & 0xFC == 0x40:
                    v = (~self.r[rs]) & 0xFF; self.r[rs] = v
                    self.Z = int(v == 0); m = f"NOT r{rs}"
                elif sub & 0xFC == 0x44:
                    t = (-self.r[rs]) & 0xFF; self.C = int(self.r[rs] != 0)
                    self.r[rs] = t; self.Z = int(t == 0); m = f"NEG r{rs}"
                elif sub & 0xFC == 0x48:
                    v = self.r[rs]; self.C, self.r[rs] = v >> 7, ((v << 1) | self.C) & 0xFF
                    self.Z = int(self.r[rs] == 0); m = f"ROL r{rs}"
                else:
                    v = self.r[rs]; self.C, self.r[rs] = v & 1, ((v >> 1) | (self.C << 7)) & 0xFF
                    self.Z = int(self.r[rs] == 0); m = f"ROR r{rs}"
            elif sub == 0x60:
                t = self.HL + self.DE; self.C = t >> 16; self.HL = t & 0xFFFF; m = "ADD HL, DE"
            elif sub == 0x61:
                self.C = int(self.HL < self.DE); self.HL = (self.HL - self.DE) & 0xFFFF; m = "SUB HL, DE"
            elif sub == 0x62:
                self.HL, self.DE = self.DE, self.HL; m = "XCHG HL, DE"
            elif sub & 0xFC in (0x80, 0x84):

                WLO, WHI = 0x0F20, 0x0F21
                wl = self.code[WLO] if WLO < len(self.code) else 0
                wh = self.code[WHI] if WHI < len(self.code) else 0
                if sub & 0xFC == 0x84:
                    if not 0 <= self.HL < len(self.code):
                        raise MachineError(f"LDC out of range {self.HL} @ {pc0:#04x}")
                    self.r[rs] = self.code[self.HL]; m = f"LDC r{rs}, [HL]"
                else:
                    if not 0 <= self.HL < len(self.code):
                        raise MachineError(f"STC out of range {self.HL} @ {pc0:#04x}")
                    if not (wl <= self.HL < wh):
                        raise MachineError(
                            f"STC outside window {self.HL:#x} not in [{wl:#x},{wh:#x}) @ {pc0:#04x}")
                    b = bytearray(self.code); b[self.HL] = self.r[rs]; self.code = bytes(b)
                    m = f"STC [HL], r{rs}"
            elif sub == 0x70:
                (k,) = self._fetch(1)
                if k >= 16:
                    raise MachineError(f"EXT k out of range {k} @ {pc0:#04x}")
                a = 0x0F00 + 2 * k
                tgt = 0
                if a + 1 < len(self.code):
                    tgt = (self.code[a + 1] << 8) | self.code[a]
                if tgt == 0:
                    raise MachineError(f"EXT handler {k} unregistered @ {pc0:#04x}")
                self._push(self.PC & 0xFF); self._push((self.PC >> 8) & 0xFF)
                self.PC = tgt; m = f"EXT {k}"
            else:
                raise MachineError(f"reserved subcode {sub:#04x} (ESC)  @ {pc0:#04x}")
        elif 0xD0 <= op <= 0xDF:
            (i,) = self._fetch(1)
            t = self.r[r] + i + (self.C if op >= 0xDC else 0)
            if op & 0xFC == 0xD0:
                self.r[r] = i; m = f"LDI r{r}, {i}"
            elif op & 0xFC == 0xD4:
                self.C = t >> 8; self.r[r] = t & 0xFF; self.Z = int((t & 0xFF) == 0); m = f"ADDI r{r}, {i}"
            elif op & 0xFC == 0xD8:
                a = self.r[r]; self.C = int(a < i); v = (a - i) & 0xFF; self.r[r] = v; self.Z = int(v == 0); m = f"SUBI r{r}, {i}"
            elif op & 0xFC == 0xDC:
                self.C = t >> 8; self.r[r] = t & 0xFF; self.Z = int((t & 0xFF) == 0); m = f"ADCI r{r}, {i}"
        elif (op & 0xFC) in (0x60, 0x64, 0x68, 0x6C) or 0xE0 <= op <= 0xFF:
            fam = op & 0xFC
            if fam == 0xE0: self._mem(self.HL); self.r[r] = self.data[self.HL]; m = f"MOV r{r}, [HL]"
            elif fam == 0xE4: self._mem(self.HL); self.data[self.HL] = self.r[r]; m = f"MOV [HL], r{r}"
            elif fam == 0xE8: self._mem(self.DE); self.r[r] = self.data[self.DE]; m = f"MOV r{r}, [DE]"
            elif fam == 0xEC: self._mem(self.DE); self.data[self.DE] = self.r[r]; m = f"MOV [DE], r{r}"
            elif fam == 0xF0: self._push(self.r[r]); m = f"PUSH r{r}"
            elif fam == 0xF4: self.r[r] = self._pop(); m = f"POP r{r}"
            elif fam == 0xF8: self.out.append(self.r[r]); m = f"OUT r{r}"
            elif fam == 0xFC:
                if self.ipos < len(self.inputs):
                    self.r[r] = self.inputs[self.ipos]; self.ipos += 1
                else:
                    self.r[r] = 0; self.C = 1
                m = f"IN r{r}"
            elif fam == 0x60:
                self.C = self.r[r] >> 7; v = (self.r[r] << 1) & 0xFF; self.r[r] = v; self.Z = int(v == 0); m = f"SHL r{r}"
            elif fam == 0x64:
                self.C = self.r[r] & 1; v = self.r[r] >> 1; self.r[r] = v; self.Z = int(v == 0); m = f"SHR r{r}"
            elif fam == 0x68:
                self.Z = int(self.r[r] == 0); m = f"TST r{r}"
            elif fam == 0x6C:
                lo, hi = self._fetch(2); t = hi << 8 | lo
                self.r[r] = (self.r[r] - 1) & 0xFF
                if self.r[r] != 0: self.PC = t
                m = f"DJNZ r{r}"
        else:
            raise MachineError(f"undefined opcode {op:#04x} @ {pc0:#04x}")

        self.trace.append(
            f"{self.tick:6d} {pc0:04X} {m:18s} | r=[{self.r[0]:3d},{self.r[1]:3d},{self.r[2]:3d},{self.r[3]:3d}]"
            f" DE={self.DE:04X} HL={self.HL:04X} SP={self.SP:04X} C={self.C} Z={self.Z}"
        )
        self.tick += 1

    def run(self):
        while self.status == "RUNNING":
            self.step()
        return self.out

    def snapshot(self):
        return dict(r=list(self.r), HL=self.HL, DE=self.DE, SP=self.SP, PC=self.PC,
                    C=self.C, Z=self.Z, tick=self.tick, status=self.status)




def _r(s):
    assert s in ("r0", "r1", "r2", "r3"), s
    return int(s[1])


def asm(src: str) -> bytes:
    pat = re.compile(r"^(\w+):$")
    items = []
    for raw in src.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        mm = pat.match(line)
        if mm:
            items.append(("label", mm.group(1)))
            continue
        name, _, args = line.partition(" ")
        args = [a.strip() for a in args.split(",")] if args.strip() else []
        items.append(("inst", name.strip(), args))

    def enc(name, args, labels):
        def _addr(x):

            if x in labels:
                return labels[x]
            try:
                return int(str(x), 0)
            except ValueError:
                return 0

        simple = {"HALT": 0x00, "NOP": 0x01, "INC HL": 0x02, "DEC HL": 0x03,
                  "INC DE": 0x04, "CLC": 0x05, "OUTM": 0x06, "OUTDE": 0x07, "RET": 0x08,
                  "LDI HL": 0x0F, "LDI DE": 0x10, "ADDI HL": 0x11, "ADDI DE": 0x12}
        if args and f"{name} {args[0]}" in simple:
            base = simple[f"{name} {args[0]}"]
            rest = args[1:]
            if base in (0x0F, 0x10):

                v = _addr(rest[0])
                return bytes([base, v & 0xFF, v >> 8])
            if base in (0x11, 0x12):
                return bytes([base, _r(rest[0])])
            return bytes([base])
        if name in simple:
            return bytes([simple[name]])
        if name == "JPHL":
            return bytes([0x13])
        if name in ("GETPC", "GETSP", "GETF"):
            base = {"GETPC": 0x14, "GETSP": 0x18, "GETF": 0x1C}[name]
            return bytes([base | _r(args[0])])
        jump = {"JMP": 0x09, "JZ": 0x0A, "JNZ": 0x0B, "JC": 0x0C, "JNC": 0x0D, "CALL": 0x0E}
        if name in jump:

            a = _addr(args[0])
            return bytes([jump[name], a & 0xFF, a >> 8])
        if name == "DJNZ":
            a = _addr(args[1])
            return bytes([0x6C | _r(args[0]), a & 0xFF, a >> 8])
        if name == "LDI":
            return bytes([0xD0 | _r(args[0]), int(str(args[1]), 0)])
        if name == "ADDI":
            return bytes([0xD4 | _r(args[0]), int(str(args[1]), 0)])
        if name == "SUBI":
            return bytes([0xD8 | _r(args[0]), int(str(args[1]), 0)])
        if name == "ADCI":
            return bytes([0xDC | _r(args[0]), int(str(args[1]), 0)])
        if name in ("ADD", "SUB") and args and args[0] == "HL":
            return bytes([0x70, {"ADD": 0x60, "SUB": 0x61}[name]])
        if name == "XCHG":
            return bytes([0x70, 0x62])
        if name == "STC":
            return bytes([0x70, 0x80 | _r(args[1])])
        if name == "LDC":
            return bytes([0x70, 0x84 | _r(args[0])])
        if name == "EXT":
            return bytes([0x70, 0x70, int(str(args[0]), 0) & 0xFF])
        rr2 = {"AND": 0x20, "OR": 0x30, "XOR": 0x40, "MUL": 0x50}
        if name in rr2:
            return bytes([rr2[name] | (_r(args[0]) << 2) | _r(args[1])])
        rr3 = {"DIV": 0x00, "MOD": 0x10, "CMP": 0x20}
        if name in rr3:
            return bytes([0x70, rr3[name] | (_r(args[0]) << 2) | _r(args[1])])
        un2 = {"NOT": 0x40, "NEG": 0x44, "ROL": 0x48, "ROR": 0x4C}
        if name in un2:
            return bytes([0x70, un2[name] | _r(args[0])])
        rr = {"ADD": 0x80, "SUB": 0x90, "ADC": 0xA0, "SBB": 0xB0, "MOV": 0xC0}
        if name == "MOV":
            mem = {"[HL]": 0xE0, "[DE]": 0xE8}
            if args[1] in ("[HL]", "[DE]"):
                return bytes([mem[args[1]] | _r(args[0])])
            if args[0] in ("[HL]", "[DE]"):
                return bytes([(mem[args[0]] + 4) | _r(args[1])])
            return bytes([0xC0 | (_r(args[0]) << 2) | _r(args[1])])
        if name in rr:
            return bytes([rr[name] | (_r(args[0]) << 2) | _r(args[1])])
        unary = {"PUSH": 0xF0, "POP": 0xF4, "OUT": 0xF8, "IN": 0xFC,
                 "SHL": 0x60, "SHR": 0x64, "TST": 0x68}
        if name in unary:
            return bytes([unary[name] | _r(args[0])])
        raise MachineError(f"unknowninstruction: {name} {args}")

    labels, addr = {}, 0
    for it in items:
        if it[0] == "label":
            labels[it[1]] = addr
        else:
            addr += len(enc(it[1], it[2], labels))
    out = bytearray()
    for it in items:
        if it[0] != "label":
            out += enc(it[1], it[2], labels)
    assert len(out) <= CODE_SIZE
    return bytes(out)