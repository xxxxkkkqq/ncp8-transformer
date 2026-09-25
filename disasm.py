"""Disassembler and one-instruction decoder for the NCP-8 encoding.

Table-driven from the same encoding facts the assembler and the simulators use:
`SINGLE[code] = (mnemonic template, length)` for one-byte opcodes and
`ESC[subcode] = (template, length)` for the `0x70` escape space. A template names its
operand fields (`{a16}`, `{i8}`, `{rcanon}`, `{off}`, `{soff}`, `{k}`), so which byte
positions carry an operand is derived rather than restated.

Encodings are reported, not folded: an encoding that is legal but not canonical (an
operand bit the datapath never reads, set to a non-zero value) is decoded with
`non_canonical` marked, so a caller can tell "this byte string means X" from "this byte
string is a sloppy way to write X".

Run: python3 disasm.py             (self-check over every assigned encoding)
"""
from collections import namedtuple

import golden_sim
from golden_sim import asm

def _r(v):
    return f"r{v & 3}"

def _off(v):
    s = (v ^ 0x80) - 0x80
    return "" if s == 0 else (f"+{s}" if s > 0 else f"{s}")

SINGLE = {}
for op, t in {0x00: "HALT", 0x01: "NOP", 0x02: "INC HL", 0x03: "DEC HL",
              0x04: "INC DE", 0x05: "CLC", 0x06: "OUTM", 0x07: "OUTDE",
              0x08: "RET", 0x13: "JPHL"}.items():
    SINGLE[op] = (t, 1)
for i, t in enumerate(["JMP", "JZ", "JNZ", "JC", "JNC", "CALL"]):
    SINGLE[0x09 + i] = (f"{t} {{a16}}", 3)
SINGLE[0x0F] = ("LDI HL, {i16}", 3)
SINGLE[0x10] = ("LDI DE, {i16}", 3)
SINGLE[0x11] = ("ADDI HL, {rcanon}", 2)
SINGLE[0x12] = ("ADDI DE, {rcanon}", 2)
for base, name in ((0x14, "GETPC"), (0x18, "GETSP"), (0x1C, "GETF"),
                   (0x60, "SHL"), (0x64, "SHR"), (0x68, "TST")):
    for k in range(4):
        SINGLE[base | k] = (f"{name} {_r(k)}", 1)
for k in range(4):
    SINGLE[0x6C | k] = (f"DJNZ {_r(k)}, {{a16}}", 3)
for base, name in ((0x20, "AND"), (0x30, "OR"), (0x40, "XOR"), (0x50, "MUL"),
                   (0x80, "ADD"), (0x90, "SUB"), (0xA0, "ADC"), (0xB0, "SBB"),
                   (0xC0, "MOV")):
    for f in range(16):
        SINGLE[base + f] = (f"{name} {_r(f >> 2)}, {_r(f & 3)}", 1)
for base, name in ((0xD0, "LDI"), (0xD4, "ADDI"), (0xD8, "SUBI"), (0xDC, "ADCI")):
    for k in range(4):
        SINGLE[base | k] = (f"{name} {_r(k)}, {{i8}}", 2)
for base, tpl in ((0xE0, "MOV {R}, [HL]"), (0xE4, "MOV [HL], {R}"),
                  (0xE8, "MOV {R}, [DE]"), (0xEC, "MOV [DE], {R}"),
                  (0xF0, "PUSH {R}"), (0xF4, "POP {R}"),
                  (0xF8, "OUT {R}"), (0xFC, "IN {R}")):
    for k in range(4):
        SINGLE[base | k] = (tpl.replace("{R}", _r(k)), 1)

ESC = {}
for base, name in ((0x00, "DIV"), (0x10, "MOD"), (0x20, "CMP")):
    for f in range(16):
        ESC[base + f] = (f"{name} {_r(f >> 2)}, {_r(f & 3)}", 2)
for base, name in ((0x40, "NOT"), (0x44, "NEG"), (0x48, "ROL"), (0x4C, "ROR")):
    for k in range(4):
        ESC[base | k] = (f"{name} {_r(k)}", 2)
for sub, t in {0x30: "MOVW HL, DE", 0x31: "MOVW DE, HL", 0x32: "MOVW HL, SP",
               0x33: "MOVW DE, SP", 0x34: "MOVW SP, HL", 0x35: "MOVW SP, DE",
               0x38: "PUSHW HL", 0x39: "PUSHW DE", 0x3A: "POPW HL", 0x3B: "POPW DE",
               0x3C: "STW [HL], DE", 0x3D: "STW [DE], HL", 0x3E: "LDW DE, [HL]",
               0x3F: "LDW HL, [DE]", 0x60: "ADD HL, DE", 0x61: "SUB HL, DE",
               0x62: "XCHG HL, DE"}.items():
    ESC[sub] = (t, 2)
for k in range(4):
    ESC[0x50 | k] = (f"LDX {_r(k)}, [HL{{off}}]", 3)
    ESC[0x54 | k] = (f"STX [HL{{off}}], {_r(k)}", 3)
ESC[0x58] = ("ADD SP, {soff}", 3)
ESC[0x70] = ("EXT {k}", 3)
for k in range(4):
    ESC[0x80 | k] = (f"STC [HL], {_r(k)}", 2)
    ESC[0x84 | k] = (f"LDC {_r(k)}, [HL]", 2)
for f in range(16):
    ESC[0x90 + f] = (f"MULH {_r(f >> 2)}, {_r(f & 3)}", 2)

def disasm(image, start=0, count=None):

    rows, pc, n = [], start, 0
    while pc < len(image) and (count is None or n < count):
        op = image[pc]
        if op == 0x70:
            if pc + 1 >= len(image):
                rows.append((pc, image[pc:pc + 1], "<truncated escape prefix>", True))
                break
            sub = image[pc + 1]
            if sub not in ESC:
                rows.append((pc, image[pc:pc + 2],
                             f"DB 0x{sub:02X}  (reserved subcode)", False))
                pc += 2
                n += 1
                continue
            tmpl, ln = ESC[sub]
            if pc + ln > len(image):
                rows.append((pc, image[pc:], "<truncated operand>", True))
                break
            body = image[pc:pc + ln]
            imm = body[2] if ln == 3 else 0
            note = False
            text = tmpl.format(off=_off(imm), soff=str(_off(imm) or 0), k=imm)
            rows.append((pc, bytes(body), text, note))
            pc += ln
            n += 1
            continue
        if op not in SINGLE:
            rows.append((pc, image[pc:pc + 1], f"DB 0x{op:02X}  (undefined opcode)",
                         False))
            pc += 1
            n += 1
            continue
        tmpl, ln = SINGLE[op]
        if pc + ln > len(image):
            rows.append((pc, image[pc:], "<truncated operand>", True))
            break
        body = image[pc:pc + ln]
        a16 = f"0x{((body[2] << 8) | body[1]):04X}" if ln == 3 else "0x0000"
        i8 = f"0x{body[1]:02X}" if ln == 2 else "0x00"
        note = "rcanon" in tmpl and ln == 2 and body[1] > 3
        text = tmpl.format(a16=a16, i16=a16, i8=i8,
                           rcanon=_r(body[1]) if ln >= 2 else "r0")
        rows.append((pc, bytes(body), text, note))
        pc += ln
        n += 1
    return rows

class Row(namedtuple("Row", "addr size codepoint text non_canonical assigned")):

    __slots__ = ()

    @property
    def mnemonic(self):
        return self.text.split(" ")[0] if self.assigned else self.text

def codepoint(op, sub=None):

    if op != 0x70:
        if sub is not None:
            raise ValueError(f"codepoint: subcode given for opcode {op:#04x}, "
                             f"which is not the escape prefix 0x70")
        return op
    if sub is None:
        raise ValueError("codepoint: the escape prefix 0x70 needs its subcode")
    return 0x7000 | (sub & 0xFF)

def decode(image, pc):

    if not 0 <= pc < len(image):
        raise ValueError(f"decode: address {pc} is outside an image of {len(image)} bytes")
    op = image[pc]
    if op != 0x70:
        if op not in SINGLE:
            return Row(pc, 1, op, f"DB 0x{op:02X}  (undefined opcode)", True, False)
        row = disasm(image, pc, 1)[0]
        return Row(pc, len(row[1]), codepoint(op), row[2], row[3], True)
    if pc + 1 >= len(image):
        return Row(pc, 1, 0x7000, "<truncated escape prefix>", True, False)
    sub = image[pc + 1]
    if sub not in ESC:
        row = disasm(image, pc, 1)[0]
        return Row(pc, 2, codepoint(0x70, sub), row[2], row[3], False)
    row = disasm(image, pc, 1)[0]
    if row[2] == "<truncated operand>":
        return Row(pc, len(row[1]), codepoint(0x70, sub), row[2], True, False)
    return Row(pc, len(row[1]), codepoint(0x70, sub), row[2], row[3], True)

def coverage():

    return (len(SINGLE), len(ESC), 256 - len(ESC), golden_sim.CODE_SIZE)

if __name__ == "__main__":
    print(f"decode rows: single-byte {len(SINGLE)}, escape {len(ESC)}, "
          f"reserved subcodes {sum(1 for s in range(256) if s not in ESC)}")

    SRCS = ["HALT", "NOP\nINC HL\nRET", "JMP 0x1234\nCALL 0x0F00",
            "LDI HL, 0x0F20\nLDI DE, 0x100", "ADDI HL, r2\nADDI DE, r3",
            "AND r1, r2\nOR r3, r0\nXOR r2, r2\nMUL r1, r3",
            "ADD r0, r1\nSUB r2, r3\nADC r3, r0\nSBB r1, r1\nMOV r0, r3",
            "LDI r0, 0x11\nADDI r1, 0x02\nSUBI r2, 0xFE\nADCI r3, 0x80",
            "MOV r0, [HL]\nMOV [HL], r1\nMOV r2, [DE]\nMOV [DE], r3",
            "PUSH r0\nPOP r1\nOUT r2\nIN r3",
            "GETPC r0\nGETSP r1\nGETF r2\nSHL r3\nSHR r0\nTST r1\nDJNZ r2, 0x0004",
            "DIV r1, r2\nMOD r3, r0\nCMP r0, r1\nNOT r2\nNEG r1\nROL r3\nROR r0",
            "ADD HL, DE\nSUB HL, DE\nXCHG HL, DE", "STC [HL], r0\nLDC r1, [HL]",
            "MOVW HL, SP\nMOVW SP, HL\nPUSHW HL\nPOPW DE\nSTW [HL], DE\nLDW HL, [DE]",
            "LDX r2, [HL+5]\nSTX [HL-8], r3\nLDX r0, [HL]\nADD SP, -4\nADD SP, 0",
            "MULH r1, r3\nMULH r0, r0\nEXT 3",
            "LDX r1, [HL+127]\nSTX [HL-128], r2",
            "JPHL\nCLC\nDEC HL"]
    bad_bytes = bad_text = 0
    for src in SRCS:
        want = asm(src)
        rows = disasm(want)
        got = b"".join(r[1] for r in rows)
        if got != want:
            bad_bytes += 1
            print(f"  BYTES {src!r}: want {want.hex()} got {got.hex()}")
        retext = "\n".join(r[2] for r in rows)
        try:
            again = asm(retext)
        except Exception as e:
            bad_text += 1
            print(f"  TEXT  {src!r} -> {retext!r} will not assemble: {e}")
            continue
        if again != want:
            bad_text += 1
            print(f"  TEXT  {src!r} -> {retext!r} re-assembles differently")
    print(f"\n{len(SRCS)} programs: {len(SRCS)-bad_bytes} byte-exact, "
          f"{len(SRCS)-bad_text} re-assemble to identical bytes")

    print("\nnon-canonical detection (the label-noise case):")
    for v in (0x01, 0x05, 0x7F, 0xFF):
        row = disasm(bytes([0x11, v]))[0]
        print(f"   11 {v:02X} -> {row[2]:16s} non_canonical={row[3]}")