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

import isa_table as ISA

DATA_SIZE = 4096
CODE_SIZE = 4096
OUT_CAP = 8192

ESCAPE_PREFIX = tl.constexpr(ISA.ESCAPE_PREFIX)
PREFIX_BYTES = tl.constexpr(ISA.PREFIX_BYTES)
_ESC_EOP_BASE = tl.constexpr(ISA.ESC_EOP_BASE)

def _escape_effective_opcodes():

    return {("ESC_" + name): tl.constexpr(v)
            for name, v in sorted(ISA.ESC_EOP_ID.items())}

globals().update(_escape_effective_opcodes())

def _fault_code_consts():

    return {("F_" + name): tl.constexpr(code)
            for name, code in sorted(ISA.CAUSE.items())}

globals().update(_fault_code_consts())

S_FAULT_REASON = tl.constexpr(14)
S_FAULT_ADDR = tl.constexpr(15)
S_MB = tl.constexpr(16)
S_SIGN = tl.constexpr(17)
S_OVFL = tl.constexpr(18)
STATE_ROWS = 19
STATE_ROWS_C = tl.constexpr(STATE_ROWS)

BANK_PAGES = 1
BANK_PAGES_C = tl.constexpr(BANK_PAGES)
BANK_OWN_C = tl.constexpr(0)

F_PACK_Z = tl.constexpr(ISA.FLAG_BITS_PACKED["Z"])
F_PACK_C = tl.constexpr(ISA.FLAG_BITS_PACKED["C"])
F_PACK_S = tl.constexpr(ISA.FLAG_BITS_PACKED["S"])
F_PACK_V = tl.constexpr(ISA.FLAG_BITS_PACKED["V"])

CFG_VEC_COUNT = tl.constexpr(ISA.VEC_COUNT)
CFG_LEN = ISA.VEC_COUNT + 2
CFG_WINLO = tl.constexpr(ISA.VEC_COUNT)
CFG_WINHI = tl.constexpr(ISA.VEC_COUNT + 1)
CFG_LEN_C = tl.constexpr(CFG_LEN)

def _cfg_row(cfg):

    row = list(cfg.vectors()) + [cfg.window()[0], cfg.window()[1]]
    return row

class DecodeTableMismatch(Exception):

    pass

def _check_decode_against_table():

    alu1, s01, s11, ln1 = ISA.single_rom()
    alu2, s02, s12, lx2 = ISA.escape_rom()
    bad = []
    for op in range(256):
        eop, d, s, ln = (int(v) for v in _dec_first.fn(op))
        if eop != op:
            bad.append(f"single {op:#04x}: effective opcode {eop:#x}, table {op:#x}")
        for field, got, want in (("s0", d, s01[op]), ("s1", s, s11[op]),
                                 ("length", ln, ln1[op])):
            if got != want:
                bad.append(f"single {op:#04x}: {field} {got}, table {want}")
    for sub in range(256):
        eop, d, s, lx = (int(v) for v in _dec_esc.fn(sub))
        row = ISA.ESCAPE.get(sub)
        want_eop = ISA.ESC_EOP_ID["BAD" if row is None else row["alu"]]
        if eop != want_eop:
            bad.append(f"escape {sub:#04x}: effective opcode {eop:#x}, table {want_eop:#x}")
        for field, got, want in (("s0", d, s02[sub]), ("s1", s, s12[sub]),
                                 ("extra bytes", lx, lx2[sub])):
            if got != want:
                bad.append(f"escape {sub:#04x}: {field} {got}, table {want}")
    for name, val in sorted(ISA.ESC_EOP_ID.items()):
        live = int(globals()["ESC_" + name])
        if live != val:
            bad.append(f"escape id {name}: kernel {live:#x}, table {val:#x}")
    if bad:
        raise DecodeTableMismatch(
            "the Triton decode disagrees with isa_table on "
            f"{len(bad)} field(s): " + "; ".join(bad[:8]))

@triton.jit
def _dec_first(op):

    eop = op
    d = 0
    s = 0
    ln = 1
    if op <= 0x08:
        pass
    elif op <= 0x10:
        ln = 3
    elif op <= 0x12:
        ln = 2
    elif op == 0x13:
        pass
    elif op <= 0x1F:
        d = op & 3; s = op & 3
    elif op <= 0x5F:
        d = (op >> 2) & 3; s = op & 3
    elif op <= 0x6B:
        d = op & 3; s = op & 3
    elif op <= 0x6F:
        d = op & 3; s = op & 3; ln = 3
    elif op == ESCAPE_PREFIX:
        ln = PREFIX_BYTES
    elif op <= 0x7F:
        pass
    elif op <= 0xCF:
        d = (op >> 2) & 3; s = op & 3
    elif op <= 0xDF:
        d = op & 3; s = op & 3; ln = 2
    else:
        d = op & 3; s = op & 3
    return eop, d, s, ln

def _power_of_two_or_die(name, size):

    if size <= 0 or size & (size - 1):
        raise ValueError(f"{name}={size} is not a positive power of two; the store "
                         f"masks would let a write leave its machine's own buffer")

for _name, _size in (("CODE_SIZE", CODE_SIZE), ("DATA_SIZE", DATA_SIZE),
                     ("OUT_CAP", OUT_CAP)):
    _power_of_two_or_die(_name, _size)
CODE_MASK = tl.constexpr(CODE_SIZE - 1)

def check_state(R, HL, DE, SP, C, Z, tick=0, PC=0, ipos=0, oplen=0, status=0,
                fault_reason=0, fault_addr=0, mb=0, s=0, v=0, where=""):

    if not 0 <= oplen <= OUT_CAP:
        raise ValueError(f"{where}state field oplen is {oplen}, outside [0, {OUT_CAP}]")
    bad = ISA.state_error({"r": list(R), "HL": HL, "DE": DE, "MB": mb, "PC": PC,
                           "SP": SP, "C": C, "Z": Z, "S": s, "V": v, "ipos": ipos, "tick": tick,
                           "status": status, "fault_reason": fault_reason,
                           "fault_addr": fault_addr},
                          where)
    if bad is not None:
        raise ValueError(bad)

STATE_ROW_OF = {"r": (0, 1, 2, 3), "HL": 4, "DE": 5, "PC": 6, "SP": 7, "C": 8, "Z": 9,
                "S": int(S_SIGN), "V": int(S_OVFL),
                "ipos": 10, "tick": 12, "status": 13, "fault_reason": 14,
                "fault_addr": 15, "MB": int(S_MB)}
STATE_ROW_OPLEN = 11
if set(STATE_ROW_OF) != set(ISA.STATE_FIELD_NAMES):
    raise ISA.DecodeTableError(
        f"the state row layout covers {sorted(STATE_ROW_OF)} while the state table "
        f"names {sorted(ISA.STATE_FIELD_NAMES)}: a field of one is missing from the "
        f"other, so a record taken on this path would be incomplete")
_ROWS_NAMED = sorted({row for at in STATE_ROW_OF.values()
                      for row in (at if isinstance(at, tuple) else (at,))}
                     | {STATE_ROW_OPLEN})
if _ROWS_NAMED != list(range(STATE_ROWS)):
    raise ISA.DecodeTableError(
        f"the state row layout names rows {_ROWS_NAMED} while a row of this machine is "
        f"{STATE_ROWS} wide: a cell of the state vector would be carried by no field, or "
        f"a field would live outside the vector")

def _row_state(row):

    out = {}
    for name, at in STATE_ROW_OF.items():
        out[name] = ([int(row[i]) for i in at] if isinstance(at, tuple) else int(row[at]))
    return out

def _write_row_state(row, st):

    for name, at in STATE_ROW_OF.items():
        if isinstance(at, tuple):
            for i, cell in zip(at, st[name]):
                row[i] = cell
        else:
            row[at] = st[name]

def _row_bytes(cells, name, size):

    values = [int(v) for v in cells]
    if len(values) != size:
        raise ValueError(f"{name} row has {len(values)} cells, this machine's region is "
                         f"{size} bytes")
    for i, v in enumerate(values):
        if not 0 <= v <= 255:
            raise ValueError(f"{name} cell {i} is {v}, which is not an 8-bit value, so "
                             f"the image cannot be recorded as bytes")
    return bytes(values)

_TRITON_RECORD_READERS = {
    **{name: (lambda m, name=name: m._record_state()[name])
       for name in ISA.STATE_FIELD_NAMES},

    "CODE": lambda m: m._record_code(),
    "DATA": lambda m: m._record_data(),
    "out": lambda m: m._record_out(),
    "inputs": lambda m: m._record_inputs(),
    "block": lambda m: m._record_block().as_dict(),
}

def _byte_image(values):

    return torch.tensor(list(values), dtype=torch.int32)

class _Row:

    __slots__ = ("b", "i")

    def __init__(self, batch, i):
        batch._row(i)
        self.b = batch
        self.i = i

    def _record_state(self):
        return _row_state(self.b.STATE[self.i])

    def _record_code(self):
        return _row_bytes(self.b.CODE[self.i], "CODE", CODE_SIZE)

    def _record_data(self):
        return _row_bytes(self.b.DATA[self.i], "DATA", DATA_SIZE)

    def _record_out(self):
        n = min(int(self.b.STATE[self.i, STATE_ROW_OPLEN].item()), self.b.out_cap)
        return bytes(int(v) for v in self.b.OUTBUF[self.i, :n].cpu().tolist())

    def _record_inputs(self):
        n = int(self.b.INLENS[self.i].item())
        return bytes(int(v) for v in self.b.INPUTS[self.i, :n].cpu().tolist())

    def _record_block(self):
        cfg = self.b.config
        lo, hi = cfg.window()
        return ISA.MachineConfig(codelen=int(self.b.CODELENS[self.i].item()),
                                 winlo=lo, winhi=hi, vec=cfg.vectors(),
                                 nbanks=self.b.nbanks, tdlim=self.b.tdlim,
                                 tickbudget=int(self.b.BUDGETS[self.i].item()),
                                 outcap=self.b.out_cap)

    def _record_bounds(self):
        return ISA.RecordBounds(where=f"machine {self.i}: ", out_cap=self.b.out_cap,
                                code_size=CODE_SIZE, data_size=DATA_SIZE,
                                inputs=self._record_inputs(),
                                block=self._record_block())

    def install(self, got):

        b, i = self.b, self.i
        b.set_state(i, oplen=len(got.out), **got.state)
        b.CODE[i].copy_(_byte_image(got.code).to(b.dev))
        b.DATA[i].copy_(_byte_image(got.data).to(b.dev))
        b.OUTBUF[i].zero_()
        if got.out:
            b.OUTBUF[i, : len(got.out)] = _byte_image(got.out).to(b.dev)

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
    elif sub >= 0x64 and sub <= 0x67:
        eop = ESC_JS + (sub - 0x64); d = 0; s = 0; lx = 1
    elif sub == 0x70:
        eop = ESC_EXT; d = 0; s = 0; lx = 1
    elif sub >= 0x80 and sub <= 0x87:
        eop = ESC_STC + ((sub >> 2) & 1); d = sub & 3; s = sub & 3; lx = 0
    elif sub >= 0x90 and sub <= 0x9F:
        eop = ESC_MULH; d = (sub >> 2) & 3; s = sub & 3; lx = 0
    elif sub >= 0xB0 and sub <= 0xB7:
        eop = ESC_LDM + ((sub >> 2) & 1); d = sub & 3; s = sub & 3; lx = 0
    elif sub >= 0xB8 and sub <= 0xBD:
        eop = ESC_LDMW_DE_HL + (sub - 0xB8); d = 0; s = 0; lx = 0
    else:
        eop = ESC_BAD; d = 0; s = 0; lx = 0
    return eop, d, s, lx

_check_decode_against_table()

@triton.jit
def _decode_refusal(eop):

    return tl.where(eop >= _ESC_EOP_BASE, F_BAD_SUBCODE, F_BAD_OPCODE)

@triton.jit
def _name_cause(errc, code):

    if errc == 0:
        errc = code
    return errc

@triton.jit
def _bank_refusal(MB, NB):

    if (MB >= NB) or (MB >= BANK_PAGES_C):
        return F_BANK_OOB
    if MB != BANK_OWN_C:
        return F_BANK_BUSY
    return 0

@triton.jit
def _tick(CODE, DATA, INPUTS, OUTBUF, S, CODELEN, INLEN, BD, DS, OC, CFG, NB):

    r0 = tl.load(S + 0); r1 = tl.load(S + 1); r2 = tl.load(S + 2); r3 = tl.load(S + 3)
    HL = tl.load(S + 4); DE = tl.load(S + 5); PC = tl.load(S + 6); SP = tl.load(S + 7)
    C = tl.load(S + 8); Z = tl.load(S + 9); IPO = tl.load(S + 10); OL = tl.load(S + 11)
    MB = tl.load(S + S_MB)
    SGN = tl.load(S + S_SIGN); OVF = tl.load(S + S_OVFL)

    nR0, nR1, nR2, nR3 = r0, r1, r2, r3
    nHL, nDE, nSP = HL, DE, SP
    nMB = MB
    nC, nZ, nIPO = C, Z, IPO
    nSGN, nOVF = SGN, OVF
    nOL = OL
    nPC = PC + 1
    NST = 0
    OVAL = 0; OEN = 0
    A1 = 0; V1 = 0; E1 = 0
    A2 = 0; V2 = 0; E2 = 0
    A3 = 0; V3 = 0; E3 = 0
    err = 0

    errc = 0

    eop = 0xFFFF
    ed = 0; es = 0; elen = 1

    if PC < CODELEN:
        op = tl.load(CODE + PC)
        eop, ed, es, elen = _dec_first(op)
        nPC = PC + elen
        if op == ESCAPE_PREFIX:

            if PC + elen > CODELEN:
                err = 1
                errc = _name_cause(errc, F_FETCH_OOB)
            else:
                sub = tl.load(CODE + PC + 1)
                eop, ed, es, elx = _dec_esc(sub)
                elen = PREFIX_BYTES + elx
                nPC = PC + elen
                if PC + elen > CODELEN:

                    eop = 0xFFFF
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)

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
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    OVAL = tl.load(DATA + HL); OEN = 1; nHL = (HL + 1) & 0xFFFF
            elif eop == 0x07:
                if DE >= DS:
                    err = 1
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    OVAL = tl.load(DATA + DE); OEN = 1; nDE = (DE + 1) & 0xFFFF
            elif eop == 0x08:
                if SP + 1 >= DS:
                    err = 1
                    errc = _name_cause(errc, F_STACK_UNDERFLOW)
                else:
                    nPC = (tl.load(DATA + SP) << 8) | tl.load(DATA + SP + 1)
                    nSP = SP + 2
            elif eop >= 0x09 and eop <= 0x0D:
                if PC + elen > CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    if eop == 0x09:
                        nPC = t
                    elif eop == 0x0A:
                        nPC = t if Z == 1 else PC + elen
                    elif eop == 0x0B:
                        nPC = t if Z == 0 else PC + elen
                    elif eop == 0x0C:
                        nPC = t if C == 1 else PC + elen
                    else:
                        nPC = t if C == 0 else PC + elen
            elif eop == 0x0E:

                if PC + elen > CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)
                elif SP < 2:
                    err = 1
                    errc = _name_cause(errc, F_STACK_OVERFLOW)
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    ret = PC + elen
                    A1 = SP - 1; V1 = ret & 0xFF; E1 = 1
                    A2 = SP - 2; V2 = (ret >> 8) & 0xFF; E2 = 1
                    nSP = SP - 2; nPC = t
            elif eop == 0x0F or eop == 0x10:
                if PC + elen > CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    nPC = PC + elen
                    if eop == 0x0F:
                        nHL = t
                    else:
                        nDE = t
            elif eop == 0x11 or eop == 0x12:
                if PC + elen > CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)
                else:
                    rs = _get4(r0, r1, r2, r3, tl.load(CODE + PC + 1) & 3)
                    nPC = PC + elen
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
                    v = Z * F_PACK_Z + C * F_PACK_C + SGN * F_PACK_S + OVF * F_PACK_V
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, d, v)
            else:
                err = 1
                errc = _name_cause(errc, _decode_refusal(eop))
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
                errc = _name_cause(errc, _decode_refusal(eop))
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
                nSGN = (v >> 7) & 1; nOVF = ((a ^ v) & (b ^ v)) >> 7
            elif k == 9:
                v = (a - b) & 255; nC = (a < b).to(tl.int32); nZ = (v == 0).to(tl.int32)
                nSGN = (v >> 7) & 1; nOVF = ((a ^ b) & (v ^ a)) >> 7
            elif k == 10:
                t = a + b + C; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
                nSGN = (v >> 7) & 1; nOVF = ((a ^ v) & (b ^ v)) >> 7
            elif k == 11:
                t = a - b - C; v = t & 255; nC = (t < 0).to(tl.int32); nZ = (v == 0).to(tl.int32)
                nSGN = (v >> 7) & 1; nOVF = ((a ^ b) & (v ^ a)) >> 7
            elif k == 12:
                v = b
                nSGN = SGN & 1; nOVF = OVF & 1
            else:
                err = 1
                errc = _name_cause(errc, _decode_refusal(eop))
            nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, d, v)
        elif eop >= 0xD0 and eop <= 0xDF:
            if PC + elen > CODELEN:
                err = 1
                errc = _name_cause(errc, F_FETCH_OOB)
            else:
                i8 = tl.load(CODE + PC + 1)
                rr = _get4(r0, r1, r2, r3, eop & 3)
                nPC = PC + elen
                if eop <= 0xD3:
                    v = i8
                    nSGN = SGN & 1; nOVF = OVF & 1
                elif eop <= 0xD7:
                    t = rr + i8; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
                    nSGN = (v >> 7) & 1; nOVF = ((rr ^ v) & (i8 ^ v)) >> 7
                elif eop <= 0xDB:
                    v = (rr - i8) & 255; nC = (rr < i8).to(tl.int32); nZ = (v == 0).to(tl.int32)
                    nSGN = (v >> 7) & 1; nOVF = ((rr ^ i8) & (v ^ rr)) >> 7
                else:
                    t = rr + i8 + C; v = t & 255; nC = t >> 8; nZ = (v == 0).to(tl.int32)
                    nSGN = (v >> 7) & 1; nOVF = ((rr ^ v) & (i8 ^ v)) >> 7
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
                if PC + elen > CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)
                else:
                    t = tl.load(CODE + PC + 1) | (tl.load(CODE + PC + 2) << 8)
                    v = (rr - 1) & 255
                    nPC = PC + elen
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
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    v = tl.load(DATA + HL)
            elif eop < 0xE8:
                if HL >= DS:
                    err = 1
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    A1 = HL; V1 = rr; E1 = 1
            elif eop < 0xEC:
                if DE >= DS:
                    err = 1
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    v = tl.load(DATA + DE)
            elif eop < 0xF0:
                if DE >= DS:
                    err = 1
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    A1 = DE; V1 = rr; E1 = 1
            elif eop < 0xF4:
                if SP <= 0:
                    err = 1
                    errc = _name_cause(errc, F_STACK_OVERFLOW)
                else:
                    A1 = SP - 1; V1 = rr; E1 = 1; nSP = SP - 1
            elif eop < 0xF8:
                if SP >= DS:
                    err = 1
                    errc = _name_cause(errc, F_STACK_UNDERFLOW)
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
        elif eop >= _ESC_EOP_BASE:

            if eop == ESC_DIV or eop == ESC_MOD:
                a = _get4(r0, r1, r2, r3, ed)
                b = _get4(r0, r1, r2, r3, es)
                if b == 0:
                    err = 1
                    errc = _name_cause(errc, F_DIV_ZERO)
                else:
                    if eop == ESC_DIV:
                        v = a // b
                    else:
                        v = a % b
                    nZ = (v == 0).to(tl.int32)
                    nC = C & 1
                    nSGN = SGN & 1; nOVF = OVF & 1
                    nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            elif eop == ESC_CMP:

                a = _get4(r0, r1, r2, r3, ed)
                b = _get4(r0, r1, r2, r3, es)
                nZ = (a == b).to(tl.int32)
                nC = (a < b).to(tl.int32)
                cv = (a - b) & 255
                nSGN = (cv >> 7) & 1; nOVF = ((a ^ b) & (cv ^ a)) >> 7
            elif eop == ESC_NOT:
                rr = _get4(r0, r1, r2, r3, ed)
                v = (~rr) & 255
                nZ = (v == 0).to(tl.int32)
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
                nC = C & 1
                nSGN = SGN & 1; nOVF = OVF & 1
            elif eop == ESC_NEG:
                rr = _get4(r0, r1, r2, r3, ed)
                v = (-rr) & 255
                nC = (rr != 0).to(tl.int32)
                nZ = (v == 0).to(tl.int32)
                nSGN = (v >> 7) & 1
                nOVF = OVF & 1
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
            elif eop == ESC_MOV_MB_HL:
                nMB = HL & 0xFFFF
            elif eop == ESC_MOV_HL_MB:
                nHL = MB & 0xFFFF
            elif eop == ESC_LDM or eop == ESC_STM:
                why = _bank_refusal(MB, NB)
                if why != 0:
                    err = 1
                    errc = _name_cause(errc, why)
                elif HL >= DS:
                    err = 1
                    errc = _name_cause(errc, F_DATA_OOB)
                elif eop == ESC_LDM:
                    v = tl.load(DATA + HL)
                    nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
                    nC = C & 1
                    nSGN = SGN & 1; nOVF = OVF & 1
                else:
                    A1 = HL; V1 = _get4(r0, r1, r2, r3, ed); E1 = 1
                    nC = C & 1
                    nSGN = SGN & 1; nOVF = OVF & 1
            elif eop == ESC_LDMW_DE_HL or eop == ESC_LDMW_HL_DE:

                why = _bank_refusal(MB, NB)
                if why != 0:
                    err = 1
                    errc = _name_cause(errc, why)
                else:
                    if eop == ESC_LDMW_DE_HL:
                        adr = HL
                    else:
                        adr = DE
                    if adr + 1 >= DS:
                        err = 1
                        errc = _name_cause(errc, F_DATA_OOB)
                    else:
                        v = tl.load(DATA + adr) | (tl.load(DATA + adr + 1) << 8)
                        if eop == ESC_LDMW_DE_HL:
                            nDE = v
                        else:
                            nHL = v
                        nC = C & 1
                        nSGN = SGN & 1; nOVF = OVF & 1
            elif eop == ESC_STMW_HL_DE or eop == ESC_STMW_DE_HL:

                why = _bank_refusal(MB, NB)
                if why != 0:
                    err = 1
                    errc = _name_cause(errc, why)
                else:
                    if eop == ESC_STMW_HL_DE:
                        adr = HL
                        v = DE
                    else:
                        adr = DE
                        v = HL
                    if adr + 1 >= DS:
                        err = 1
                        errc = _name_cause(errc, F_DATA_OOB)
                    else:
                        A1 = adr; V1 = v & 0xFF; E1 = 1
                        A2 = adr + 1; V2 = (v >> 8) & 0xFF; E2 = 1
                        nC = C & 1
                        nSGN = SGN & 1; nOVF = OVF & 1
            elif (eop == ESC_JS or eop == ESC_JNS or eop == ESC_VS or eop == ESC_VC):

                if PC + elen > CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)
                else:
                    d = tl.load(CODE + PC + 2)
                    d = d - 256 * (d >> 7)
                    tgt = (PC + elen + d) & 0xFFFF
                    if eop == ESC_JS:
                        hit = SGN
                    elif eop == ESC_JNS:
                        hit = 1 - SGN
                    elif eop == ESC_VS:
                        hit = OVF
                    else:
                        hit = 1 - OVF
                    nPC = tgt if hit == 1 else PC + elen
                    nSGN = SGN & 1
                    nOVF = OVF & 1
            elif eop == ESC_EXT:

                if PC + elen > CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_FETCH_OOB)
                else:
                    k = tl.load(CODE + PC + 2)
                    if k >= CFG_VEC_COUNT:
                        err = 1
                        errc = _name_cause(errc, F_TRAP_UNREG)
                    else:

                        tgt = tl.load(CFG + k)

                        if tgt == 0:
                            err = 1
                            errc = _name_cause(errc, F_TRAP_UNREG)
                        elif SP < 2:
                            err = 1
                            errc = _name_cause(errc, F_STACK_OVERFLOW)
                        else:
                            ret = PC + elen
                            A1 = SP - 1; V1 = ret & 0xFF; E1 = 1
                            A2 = SP - 2; V2 = (ret >> 8) & 0xFF; E2 = 1
                            nSP = SP - 2; nPC = tgt
            elif eop == ESC_STC:

                if HL >= CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_CODE_OOB)
                else:
                    wlo = tl.load(CFG + CFG_WINLO)
                    whi = tl.load(CFG + CFG_WINHI)
                    if (HL < wlo) or (HL >= whi):
                        err = 1
                        errc = _name_cause(errc, F_WINDOW)
                    else:
                        A3 = HL; V3 = _get4(r0, r1, r2, r3, ed); E3 = 1
            elif eop == ESC_LDC:

                if HL >= CODELEN:
                    err = 1
                    errc = _name_cause(errc, F_CODE_OOB)
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
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    nSP = HL
            elif eop == ESC_MOVW_SP_DE:
                if DE > DS:
                    err = 1
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    nSP = DE
            elif eop == ESC_PUSHW_HL or eop == ESC_PUSHW_DE:

                if SP < 2:
                    err = 1
                    errc = _name_cause(errc, F_STACK_OVERFLOW)
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
                    errc = _name_cause(errc, F_STACK_UNDERFLOW)
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
                    errc = _name_cause(errc, F_DATA_OOB)
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
                    errc = _name_cause(errc, F_DATA_OOB)
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
                    errc = _name_cause(errc, F_DATA_OOB)
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
                    errc = _name_cause(errc, F_DATA_OOB)
                else:
                    nSP = v
            elif eop == ESC_MULH:

                v = ((_get4(r0, r1, r2, r3, ed) * _get4(r0, r1, r2, r3, es)) >> 8) & 255
                nZ = (v == 0).to(tl.int32)
                nC = C & 1
                nSGN = SGN & 1; nOVF = OVF & 1
                nR0, nR1, nR2, nR3 = _wr(r0, r1, r2, r3, ed, v)
            else:
                err = 1
                errc = _name_cause(errc, F_BAD_SUBCODE)
        else:
            err = 1
            errc = _name_cause(errc, _decode_refusal(eop))
    else:
        err = 1
        errc = _name_cause(errc, F_FETCH_OOB)

    OT = tl.load(S + 12)
    ST = tl.load(S + 13)
    NT = OT + 1
    if OEN == 1 and OL >= OC:

        err = 1
        errc = _name_cause(errc, F_OUT_CAP)
    if ST != 0:

        RST = ST
        RTK = OT
    elif OT >= BD:

        tl.store(S + 13, 2)
        RST = 2
        RTK = OT
    elif err == 1:

        tl.store(S + 13, 3)
        tl.store(S + S_FAULT_REASON, errc)
        tl.store(S + S_FAULT_ADDR, PC)
        RST = 3
        RTK = OT
    else:
        nOL = OL + OEN
        tl.store(S + 0, nR0); tl.store(S + 1, nR1); tl.store(S + 2, nR2); tl.store(S + 3, nR3)
        tl.store(S + 4, nHL); tl.store(S + 5, nDE); tl.store(S + 6, nPC); tl.store(S + 7, nSP)
        tl.store(S + 8, nC); tl.store(S + 9, nZ); tl.store(S + 10, nIPO); tl.store(S + 11, nOL)
        tl.store(S + S_SIGN, nSGN); tl.store(S + S_OVFL, nOVF)
        tl.store(S + 12, NT)
        tl.store(S + 13, NST)
        tl.store(S + S_MB, nMB)

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
def ncp_step_kernel(CODE, DATA, INPUTS, OUTBUF, S, BUDGET, CODELEN, INLEN, DS, OC,
                    CFG, NB):

    BD = tl.load(BUDGET + 0)
    _tick(CODE, DATA, INPUTS, OUTBUF, S, CODELEN, INLEN, BD, DS, OC, CFG, NB)

@triton.jit
def ncp_resident_kernel(CODE, DATA, INPUTS, OUTBUF, STATES, CODELENS, INLENS, BUDGETS,
                        STEP_LIMIT, CFG, NB,
                        CS: tl.constexpr, DS: tl.constexpr, INS: tl.constexpr,
                        OCS: tl.constexpr):

    pid = tl.program_id(0)
    CP = CODE + pid * CS
    DP = DATA + pid * DS
    IP = INPUTS + pid * INS
    OP = OUTBUF + pid * OCS
    ST = STATES + pid * STATE_ROWS_C
    CF = CFG + pid * CFG_LEN_C
    CL = tl.load(CODELENS + pid)
    IL = tl.load(INLENS + pid)
    BD = tl.load(BUDGETS + pid)
    st = tl.load(ST + 13)
    tk = tl.load(ST + 12)
    n = 0
    while (st == 0) & ((STEP_LIMIT <= 0) | (n < STEP_LIMIT)):
        st, tk = _tick(CP, DP, IP, OP, ST, CL, IL, BD, DS, OCS, CF, NB)
        n += 1

class BatchResult(NamedTuple):

    outs: list
    status: list
    ticks: list
    oplens: list

class TritonBatch:

    def __init__(self, n, device="cuda", max_in=1, tick_budget=ISA.TICK_BUDGET_DEFAULT,
                 num_warps=1, out_cap=OUT_CAP, config=None):
        if n < 1:
            raise ValueError("batch size must be >= 1")
        cfg = ISA.MachineConfig() if config is None else config
        if config is not None and not isinstance(config, ISA.MachineConfig):
            raise ISA.ConfigError(
                f"config must be an isa_table.MachineConfig, got a "
                f"{type(config).__name__}: configuration is validated at load, and a "
                f"mapping or tuple would arrive unchecked")
        if cfg.codelen is not None:
            raise ISA.ConfigError(
                f"CODELEN={cfg.codelen} cannot be declared for a batch: each row's "
                f"instruction bound is the length of the program set_program loaded "
                f"into it, so one number for all {n} rows would describe {n} machines "
                f"that are not this batch")
        self.dev = torch.device(device)
        self.n = n
        self.config = cfg
        self.out_cap = ISA.resolve_constraint("OUTCAP", "out_cap", out_cap, OUT_CAP,
                                              cfg.outcap)
        ISA.check_capacity(self.out_cap, OUT_CAP)
        self.nbanks = 1 if cfg.nbanks is None else cfg.nbanks
        self.tdlim = 0 if cfg.tdlim is None else cfg.tdlim
        self.max_in = max(1, int(max_in))
        self.num_warps = num_warps
        i32 = torch.int32
        self.CODE = torch.zeros((n, CODE_SIZE), dtype=i32, device=self.dev)
        self.DATA = torch.zeros((n, DATA_SIZE), dtype=i32, device=self.dev)
        self.INPUTS = torch.zeros((n, self.max_in), dtype=i32, device=self.dev)
        self.OUTBUF = torch.zeros((n, self.out_cap), dtype=i32, device=self.dev)
        self.STATE = torch.zeros((n, STATE_ROWS), dtype=i32, device=self.dev)
        self.STATE[:, 7] = DATA_SIZE
        self.CODELENS = torch.zeros(n, dtype=i32, device=self.dev)
        self.INLENS = torch.zeros(n, dtype=i32, device=self.dev)
        tb = ISA.resolve_constraint("TICKBUDGET", "tick_budget", int(tick_budget),
                                    ISA.TICK_BUDGET_DEFAULT, cfg.tickbudget)
        self.BUDGETS = torch.full((n,), int(tb), dtype=i32, device=self.dev)

        self.CFG = torch.tensor([_cfg_row(cfg) for _ in range(n)], dtype=i32,
                                device=self.dev)

    def _row(self, i):

        if not isinstance(i, int) or not 0 <= i < self.n:
            raise ValueError(f"machine index {i} is outside a batch of {self.n} machines")

    def set_program(self, i, code, data=None, inputs=None):

        self._row(i)
        cb = bytes(code)
        if len(cb) > CODE_SIZE:
            raise ValueError(f"machine {i}: code is {len(cb)} bytes, above CODE_SIZE {CODE_SIZE}")

        bad = self.config.check_for_program(len(cb), f"machine {i}: ")
        if bad is not None:
            raise ISA.ConfigError(bad)
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
                  C=0, Z=0, S=0, V=0, ipos=0, oplen=0, tick=0, status=0, fault_reason=0,
                  fault_addr=0, MB=0):

        self._row(i)
        check_state(r, HL, DE, SP, C, Z, tick=tick, PC=PC, ipos=ipos, oplen=oplen,
                    status=status, fault_reason=fault_reason, fault_addr=fault_addr,
                    mb=MB, s=S, v=V, where=f"machine {i}: ")
        self.STATE[i, 0:4] = torch.tensor(list(r), dtype=torch.int32, device=self.dev)
        self.STATE[i, 4] = HL
        self.STATE[i, 5] = DE
        self.STATE[i, 6] = PC
        self.STATE[i, 7] = SP
        self.STATE[i, 8] = C
        self.STATE[i, 9] = Z
        self.STATE[i, int(S_SIGN)] = S
        self.STATE[i, int(S_OVFL)] = V
        self.STATE[i, 10] = ipos
        self.STATE[i, 11] = oplen
        self.STATE[i, 12] = tick
        self.STATE[i, 13] = status
        self.STATE[i, 14] = fault_reason
        self.STATE[i, 15] = fault_addr
        self.STATE[i, int(S_MB)] = MB

    def set_budget(self, i, budget):

        self._row(i)
        self.BUDGETS[i] = int(budget)

    def record_state(self, i):

        row = _Row(self, i)
        return ISA.publish_record(row, _TRITON_RECORD_READERS, row._record_bounds())

    def install_state(self, i, snap):

        row = _Row(self, i)
        row.install(ISA.check_record(snap, row._record_bounds()))

    def run(self):

        self._launch(0)
        return self.results()

    def step(self, steps=1):

        self._launch(int(steps))
        return self.results()

    def _launch(self, step_limit):
        ncp_resident_kernel[(self.n,)](
            self.CODE, self.DATA, self.INPUTS, self.OUTBUF, self.STATE,
            self.CODELENS, self.INLENS, self.BUDGETS, step_limit, self.CFG,
            self.nbanks,
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
        s = _row_state(self.STATE[i])
        s["oplen"] = int(self.STATE[i, STATE_ROW_OPLEN].item())
        return s

    def data(self, i):

        self._row(i)
        return self.DATA[i].cpu().tolist()

    def code(self, i):

        self._row(i)
        return self.CODE[i].cpu().tolist()

def run_batch(codes, datas=None, inputs=None, budgets=None, states=None,
              tick_budget=ISA.TICK_BUDGET_DEFAULT, device="cuda", num_warps=1,
              config=None):

    n = len(codes)
    max_in = 1
    if inputs:
        max_in = max(1, max((len(v) for v in inputs), default=1))
    b = TritonBatch(n, device=device, max_in=max_in,
                    tick_budget=tick_budget, num_warps=num_warps, config=config)
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
            if len(row) != STATE_ROWS:
                raise ValueError(f"machine {i}: state row has {len(row)} values, "
                                 f"need {STATE_ROWS}")
            b.set_state(i, r=row[0:4], HL=row[4], DE=row[5], PC=row[6], SP=row[7],
                        C=row[8], Z=row[9], S=row[int(S_SIGN)], V=row[int(S_OVFL)],
                        ipos=row[10], oplen=row[11], tick=row[12], status=row[13],
                        fault_reason=row[14], fault_addr=row[15], MB=row[int(S_MB)])
    return b.run()

class TritonCircuit:

    def __init__(self, code, data=None, inputs=b"",
                 tick_budget=ISA.TICK_BUDGET_DEFAULT, device="cuda",
                 out_cap=OUT_CAP, config=None):

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

        bad = cfg.check_for_program(self.codelen, "TritonCircuit: ")
        if bad is not None:
            raise ISA.ConfigError(bad)
        self.out_cap = ISA.resolve_constraint("OUTCAP", "out_cap", out_cap, OUT_CAP,
                                              cfg.outcap)
        ISA.check_capacity(self.out_cap, OUT_CAP)
        self.nbanks = 1 if cfg.nbanks is None else cfg.nbanks
        self.tdlim = 0 if cfg.tdlim is None else cfg.tdlim
        self.CODE = torch.zeros(max(1, CODE_SIZE), dtype=i32, device=dev)
        self.CODE[: len(code)] = torch.tensor(list(code), dtype=i32, device=dev)
        self.DATA = torch.zeros(DATA_SIZE, dtype=i32, device=dev)
        if data:
            self.DATA[: len(data)] = torch.tensor(list(data), dtype=i32, device=dev)
        self.INP = torch.tensor(list(inputs) if inputs else [0], dtype=i32, device=dev)
        self.inlen = len(inputs)
        self.OUTBUF = torch.zeros(self.out_cap, dtype=i32, device=dev)
        self.S = torch.zeros(STATE_ROWS, dtype=i32, device=dev)
        self.S[7] = DATA_SIZE
        self.BUDGET = torch.tensor(
            [ISA.resolve_constraint("TICKBUDGET", "tick_budget", int(tick_budget),
                                    ISA.TICK_BUDGET_DEFAULT, cfg.tickbudget)],
            dtype=i32, device=dev)
        self.tb = int(self.BUDGET.item())

        self.CFG = torch.tensor(_cfg_row(cfg), dtype=i32, device=dev)

    def load_state(self, R, HL, DE, SP, C, Z, tick, PC=0, fault_reason=None,
                   fault_addr=None, S=None, V=None):

        fr = int(self.S[14].item()) if fault_reason is None else fault_reason
        fa = int(self.S[15].item()) if fault_addr is None else fault_addr
        s0 = int(self.S[int(S_SIGN)].item()) if S is None else S
        v0 = int(self.S[int(S_OVFL)].item()) if V is None else V
        check_state(R, HL, DE, SP, C, Z, tick=tick, PC=PC,
                    ipos=int(self.S[10].item()), oplen=int(self.S[11].item()),
                    status=int(self.S[13].item()), fault_reason=fr, fault_addr=fa,
                    mb=int(self.S[int(S_MB)].item()), s=s0, v=v0)
        self.S[0:4] = torch.tensor(list(R), dtype=torch.int32, device=self.dev)
        self.S[4] = HL; self.S[5] = DE; self.S[6] = PC
        self.S[7] = SP; self.S[8] = C; self.S[9] = Z
        self.S[12] = tick
        self.S[14] = fr; self.S[15] = fa
        self.S[int(S_SIGN)] = s0; self.S[int(S_OVFL)] = v0

    def _record_state(self):
        return _row_state(self.S)

    def _record_code(self):
        return _row_bytes(self.CODE, "CODE", CODE_SIZE)

    def _record_data(self):
        return _row_bytes(self.DATA, "DATA", DATA_SIZE)

    def _record_out(self):
        n = min(int(self.S[STATE_ROW_OPLEN].item()), self.out_cap)
        return bytes(int(v) for v in self.OUTBUF[:n].cpu().tolist())

    def _record_inputs(self):
        return bytes(int(v) for v in self.INP[: self.inlen].cpu().tolist())

    def _record_block(self):

        lo, hi = self.config.window()
        return ISA.MachineConfig(codelen=self.codelen, winlo=lo, winhi=hi,
                                 vec=self.config.vectors(), nbanks=self.nbanks,
                                 tdlim=self.tdlim, tickbudget=self.tb,
                                 outcap=self.out_cap)

    def _record_bounds(self):

        return ISA.RecordBounds(where="TritonCircuit: ", out_cap=self.out_cap,
                                code_size=CODE_SIZE, data_size=DATA_SIZE,
                                inputs=self._record_inputs(),
                                block=self._record_block())

    def record_state(self):

        return ISA.publish_record(self, _TRITON_RECORD_READERS, self._record_bounds())

    def install_state(self, snap):

        got = ISA.check_record(snap, self._record_bounds())
        st = got.state
        check_state(st["r"], st["HL"], st["DE"], st["SP"], st["C"], st["Z"],
                    tick=st["tick"], PC=st["PC"], ipos=st["ipos"], oplen=len(got.out),
                    status=st["status"], fault_reason=st["fault_reason"],
                    fault_addr=st["fault_addr"], mb=st["MB"], s=st["S"], v=st["V"],
                    where="TritonCircuit: ")
        _write_row_state(self.S, st)
        self.S[STATE_ROW_OPLEN] = len(got.out)
        self.CODE.copy_(_byte_image(got.code).to(self.dev))
        self.DATA.copy_(_byte_image(got.data).to(self.dev))
        self.OUTBUF.zero_()
        if got.out:
            self.OUTBUF[: len(got.out)] = _byte_image(got.out).to(self.dev)

    def step(self):
        ncp_step_kernel[(1,)](self.CODE, self.DATA, self.INP, self.OUTBUF, self.S,
                              self.BUDGET, self.codelen, self.inlen, DATA_SIZE,
                              self.out_cap, self.CFG, self.nbanks)

    @property
    def status(self):
        return self.S[13:14]

    @property
    def tick(self):
        return self.S[12:13]

    @property
    def oplen(self):
        return self.S[11:12]

    @property
    def fault_reason(self):
        return self.S[14:15]

    @property
    def fault_addr(self):
        return self.S[15:16]

    def snapshot(self):

        s = _row_state(self.S)
        s["oplen"] = int(self.S[STATE_ROW_OPLEN].item())
        return s

    def out(self):

        n = min(int(self.S[11].item()), OUT_CAP)
        return bytes(self.OUTBUF[:n].cpu().tolist())

    def run(self):

        while int(self.status.item()) == 0:
            self.step()
        return self.out()

    def run_resident(self):

        cfg = self.config
        if cfg.codelen is not None:
            cfg = ISA.MachineConfig(**{n: getattr(cfg, n) for n in cfg.__slots__
                                       if n != "codelen"})
        b = TritonBatch(1, device=self.dev, max_in=max(1, int(self.INP.numel())),
                        out_cap=self.out_cap, config=cfg)
        b.CODE[0].copy_(self.CODE)
        b.CODELENS[0] = self.codelen
        b.DATA[0].copy_(self.DATA)
        b.OUTBUF[0].copy_(self.OUTBUF)
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