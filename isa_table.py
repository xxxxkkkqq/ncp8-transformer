"""The NCP-8 decode table: one statement of what each code point means.

Every fact about an encoding lives here once - the mnemonic, the ALU selector, the two
operand selectors, the instruction length - for the single-byte opcodes and for the
`0x70` escape space. The reference simulator dispatches on the selector this table gives
it, the tensor implementation builds its decode ROMs from these rows, and the Triton
implementation checks its in-kernel decode against them for every code point before it
will load.

Self-test: `python3 isa_table.py` compares the table against the ROMs and dispatch of the
implementations as they are actually loaded, and reports the escape subcodes still free.

The fault causes live here too: `FAULT_CAUSES` numbers every cause a machine can name
(dense from 0, where 0 means no fault) and `FAULT_SITE_ORDER` states the sites in
precedence order, so the tick that stops a machine names exactly one cause. The order is
*a statement* here, not a driver: each implementation sequences its own checks, and the
claim that all three sequence them the way this table lists them is held by a searched
sweep over code points and bound values, not by construction. Reordering a check inside
an implementation, or a row in this table, is a change to the contract and has to be
shown by that sweep.
`fault_state_error` is the single legality rule the three `check_state()`s share: widths,
cause-in-table, and the pairing `fault_reason != 0` if and only if `status == 3`.

`MachineConfig` is the load-time configuration block the same three implementations
read: CODELEN, the two window bounds, the trap vector table, the bank count, the trap
depth limit, the tick budget and the output capacity. Constructing it is the gate -
each field has one stated range, a half-supplied window and a reversed one are refused,
and `resolve_constraint` refuses a bound that the block and a moved constructor
argument state differently. `as_dict()`/`from_dict()` are how a block is carried as
data, so a record or a recording can hold the machine it describes.
"""

from collections import namedtuple

DATA_SIZE = 4096
ESCAPE_PREFIX = 0x70
PREFIX_BYTES = 2

class DecodeTableError(Exception):

    pass

class UndefinedCode(DecodeTableError):

    pass

ALU_NAMES = (
    "NOP", "ADD", "SUB", "ADC", "SBB", "MOV", "TST", "SHL", "SHR", "DJNZ", "CLC",
    "LDI", "ADDI", "SUBI", "ADCI",
    "MOV_R_HL", "MOV_HL_R", "MOV_R_DE", "MOV_DE_R",
    "PUSH", "POP", "OUT", "IN",
    "INC_HL", "DEC_HL", "INC_DE",
    "OUTM", "OUTDE",
    "JMP", "JZ", "JNZ", "JC", "JNC",
    "CALL", "RET",
    "LDI_HL", "LDI_DE", "ADDI_HL", "ADDI_DE",
    "HALT", "BAD",
    "JPHL", "GETPC", "GETSP", "GETF",
    "AND", "OR", "XOR", "MUL",
    "DIV", "MOD", "CMP",
    "NOT", "NEG", "ROL", "ROR",
    "ADD_HLDE", "SUB_HLDE", "XCHG", "EXT",
    "STC", "LDC",
    "MOVW_HL_DE", "MOVW_DE_HL", "MOVW_HL_SP", "MOVW_DE_SP", "MOVW_SP_HL",
    "MOVW_SP_DE",
    "PUSHW_HL", "PUSHW_DE", "POPW_HL", "POPW_DE",
    "STW_HLDE", "STW_DEHL", "LDW_DEHL", "LDW_HLDE",
    "LDX", "STX", "ADD_SP", "MULH",
    "LDM", "STM", "LDMW_DE_HL", "LDMW_HL_DE", "STMW_HL_DE", "STMW_DE_HL",
    "MOV_MB_HL", "MOV_HL_MB",
    "JS", "JNS", "VS", "VC",
    "TRAPRET", "CALL_HL",
)
ALU_ID = {name: i for i, name in enumerate(ALU_NAMES)}
K = len(ALU_NAMES)
BAD = ALU_ID["BAD"]

ESC_EOP_BASE = 0x100
ESC_EOP_NAMES = (
    "DIV", "MOD", "CMP", "NOT", "NEG", "ROL", "ROR",
    "ADD_HLDE", "SUB_HLDE", "XCHG", "EXT", "STC", "LDC", "BAD",
    "MOVW_HL_DE", "MOVW_DE_HL", "MOVW_HL_SP", "MOVW_DE_SP", "MOVW_SP_HL",
    "MOVW_SP_DE", "PUSHW_HL", "PUSHW_DE", "POPW_HL", "POPW_DE",
    "STW_HLDE", "STW_DEHL", "LDW_DEHL", "LDW_HLDE", "LDX", "STX", "ADD_SP", "MULH",
    "LDM", "STM", "LDMW_DE_HL", "LDMW_HL_DE", "STMW_HL_DE", "STMW_DE_HL",
    "MOV_MB_HL", "MOV_HL_MB",
    "JS", "JNS", "VS", "VC",
    "TRAPRET", "CALL_HL",
)
ESC_EOP_ID = {name: ESC_EOP_BASE + i for i, name in enumerate(ESC_EOP_NAMES)}

SINGLE = {}
ESCAPE = {}

def _single(op, alu, mnem, s0=0, s1=0, l=0, kind=""):
    if op in SINGLE:
        raise DecodeTableError(f"single-byte code {op:#04x} is assigned twice")
    SINGLE[op] = dict(op=op, space="single", alu=alu, mnem=mnem, s0=s0, s1=s1,
                      l=l, kind=kind)

def _escape(sub, alu, mnem, s0=0, s1=0, l=0, kind=""):
    if sub in ESCAPE:
        raise DecodeTableError(f"escape subcode {sub:#04x} is assigned twice")
    ESCAPE[sub] = dict(op=sub, space="escape", alu=alu, mnem=mnem, s0=s0, s1=s1,
                       l=l, kind=kind)

for _i, (_alu, _mn) in enumerate([("HALT", "HALT"), ("NOP", "NOP"), ("INC_HL", "INC HL"),
                                  ("DEC_HL", "DEC HL"), ("INC_DE", "INC DE"),
                                  ("CLC", "CLC"), ("OUTM", "OUTM"),
                                  ("OUTDE", "OUTDE"), ("RET", "RET")]):
    _single(_i, _alu, _mn)
for _i, _alu in enumerate(["JMP", "JZ", "JNZ", "JC", "JNC", "CALL"]):
    _single(0x09 + _i, _alu, _alu, l=2, kind="a16")
_single(0x0F, "LDI_HL", "LDI HL", l=2, kind="i16")
_single(0x10, "LDI_DE", "LDI DE", l=2, kind="i16")
_single(0x11, "ADDI_HL", "ADDI HL", l=1, kind="roff")
_single(0x12, "ADDI_DE", "ADDI DE", l=1, kind="roff")
_single(0x13, "JPHL", "JPHL")
for _base, _alu in ((0x14, "GETPC"), (0x18, "GETSP"), (0x1C, "GETF")):
    for _k in range(4):
        _single(_base | _k, _alu, f"{_alu} r{_k}", _k, _k)
for _base, _alu in ((0x20, "AND"), (0x30, "OR"), (0x40, "XOR"), (0x50, "MUL"),
                    (0x80, "ADD"), (0x90, "SUB"), (0xA0, "ADC"), (0xB0, "SBB"),
                    (0xC0, "MOV")):
    for _f in range(16):
        _single(_base + _f, _alu, f"{_alu} r{(_f >> 2) & 3}, r{_f & 3}",
                (_f >> 2) & 3, _f & 3)
for _base, _alu in ((0x60, "SHL"), (0x64, "SHR"), (0x68, "TST")):
    for _k in range(4):
        _single(_base | _k, _alu, f"{_alu} r{_k}", _k, _k)
for _k in range(4):
    _single(0x6C | _k, "DJNZ", f"DJNZ r{_k}", _k, _k, l=2, kind="a16")
for _base, _alu in ((0xD0, "LDI"), (0xD4, "ADDI"), (0xD8, "SUBI"), (0xDC, "ADCI")):
    for _k in range(4):
        _single(_base | _k, _alu, f"{_alu} r{_k}", _k, _k, l=1, kind="i8")
for _base, _alu, _tpl in ((0xE0, "MOV_R_HL", "MOV r{k}, [HL]"),
                          (0xE4, "MOV_HL_R", "MOV [HL], r{k}"),
                          (0xE8, "MOV_R_DE", "MOV r{k}, [DE]"),
                          (0xEC, "MOV_DE_R", "MOV [DE], r{k}"),
                          (0xF0, "PUSH", "PUSH r{k}"), (0xF4, "POP", "POP r{k}"),
                          (0xF8, "OUT", "OUT r{k}"), (0xFC, "IN", "IN r{k}")):
    for _k in range(4):
        _single(_base | _k, _alu, _tpl.replace("{k}", str(_k)), _k, _k)

for _base, _alu in ((0x00, "DIV"), (0x10, "MOD"), (0x20, "CMP")):
    for _f in range(16):
        _escape(_base + _f, _alu, f"{_alu} r{(_f >> 2) & 3}, r{_f & 3}",
                (_f >> 2) & 3, _f & 3)
for _base, _alu in ((0x40, "NOT"), (0x44, "NEG"), (0x48, "ROL"), (0x4C, "ROR")):
    for _k in range(4):
        _escape(_base | _k, _alu, f"{_alu} r{_k}", _k, _k)
for _sub, (_alu, _mn) in ((0x30, ("MOVW_HL_DE", "MOVW HL, DE")),
                          (0x31, ("MOVW_DE_HL", "MOVW DE, HL")),
                          (0x32, ("MOVW_HL_SP", "MOVW HL, SP")),
                          (0x33, ("MOVW_DE_SP", "MOVW DE, SP")),
                          (0x34, ("MOVW_SP_HL", "MOVW SP, HL")),
                          (0x35, ("MOVW_SP_DE", "MOVW SP, DE")),
                          (0x38, ("PUSHW_HL", "PUSHW HL")),
                          (0x39, ("PUSHW_DE", "PUSHW DE")),
                          (0x3A, ("POPW_HL", "POPW HL")),
                          (0x3B, ("POPW_DE", "POPW DE")),
                          (0x3C, ("STW_HLDE", "STW [HL], DE")),
                          (0x3D, ("STW_DEHL", "STW [DE], HL")),
                          (0x3E, ("LDW_DEHL", "LDW DE, [HL]")),
                          (0x3F, ("LDW_HLDE", "LDW HL, [DE]")),
                          (0x60, ("ADD_HLDE", "ADD HL, DE")),
                          (0x61, ("SUB_HLDE", "SUB HL, DE")),
                          (0x62, ("XCHG", "XCHG HL, DE"))):
    _escape(_sub, _alu, _mn)
for _k in range(4):
    _escape(0x50 | _k, "LDX", f"LDX r{_k}, [HL{{off}}]", _k, _k, l=1, kind="off")
    _escape(0x54 | _k, "STX", f"STX [HL{{off}}], r{_k}", _k, _k, l=1, kind="off")
_escape(0x58, "ADD_SP", "ADD SP, {soff}", l=1, kind="off")
_escape(0x70, "EXT", "EXT {k}", l=1, kind="k")
for _k in range(4):
    _escape(0x80 | _k, "STC", f"STC [HL], r{_k}", _k, _k)
    _escape(0x84 | _k, "LDC", f"LDC r{_k}, [HL]", _k, _k)
for _f in range(16):
    _escape(0x90 + _f, "MULH", f"MULH r{(_f >> 2) & 3}, r{_f & 3}",
            (_f >> 2) & 3, _f & 3)
for _k in range(4):
    _escape(0xB0 | _k, "LDM", f"LDM r{_k}, [HL]", _k, _k)
    _escape(0xB4 | _k, "STM", f"STM [HL], r{_k}", _k, _k)

for _sub, _alu in zip(range(0x64, 0x68), ("JS", "JNS", "VS", "VC")):
    _escape(_sub, _alu, f"{_alu} {{soff}}", l=1, kind="off")

_escape(0xA8, "TRAPRET", "TRAPRET")
_escape(0xA9, "CALL_HL", "CALL HL")
for _sub, (_alu, _mn) in ((0xB8, ("LDMW_DE_HL", "LDMW DE, [HL]")),
                          (0xB9, ("LDMW_HL_DE", "LDMW HL, [DE]")),
                          (0xBA, ("STMW_HL_DE", "STMW [HL], DE")),
                          (0xBB, ("STMW_DE_HL", "STMW [DE], HL")),
                          (0xBC, ("MOV_MB_HL", "MOV MB, HL")),
                          (0xBD, ("MOV_HL_MB", "MOV HL, MB"))):
    _escape(_sub, _alu, _mn)

V4_RESERVED = tuple(range(0xA0, 0xA8))

FAULT_CAUSES = (
    ("OK", "no fault"),
    ("BAD_OPCODE", "unassigned single-byte opcode"),
    ("BAD_SUBCODE", "reserved escape subcode"),
    ("FETCH_OOB", "instruction bytes past the image end"),
    ("DATA_OOB", "address outside [0, DATA_SIZE), from a read/write either byte of a "
                  "pair, or from a pointer value that left the span (an SP write)"),
    ("STACK_UNDERFLOW", "pop/RET with no stored slots"),
    ("STACK_OVERFLOW", "push/CALL with no room"),
    ("DIV_ZERO", "DIV/MOD divisor zero"),
    ("CODE_OOB", "LDC/STC address past the image"),
    ("WINDOW", "STC outside the declared window"),
    ("TRAP_UNREG", "EXT k with zero vector, or k >= 16"),
    ("TRAP_DEPTH", "EXT k with TDEPTH == TDLIM, before any push"),
    ("TRAP_FRAME", "TRAPRET read a tag that is not 0xA5"),
    ("TRAP_UNBALANCED", "TRAPRET with TDEPTH == 0"),
    ("OUT_CAP", "producing byte number OUT_CAP+1"),
    ("BANK_OOB", "MB >= NBANKS"),
    ("BANK_BUSY", "cross-bank access while the owner is running"),
    ("PC_ILLEGAL", "committed PC outside the image"),
)
CAUSE = {name: code for code, (name, _d) in enumerate(FAULT_CAUSES)}
CAUSE_NAME = {code: name for code, (name, _d) in enumerate(FAULT_CAUSES)}
CAUSE_DESC = {name: desc for name, desc in FAULT_CAUSES}
FAULT_CODES = tuple(range(len(FAULT_CAUSES)))

CAUSES_AWAITING_FEATURE = {
    "BANK_BUSY": "the ownership rule is answered on all four paths, but the three "
                 "circuit paths hold one page each -- their own DATA, at index 0 -- so a "
                 "selector that passes the declared bound names the page it owns and a "
                 "selector past it names BANK_OOB before ownership is read. Only the "
                 "reference's group driver hands a machine a foreign page, and no group "
                 "driver exists on a circuit path, so no bank access there can name this "
                 "cause",
}

FAULT_SITE_ORDER = (
    ("PC_ILLEGAL", "PC_ILLEGAL"),
    ("FETCH_CODE", "FETCH_OOB"),
    ("BAD_OPCODE", "BAD_OPCODE"),
    ("BAD_SUBCODE", "BAD_SUBCODE"),
    ("FETCH_OPERAND", "FETCH_OOB"),
    ("BANK_OOB", "BANK_OOB"),
    ("BANK_BUSY", "BANK_BUSY"),
    ("TRAP_UNREG", "TRAP_UNREG"),
    ("TRAP_DEPTH", "TRAP_DEPTH"),
    ("TRAP_UNBALANCED", "TRAP_UNBALANCED"),
    ("DATA_OOB", "DATA_OOB"),
    ("STACK_PUSH", "STACK_OVERFLOW"),
    ("STACK_POP", "STACK_UNDERFLOW"),
    ("TRAP_FRAME", "TRAP_FRAME"),
    ("DIV_ZERO", "DIV_ZERO"),
    ("CODE_OOB", "CODE_OOB"),
    ("WINDOW", "WINDOW"),
    ("SP_RANGE", "DATA_OOB"),
    ("OUT_CAP", "OUT_CAP"),
)
FAULT_SITE_NAMES = tuple(site for site, _cause in FAULT_SITE_ORDER)
FAULT_SITE_CAUSE = {site: CAUSE[cause] for site, cause in FAULT_SITE_ORDER}
SITE_RANK = {site: i for i, site in enumerate(FAULT_SITE_NAMES)}

BANK_FAULT_SITES = (("BANK_OOB", "BANK_OOB"), ("BANK_BUSY", "BANK_BUSY"))
BANK_SITE_ANCHOR = "FETCH_OPERAND"

def full_fault_site_order():

    return FAULT_SITE_ORDER

def fault_name(code):

    if code not in CAUSE_NAME:
        raise DecodeTableError(f"fault code {code} is not in the cause table")
    return CAUSE_NAME[code]

def fault_code(name):

    if name not in CAUSE:
        raise DecodeTableError(f"fault cause {name!r} is not in the cause table")
    return CAUSE[name]

FAULT_BITS = 8
FAULT_ADDR_BITS = 16

VEC_COUNT = 16

DEFAULT_WINDOW = (0, 0)
DEFAULT_VECTORS = (0,) * VEC_COUNT
DEFAULT_ENTRY = 0
DEFAULT_TDLIM = 64

TRAP_TAG = 0xA5

CONFIG_NAMES = ("CODELEN", "ENTRY", "WINLO", "WINHI", "VEC", "NBANKS", "TDLIM",
                "SPLIM", "TICKBUDGET", "OUTCAP")

CONFIG_WIDTHS = {"CODELEN": 16, "ENTRY": 16, "WINLO": 16, "WINHI": 16, "VEC": 16,
                 "NBANKS": 16, "TDLIM": 8, "SPLIM": 16, "TICKBUDGET": None,
                 "OUTCAP": 16}

OUT_CAP_LIMIT = 1 << 16

class ConfigError(ValueError):

    pass

def _cfg_int(name, value, lo, hi):

    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"configuration {name} must be an int in {lo}..{hi}, "
                          f"got {value!r}")
    if not lo <= value <= hi:
        raise ConfigError(f"configuration {name} is {value}, outside {lo}..{hi}")
    return value

def check_splim(value, where=""):

    return _cfg_int(f"{where}SPLIM", value, 0, DATA_SIZE)

def check_capacity(value, width, field="out_cap"):

    v = _cfg_int(field, value, 0, OUT_CAP_LIMIT - 1)
    if v < 1:
        raise ValueError(f"{field}={value} is below 1: a machine with no output "
                         f"capacity cannot emit the byte that would tell you so")
    if v & (v - 1):
        raise ValueError(f"{field}={value} is not a power of two; the store masks "
                         f"would let a write leave its own output buffer")
    if v > width:
        raise ValueError(f"{field}={value} exceeds the {width}-byte allocated output "
                         f"buffer, so the capacity would not be the bound in force")
    return v

TICK_BUDGET_DEFAULT = 200_000

def resolve_constraint(cfg_name, ctor_name, ctor_value, ctor_default, cfg_value):

    if cfg_value is None:
        return ctor_value
    if ctor_value != ctor_default and ctor_value != cfg_value:
        raise ConfigError(
            f"{cfg_name} is declared twice with different values: the constructor "
            f"argument {ctor_name}={ctor_value} and the configuration block "
            f"{cfg_name}={cfg_value}; a bound with two sources has no defined "
            f"winner, so pass one of them or leave {ctor_name} at its default "
            f"{ctor_default}")
    return cfg_value

def window_error(winlo, winhi, codelen, where=""):

    if winhi > codelen:
        return (f"{where}window [0x{winlo:04X},0x{winhi:04X}) does not fit inside a "
                f"program of {codelen} bytes (0x{codelen:04X}): WINHI={winhi} is past "
                f"the end of the program the machine was loaded with, so the span names "
                f"cells no program occupies")
    return None

def entry_error(entry, codelen, where=""):

    if codelen > 0 and entry >= codelen:
        return (f"{where}ENTRY={entry} (0x{entry:04X}) does not fit inside a program "
                f"of {codelen} bytes (0x{codelen:04X}): the boot address is at or past "
                f"the end of the program the machine was loaded with, so the first "
                f"fetch would fault")
    return None

class MachineConfig:

    __slots__ = ("codelen", "entry", "winlo", "winhi", "vec", "nbanks", "tdlim",
                 "splim", "tickbudget", "outcap")

    def __init__(self, *, codelen=None, entry=None, winlo=None, winhi=None, vec=None,
                 nbanks=None, tdlim=None, splim=None, tickbudget=None, outcap=None):
        self.codelen = (None if codelen is None
                        else _cfg_int("CODELEN", codelen, 0, (1 << 16) - 1))
        self.entry = (None if entry is None
                      else _cfg_int("ENTRY", entry, 0, (1 << 16) - 1))
        if self.entry is not None and self.codelen is not None:
            bad = entry_error(self.entry, self.codelen)
            if bad is not None:
                raise ConfigError(bad)
        if (winlo is None) != (winhi is None):
            raise ConfigError("WINLO and WINHI are one declaration: got "
                              f"WINLO={winlo!r} without WINHI, or WINHI={winhi!r} "
                              "without WINLO, which would leave half a bound")
        if winlo is not None:
            lo = _cfg_int("WINLO", winlo, 0, (1 << 16) - 1)
            hi = _cfg_int("WINHI", winhi, 0, (1 << 16) - 1)
            if hi < lo:
                raise ConfigError(
                    f"WINDOW: WINHI={hi} (0x{hi:04X}) is below WINLO={lo} "
                    f"(0x{lo:04X}); a reversed window is refused at load rather than "
                    f"run as the empty window [0x{lo:04X},0x{lo:04X}), which is how a "
                    f"typo in one bound would come out looking like disabled "
                    f"self-modification")
            if self.codelen is not None:
                bad = window_error(lo, hi, self.codelen)
                if bad is not None:
                    raise ConfigError(bad)
            self.winlo, self.winhi = lo, hi
        else:
            self.winlo = self.winhi = None
        self.vec = None if vec is None else self._checked_vec(vec)
        self.nbanks = (None if nbanks is None
                       else _cfg_int("NBANKS", nbanks, 1, (1 << 16) - 1))
        self.tdlim = (None if tdlim is None
                      else _cfg_int("TDLIM", tdlim, 0, (1 << 8) - 1))
        self.splim = None if splim is None else check_splim(splim)
        if tickbudget is not None:
            tickbudget = _cfg_int("TICKBUDGET", tickbudget, 0, 1 << 62)
        self.tickbudget = tickbudget
        if outcap is not None:
            try:
                outcap = check_capacity(outcap, OUT_CAP_LIMIT - 1, "OUTCAP")
            except ValueError as e:

                raise ConfigError(str(e)) from e
        self.outcap = outcap

    @staticmethod
    def _checked_vec(vec):

        table = [None] * VEC_COUNT
        if isinstance(vec, dict):
            for k, v in vec.items():
                k = _cfg_int("VEC index", k, 0, VEC_COUNT - 1)
                table[k] = _cfg_int(f"VEC[{k}]", v, 0, (1 << 16) - 1)
        else:
            entries = list(vec)
            if len(entries) > VEC_COUNT:
                raise ConfigError(f"VEC has {len(entries)} entries, this machine "
                                  f"declares {VEC_COUNT} traps")
            for k, v in enumerate(entries):
                table[k] = _cfg_int(f"VEC[{k}]", v, 0, (1 << 16) - 1)
        return tuple(0 if v is None else v for v in table)

    @property
    def has_window(self):
        return self.winlo is not None

    @property
    def has_vec(self):
        return self.vec is not None

    def vector(self, k):

        return DEFAULT_VECTORS[k] if self.vec is None else self.vec[k]

    def vectors(self):

        return DEFAULT_VECTORS if self.vec is None else self.vec

    def window(self):

        return DEFAULT_WINDOW if self.winlo is None else (self.winlo, self.winhi)

    def check_for_program(self, codelen, where=""):

        if self.winlo is None:
            return None
        return window_error(self.winlo, self.winhi, codelen, where)

    def equivalent_to_default(self):

        return all(getattr(self, n) is None for n in self.__slots__)

    def set_fields(self):

        return [n for n in self.__slots__ if getattr(self, n) is not None]

    def as_dict(self):

        out = {}
        for name in self.__slots__:
            value = getattr(self, name)
            if value is not None:
                out[name] = list(value) if name == "vec" else value
        return out

    @classmethod
    def from_dict(cls, fields):

        if fields is None:
            return None
        if not isinstance(fields, dict):
            raise ConfigError(f"a configuration carried as data must be a dict of "
                              f"fields, got a {type(fields).__name__}")
        unknown = sorted(set(fields) - set(cls.__slots__))
        if unknown:
            raise ConfigError(f"configuration names constraints outside the field set: "
                              f"{unknown}")
        fields = dict(fields)
        vec = fields.get("vec")
        if isinstance(vec, dict):

            normalized = {}
            for k, v in vec.items():
                if isinstance(k, str):
                    if not k.isdigit():
                        raise ConfigError(f"the vector index {k!r} arriving as "
                                          f"carried data is not a non-negative "
                                          f"integer")
                    k = int(k)
                normalized[k] = v
            fields["vec"] = normalized
        return cls(**fields)

    def __eq__(self, other):
        if not isinstance(other, MachineConfig):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in self.__slots__)

    def __hash__(self):
        return hash(tuple(getattr(self, n) for n in self.__slots__))

    def __repr__(self):
        return ("MachineConfig(" + ", ".join(
            f"{n}={getattr(self, n)!r}" for n in self.__slots__
            if getattr(self, n) is not None) + ")")

def config_difference(before, after):

    if before == after:
        return None
    moved = [n for n in MachineConfig.__slots__
             if getattr(before, n) != getattr(after, n)]
    return (f"configuration moved under the machine: {moved} changed, so a tick "
            f"reached load-time input (before "
            f"{ {n: getattr(before, n) for n in moved} }, after "
            f"{ {n: getattr(after, n) for n in moved} })")

def fault_state_error(status, fault_reason, fault_addr=0, where=""):

    def outside(field, value, lo, hi):
        return f"{where}state field {field} is {value}, outside [{lo}, {hi}]"

    if not isinstance(fault_reason, int) or isinstance(fault_reason, bool):
        return (f"{where}state field fault_reason is {fault_reason!r}, which is not an "
                f"integer cause code")
    if not 0 <= fault_reason < (1 << FAULT_BITS):
        return outside("fault_reason", fault_reason, 0, (1 << FAULT_BITS) - 1)
    if fault_reason not in CAUSE_NAME:
        return (f"{where}state field fault_reason is {fault_reason}, which the "
                f"{len(FAULT_CAUSES)}-entry cause table does not assign")
    if not 0 <= fault_addr < (1 << FAULT_ADDR_BITS):
        return outside("fault_addr", fault_addr, 0, (1 << FAULT_ADDR_BITS) - 1)
    if fault_reason != 0 and status != 3:
        return (f"{where}fault cause {fault_reason} "
                f"({CAUSE_NAME.get(fault_reason, 'unknown')}) is recorded while status "
                f"is {status}, which is not the error status 3: a cause is recorded "
                f"exactly when the machine stopped for it")
    if status == 3 and fault_reason == 0:
        return (f"{where}status is the error status 3 while fault_reason is 0 (OK): an "
                f"error tick must name the cause it stopped for")
    return None

StateField = namedtuple("StateField", "name lo hi cells")

STATE_FIELDS = (
    StateField("r", 0, 255, 4),
    StateField("HL", 0, 65535, None),
    StateField("DE", 0, 65535, None),
    StateField("MB", 0, 65535, None),
    StateField("PC", 0, 65535, None),
    StateField("SP", 0, DATA_SIZE, None),
    StateField("C", 0, 1, None),
    StateField("Z", 0, 1, None),
    StateField("S", 0, 1, None),
    StateField("V", 0, 1, None),
    StateField("ipos", 0, None, None),
    StateField("tick", 0, None, None),
    StateField("status", 0, 3, None),
    StateField("fault_reason", 0, (1 << FAULT_BITS) - 1, None),
    StateField("fault_addr", 0, (1 << FAULT_ADDR_BITS) - 1, None),

    StateField("TDEPTH", 0, (1 << 8) - 1, None),
)

STATE_FIELD_NAMES = tuple(f.name for f in STATE_FIELDS)

FLAG_FIELDS = ("Z", "C", "S", "V")
FLAG_BITS_PACKED = {name: 1 << i for i, name in enumerate(FLAG_FIELDS)}

FLAG_WRITES = {
    "CLC": ("C",),
    "ADD": ("C", "Z", "S", "V"),
    "ADC": ("C", "Z", "S", "V"),
    "SUB": ("C", "Z", "S", "V"),
    "SBB": ("C", "Z", "S", "V"),
    "ADDI": ("C", "Z", "S", "V"),
    "SUBI": ("C", "Z", "S", "V"),
    "ADCI": ("C", "Z", "S", "V"),
    "CMP": ("C", "Z", "S", "V"),
    "NEG": ("C", "Z", "S"),
    "AND": ("Z",),
    "OR": ("Z",),
    "XOR": ("Z",),
    "MUL": ("C", "Z"),
    "MULH": ("Z",),
    "DIV": ("Z",),
    "MOD": ("Z",),
    "NOT": ("Z",),
    "ROL": ("C", "Z"),
    "ROR": ("C", "Z"),
    "SHL": ("C", "Z"),
    "SHR": ("C", "Z"),
    "TST": ("Z",),
    "ADD_HLDE": ("C",),
    "SUB_HLDE": ("C",),
    "IN": ("C",),
}

def flag_writes(row):

    declared = FLAG_WRITES.get(row["alu"], ())
    return tuple(f for f in FLAG_FIELDS if f in declared)

def flags_byte(values):

    out = 0
    for name, bit in FLAG_BITS_PACKED.items():
        out |= (int(values[name]) & 1) * bit
    return out

RECORD_IMAGES = ("CODE", "DATA")
RECORD_STREAMS = ("out", "inputs")
RECORD_BLOCK = "block"
RECORD_COMPONENTS = (STATE_FIELD_NAMES + RECORD_IMAGES + RECORD_STREAMS
                     + (RECORD_BLOCK,))

RecordBounds = namedtuple("RecordBounds",
                          "where out_cap code_size data_size inputs block")

CheckedRecord = namedtuple("CheckedRecord", "state code data out inputs")

def state_error(values, where=""):

    unknown = sorted(set(values) - set(STATE_FIELD_NAMES))
    if unknown:
        return (f"{where}state carries {unknown}, which this table declares no width "
                f"for: add the field to isa_table.STATE_FIELDS or stop carrying it")
    for f in STATE_FIELDS:
        if f.name not in values:
            return f"{where}state field {f.name} is missing"
        v = values[f.name]
        if f.cells is None:
            bad = _width_error(f, v, f.name, where)
            if bad is not None:
                return bad
            continue
        try:
            cells = list(v)
        except TypeError:
            return (f"{where}state field {f.name} is {v!r}, which is not the "
                    f"{f.cells} cells this machine declares")
        if len(cells) != f.cells:
            return (f"{where}state field {f.name} has {len(cells)} cells, this machine "
                    f"declares {f.cells}")
        for i, cell in enumerate(cells):
            bad = _width_error(f, cell, f"{f.name}[{i}]", where)
            if bad is not None:
                return bad
    return fault_state_error(values["status"], values["fault_reason"],
                             values["fault_addr"], where)

def _width_error(f, v, label, where):

    try:
        bad = v < f.lo if f.hi is None else not f.lo <= v <= f.hi
    except TypeError:
        bad = True
    if bad:
        hi = "unbounded" if f.hi is None else f.hi
        return (f"{where}state field {label} is {v}, outside [{f.lo}, {hi}]")
    return None

def state_shape(values, where=""):

    unknown = sorted(set(values) - set(STATE_FIELD_NAMES))
    if unknown:
        return (f"{where}state carries {unknown}, which this table declares no width "
                f"for: add the field to isa_table.STATE_FIELDS or stop carrying it")
    for f in STATE_FIELDS:
        if f.name not in values:
            return f"{where}state field {f.name} is missing"
        v = values[f.name]
        if f.cells is None:
            if isinstance(v, bool) or not isinstance(v, int):
                return (f"{where}state field {f.name} is {v!r}, which is not the integer "
                        f"this machine stores")
        else:
            try:
                cells = list(v)
            except TypeError:
                return (f"{where}state field {f.name} is {v!r}, which is not the "
                        f"{f.cells} cells this machine declares")
            if len(cells) != f.cells:
                return (f"{where}state field {f.name} has {len(cells)} cells, this "
                        f"machine declares {f.cells}")
            if any(isinstance(c, bool) or not isinstance(c, int) for c in cells):
                return (f"{where}state field {f.name} is {cells!r}, and every cell of it "
                        f"is an integer")
    return None

def record_from(machine, readers, where=""):

    missing = [name for name in RECORD_COMPONENTS if name not in readers]
    if missing:
        raise ValueError(f"{where}records no reader for {missing}, so this machine's "
                         f"state is not fully recorded")
    extra = sorted(set(readers) - set(RECORD_COMPONENTS))
    if extra:
        raise ValueError(f"{where}records {extra}, which is not a component of a state "
                         f"record")
    return {name: readers[name](machine) for name in RECORD_COMPONENTS}

def check_record(snap, bounds):

    where = bounds.where
    if not isinstance(snap, dict):
        raise ValueError(f"{where}a state record is the dict of components "
                         f"record_state() publishes, not a {type(snap).__name__}")
    missing = [name for name in RECORD_COMPONENTS if name not in snap]
    if missing:
        raise ValueError(f"{where}record has no component {missing}: a record is "
                         f"refused rather than installed with those fields left at "
                         f"their reset values")
    extra = sorted(set(snap) - set(RECORD_COMPONENTS))
    if extra:
        raise ValueError(f"{where}record carries {extra}, which is not a component of a "
                         f"state record")
    state = {name: snap[name] for name in STATE_FIELD_NAMES}
    bad = state_shape(state, where)
    if bad is not None:
        raise ValueError(bad)
    state = {f.name: (list(state[f.name]) if f.cells is not None else state[f.name])
             for f in STATE_FIELDS}

    code = _record_image(snap["CODE"], "CODE", bounds.code_size, where)
    data = _record_image(snap["DATA"], "DATA", bounds.data_size, where)
    out = _record_bytes(snap["out"], "out", where)
    if len(out) > bounds.out_cap:
        raise ValueError(f"{where}record's output stream is {len(out)} bytes, above "
                         f"this machine's output capacity {bounds.out_cap}: the stream "
                         f"a machine emitted is bounded by the capacity it was built "
                         f"with")
    given = _record_bytes(bounds.inputs, "input stream", where)
    taken = _record_bytes(snap["inputs"], "inputs", where)
    if state["ipos"] > len(taken):
        raise ValueError(f"{where}record's ipos is {state['ipos']}, past the end of the "
                         f"{len(taken)}-byte input stream it was recorded with")
    if taken != given:
        raise ValueError(f"{where}record consumed {state['ipos']} bytes of a "
                         f"{len(taken)}-byte input stream and this machine's stream is "
                         f"{len(given)} bytes: the cursor means the same byte only over "
                         f"the same stream")
    block = snap[RECORD_BLOCK]
    if not isinstance(block, dict):
        raise ValueError(f"{where}record's configuration block is {block!r}, which is "
                         f"not the dict of declared fields as_dict() publishes")
    want = MachineConfig.from_dict(block)
    moved = sorted(n for n in MachineConfig.__slots__
                   if getattr(want, n) != getattr(bounds.block, n))
    if moved:
        raise ValueError(
            f"{where}record was taken under a different configuration: {moved} "
            f"{'differs' if len(moved) == 1 else 'differ'} (record "
            f"{ {n: getattr(want, n) for n in moved} }, this machine "
            f"{ {n: getattr(bounds.block, n) for n in moved} })")
    return CheckedRecord(state=state, code=code, data=data, out=out, inputs=given)

def publish_record(machine, readers, bounds):

    snap = record_from(machine, readers, bounds.where)
    got = check_record(snap, bounds)
    rec = dict(got.state)
    rec["CODE"], rec["DATA"] = got.code, got.data
    rec["out"], rec["inputs"] = got.out, got.inputs
    rec[RECORD_BLOCK] = dict(snap[RECORD_BLOCK])
    return {name: rec[name] for name in RECORD_COMPONENTS}

def _record_bytes(value, name, where):

    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, list, tuple)):
        try:
            return bytes(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}record's {name} is not a byte stream: {e}") from e
    raise ValueError(f"{where}record's {name} is a {type(value).__name__}, which is not "
                     f"a byte stream")

def _record_image(value, name, size, where):

    data = _record_bytes(value, name, where)
    if len(data) != size:
        raise ValueError(f"{where}record's {name} image is {len(data)} bytes, and this "
                         f"machine's {name} region is {size} bytes: an image is "
                         f"recorded whole, because a prefix would install the rest of "
                         f"the region at its reset value")
    return data

def _illegal_state_values(f):

    out = [f.lo - 1]
    if f.cells is None:
        out.append("x" if f.hi is None else f.hi + 1)
    else:
        out.append([f.hi + 1] + [0] * (f.cells - 1))
        out.append([0] * (f.cells - 1))
    return out

def check_state_table():

    if len(set(STATE_FIELD_NAMES)) != len(STATE_FIELD_NAMES):
        raise DecodeTableError("the state table names a field twice")
    legal = {f.name: ([f.lo] * f.cells if f.cells else f.lo) for f in STATE_FIELDS}
    if state_error(dict(legal)) is not None:
        raise DecodeTableError("the state table refuses its own reset values: "
                               f"{state_error(dict(legal))}")
    for f in STATE_FIELDS:
        for value in _illegal_state_values(f):
            probe = dict(legal)
            probe[f.name] = value
            msg = state_error(probe)
            if msg is None or f.name not in msg:
                raise DecodeTableError(f"the state table accepts {f.name}={value!r} "
                                       f"without refusing it by name")
    short = dict(legal)
    del short["SP"]
    if "SP" not in (state_error(short) or ""):
        raise DecodeTableError("the state table does not refuse a missing field by name")
    if "extra_field" not in (state_error(dict(legal, extra_field=1)) or ""):
        raise DecodeTableError("the state table accepts a field it declares no width for")
    if state_shape(dict(legal, SP=legal["SP"] + 1)) is not None:
        raise DecodeTableError("the shape check refused a value above a width, and "
                               "check_state() is the gate that owns that refusal")
    if state_shape(dict(legal, r=[0, 0, 0])) is None:
        raise DecodeTableError("the shape check accepted a register file of the wrong "
                               "cell count")
    pairing = dict(legal, status=3, fault_reason=0)
    if state_error(pairing) is None:
        raise DecodeTableError("the state table accepts status 3 with no cause named")
    if set(RECORD_COMPONENTS) != set(STATE_FIELD_NAMES) | set(RECORD_IMAGES) \
            | set(RECORD_STREAMS) | {RECORD_BLOCK}:
        raise DecodeTableError("the record's component list is not the state table plus "
                               "the images, streams and block it names")
    if len(RECORD_COMPONENTS) != len(set(RECORD_COMPONENTS)):
        raise DecodeTableError("the record lists a component twice")
    if "oplen" in RECORD_COMPONENTS:
        raise DecodeTableError("a record carries its output stream as `out`, so an "
                               "`oplen` component would be a second copy of the same quantity "
                               "under the name the paths that keep a cursor use")
    declared = {n.lower() for n in CONFIG_NAMES} | {"out_cap", "tick_budget", "tb",
                                                    "config", "cfg"}
    leaked = sorted({n.lower() for n in RECORD_COMPONENTS} & declared)
    if leaked:
        raise DecodeTableError(f"a record component names load-time configuration "
                               f"{leaked}")
    check_flag_table()
    return True

def check_flag_table():

    if set(FLAG_FIELDS) - set(STATE_FIELD_NAMES):
        raise DecodeTableError("a condition flag is not a field of the state table: "
                               f"{sorted(set(FLAG_FIELDS) - set(STATE_FIELD_NAMES))}")
    for f in STATE_FIELDS:
        if f.name in FLAG_FIELDS and (f.lo, f.hi, f.cells) != (0, 1, None):
            raise DecodeTableError(f"flag {f.name} is not declared as one bit")
    if len(set(FLAG_FIELDS)) != len(FLAG_FIELDS):
        raise DecodeTableError("the flag list names a flag twice")
    if sorted(FLAG_BITS_PACKED) != sorted(FLAG_FIELDS):
        raise DecodeTableError("the GETF packing and the flag list name different flags")
    unknown = sorted(set(FLAG_WRITES) - set(ALU_ID))
    if unknown:
        raise DecodeTableError(f"the flag declaration names selectors the ALU vocabulary "
                               f"does not have: {unknown}")
    for alu, flags in FLAG_WRITES.items():
        extra = sorted(set(flags) - set(FLAG_FIELDS))
        if extra:
            raise DecodeTableError(f"{alu} declares flags that are not condition flags: "
                                   f"{extra}")
        if not flags:
            raise DecodeTableError(f"{alu} declares no flag, so it is not a writer")
    if flags_byte({n: 0 for n in FLAG_FIELDS}) != 0:
        raise DecodeTableError("the reset flags do not pack as a zero byte")
    for name, bit in FLAG_BITS_PACKED.items():
        got = flags_byte({n: int(n == name) for n in FLAG_FIELDS})
        if got != bit:
            raise DecodeTableError(f"flag {name} packs as {got}, the byte gives it {bit}")
    return True

check_state_table()

def single_row(op):

    row = SINGLE.get(op)
    if row is None:
        raise UndefinedCode(f"single-byte code point {op:#04x} is not assigned")
    return row

def escape_row(sub):

    row = ESCAPE.get(sub)
    if row is None:
        raise UndefinedCode(f"escape subcode {sub:#04x} is not assigned")
    return row

def length(op):

    return 1 + single_row(op)["l"]

def escape_length(sub):

    return PREFIX_BYTES + escape_row(sub)["l"]

def operands(op):

    r = single_row(op)
    return r["s0"], r["s1"]

def escape_operands(sub):

    r = escape_row(sub)
    return r["s0"], r["s1"]

def mnemonic(op):

    return single_row(op)["mnem"]

def escape_mnemonic(sub):

    return escape_row(sub)["mnem"]

def operand_kind(op):

    return single_row(op)["kind"]

def escape_operand_kind(sub):

    return escape_row(sub)["kind"]

def instruction_length(image, at=0):

    if at >= len(image):
        return None
    op = image[at]
    if op == ESCAPE_PREFIX:
        if at + 1 >= len(image):
            return None
        sub = image[at + 1]
        return None if sub not in ESCAPE else PREFIX_BYTES + ESCAPE[sub]["l"]
    return None if op not in SINGLE else 1 + SINGLE[op]["l"]

def unassigned_single():

    return tuple(o for o in range(256) if o != ESCAPE_PREFIX and o not in SINGLE)

def unassigned_escape():

    return tuple(s for s in range(256) if s not in ESCAPE)

def single_rom():

    alu = [BAD] * 256
    s0 = [0] * 256
    s1 = [0] * 256
    ln = [1] * 256
    ln[ESCAPE_PREFIX] = PREFIX_BYTES
    for op, r in SINGLE.items():
        alu[op], s0[op], s1[op], ln[op] = ALU_ID[r["alu"]], r["s0"], r["s1"], 1 + r["l"]
    return alu, s0, s1, ln

def escape_rom():

    alu = [BAD] * 256
    s0 = [0] * 256
    s1 = [0] * 256
    lx = [0] * 256
    for sub, r in ESCAPE.items():
        alu[sub], s0[sub], s1[sub] = ALU_ID[r["alu"]], r["s0"], r["s1"]
        lx[sub] = r["l"]
    return alu, s0, s1, lx

def check_structure():

    for space, table in (("single", SINGLE), ("escape", ESCAPE)):
        for code, r in sorted(table.items()):
            if not 0 <= code <= 255:
                raise DecodeTableError(f"{space} code {code} is outside one byte")
            if not isinstance(r["mnem"], str) or not r["mnem"]:
                raise DecodeTableError(f"{space} code {code:#04x} has no mnemonic")
            if r["alu"] not in ALU_ID:
                raise DecodeTableError(f"{space} code {code:#04x} selects unknown ALU "
                                       f"{r['alu']!r}")
            if r["alu"] == "BAD":
                raise DecodeTableError(f"{space} code {code:#04x} is assigned BAD")
            for field in ("s0", "s1"):
                if not 0 <= r[field] <= 3:
                    raise DecodeTableError(f"{space} code {code:#04x} {field}="
                                           f"{r[field]} is outside 0..3")
            if not 0 <= r["l"] <= 2:
                raise DecodeTableError(f"{space} code {code:#04x} has l={r['l']}, "
                                       f"outside 0..2")
    if ESCAPE_PREFIX in SINGLE:
        raise DecodeTableError("the escape prefix is not an instruction of its own")
    esc_alu = {r["alu"] for r in ESCAPE.values()}
    missing = sorted(esc_alu - (set(ESC_EOP_NAMES) - {"BAD"}))
    if missing:
        raise DecodeTableError("escape rows select effective opcodes the Triton "
                               f"vocabulary does not name: {missing}")
    unused = sorted((set(ESC_EOP_NAMES) - {"BAD"}) - esc_alu)
    if unused:
        raise DecodeTableError(f"Triton escape vocabulary names no escape row: {unused}")
    clash = [s for s in V4_RESERVED if s in ESCAPE]
    if clash:
        raise DecodeTableError("held subcodes are already assigned: "
                               f"{[hex(c) for c in clash]}")
    if len(V4_RESERVED) != len(set(V4_RESERVED)):
        raise DecodeTableError("v4 reserved subcodes are not distinct")
    if K != len(set(ALU_NAMES)):
        raise DecodeTableError("ALU vocabulary has duplicate names")
    if len(ESC_EOP_NAMES) != len(set(ESC_EOP_NAMES)):
        raise DecodeTableError("Triton escape vocabulary has duplicate names")
    if max(ESC_EOP_ID.values()) > 0x1FF:
        raise DecodeTableError("Triton escape effective opcodes exceed the 9-bit range")
    check_fault_table()
    return True

def check_config_table():

    if tuple(n.upper() for n in MachineConfig.__slots__) != CONFIG_NAMES:
        raise DecodeTableError("the configuration block's fields and CONFIG_NAMES "
                               "disagree, so a survey of one misses a field of the "
                               "other")
    if set(CONFIG_WIDTHS) != set(CONFIG_NAMES):
        raise DecodeTableError("a configuration name has no declared width")
    if CONFIG_WIDTHS["VEC"] != 16 or VEC_COUNT != 16:
        raise DecodeTableError("the vector entries and the trap count disagree about "
                               "how wide a vector is")
    if len(DEFAULT_VECTORS) != VEC_COUNT or any(v != 0 for v in DEFAULT_VECTORS):
        raise DecodeTableError("the default vector table does not declare one "
                               "unregistered entry per trap")
    if DEFAULT_WINDOW != (0, 0) or DEFAULT_WINDOW[1] < DEFAULT_WINDOW[0]:
        raise DecodeTableError("the default window is not the empty span")
    empty = MachineConfig()
    if not empty.equivalent_to_default():
        raise DecodeTableError("an empty configuration block is not the default one")
    if empty.window() != DEFAULT_WINDOW or empty.vectors() != DEFAULT_VECTORS:
        raise DecodeTableError("an absent field does not resolve to its default")
    if any(empty.vector(k) != 0 for k in range(VEC_COUNT)):
        raise DecodeTableError("an absent vector table registers a handler")
    try:
        MachineConfig(winlo=8, winhi=4)
    except ConfigError:
        pass
    else:
        raise DecodeTableError("a reversed window is not refused at load, so a reversed "
                               "bound pair has no gate")
    try:
        MachineConfig(codelen=1, winlo=0, winhi=0x2000)
    except ConfigError as e:
        if "0x2000" not in str(e) or "program of 1 bytes" not in str(e):
            raise DecodeTableError("a window past a declared CODELEN is refused without "
                                   f"naming the span and the program: {e}")
    else:
        raise DecodeTableError("a window past a declared CODELEN is accepted, so a span "
                               "naming cells no program occupies has no gate")
    if MachineConfig(codelen=1, winlo=0, winhi=1).check_for_program(1) is not None:
        raise DecodeTableError("a window ending at the last program byte is refused, "
                               "though WINHI is exclusive")
    if MachineConfig(winlo=0, winhi=0x2000).check_for_program(1) is None:
        raise DecodeTableError("a block that leaves CODELEN absent states no window bound "
                               "for the machine that resolves it to the loaded length")
    if MachineConfig().check_for_program(0) is not None:
        raise DecodeTableError("an undeclared window is refused against some program")
    if MachineConfig().splim is not None:
        raise DecodeTableError("an absent stack floor reads as a declared one, so the "
                               "machine's own default would never resolve")
    if MachineConfig(splim=DATA_SIZE).splim != DATA_SIZE:
        raise DecodeTableError("the declared stack floor does not survive validation")
    for bad_floor in (-1, DATA_SIZE + 1, DATA_SIZE * 2):
        try:
            MachineConfig(splim=bad_floor)
        except ConfigError:
            continue
        raise DecodeTableError(f"a stack floor of {bad_floor} is accepted, though it is "
                               f"outside the DATA span the floor is bounded by")
    if MachineConfig().entry is not None:
        raise DecodeTableError("an absent boot address reads as a declared one, so the "
                               "machine's own default would never resolve")
    if MachineConfig(entry=0x0100).entry != 0x0100 or DEFAULT_ENTRY != 0:
        raise DecodeTableError("the declared boot address does not survive validation, "
                               "or the default is not the address the machine boots")
    for bad_entry in (-1, 1 << 16):
        try:
            MachineConfig(entry=bad_entry)
        except ConfigError:
            continue
        raise DecodeTableError(f"a boot address of {bad_entry} is accepted, though it "
                               f"is outside the 16 bits the field declares")
    if MachineConfig().tdlim is not None:
        raise DecodeTableError("an absent trap-depth limit reads as a declared one, so "
                               "the machine's own default would never resolve")
    if MachineConfig(tdlim=9).tdlim != 9 or DEFAULT_TDLIM != 64:
        raise DecodeTableError("the declared trap-depth limit does not survive "
                               "validation, or the default is not the 64 traps an "
                               "absent block declares")
    for bad in (-1, 1 << 8):
        try:
            MachineConfig(tdlim=bad)
        except ConfigError:
            continue
        raise DecodeTableError(f"a trap-depth limit of {bad} is accepted, though it is "
                               f"outside the 8 bits the field declares")
    try:
        MachineConfig(codelen=4, entry=4)
    except ConfigError as e:
        if "ENTRY=4" not in str(e) or "4 bytes" not in str(e):
            raise DecodeTableError("a boot address at the program end is refused without "
                                   f"naming the address and the program: {e}")
    else:
        raise DecodeTableError("a boot address at the program end is accepted, so a "
                               "machine whose first fetch would fault has no gate")
    for bad in (0, 3, 6, OUT_CAP_LIMIT):
        try:
            check_capacity(bad, OUT_CAP_LIMIT - 1)
        except ValueError:
            continue
        raise DecodeTableError(f"a capacity of {bad} is accepted, though it is neither "
                               f"a positive power of two within the output buffer")
    return True

def check_fault_table():

    names = [name for name, _d in FAULT_CAUSES]
    if len(names) != len(set(names)):
        raise DecodeTableError("the cause table has a duplicate cause name")
    if [fault_code(n) for n in names] != list(FAULT_CODES):
        raise DecodeTableError("the cause table is not dense from 0")
    if names[0] != "OK" or CAUSE["OK"] != 0:
        raise DecodeTableError("cause 0 must be OK")
    if max(FAULT_CODES) > 255:
        raise DecodeTableError("the cause table exceeds the 8-bit fault_reason field")
    if len(FAULT_SITE_NAMES) != len(set(FAULT_SITE_NAMES)):
        raise DecodeTableError("the fault precedence list names a site twice")
    for site, cause in FAULT_SITE_ORDER:
        if cause not in CAUSE:
            raise DecodeTableError(f"fault site {site} names unknown cause {cause!r}")
        if cause == "OK":
            raise DecodeTableError(f"fault site {site} names OK, which is not a fault")
    for name in CAUSES_AWAITING_FEATURE:
        if name not in CAUSE:
            raise DecodeTableError(f"waiting list names unknown cause {name!r}")
    if BANK_SITE_ANCHOR not in SITE_RANK:
        raise DecodeTableError("the bank sites are stated to rank after "
                               f"{BANK_SITE_ANCHOR!r}, which the precedence list does "
                               f"not name")
    for site, cause in BANK_FAULT_SITES:
        if cause not in CAUSE:
            raise DecodeTableError(f"bank site {site} names unknown cause {cause!r}")
        if cause == "OK":
            raise DecodeTableError(f"bank site {site} names OK, which is not a fault")
        if site not in SITE_RANK:
            raise DecodeTableError(f"bank site {site} is not a row of the precedence "
                                   f"list, which is the list a path stacks into")
    full = full_fault_site_order()
    if full != FAULT_SITE_ORDER:
        raise DecodeTableError("the order that includes the bank sites is not the order "
                               "a path stacks one signal per site into, so a stacked "
                               "cause could be named at a rank the rule does not state")
    keep = SITE_RANK[BANK_SITE_ANCHOR] + 1
    if FAULT_SITE_ORDER[keep:keep + len(BANK_FAULT_SITES)] != BANK_FAULT_SITES:
        raise DecodeTableError(f"the bank sites are not stacked directly after "
                               f"{BANK_SITE_ANCHOR!r}: "
                               f"{[s for s, _c in FAULT_SITE_ORDER[keep:keep + 2]]}")

    if not SITE_RANK["TRAP_UNREG"] < SITE_RANK["TRAP_DEPTH"] < SITE_RANK["STACK_PUSH"]:
        raise DecodeTableError("the EXT chain tests the vector, then the depth counter, "
                               "then the four-slot push room, so the precedence list "
                               "must rank TRAP_UNREG < TRAP_DEPTH < STACK_PUSH: "
                               f"{[s for s in FAULT_SITE_NAMES if s in (
                                   'TRAP_UNREG', 'TRAP_DEPTH', 'STACK_PUSH')]}")
    if not SITE_RANK["TRAP_UNBALANCED"] < SITE_RANK["STACK_POP"] \
            < SITE_RANK["TRAP_FRAME"]:
        raise DecodeTableError("the TRAPRET chain tests the depth counter, then the "
                               "four-slot pop room, then the frame tag, so the "
                               "precedence list must rank TRAP_UNBALANCED < STACK_POP "
                               "< TRAP_FRAME: "
                               f"{[s for s in FAULT_SITE_NAMES if s in (
                                   'TRAP_UNBALANCED', 'STACK_POP', 'TRAP_FRAME')]}")
    order = full_fault_cause_order()
    if order.index(CAUSE["BANK_OOB"]) > order.index(CAUSE["BANK_BUSY"]) \
            or order.index(CAUSE["BANK_BUSY"]) > order.index(CAUSE["DATA_OOB"]):
        raise DecodeTableError("the bank causes do not rank ahead of the address cause: "
                               f"{[CAUSE_NAME[c] for c in order]}")
    return True

def full_fault_cause_order():

    return [CAUSE[cause] for _site, cause in full_fault_site_order()]

check_structure()

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import circuit_torch as CT

    print(f"table: {len(SINGLE)} single-byte codes, {len(ESCAPE)} escape subcodes")
    miss = unassigned_single()
    print("  unassigned single-byte codes:", len(miss), [hex(o) for o in miss[:4]])

    live = tuple(t.cpu().tolist() for t in (CT._ALU, CT._S0, CT._S1, CT._LEN))
    live2 = tuple(t.cpu().tolist() for t in (CT._ALU2, CT._S02, CT._S12, CT._LX2))
    want = single_rom()
    want2 = escape_rom()

    bad = []
    for k, nm in enumerate(("alu", "s0", "s1", "ln")):
        if live[k] != list(want[k]):
            for op in range(256):
                if live[k][op] != want[k][op]:
                    bad.append(("single", nm, hex(op), live[k][op], want[k][op]))
    for k, nm in enumerate(("alu", "s0", "s1", "lx")):
        if live2[k] != list(want2[k]):
            for sub in range(256):
                if live2[k][sub] != want2[k][sub]:
                    bad.append(("escape", nm, hex(sub), live2[k][sub], want2[k][sub]))
    print(f"\ntable vs live tensor ROM (assignment, length, s0/s1): {len(bad)} disagreements")
    for b in bad[:12]:
        print("   ", b)

    free = len(unassigned_escape())
    held = [s for s in V4_RESERVED if s in ESCAPE]
    print(f"\nbank family subcodes 0xB0-0xBD assigned: "
          f"{sum(1 for s in range(0xB0, 0xBE) if s in ESCAPE)} of 14")
    print(f"held subcodes {V4_RESERVED[0]:#04x}-{V4_RESERVED[-1]:#04x} claimed by the "
          f"table: {[hex(h) for h in held] or 'none'}")
    print("free escape subcodes:", free)
    print("VERDICT:", "table reproduces the live decode" if not bad
          else "TABLE DIVERGES FROM LIVE DECODE")