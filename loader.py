"""Loader for NCP-8: source text to a placed image, with symbols and metadata.

Two-pass assembler front end adding what `golden_sim.asm` does not do: directives
(`.org`, `.byte`, `.word`, `.ascii`, `.equ`), an expression parser for absolute and
symbol-relative operands, a symbol table, and the configuration the machine obeys but
cannot write - the trap vector entries, the self-modification window bounds, and the
length of the placed content. A declaration costs no byte of the image: `vectors=` and
`window=` are handed to the machine beside it, `config()` builds the block, and every
address in the image is program content. `entry=` states where the load's `main` is;
the machine boots at CODE[0], so a program that starts elsewhere says so with a jump.

Every refusal is a `LoaderError`, which is deliberately not a `MachineError`: a program
that never assembled must not be catchable by a handler written for a program that ran
and faulted. Placements are checked before anything is written, so a collision between
two declarations is reported rather than resolved by whichever one came last.

Run: python3 loader.py             (self-check)
"""
from __future__ import annotations

import re

import disasm
import isa_forms
import isa_table as ISA
from golden_sim import AssemblyError, CODE_SIZE, asm

VEC_COUNT = ISA.VEC_COUNT
BOUND_BITS = 16

class LoaderError(AssemblyError):

    pass

_TOKEN = re.compile(r"\s*(?:(?P<num>0[xXbBoO][0-9a-fA-F_]+|[0-9][0-9_]*)"
                    r"|(?P<name>[A-Za-z_]\w*)"
                    r"|(?P<op><<|>>|//|[-+*%&|^~(),/])"
                    r"|(?P<bad>.))")
_FUNCS = {"low": lambda v: v & 0xFF, "high": lambda v: (v >> 8) & 0xFF, "abs": abs}

class _Tok:
    __slots__ = ("kind", "text")

    def __init__(self, kind, text):
        self.kind, self.text = kind, text

    def __repr__(self):
        return f"{self.kind}({self.text!r})"

def _lex(expr, lineno):
    out, pos = [], 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if m is None:
            raise LoaderError(f"cannot tokenize {expr[pos:pos + 8]!r} in {expr!r}", lineno)
        pos = m.end()
        for kind in ("num", "name", "op"):
            if m.group(kind) is not None:
                out.append(_Tok(kind, m.group(kind)))
                break
        else:
            raise LoaderError(f"unexpected character {m.group('bad')!r} in {expr!r}", lineno)
    out.append(_Tok("end", ""))
    return out

class _Expr:

    def __init__(self, toks, text, lineno, symbols):
        self.toks, self.i = toks, 0
        self.text, self.lineno, self.symbols = text, lineno, symbols

    def parse(self):
        v = self.or_()
        if self.peek().kind != "end":
            raise LoaderError(f"trailing {self.peek().text!r} in {self.text!r}", self.lineno)
        return v

    def peek(self):
        return self.toks[self.i]

    def take(self, text):
        t = self.peek()
        if t.kind == "op" and t.text == text:
            self.i += 1
            return True
        return False

    def expect(self, text):
        if not self.take(text):
            raise LoaderError(f"expected {text!r} in {self.text!r}", self.lineno)

    def or_(self):
        v = self.xor()
        while self.take("|"):
            v |= self.xor()
        return v

    def xor(self):
        v = self.and_()
        while self.take("^"):
            v ^= self.and_()
        return v

    def and_(self):
        v = self.shift()
        while self.take("&"):
            v &= self.shift()
        return v

    def shift(self):
        v = self.add()
        while True:
            if self.take("<<"):
                v <<= self.add()
            elif self.take(">>"):
                s = self.add()
                if s < 0:
                    raise LoaderError(f"negative right shift in {self.text!r}", self.lineno)
                v >>= s
            else:
                return v

    def add(self):
        v = self.mul()
        while True:
            if self.take("+"):
                v += self.mul()
            elif self.take("-"):
                v -= self.mul()
            else:
                return v

    def mul(self):
        v = self.unary()
        while True:
            if self.take("*"):
                v *= self.unary()
            elif self.take("//"):
                d = self.unary()
                if d == 0:
                    raise LoaderError(f"division by zero in {self.text!r}", self.lineno)
                v //= d
            elif self.take("%"):
                d = self.unary()
                if d == 0:
                    raise LoaderError(f"division by zero in {self.text!r}", self.lineno)
                v %= d
            elif self.take("/"):
                raise LoaderError(f"'/' yields a float, use // in {self.text!r}", self.lineno)
            else:
                return v

    def unary(self):
        if self.take("+"):
            return self.unary()
        if self.take("-"):
            return -self.unary()
        if self.take("~"):
            return ~self.unary()
        return self.primary()

    def primary(self):
        t = self.peek()
        if t.kind == "num":
            self.i += 1
            return int(t.text.replace("_", ""), 0)
        if t.kind == "name":
            self.i += 1
            if self.take("("):
                if t.text not in _FUNCS:
                    raise LoaderError(f"unknown function {t.text!r} in {self.text!r}",
                                      self.lineno)
                arg = self.or_()
                self.expect(")")
                return _FUNCS[t.text](arg)
            if t.text in self.symbols:
                return self.symbols[t.text]
            raise LoaderError(f"undefined symbol {t.text!r} in {self.text!r}", self.lineno)
        if t.kind == "op" and t.text == "(":
            self.i += 1
            v = self.or_()
            self.expect(")")
            return v
        raise LoaderError(f"cannot parse {t.text!r} in {self.text!r}", self.lineno)

class Symbols:

    def __init__(self):
        self._addr = {}
        self._const = {}

    def add(self, name, value, lineno, const):
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise LoaderError(f"invalid symbol name {name!r}", lineno)
        if name in self._addr or name in self._const:
            kind = "constant" if name in self._const else "label"
            raise LoaderError(f"symbol {name!r} defined twice (already a {kind})", lineno)
        (self._const if const else self._addr)[name] = int(value)

    def value(self, name):
        if name in self._addr:
            return self._addr[name]
        if name in self._const:
            return self._const[name]
        raise LoaderError(f"undefined symbol {name!r}")

    def is_address(self, name):
        return name in self._addr

    def flat(self):
        out = dict(self._addr)
        out.update(self._const)
        return out

    def as_dict(self):
        return self.flat()

    def names(self):
        return set(self._addr) | set(self._const)

    def kind_of(self, name):
        if name in self._const:
            return "constant"
        if name in self._addr:
            return "address"
        raise LoaderError(f"undefined symbol {name!r}")

    def __contains__(self, name):
        return name in self.names()

def expression_names(expr):

    return {m.group("name") for m in _TOKEN.finditer(str(expr)) if m.group("name")}

def evaluate(expr, symbols, lineno, *, forward=None):

    text = str(expr).strip()
    if not text:
        raise LoaderError("empty expression", lineno)
    try:
        toks = _lex(text, lineno)
    except LoaderError as e:
        raise e if e.line is not None else LoaderError(e.msg, lineno)
    try:
        return _Expr(toks, text, lineno, symbols).parse()
    except LoaderError as e:
        if e.line is not None:
            raise
        raise LoaderError(e.msg, lineno)

_SHAPE_ENCODING = isa_forms.shape_info()

_FRAME = "[HL+-i8]"
_PASSTHROUGH = ("r", "rcanon", "HL", "DE", "SP", "[HL]", "[DE]")
_FRAME_RE = re.compile(r"^\[HL(?:([+-])([^\]]+))?\]$")

def _fits(kind, text):

    return isa_forms.fits(kind, text, value_loose=True)

def match_line(name, args, lineno, text):

    cands = [s for s in isa_forms.FORMS.get(name, ()) if len(s) == len(args)
             and all(_fits(k, a) for k, a in zip(s, args))]
    if len(cands) > 1:
        raise LoaderError(f"{text!r}: matches several encodings {sorted(cands)}, the "
                          "text is ambiguous", lineno)
    if not cands:
        _reason, pos, kind = isa_forms.blame(name, args)
        detail = None
        if kind in ("a16", "i16") and args[pos].strip() in isa_forms.RESERVED:
            detail = f"{args[pos].strip()!r} names a register or pointer, not a target"
        raise LoaderError(isa_forms.refusal(name, args, text, detail), lineno)
    return cands[0], [a.strip() for a in args]

def shape_size(shape_name):

    return _SHAPE_ENCODING[shape_name]

def _render_operand(kind, text, symbols, lineno, name):

    if kind in _PASSTHROUGH:
        return text.strip()
    if kind == _FRAME:
        m = _FRAME_RE.match(text.replace(" ", ""))
        if not m:
            raise LoaderError(f"{name} needs an [HL+i8] address operand, {text!r} is "
                              "not one", lineno)
        sign, num = m.group(1), m.group(2)
        if num is None:
            return "[HL]"
        _reject_address_symbols(num, symbols, lineno, name, "frame offset")
        v = evaluate(num, symbols.flat(), lineno)
        if sign == "-":
            v = -v
        if not -128 <= v <= 127:
            raise LoaderError(f"{name} frame offset {v} is outside -128..127 "
                              f"in {text!r}", lineno)
        return "[HL]" if v == 0 else f"[HL{v:+d}]"
    if kind in ("i8", "soff", "k"):
        _reject_address_symbols(text, symbols, lineno, name, kind)
    v = evaluate(text, symbols.flat(), lineno)
    if kind in ("a16", "i16"):
        if not 0 <= v <= 0xFFFF:
            raise LoaderError(f"{name} address {v} is outside 0..65535 in {text!r}",
                              lineno)
        return f"0x{v:04X}"
    if kind == "i8":
        if not 0 <= v <= 0xFF:
            raise LoaderError(f"{name} immediate {v} is outside 0..255 in {text!r}",
                              lineno)
        return f"0x{v:02X}"
    if kind == "k":
        if not 0 <= v <= 0xFF:
            raise LoaderError(f"{name} trap number {v} is outside 0..255 in {text!r}",
                              lineno)
        return f"0x{v:02X}"
    if kind == "soff":
        if not -128 <= v <= 127:
            raise LoaderError(f"{name} needs a signed 8-bit value, {v} is outside "
                              f"-128..127 in {text!r}", lineno)
        return str(v)
    raise LoaderError(f"internal: cannot render operand kind {kind!r}", lineno)

def _reject_address_symbols(text, symbols, lineno, name, slot):

    for ref in sorted(expression_names(text)):
        if ref in symbols and symbols.is_address(ref):
            raise LoaderError(f"{name} needs a numeric value in the {slot} slot, but "
                              f"{ref!r} is a label (a code address) in {text!r}", lineno)

_LABEL = re.compile(r"^([A-Za-z_]\w*):$")
_DIRECTIVES = {".org", ".byte", ".word", ".ascii", ".equ"}
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "\\": "\\", '"': '"'}

def _strip_comment(raw):

    out, i, in_str = [], 0, False
    while i < len(raw):
        ch = raw[i]
        if in_str:
            if ch == "\\":
                out.append(ch)
                if i + 1 < len(raw):
                    out.append(raw[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == ";":
            break
        out.append(ch)
        i += 1
    return "".join(out).strip()

def _split_args(s):

    out, buf, i, in_str = [], [], 0, False
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == "\\":
                buf.append(ch)
                if i + 1 < len(s):
                    buf.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == ",":
            out.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf).strip())
    return out

class _Item:
    __slots__ = ("kind", "payload", "lineno", "text")

    def __init__(self, kind, payload, lineno, text):
        self.kind, self.payload = kind, payload
        self.lineno, self.text = lineno, text

    def __repr__(self):
        return f"<{self.kind} {self.payload!r} line {self.lineno}>"

def _ascii_bytes(payload, lineno):
    payload = payload.strip()
    if len(payload) < 2 or not payload.startswith('"') or not payload.endswith('"'):
        raise LoaderError(f'.ascii needs one double-quoted string, got {payload!r}', lineno)
    body = payload[1:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            if i + 1 >= len(body) or body[i + 1] not in _ESCAPES:
                raise LoaderError(f'.ascii unknown escape {body[i:i + 2]!r}', lineno)
            out.append(ord(_ESCAPES[body[i + 1]]))
            i += 2
            continue
        if ch == '"':
            raise LoaderError('.ascii string contains an unescaped quote', lineno)
        out.append(ord(ch))
        i += 1
    return bytes(out)

def _parse(src):
    items = []
    for lineno, raw in enumerate(src.splitlines(), 1):
        line = _strip_comment(raw)
        if not line:
            continue
        m = _LABEL.match(line)
        if m:
            items.append(_Item("label", m.group(1), lineno, line))
            continue
        name, _, rest = line.partition(" ")
        name = name.strip()
        if name in _DIRECTIVES:
            items.append(_Item(name[1:], rest.strip(), lineno, line))
            continue
        if name.startswith("."):
            raise LoaderError(f"unknown directive {name!r}", lineno)
        args = [a for a in _split_args(rest)] if rest.strip() else []
        items.append(_Item("inst", (name, args), lineno, line))
    return items

def _checked_int(value, what, lo, hi, lineno=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise LoaderError(f"{what} must be an int in {lo}..{hi}, got {value!r}", lineno)
    if not lo <= value <= hi:
        raise LoaderError(f"{what} is {value}, outside {lo}..{hi}", lineno)
    return value

def _resolve(value, symbols, what, lo, hi, lineno=None):

    if isinstance(value, str):
        if value not in symbols:
            raise LoaderError(f"{what} refers to undefined symbol {value!r}", lineno)
        value = symbols.value(value)
    return _checked_int(value, what, lo, hi, lineno)

class LoadResult:

    __slots__ = ("image", "symbols", "entry", "report", "vectors", "window",
                 "origins", "entry_explicit", "content_extent", "needed")

    def __init__(self, image, symbols, entry, report, vectors, window, origins,
                 entry_explicit, content_extent, needed):
        self.image = bytes(image)
        self.symbols = symbols
        self.entry = entry
        self.report = list(report)
        self.vectors = dict(vectors)
        self.window = None if window is None else tuple(window)
        self.origins = origins
        self.entry_explicit = entry_explicit
        self.content_extent = content_extent
        self.needed = needed

    def __len__(self):
        return len(self.image)

    @property
    def length(self):
        return len(self.image)

    def __repr__(self):
        return (f"<LoadResult {len(self.image)} bytes, entry=0x{self.entry:04X}, "
                f"{len(self.symbols)} symbols, {len(self.vectors)} vectors>")

    def summary(self):
        return "\n".join(self.report)

    def to_machine(self, **kw):

        from golden_sim import NCP8
        kw.setdefault("config", self.config())
        return NCP8(self.image, **kw)

    def config(self):

        if self.window is None:
            winlo = winhi = None
        else:
            winlo, winhi = self.window
        vec = dict(self.vectors) if self.vectors else None

        return ISA.MachineConfig(codelen=self.content_extent, winlo=winlo, winhi=winhi,
                                 vec=vec)

def _next_pow2(v):
    if v <= 1:
        return 1
    return 1 << (v - 1).bit_length()

def image_needed(content_extent=1):

    return max(int(content_extent), 1)

def assemble(src, *, vectors=None, window=None, image=None, entry=None):

    if not isinstance(src, str):
        raise LoaderError(f"src must be a string, got a {type(src).__name__}")
    if vectors is not None and not isinstance(vectors, dict):
        raise LoaderError(f"vectors must be a dict of index -> label_or_int, got "
                          f"a {type(vectors).__name__}")
    vectors = {} if vectors is None else dict(vectors)
    if window is not None and not (isinstance(window, (tuple, list)) and len(window) == 2):
        raise LoaderError(f"window must be a (lo, hi) pair, got {window!r}")
    items = _parse(src)
    symbols = Symbols()

    later = set()
    for it in items:
        if it.kind == "label":
            later.add(it.payload)
        elif it.kind == "equ":
            later.add(it.payload.partition(",")[0].strip())

    def value_now(expr, lineno, where):

        try:
            return evaluate(expr, symbols.flat(), lineno)
        except LoaderError as e:
            for ref in expression_names(expr) & later:
                if ref not in symbols:
                    raise LoaderError(f"{where} cannot refer to {ref!r}: it is defined "
                                      f"later in the source", lineno) from None
            raise

    sizes = []
    pc = 0
    for it in items:
        if it.kind == "label":
            symbols.add(it.payload, pc, it.lineno, const=False)
            sizes.append((it, 0))
            continue
        if it.kind == "equ":
            name, _, expr = it.payload.partition(",")
            name = name.strip()
            if not expr.strip():
                raise LoaderError(".equ needs NAME, <expr>", it.lineno)
            symbols.add(name, value_now(expr, it.lineno, ".equ"), it.lineno, const=True)
            sizes.append((it, 0))
            continue
        if it.kind == "org":
            target = value_now(it.payload, it.lineno, ".org")
            _checked_int(target, ".org target", 0, CODE_SIZE, it.lineno)
            pc = target
            sizes.append((it, 0))
            continue
        if it.kind in ("byte", "word"):
            args = [a for a in _split_args(it.payload)]
            if not args or not any(args):
                raise LoaderError(f".{it.kind} needs at least one expression", it.lineno)
            for a in args:
                if not a:
                    raise LoaderError(f".{it.kind} has an empty expression "
                                      f"(check for a trailing comma)", it.lineno)
            n = (1 if it.kind == "byte" else 2) * len(args)
            sizes.append((it, n))
            pc += n
            continue
        if it.kind == "ascii":
            n = len(_ascii_bytes(it.payload, it.lineno))
            sizes.append((it, n))
            pc += n
            continue
        name, args = it.payload
        shape, _operands = match_line(name, args, it.lineno, it.text)
        size = shape_size((name, shape))[0]
        sizes.append((it, size))
        pc += size

    out, origins, blocks = {}, {}, []
    hi_water = 0

    def place(kind, addr, data, lineno, text):
        nonlocal hi_water
        for i, byte in enumerate(data):
            a = addr + i
            if not 0 <= a < CODE_SIZE:
                raise LoaderError(f"{kind} reaches address 0x{a:X}, outside CODE "
                                  f"(0..{CODE_SIZE - 1})", lineno)
            if a in out:
                raise LoaderError(f"{kind} at 0x{a:04X} overlaps a byte already placed "
                                  f"there ({origins[a]})", lineno)
            out[a] = byte
            origins[a] = f"{kind} at line {lineno}: {text}"
        blocks.append((kind, addr, len(data), lineno, text))
        hi_water = max(hi_water, addr + len(data))

    pc = 0
    for it, _size in sizes:
        if it.kind == "org":
            pc = value_now(it.payload, it.lineno, ".org")
            continue
        if it.kind in ("label", "equ"):
            continue
        if it.kind == "byte":
            for expr in _split_args(it.payload):
                v = evaluate(expr, symbols.flat(), it.lineno)
                _checked_int(v, ".byte value", 0, 0xFF, it.lineno)
                place("byte", pc, bytes([v]), it.lineno, f"{expr} = {v}")
                pc += 1
            continue
        if it.kind == "word":
            for expr in _split_args(it.payload):
                v = evaluate(expr, symbols.flat(), it.lineno)
                _checked_int(v, ".word value", 0, 0xFFFF, it.lineno)
                place("word", pc, (v & 0xFFFF).to_bytes(2, "little"), it.lineno,
                      f"{expr} = {v}")
                pc += 2
            continue
        if it.kind == "ascii":
            data = _ascii_bytes(it.payload, it.lineno)
            place("ascii", pc, data, it.lineno, it.payload)
            pc += len(data)
            continue
        name, args = it.payload
        shape, operands = match_line(name, args, it.lineno, it.text)
        size, cps = shape_size((name, shape))
        rendered = [_render_operand(k, a, symbols, it.lineno, name)
                    for k, a in zip(shape, operands)]
        text = name if not rendered else f"{name} {', '.join(rendered)}"
        try:
            data = asm(text)
        except AssemblyError as e:
            detail = e.msg if isinstance(e, AssemblyError) else str(e)
            raise LoaderError(f"{it.text!r} as canonical {text!r}: {detail}",
                              it.lineno) from e
        if len(data) != size:
            raise LoaderError(f"{text!r} encoded to {len(data)} bytes, but code point "
                              f"0x{cps[0]:04X} is {size} bytes in the decoder", it.lineno)
        if disasm.decode(data, 0).size != size:
            raise LoaderError(f"internal: {text!r} encoded to {data.hex()} but the decoder "
                              f"reads {disasm.decode(data, 0).size} bytes at 0x{cps[0]:04X}",
                              it.lineno)
        place("code", pc, data, it.lineno, text)
        pc += size

    content_extent = hi_water
    for k in sorted(vectors):
        _checked_int(k, "vector index", 0, VEC_COUNT - 1)
    needed = image_needed(content_extent)

    if image is not None:
        _checked_int(image, "image length", 1, CODE_SIZE)
        length = image
        if length < needed:
            raise LoaderError(_short_image_message(length, needed, content_extent), None)
    else:
        length = _next_pow2(needed)
        if length > CODE_SIZE:
            raise LoaderError(f"content needs {needed} bytes, so the image would be "
                              f"{length}, above CODE_SIZE {CODE_SIZE}")
    if length > CODE_SIZE:
        raise LoaderError(f"image length {length} is above CODE_SIZE {CODE_SIZE}")
    if content_extent > length:
        raise LoaderError(f"content reaches 0x{content_extent - 1:04X}, past the end of "
                          f"the {length}-byte image", None)

    image_bytes = bytearray(length)
    for a in sorted(out):
        image_bytes[a] = out[a]

    placed_vectors = {}
    declarations = []
    for k in sorted(vectors):
        tgt = _resolve(vectors[k], symbols, f"vector {k} target", 0, (1 << BOUND_BITS) - 1)
        if tgt == 0:
            raise LoaderError(
                f"vector {k} targets address 0, which the machine reads as 'handler {k} "
                f"not registered', so the trap faults instead of running anything. Point it "
                f"at a handler, or leave {k} out of the table to declare it unregistered.",
                None)
        if tgt >= content_extent:
            raise LoaderError(f"vector {k} targets 0x{tgt:04X}, past the end of the "
                              f"{content_extent}-byte content: the handler's first fetch "
                              f"would fault as an out-of-range PC instead of running", None)
        placed_vectors[k] = tgt
        declarations.append(f"vector {k} -> 0x{tgt:04X}")

    placed_window = None
    if window is not None:
        lo = _resolve(window[0], symbols, "window lo", 0, (1 << BOUND_BITS) - 1)
        hi = _resolve(window[1], symbols, "window hi", 0, (1 << BOUND_BITS) - 1)
        if hi < lo:
            raise LoaderError(f"window bounds are reversed: lo=0x{lo:04X} hi=0x{hi:04X}. "
                              f"A reversed span is refused here rather than loaded as the "
                              f"empty window, because the empty window is a declaration "
                              f"meaning 'no STC writes anything'", None)
        placed_window = (lo, hi)
        declarations.append(f"window [0x{lo:04X}, 0x{hi:04X})")

    if entry is None:
        entry_explicit = False
        entry = symbols.value("main") if "main" in symbols else 0
    else:
        entry_explicit = True
        entry = _resolve(entry, symbols, "entry", 0, 0xFFFF)
        if entry >= content_extent:
            raise LoaderError(f"entry 0x{entry:04X} is past the end of the "
                              f"{content_extent}-byte content: padding past the content is "
                              f"a buffer, not a place to start", None)

    rep = [f"image: {length} bytes "
           + ("(length requested exactly)" if image is not None
              else f"(default: smallest power of two covering the needed "
                   f"{needed} bytes)")]
    rep.append(f"content: {content_extent} bytes of code/data, needing "
               f"{needed} bytes")
    rep.append(f"configuration: {len(placed_vectors)} trap vector(s), "
               + ("no window declared, so no STC writes" if placed_window is None
                  else f"window [0x{placed_window[0]:04X}, 0x{placed_window[1]:04X})")
               + "; none of it is a byte of the image")
    for kind, addr, size, lineno, text in blocks:
        where = f"line {lineno}" if lineno is not None else "declared by argument"
        rep.append(f"{kind:5s} 0x{addr:04X} {size:3d}B  {text}   [{where}]")
    for text in declarations:
        rep.append(f"decl  ---- ----  {text}   [declared by argument]")
    rep.append(f"entry: 0x{entry:04X} "
               + ("(given)" if entry_explicit else
                  ("(symbol main)" if "main" in symbols else
                   "(no entry given and no symbol main: the load address 0)"))
               + (" - where this load's `main` is, not where the machine starts: every "
                  "path boots at CODE[0] until a start address is configuration"
                  if entry else ""))
    rep.append(f"symbols: {len(symbols.flat())}")
    for name in sorted(symbols.flat()):
        rep.append(f"        {name:16s} 0x{symbols.value(name):04X} ({symbols.kind_of(name)})")
    return LoadResult(image_bytes, symbols.as_dict(), entry, rep, placed_vectors,
                      placed_window, origins, entry_explicit, content_extent, needed)

def _short_image_message(length, needed, content_extent):

    parts = [f"image length {length} bytes is too short for this program"]
    if content_extent > length:
        parts.append(f"the code/data itself reaches 0x{content_extent - 1:04X}")
    parts.append(f"needs at least {needed} bytes")
    return "; ".join(parts)

def describe(result):

    return "\n".join(result.report)

if __name__ == "__main__":
    r = assemble("main:\n  LDI r0, 1\n  HALT\nhandler:\n  RET\n",
                 vectors={0: "handler"}, window=(0x00, 0x08), entry="main")
    print(describe(r))