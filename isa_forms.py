"""The accepted operand shapes of every NCP-8 mnemonic, as data: `FORMS`.

This is the single statement of which instruction spellings the assembler accepts, and
both front ends consult it: `golden_sim.asm` for the bytes of a line, `loader.assemble`
for the same decision before it evaluates an operand's expression. A line is legal only
when one shape of its mnemonic matches it with the whole argument list consumed, so
arity is exact and no argument may be ignored; `refusal()` is what a front end raises,
and it quotes the forms `syntax()` renders rather than inventing one. Addressing modes
are operand kinds of their own, never folded into a register kind.

Self-test: `python3 isa_forms.py` checks the table against the encodings `isa_table`
assigns -- byte accounting per shape, every assigned code point claimed, every claimed
shape backed by a code point, the matcher accepting each shape's own spelling with no
ambiguity, each declared operand domain matching what the decoder reports, and the
decoder naming the same shape for every code point -- and exits non-zero on any gap.
"""
import re

import isa_table

KINDS = {
    "r": (0, None, "r", "r0-r3, index carried by the code byte", False),
    "HL": (0, None, "HL", "the 16-bit pointer HL", False),
    "DE": (0, None, "DE", "the 16-bit pointer DE", False),
    "SP": (0, None, "SP", "the stack pointer", False),
    "[HL]": (0, None, "[HL]", "indirect through DATA[HL]", False),
    "[DE]": (0, None, "[DE]", "indirect through DATA[DE]", False),
    "a16": (2, "{a16}", "a16", "0..65535 branch/call target, or a label that is not a "
                             "reserved register name", True),
    "i16": (2, "{i16}", "i16", "0..65535 16-bit address, or a label that is not a "
                               "reserved register name", True),
    "i8": (1, "{i8}", "i8", "0..255 byte literal", False),
    "rcanon": (1, "{rcanon}", "r", "r0-r3 in the operand byte; bits 2..7 must be 0",
               False),
    "soff": (1, "{soff}", "i8", "-128..127 signed byte literal", False),
    "k": (1, "{k}", "k", "0..255 trap number, the whole operand byte", False),
    "[HL+-i8]": (1, "{off}", "[HL+i8]", "[HL] or [HL+d]/[HL-d], d in -128..127", False),
}

FORMS = {

    "HALT": ((),),
    "NOP": ((),),
    "INC": (("HL",), ("DE",)),
    "DEC": (("HL",),),
    "CLC": ((),),
    "OUTM": ((),),
    "OUTDE": ((),),
    "RET": ((),),
    "JMP": (("a16",),),
    "JZ": (("a16",),),
    "JNZ": (("a16",),),
    "JC": (("a16",),),
    "JNC": (("a16",),),
    "CALL": (("a16",),),
    "LDI": (("HL", "i16"), ("DE", "i16"), ("r", "i8")),
    "ADDI": (("HL", "rcanon"), ("DE", "rcanon"), ("r", "i8")),
    "JPHL": ((),),
    "GETPC": (("r",),),
    "GETSP": (("r",),),
    "GETF": (("r",),),

    "AND": (("r", "r"),),
    "OR": (("r", "r"),),
    "XOR": (("r", "r"),),
    "MUL": (("r", "r"),),

    "SHL": (("r",),),
    "SHR": (("r",),),
    "TST": (("r",),),
    "DJNZ": (("r", "a16"),),

    "ADD": (("r", "r"), ("HL", "DE"), ("SP", "soff")),
    "SUB": (("r", "r"), ("HL", "DE")),
    "ADC": (("r", "r"),),
    "SBB": (("r", "r"),),
    "MOV": (("r", "r"), ("r", "[HL]"), ("[HL]", "r"), ("r", "[DE]"), ("[DE]", "r")),
    "SUBI": (("r", "i8"),),
    "ADCI": (("r", "i8"),),

    "PUSH": (("r",),),
    "POP": (("r",),),
    "OUT": (("r",),),
    "IN": (("r",),),

    "DIV": (("r", "r"),),
    "MOD": (("r", "r"),),
    "CMP": (("r", "r"),),
    "MOVW": (("HL", "DE"), ("DE", "HL"), ("HL", "SP"), ("DE", "SP"), ("SP", "HL"),
             ("SP", "DE")),
    "PUSHW": (("HL",), ("DE",)),
    "POPW": (("HL",), ("DE",)),
    "STW": (("[HL]", "DE"), ("[DE]", "HL")),
    "LDW": (("DE", "[HL]"), ("HL", "[DE]")),
    "NOT": (("r",),),
    "NEG": (("r",),),
    "ROL": (("r",),),
    "ROR": (("r",),),
    "LDX": (("r", "[HL+-i8]"),),
    "STX": (("[HL+-i8]", "r"),),
    "XCHG": (("HL", "DE"),),
    "EXT": (("k",),),
    "STC": (("[HL]", "r"),),
    "LDC": (("r", "[HL]"),),
    "MULH": (("r", "r"),),
}

_NUM = re.compile(r"^[+-]?(?:0[xXoObB][0-9a-fA-F_]+|[0-9][0-9_]*)$")
_SYM = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_FRAME = re.compile(r"^\[HL(?:([+-])([^\]]+))?\]$")
_REG_NAME = re.compile(r"^r[0-9]+$")

RESERVED = frozenset(("r0", "r1", "r2", "r3", "HL", "DE", "SP"))

_FIELD = {"a16": "{a16}", "i16": "{i16}", "i8": "{i8}", "roff": "{rcanon}",
          "soff": "{soff}", "k": "{k}", "off": "{off}"}

VALUE_KINDS = frozenset(("a16", "i16", "i8", "soff", "k", "[HL+-i8]"))

_WORD = {"r": "register operand (want r0-r3)",
         "rcanon": "register operand (want r0-r3)",
         "HL": "pointer operand (must read HL)",
         "DE": "pointer operand (must read DE)",
         "SP": "pointer operand (must read SP)",
         "[HL]": "memory operand (must read [HL])",
         "[DE]": "memory operand (must read [DE])",
         "a16": "address operand",
         "i16": "16-bit address operand",
         "i8": "8-bit immediate",
         "soff": "signed 8-bit immediate",
         "k": "trap number operand",
         "[HL+-i8]": "frame operand (want [HL], [HL+i8] or [HL-i8])",
}

def _int(text):

    t = text.strip()
    if not _NUM.match(t):
        return None
    try:
        return int(t.replace("_", ""), 0)
    except ValueError:
        return None

def check(kind, text):

    t = text.strip()
    if kind == "r" or kind == "rcanon":
        return bool(re.fullmatch(r"r[0-3]", t))
    if kind in ("HL", "DE", "SP", "[HL]", "[DE]"):
        return t == kind
    if kind == "[HL+-i8]":
        m = _FRAME.fullmatch(t.replace(" ", ""))
        if not m:
            return False
        if m.group(2) is None:
            return True
        v = _int(m.group(2))
        if v is None:
            return False
        return -128 <= (-v if m.group(1) == "-" else v) <= 127
    if kind in ("a16", "i16"):
        v = _int(t)
        if v is not None:
            return 0 <= v <= 0xFFFF
        return bool(_SYM.fullmatch(t)) and t not in RESERVED
    if kind == "i8":
        v = _int(t)
        return v is not None and 0 <= v <= 0xFF
    if kind == "soff":
        v = _int(t)
        return v is not None and -128 <= v <= 127
    if kind == "k":
        v = _int(t)
        return v is not None and 0 <= v <= 0xFF
    raise KeyError(f"isa_forms: undeclared operand kind {kind!r}")

def fits(kind, text, value_loose=False):

    if not value_loose or kind not in VALUE_KINDS:
        return check(kind, text)
    t = str(text).strip()
    if kind == "[HL+-i8]":
        return bool(_FRAME.fullmatch(t.replace(" ", "")))
    if kind in ("a16", "i16"):
        return bool(t) and t not in RESERVED
    return bool(t)

def match(name, args):

    hits = [s for s in FORMS.get(name, ())
            if len(s) == len(args) and all(check(k, a) for k, a in zip(s, args))]
    return hits[0] if len(hits) == 1 else None

def accepts(name, args):

    return match(name, args) is not None

def arieties(name):

    return sorted({len(s) for s in FORMS.get(name, ())})

def syntax(name):

    if name not in FORMS:
        return "no such mnemonic"
    out = []
    for shape in FORMS[name]:
        operands = ", ".join(KINDS[k][2] for k in shape)
        out.append(f"{name} {operands}".strip())
    return " | ".join(out)

def blame(name, args):

    if name not in FORMS:
        return ("no such mnemonic", None, None)
    shapes = [s for s in FORMS[name] if len(s) == len(args)]
    if not shapes:
        return ("wrong operand count", None, None)
    for pos, arg in enumerate(args):
        fits = [s for s in shapes if check(s[pos], arg)]
        if fits:
            shapes = fits
            continue
        kinds = {s[pos] for s in shapes}
        if len(kinds) == 1:
            return ("operand mismatch", pos, kinds.pop())
        if _REG_NAME.match(arg.strip()):
            return ("operand mismatch", pos, "r")
        return ("operands mismatch", None, None)
    return (None, None, None)

def refusal(name, args, text, detail=None):

    reason, pos, kind = blame(name, args)
    if reason == "no such mnemonic":
        return f"{text!r}: unknown instruction {name!r}"
    if detail is not None:
        line = detail
    elif reason == "wrong operand count":
        want = "/".join(str(n) for n in arieties(name))
        line = f"{name} takes {want} operands, {len(args)} given"
    elif reason == "operand mismatch":
        line = f"invalid {_WORD[kind]} {args[pos]!r} in position {pos + 1}"
    else:
        line = f"unknown operand combination {list(args)!r}"
    return f"{text!r}: {line}; {name} accepts {syntax(name)}"

def prefix_of(codepoint):

    return 1 if codepoint < 0x100 else 2

def row_template(row):

    text, field = row["mnem"], row["kind"]
    if field and "{" not in text:
        _name, _, rest = text.partition(" ")
        text += (" " if not rest.strip() else ", ") + _FIELD[field]
    return text

def template_shape(template):

    name, _, rest = template.partition(" ")
    name = name.strip()
    pieces = [p.strip() for p in rest.split(",")] if rest.strip() else []
    shape = []
    for piece in pieces:
        if piece == "[HL{off}]":
            shape.append("[HL+-i8]")
            continue
        for field in ("{a16}", "{i16}", "{i8}", "{rcanon}", "{soff}", "{k}"):
            if field in piece:
                if piece != field:
                    raise ValueError(f"template operand {piece!r} mixes a field with "
                                     "literal text")
                shape.append({"{a16}": "a16", "{i16}": "i16", "{i8}": "i8",
                              "{rcanon}": "rcanon", "{soff}": "soff",
                              "{k}": "k"}[field])
                break
        else:
            if re.fullmatch(r"r[0-3]", piece):
                shape.append("r")
            elif piece in ("HL", "DE", "SP", "[HL]", "[DE]"):
                shape.append(piece)
            else:
                raise ValueError(f"template operand {piece!r} matches no operand kind")
    return name, tuple(shape)

def encodings():

    out = {}
    for op in isa_table.SINGLE:
        row = isa_table.SINGLE[op]
        name, shape = template_shape(row_template(row))
        out[op] = (name, shape, prefix_of(op) + row["l"], row_template(row))
    for sub in isa_table.ESCAPE:
        row = isa_table.ESCAPE[sub]
        tmpl = row_template(row)
        name, shape = template_shape(tmpl)
        out[0x7000 | sub] = (name, shape, prefix_of(0x7000 | sub) + row["l"], tmpl)
    return out

def shape_info():

    out = {}
    for cp, (name, shape, size, _tmpl) in encodings().items():
        got = out.get((name, shape))
        if got is None:
            out[(name, shape)] = (size, [cp])
        else:
            got[1].append(cp)
    return {(k): (v[0], tuple(sorted(v[1]))) for k, v in out.items()}

SAMPLE = {"r": "r2", "HL": "HL", "DE": "DE", "SP": "SP", "[HL]": "[HL]", "[DE]": "[DE]",
          "a16": "0x0F00", "i16": "0x100", "i8": "200", "rcanon": "r3", "soff": "-8",
          "k": "9", "[HL+-i8]": "[HL+4]"}

def _decoder_tables():

    import disasm

    out = {}
    for op, (tmpl, size) in disasm.SINGLE.items():
        name, shape = template_shape(tmpl)
        out[op] = (name, shape, size, tmpl)
    for sub, (tmpl, size) in disasm.ESC.items():
        name, shape = template_shape(tmpl)
        out[0x7000 | sub] = (name, shape, size, tmpl)
    return out

def self_test(verbose=True):

    enc = encodings()
    fails = []

    for cp, (name, shape, size, tmpl) in sorted(enc.items()):
        try:
            implied = prefix_of(cp) + sum(KINDS[k][0] for k in shape)
        except KeyError as e:
            fails.append(("kind", cp, str(e)))
            continue
        if implied != size:
            fails.append(("bytes", cp, tmpl, size, implied))

    for cp, (name, shape, size, tmpl) in sorted(enc.items()):
        if name not in FORMS:
            fails.append(("mnemonic-absent", cp, name, tmpl))
        elif shape not in FORMS[name]:
            fails.append(("shape-absent", cp, name, tmpl, shape))

    for name in sorted({n for n, _s, _z, _t in enc.values()}):
        if name not in FORMS:
            fails.append(("rendered-mnemonic-uncovered", name))

    for name in sorted(FORMS):
        for shape in FORMS[name]:
            if not any(n == name and s == shape for n, s, _z, _t in enc.values()):
                fails.append(("unencoded-shape", name, shape))
            for kind in shape:
                if kind not in KINDS:
                    fails.append(("undeclared-kind", name, kind))

    for name in sorted(FORMS):
        for shape in FORMS[name]:
            try:
                args = [SAMPLE[k] for k in shape]
            except KeyError as e:
                fails.append(("no-sample-for-kind", name, str(e)))
                continue
            if match(name, args) != shape:
                fails.append(("shape-not-self-matching", name, shape, args, match(name, args)))
            for other in FORMS[name]:
                if other == shape or len(other) != len(shape):
                    continue
                if all(check(a, b) for a, b in zip(other, args)):
                    fails.append(("ambiguous-shape-pair", name, shape, other))

        for i, s in enumerate(FORMS[name]):
            for t in FORMS[name][i + 1:]:
                if len(s) != len(t):
                    continue
                if all(a == b or a in VALUE_KINDS or b in VALUE_KINDS
                       for a, b in zip(s, t)):
                    fails.append(("structural-ambiguity-pair", name, s, t))

    claimed = {cp for cp, (name, shape, _z, _t) in enc.items()
               if name in FORMS and shape in FORMS[name]}
    for cp in sorted(set(enc) - claimed):
        fails.append(("encoding-unclaimed", cp, enc[cp][:2]))
    extra = set(claimed) - set(enc)
    for cp in sorted(extra):
        fails.append(("claimed-encoding-unassigned", cp))
    if len(enc) != len(isa_table.SINGLE) + len(isa_table.ESCAPE):
        fails.append(("assigned-count", len(enc),
                      len(isa_table.SINGLE) + len(isa_table.ESCAPE)))
    try:
        dec = _decoder_tables()
    except Exception as e:
        dec = {}
        fails.append(("decoder-unreadable", type(e).__name__, str(e)))
    if set(dec) != set(enc):
        fails.append(("decoder-encoding-set",
                      sorted(set(enc) - set(dec))[:4], sorted(set(dec) - set(enc))[:4]))
    for cp in sorted(set(dec) & set(enc)):
        if dec[cp][:3] != enc[cp][:3]:
            fails.append(("decoder-disagrees", cp, enc[cp][:3], dec[cp][:3]))

    if verbose:
        shapes = sum(len(v) for v in FORMS.values())
        kinds5 = ("shape-not-self-matching", "ambiguous-shape-pair", "no-sample-for-kind",
              "structural-ambiguity-pair")
        domain = _domain_checks()
        fails.extend(domain)
        print(f"mnemonics in FORMS: {len(FORMS)}")
        print(f"shapes in FORMS: {shapes}")
        print(f"assigned encodings in isa_table: {len(enc)} "
              f"(single-byte {len(isa_table.SINGLE)}, escape {len(isa_table.ESCAPE)})")
        print(f"assigned encodings claimed by FORMS: {len(claimed)} of {len(enc)}")
        print(f"distinct (mnemonic, shape) pairs backed by an encoding: {len(shape_info())}")
        print(f"cross-check 1 (operand bytes + prefix == declared size): "
              f"{sum(1 for f in fails if f[0] == 'bytes')} mismatches")
        print(f"cross-check 2 (every assigned encoding claimed): "
              f"{sum(1 for f in fails if f[0] in ('shape-absent', 'mnemonic-absent', 'encoding-unclaimed'))}"
              f" unclaimed")
        print(f"cross-check 3 (every rendered mnemonic covered): "
              f"{sum(1 for f in fails if f[0] == 'rendered-mnemonic-uncovered')} uncovered")
        print(f"cross-check 4 (every FORMS shape has an encoding): "
              f"{sum(1 for f in fails if f[0] in ('unencoded-shape', 'undeclared-kind', 'claimed-encoding-unassigned'))}"
              f" orphan shapes")
        print(f"cross-check 5 (matcher accepts each shape's own spelling, no ambiguity): "
              f"{sum(1 for f in fails if f[0] in kinds5)} failures")
        print(f"cross-check 6 (each kind's declared value domain matches what the decoder "
              f"reports): {sum(1 for f in fails if f[0] == 'domain-disagrees')} disagreements")
        print(f"cross-check 7 (claim set equals isa_table's assigned encodings, and the "
              f"decoder names the same shape for each): "
              f"{sum(1 for f in fails if f[0] in ('decoder-encoding-set', 'decoder-disagrees', 'assigned-count', 'decoder-unreadable'))}"
              f" disagreements")
        for f in fails[:20]:
            print("   GAP", f)
        print(f"TOTAL failures: {len(fails)}")
    return fails

def _domain_checks():

    import disasm

    def non_canonical(image):
        return bool(disasm.decode(image, 0).non_canonical)

    checks = [

        ("k", lambda v: bytes([0x70, 0x70, v]), range(0, isa_table.VEC_COUNT),
         [isa_table.VEC_COUNT, isa_table.VEC_COUNT + 1, 200, 255]),
        ("rcanon", lambda v: bytes([0x11, v]), range(0, 4), [4, 8, 0x7F, 0xFF]),
    ]
    out = []
    for kind, enc, good, bad in checks:
        for v in good:
            if non_canonical(enc(v)):
                out.append(("domain-disagrees", kind, v, "expected canonical",
                            "reported non-canonical"))
        for v in bad:
            if not non_canonical(enc(v)):
                out.append(("domain-disagrees", kind, v, "expected non-canonical",
                            "reported canonical"))
    return out

if __name__ == "__main__":
    import sys

    sys.exit(1 if self_test() else 0)