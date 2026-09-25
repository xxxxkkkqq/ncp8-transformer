"""Form-domain acceptance: the encodings the decoder assigns against the spellings

Every assigned code point is walked over the boundary values of its operand field, rendered by
the disassembler, and handed to both assembler front ends -- `golden_sim.asm` and
`loader.assemble` -- which must reproduce those bytes. In the other direction every accepted
spelling must encode to a code point the table assigns, and the value range an operand kind
declares in `isa_forms.KINDS` must be the range both front ends accept, one value past each
end refused by name. The counts of each set walked are printed, so the suite cannot pass by
covering a subset nobody counted.

Run: python3 test_form_domain.py
"""
import re

import disasm
import golden_sim as G
import isa_forms
import isa_table
import loader

_RANGE_AT_HEAD = re.compile(r"^(-?\d+)\.\.(-?\d+)")
_RANGE_FOR_VALUE = re.compile(r"\bd in (-?\d+)\.\.(-?\d+)")

TABLES = "isa_table.SINGLE + isa_table.ESCAPE"

FRONTS = ("asm", "loader")

_CACHE = {}

def require(cond, msg):

    if not cond:
        raise AssertionError(msg)

def boundary(width):

    hi = (1 << (8 * width)) - 1
    return sorted({v for v in (0, 1, 2, hi - 2, hi - 1, hi) if 0 <= v <= hi})

def code_bytes(cp):

    return bytes((cp,)) if cp < 0x100 else bytes((0x70, cp & 0xFF))

def fields(shape):

    out, start = [], 0
    for kind in shape:
        width = isa_forms.KINDS[kind][0]
        if width:
            out.append((start, kind, width))
            start += width
    return out

def image_of(cp, shape, position=None, value=0):

    ops = bytearray(sum(isa_forms.KINDS[k][0] for k in shape))
    if position is not None:
        start, width = position[0], position[2]
        ops[start:start + width] = int(value).to_bytes(width, "little")
    return code_bytes(cp) + bytes(ops)

def emit(front, text):

    if front == "asm":
        return G.asm(text)
    return loader.assemble(text + "\n").image

def refusal_of(front, text):

    try:
        emit(front, text)
    except Exception as exc:
        return str(exc)
    return None

def decode(image):

    return disasm.decode(image, 0)

def assigned_encodings():

    return isa_forms.encodings()

def field_text_classes(cp, shape, position):

    _start, _kind, width = position
    values = range(1 << (8 * width)) if width == 1 else boundary(width)
    classes = {}
    for value in values:
        text = decode(image_of(cp, shape, position, value)).text
        classes.setdefault(text, []).append(value)
    return classes

def probe_images(cp, shape):

    probes = [(image_of(cp, shape), None, 0)]
    for position in fields(shape):
        for value in boundary(position[2]):
            if value:
                probes.append((image_of(cp, shape, position, value), position, value))
    return probes

def walk_assigned():

    enc = assigned_encodings()
    problems, probed, exact_code_points = [], set(), set()
    read_probes = unread_probes = normalised = unread_canonical = 0
    unread_fields, unflagged = set(), []
    for cp in sorted(enc):
        name, shape, _size, _tmpl = enc[cp]
        probed.add(cp)
        classes = {p: field_text_classes(cp, shape, p) for p in fields(shape)}
        for position in classes:
            if any(len(vs) > 1 for vs in classes[position].values()):
                unread_fields.add((name, position[1], cp))
        images = set()
        for image, position, value in probe_images(cp, shape):
            if image in images:
                continue
            images.add(image)
            row = decode(image)
            if not row.assigned:
                problems.append(f"{name} {image.hex()}: the table assigns code point "
                                f"{cp:#06x} but this image decodes as unassigned")
                continue
            if row.codepoint != cp or row.size != len(image):
                problems.append(f"{name} {image.hex()}: decoded as code point "
                                f"{row.codepoint:#06x} in {row.size} bytes, the table "
                                f"assigns {cp:#06x} in {len(image)} bytes")
                continue
            text = row.text
            if position is None:
                read, canonical = True, image
            else:
                values = classes[position].get(text)
                if not values:
                    problems.append(f"{name} {image.hex()} -> {text!r}: no value of the "
                                    f"{position[1]} field renders this text")
                    continue
                read = len(values) == 1
                canonical = image if read else image_of(cp, shape, position, min(values))
            if read:
                read_probes += 1
            else:
                unread_probes += 1
            for front in FRONTS:
                msg = refusal_of(front, text)
                if msg is not None:
                    problems.append(f"{name} {image.hex()} -> {text!r}: {front} refused a "
                                    f"rendering of an assigned encoding: {msg}")
                    continue
                got = emit(front, text)
                if len(got) < len(canonical):
                    problems.append(f"{name} {image.hex()} -> {text!r}: {front} emitted "
                                    f"{len(got)} bytes, below the {len(canonical)} the "
                                    "encoding takes")
                elif got[:len(canonical)] != canonical:
                    problems.append(f"{name} {image.hex()} -> {text!r}: {front} emitted "
                                    f"{got[:len(canonical)].hex()}, not {canonical.hex()}")
                elif any(got[len(canonical):]):
                    problems.append(f"{name} {image.hex()} -> {text!r}: {front} padded past "
                                    "the encoding with bytes that are not zero")
            if image == canonical:
                exact_code_points.add(cp)
                if not read:
                    unread_canonical += 1
            else:
                normalised += 1
                if not row.non_canonical:
                    unflagged.append(f"{name} {image.hex()} -> {text!r}: normalised to "
                                     f"{canonical.hex()} and the decoder flagged neither")
    counts = {"probed": len(probed), "probed_set": probed, "assigned": len(enc),
              "read_probes": read_probes, "unread_probes": unread_probes,
              "probes": read_probes + unread_probes, "normalised": normalised,
              "unread_canonical": unread_canonical, "unflagged": unflagged,
              "exact_code_points": len(exact_code_points),
              "unread_fields": sorted(unread_fields)}
    return problems, counts

def assigned_walk():

    if "assigned" not in _CACHE:
        _CACHE["assigned"] = walk_assigned()
    return _CACHE["assigned"]

def declared_domain(kind):

    text = isa_forms.KINDS[kind][3]
    for pattern in (_RANGE_AT_HEAD, _RANGE_FOR_VALUE):
        m = pattern.search(text)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None

def numeric_positions(shape):

    out = []
    for pos, kind in enumerate(shape):
        domain = declared_domain(kind)
        if domain is not None:
            out.append((pos, kind, domain))
    return out

def spell(kind, value):

    if kind == "[HL+-i8]":
        return f"[HL{value}]" if value < 0 else f"[HL+{value}]"
    return str(value)

def spelling(name, args):

    return f"{name} {', '.join(args)}" if args else name

def parsed(text):

    name, _, rest = text.partition(" ")
    args = [a.strip() for a in rest.split(",")] if rest.strip() else []
    return name.strip(), args

def shape_holder(kind):

    info = isa_forms.shape_info()
    for name in sorted(isa_forms.FORMS):
        for shape in isa_forms.FORMS[name]:
            if (name, shape) not in info:
                continue
            for pos, k in enumerate(shape):
                if k == kind:
                    return (name, shape, pos)
    return None

def walk_accepted_spellings():

    enc = assigned_encodings()
    problems, probed = [], 0
    for (name, shape), (size, cps) in sorted(isa_forms.shape_info().items()):
        base = [isa_forms.SAMPLE[k] for k in shape]
        spellings = [list(base)]
        for pos, kind, (lo, hi) in numeric_positions(shape):
            for value in sorted({lo, lo + 1, hi - 1, hi, 0}):
                args = list(base)
                args[pos] = spell(kind, value)
                spellings.append(args)
        for args in spellings:
            text = spelling(name, args)
            probed += 1
            matched = isa_forms.match(name, list(args))
            if matched != shape:
                problems.append(f"{text!r}: FORMS matches {matched!r}, not the shape "
                                f"{shape!r} this spelling was built from")
                continue
            for front in FRONTS:
                try:
                    emitted = emit(front, text)
                except Exception as exc:
                    problems.append(f"{text!r}: {front} refused an accepted spelling: {exc}")
                    continue
                if len(emitted) < size:
                    problems.append(f"{text!r}: {front} emitted {len(emitted)} bytes for an "
                                    f"encoding the decoder reads in {size}")
                    continue
                image = emitted[:size]
                row = decode(image)
                if row.codepoint not in enc:
                    problems.append(f"{text!r}: {front} emitted {image.hex()}, code point "
                                    f"{row.codepoint:#06x} the decode table does not assign")
                elif row.codepoint not in cps:
                    problems.append(f"{text!r}: {front} emitted code point "
                                    f"{row.codepoint:#06x}, which shape ({name}, {shape}) "
                                    f"does not claim: {sorted(hex(c) for c in cps)}")
                elif isa_forms.match(*parsed(row.text)) != shape:
                    problems.append(f"{text!r}: {front} emitted {image.hex()}, which renders "
                                    f"as {row.text!r} and matches no accepted shape")
    return problems, probed

def walk_declared_domains():

    problems, rows, covered = [], 0, []
    for kind in sorted(isa_forms.KINDS):
        domain = declared_domain(kind)
        if domain is None:
            if kind in isa_forms.VALUE_KINDS:
                problems.append(f"kind {kind!r} is a value kind whose KINDS entry declares "
                                "no numeric range, so its accepted domain goes unchecked")
            continue
        covered.append(kind)
        lo, hi = domain
        holder = shape_holder(kind)
        if holder is None:
            problems.append(f"kind {kind!r} declares a value domain no FORMS shape uses")
            continue
        name, shape, pos = holder
        base = [isa_forms.SAMPLE[k] for k in shape]
        for value, inside in ((lo, True), (lo + 1, True), (hi - 1, True), (hi, True),
                              (lo - 1, False), (hi + 1, False)):
            args = list(base)
            args[pos] = spell(kind, value)
            text = spelling(name, args)
            rows += 1
            if inside:
                if isa_forms.match(name, args) != shape:
                    problems.append(f"{text!r}: {kind} value {value} is inside the declared "
                                    f"domain {lo}..{hi} and FORMS refuses the spelling")
                    continue
                for front in FRONTS:
                    msg = refusal_of(front, text)
                    if msg is not None:
                        problems.append(f"{text!r}: {front} refused a value inside the "
                                        f"declared domain {lo}..{hi}: {msg}")
                continue
            if isa_forms.check(kind, args[pos]):
                problems.append(f"{text!r}: {kind} value {value} is outside the declared "
                                f"domain {lo}..{hi} and FORMS accepts it")
                continue
            for front in FRONTS:
                msg = refusal_of(front, text)
                if msg is None:
                    problems.append(f"{text!r}: {front} assembled a {kind} value outside "
                                    f"the declared domain {lo}..{hi}")
                elif name not in msg or str(value) not in msg:
                    problems.append(f"{text!r}: {front} refused it without naming the "
                                    f"mnemonic and the value: {msg}")
    return problems, rows, covered

EXT_ROWS = (
    ("EXT 0", bytes([0x70, 0x70, 0x00])),
    ("EXT 4", bytes([0x70, 0x70, 0x04])),
    ("EXT 15", bytes([0x70, 0x70, 0x0F])),
    ("EXT 16", bytes([0x70, 0x70, 0x10])),
    ("EXT 17", bytes([0x70, 0x70, 0x11])),
    ("EXT 255", bytes([0x70, 0x70, 0xFF])),
)

def test_ext_operand_byte_round_trips():

    for text, want in EXT_ROWS:
        row = decode(want)
        require(row.assigned and row.text == text,
                f"{want.hex()} decodes as {row.text!r}, the rendering this row spells "
                f"{text!r}")
        for front in FRONTS:
            msg = refusal_of(front, text)
            require(msg is None, f"{front} refused {text!r}: {msg}")
            got = emit(front, text)
            require(got[:len(want)] == want,
                    f"{front} emitted {got[:len(want)].hex()} for {text!r}, the encoding "
                    f"is {want.hex()}")
    operands = "/".join(t.split(" ")[1] for t, _w in EXT_ROWS)
    print(f"  EXT rows: {len(EXT_ROWS)} spellings of the operand byte ({operands}) decode "
          "and assemble to their own bytes on both front ends")

def test_assigned_encodings_round_trip():

    problems, counts = assigned_walk()
    require(counts["probed"] == counts["assigned"],
            f"{TABLES} assigns {counts['assigned']} code points but the walk probed "
            f"{counts['probed']}")
    require(counts["exact_code_points"] == counts["assigned"],
            f"only {counts['exact_code_points']} of {counts['assigned']} assigned "
            f"encodings have a spelling that assembles byte for byte: {problems[:4]}")
    require(not counts["unflagged"], str(counts["unflagged"][:12]))
    require(not problems, str(problems[:12]))

def test_accepted_spellings_are_assigned():

    problems, probed = walk_accepted_spellings()
    _CACHE["spellings"] = probed
    require(probed > 0, "the FORMS walk probed no spelling")
    require(not problems, str(problems[:12]))

def test_declared_domains_are_the_accepted_domains():

    problems, rows, covered = walk_declared_domains()
    _CACHE["domain_rows"] = rows
    _CACHE["domain_kinds"] = covered
    require(rows > 0, "the declared-domain walk checked no boundary")
    require(set(covered) == set(isa_forms.VALUE_KINDS),
            f"the value domains walked {sorted(covered)}, which is not the value kinds "
            f"isa_forms declares: {sorted(isa_forms.VALUE_KINDS)}")
    require(not problems, str(problems[:12]))

def test_probed_count_is_the_assigned_count():

    enc = assigned_encodings()
    want = len(isa_table.SINGLE) + len(isa_table.ESCAPE)
    require(len(enc) == want, f"{TABLES} holds {want} rows and {len(enc)} encodings were "
                              "derived from them")
    _problems, counts = assigned_walk()
    require(counts["probed_set"] == set(enc),
            f"the walk probed {counts['probed']} code points, not the {len(enc)} assigned")
    require(counts["probes"] > want,
            f"{counts['probes']} probes over {want} code points means operand fields were "
            "walked by nobody")

FAILS = []

def check(name, fn):

    try:
        fn()
    except AssertionError as exc:
        FAILS.append(f"{name}: {exc}")
        print(f"  FAIL {name}  {str(exc)[:200]}")

def main():
    print("form-domain acceptance (decode table against both assembler front ends):")
    check("EXT operand byte round trips", test_ext_operand_byte_round_trips)
    check("assigned encodings round trip", test_assigned_encodings_round_trip)
    check("accepted spellings are assigned", test_accepted_spellings_are_assigned)
    check("declared domains are the accepted domains",
          test_declared_domains_are_the_accepted_domains)
    check("probed count is the assigned count", test_probed_count_is_the_assigned_count)

    counts = {}

    def walk():
        nonlocal counts
        _problems, counts = assigned_walk()

    check("decode-table walk over both front ends", walk)
    spellings = _CACHE.get("spellings", 0)
    domain_rows = _CACHE.get("domain_rows", 0)
    kinds = _CACHE.get("domain_kinds", ())
    if counts:
        unread = ", ".join(f"{name} {{{kind}}} at {cp:#06x}"
                           for name, kind, cp in counts["unread_fields"]) or "none"
        print(f"  {counts['probed']} code points probed == {TABLES} assigned "
              f"{counts['assigned']} (single-byte {len(isa_table.SINGLE)}, escape "
              f"{len(isa_table.ESCAPE)}); {counts['probes']} operand images walked through "
              f"both front ends: {counts['read_probes']} were required to round-trip byte "
              f"for byte and {counts['normalised']} normalised to the canonical form of "
              f"their rendering ({counts['unread_probes']} probes sit in a field with bits "
              f"the decoder does not read, {counts['unread_canonical']} of them already "
              f"canonical)")
        print(f"  fields carrying bits the decoder does not read: {unread}")
    print(f"  {spellings} accepted spellings encoded to code points the table assigns; "
          f"{domain_rows} declared-domain boundary rows over kinds {', '.join(kinds)}, "
          "each end assembling and one past each end refused by name on both front ends")
    print(f"\nform-domain acceptance: {len(FAILS)} failure(s)")
    for f in FAILS:
        print("  !!", f)
    return 1 if FAILS else 0

if __name__ == "__main__":
    raise SystemExit(main())