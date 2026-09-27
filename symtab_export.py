"""Symbol-table export for NCP-8: a load's symbols as a standalone, byte-stable file.

`export` writes the typed table a load resolved against (a multi-unit link exports its
merged table) as a header line plus one sorted `name kind value` record per symbol, and
`read` parses that text back into a real loader.Symbols - names, values and the
address/constant kinds, which are semantics (an address symbol may not reach an 8-bit
immediate) and which the flat `symbols` dict on a LoadResult folds away. Two exports of
one table are one string; two tables that differ only in insertion order export the same
bytes. Refusals are named and carry the file line: a duplicate symbol names both of its
lines, an unknown kind, a malformed line, a non-hex value, an invalid name and a missing
header are refused, and a flat dict is refused at export because a text built without
kinds could not be read back into its own table.

Run: python3 symtab_export.py      (self-check)
"""
from __future__ import annotations

import re

import loader
from loader import Symbols

HEADER = "ncp8-symtab 1"
KINDS = ("address", "constant")
_NAME = re.compile(r"[A-Za-z_]\w*")
_VALUE = re.compile(r"-?0x[0-9a-f]+")

class SymtabError(ValueError):

    def __init__(self, message, lineno=None):
        self.lineno = lineno
        super().__init__(f"line {lineno}: {message}" if lineno is not None else message)

def _records(table):

    return [(name, table.kind_of(name), table.value(name))
            for name in sorted(table.names())]

def _render(name, kind, value):
    sign = "-" if value < 0 else ""
    return f"{name} {kind} {sign}0x{abs(value):x}"

def export(source):

    if isinstance(source, loader.LoadResult):
        table = source.symtab
    elif isinstance(source, Symbols):
        table = source
    else:
        raise SymtabError(
            f"export takes a loader.LoadResult or a loader.Symbols, got a "
            f"{type(source).__name__}: a flat name -> value dict carries no kinds, so "
            f"an export built from one could not be read back into its own table")
    lines = [HEADER]
    lines.extend(_render(name, kind, value) for name, kind, value in _records(table))
    return "\n".join(lines) + "\n"

def read(text):

    lines = text.split("\n")
    if not lines or lines[0] != HEADER:
        got = lines[0] if lines else "<empty>"
        raise SymtabError(f"not an ncp8 symbol table: the first line is {got!r} "
                          f"(expected {HEADER!r})")
    table = Symbols()
    seen = {}
    for lineno, raw in enumerate(lines[1:], start=2):

        if raw == "" and lineno == len(lines):
            continue
        fields = raw.split()
        if len(fields) != 3:
            raise SymtabError(f"expected 'name kind value', got {raw!r}", lineno)
        name, kind, value = fields
        if not _NAME.fullmatch(name):
            raise SymtabError(f"invalid symbol name {name!r}", lineno)
        if kind not in KINDS:
            raise SymtabError(f"unknown kind {kind!r} (expected "
                              f"{' or '.join(repr(k) for k in KINDS)})", lineno)
        if not _VALUE.fullmatch(value):
            raise SymtabError(f"value {value!r} is not the hex integer this format "
                              f"writes (0x-prefixed, lowercase)", lineno)
        if name in seen:
            raise SymtabError(f"symbol {name!r} defined twice: lines {seen[name]} "
                              f"and {lineno}", lineno)
        seen[name] = lineno
        table.add(name, int(value, 16), lineno, const=(kind == "constant"))
    return table

if __name__ == "__main__":
    import sys

    src = Symbols()
    src.add("main", 0x2D, 1, const=False)
    src.add("handler", 0x28, 2, const=False)
    src.add("SCALE", 0x20, 3, const=True)
    src.add("NEG", -1, 4, const=True)
    one, two = export(src), export(src)
    bad = []
    if one != two:
        bad.append(f"two exports of one table differ ({one!r} vs {two!r})")
    back = read(one)
    for name in sorted(src.names()):
        if back.value(name) != src.value(name) or back.kind_of(name) != src.kind_of(name):
            bad.append(f"{name}: read back as "
                       f"({back.value(name)}, {back.kind_of(name)}), "
                       f"was ({src.value(name)}, {src.kind_of(name)})")
    if back.names() != src.names():
        bad.append(f"names {sorted(back.names())} vs {sorted(src.names())}")

    def refuses(text):

        try:
            read(text)
        except SymtabError as e:
            return str(e)
        return None

    for label, text, *needle in (
            ("duplicate symbol", HEADER + "\nx address 0x0\nx constant 0x1\n",
             "defined twice", "lines 2 and 3"),
            ("unknown kind", HEADER + "\nx addr 0x0\n", "unknown kind", "'addr'"),
            ("malformed line", HEADER + "\nx address\n", "name kind value"),
            ("bad value", HEADER + "\nx address 28\n", "hex integer"),
            ("bad name", HEADER + "\n0x address 0x0\n", "invalid symbol name"),
            ("no header", "x address 0x0\n", "not an ncp8 symbol table")):
        msg = refuses(text)
        if msg is None:
            bad.append(f"{label} was accepted")
        elif not all(n in msg for n in needle):
            bad.append(f"{label} refused without naming {needle}: {msg}")
    print(f"export -> read -> compare: {len(src.names())} symbols, both kinds, "
          "byte-stable across two exports")
    for b in bad:
        print("  PROBLEM", b)
    print("VERDICT:", "a symbol table is a file: byte-stable out, kind-faithful back"
          if not bad else f"{len(bad)} problem(s)")
    if bad:
        sys.exit(1)