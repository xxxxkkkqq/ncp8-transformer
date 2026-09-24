"""Acceptance for the loader, disassembler, profiler and debugger.

The central check is a round trip that assumes no fixed point: every assigned encoding is
assembled, disassembled and re-assembled, and the byte strings must agree, with the
non-canonical forms reported as the exceptions they are rather than folded into the pass
count. Coverage is asserted as set equality against the decode tables, so a sweep that
visits the wrong members of the encoding space fails even when it visits the right number
of them.

Also checked: the debugger's replay fails on a tampered frame, a tampered pre-state, a
tampered write set and a dropped frame; a recording that stopped at its step cap replays
exactly; input validation behaves identically under `python` and `python -O`; and no
module in this package reaches outside its own directory to import a sibling.

Run: python3 test_toolchain.py     (also under -O)
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

import debug
import disasm
import loader
import profile
from golden_sim import AssemblyError, CODE_SIZE, DATA_SIZE, MachineError, NCP8, asm

IMMS = (0x00, 0x01, 0x7F, 0x80, 0xFF)
A16_HIGH = 0x80
SEEDS = {"noncanonical": 20260924}

ASSIGNED_SINGLE = len(disasm.SINGLE)
ASSIGNED_ESC = len(disasm.ESC)
ASSIGNED = ASSIGNED_SINGLE + ASSIGNED_ESC

def require(cond, msg):

    if not cond:
        raise AssertionError(msg)

def refuses(fn, *args, **kw):

    try:
        fn(*args, **kw)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None

OPERAND_FIELDS = ("{a16}", "{i16}", "{i8}", "{rcanon}", "{off}", "{soff}", "{k}")

REG_FIELD_MAX = (1 << 2) - 1

def byte_list(values):

    return ", ".join(f"0x{v:02X}" for v in sorted(values)) if values else "nothing"

def encoding_labels(keys):

    parts = []
    for key in sorted(keys):
        if key >= 0x100:
            parts.append(f"0x70 {key & 0xFF:02X} {disasm.ESC[key & 0xFF][0]!r}")
        else:
            parts.append(f"{key:02X} {disasm.SINGLE[key][0]!r}")
    return " | ".join(parts) if parts else "nothing"

def pair_labels(keys):

    parts = []
    for key, imm in sorted(keys):
        parts.append(f"{encoding_labels([key])} at operand {imm:02X}")
    return " | ".join(parts) if parts else "nothing"

def operand_positions(prefix, size):

    first = 2 if prefix == 0x70 else 1
    return tuple(range(first, size))

def expected_operand_encodings():

    by_template, by_size = {}, {}
    for prefix, table in ((None, disasm.SINGLE), (0x70, disasm.ESC)):
        for cp, (tmpl, size) in sorted(table.items()):
            key = cp if prefix is None else disasm.codepoint(0x70, cp)
            record = (prefix, cp, tmpl, size, operand_positions(prefix, size))
            if any(field in tmpl for field in OPERAND_FIELDS):
                by_template[key] = record
            if record[4]:
                by_size[key] = record
    require(set(by_template) == set(by_size),
            "the templates and the declared sizes disagree about which encodings take an "
            "operand byte: named only by a template ["
            f"{encoding_labels(set(by_template) - set(by_size))}], sized only ["
            f"{encoding_labels(set(by_size) - set(by_template))}]. Whichever set the sweep "
            "walks, it would then be measuring itself.")
    return by_template

def expected_flag_and_fold_keys():

    flagged, folds = set(), set()
    for cp, (tmpl, size) in disasm.SINGLE.items():
        if "{rcanon}" in tmpl and size == 2:
            folds.update((cp, imm) for imm in IMMS if imm > REG_FIELD_MAX)
    flagged.update(folds)
    for sub, (tmpl, size) in disasm.ESC.items():
        if "{k}" in tmpl and size == 3:
            key = disasm.codepoint(0x70, sub)
            flagged.update((key, imm) for imm in IMMS if imm >= loader.VEC_COUNT)
    return flagged, folds

def sweep_images():

    rows = []
    for op in sorted(disasm.SINGLE):
        _tmpl, size = disasm.SINGLE[op]
        body = bytearray([op] + [0x00] * (size - 1))
        if size >= 2:
            body[1] = 0x01
        rows.append(bytes(body))
    for sub in sorted(disasm.ESC):
        _tmpl, size = disasm.ESC[sub]
        body = bytearray([0x70, sub] + [0x00] * (size - 2))
        if size == 3:
            body[2] = 0x01
        rows.append(bytes(body))
    return rows

def test_round_trip_all_encodings():

    combos = 0
    flagged_seen = set()
    folded = {}
    want_flagged, want_folded = expected_flag_and_fold_keys()
    tables = [(None, disasm.SINGLE), (0x70, disasm.ESC)]
    for prefix, table in tables:
        for cp in sorted(table):
            _tmpl, size = table[cp]
            key = cp if prefix is None else disasm.codepoint(0x70, cp)
            for imm in IMMS:
                if prefix is None:
                    body = bytearray([cp] + [0x00] * (size - 1))
                    if size == 2:
                        body[1] = imm
                    elif size == 3:
                        body[1], body[2] = imm, A16_HIGH
                else:
                    body = bytearray([0x70, cp] + [0x00] * (size - 2))
                    if size == 3:
                        body[2] = imm
                img = bytes(body)
                combos += 1
                rows = disasm.disasm(img, 0, 1)
                require(len(rows) == 1, f"{img.hex()} decoded to {len(rows)} rows, want 1")
                addr, got_bytes, text, non_canonical = rows[0]
                require(addr == 0, f"{img.hex()} decoded at address {addr}")
                require(got_bytes == img,
                        f"{img.hex()} decoded to {got_bytes.hex()!r}: the decoder did not "
                        f"consume the whole encoding")
                if non_canonical:
                    flagged_seen.add((key, imm))
                again = refuses(lambda: asm(text))
                require(again is None, f"{img.hex()} -> {text!r} will not assemble: {again}")
                back = asm(text)
                if back == img:
                    continue

                require(non_canonical,
                        f"{img.hex()} -> {text!r} re-assembles to {back.hex()} but the "
                        f"decoder did not flag it non-canonical")
                folded[(key, imm)] = (img, text, back)
                rows2 = disasm.disasm(back, 0, 1)
                require(rows2[0][2] == text,
                        f"canonical {back.hex()} of non-canonical {img.hex()} renders as "
                        f"{rows2[0][2]!r}, not {text!r}")
                require(not rows2[0][3],
                        f"the canonical form {back.hex()} of {img.hex()} is itself flagged "
                        f"non-canonical")
                require(asm(rows2[0][2]) == back, "canonical form is not a fixed point")
    require(ASSIGNED == 355, f"assigned encoding count moved: {ASSIGNED}, expected 355 "
                             f"(single {ASSIGNED_SINGLE}, escape {ASSIGNED_ESC})")
    require(combos == ASSIGNED * len(IMMS) == 1775,
            f"sweep covered {combos} combinations, want {ASSIGNED}*{len(IMMS)}")
    require(flagged_seen == want_flagged,
            f"the decoder flagged {len(flagged_seen)} of the {combos} swept combinations, "
            f"which is not the set the operand fields predict: unexpected ["
            f"{pair_labels(flagged_seen - want_flagged)}], never flagged ["
            f"{pair_labels(want_flagged - flagged_seen)}]")
    require(set(folded) == want_folded,
            f"the swept bytes that did not re-assemble to themselves are not the set the "
            f"don't-care field predicts: unexpected ["
            f"{pair_labels(set(folded) - want_folded)}], missing ["
            f"{pair_labels(want_folded - set(folded))}]")
    print(f"  asm->disasm->asm byte-exact on {combos - len(folded)}/{combos} combinations "
          f"over {ASSIGNED} assigned encodings ({ASSIGNED_SINGLE} single-byte, "
          f"{ASSIGNED_ESC} escape), operand set {tuple(hex(v) for v in IMMS)}; the "
          f"{len(folded)} others, each one flagged and folded to a form that round trips: "
          + ", ".join(f"{img.hex()} -> {back.hex()}" for img, _t, back
                      in (folded[k] for k in sorted(folded))))

def test_zero_operand_swept():

    want = expected_operand_encodings()
    swept = {}
    for prefix, table in ((None, disasm.SINGLE), (0x70, disasm.ESC)):
        for cp in sorted(table):
            _tmpl, size = table[cp]
            positions = operand_positions(prefix, size)
            if not positions:
                continue
            key = cp if prefix is None else disasm.codepoint(0x70, cp)
            head = [cp] if prefix is None else [0x70, cp]
            img = bytes(bytearray(head + [0x00] * (size - len(head))))
            require(all(img[p] == 0x00 for p in positions),
                    f"operand 0x00 sweep reached {img.hex()} for "
                    f"{encoding_labels([key])}, whose operand byte at offset "
                    f"{[p for p in positions if img[p] != 0x00]} is not 0x00")
            row = disasm.decode(img, 0)
            require(row.assigned and row.codepoint == key and row.size == size,
                    f"{img.hex()} ({encoding_labels([key])}) is not the encoding the table "
                    f"swept: decoded as {row.text!r}, {row.size} bytes, code point "
                    f"0x{row.codepoint:04X}, assigned={row.assigned}")
            require(not row.non_canonical,
                    f"operand 0x00 of {img.hex()} ({row.text!r}) is flagged non-canonical, "
                    f"so all-zero operands are not this encoding's canonical spelling")
            text = row.text
            back = asm(text)
            require(back == img, f"operand 0x00: {img.hex()} -> {text!r} -> {back.hex()}, "
                                 f"not the bytes the tables predict")
            fixed = disasm.decode(back, 0)
            require(fixed.text == text and fixed.codepoint == key
                    and not fixed.non_canonical,
                    f"the bytes {back.hex()} that asm made for {text!r} decode as "
                    f"{fixed.text!r}, so the zero-operand round trip has no fixed point")
            for p in positions:
                patch = bytearray(img)
                patch[p] = 0x01
                moved = disasm.decode(bytes(patch), 0)
                require(moved.text != text or moved.non_canonical,
                        f"{img.hex()} offset {p}: operand 0x01 renders as {moved.text!r}, "
                        f"exactly as 0x00 does and not flagged, so offset {p} is not an "
                        f"operand byte of {encoding_labels([key])}")
                require(moved.size == size and asm(moved.text) in (bytes(patch), back),
                        f"the perturbed {bytes(patch).hex()} decodes as {moved.text!r}, "
                        f"which assembles to neither itself nor the canonical {back.hex()}")
            swept[key] = img
    require(set(swept) == set(want),
            f"the zero-operand sweep ran on {len(swept)} of the {len(want)} operand-taking "
            f"encodings: never swept [{encoding_labels(set(want) - set(swept))}], swept "
            f"without an operand byte [{encoding_labels(set(swept) - set(want))}]")
    n_single = sum(1 for prefix, _c, _t, _s, _p in want.values() if prefix is None)

    require(disasm.disasm(bytes([0x70, 0x50, 0x00]), 0, 1)[0][2] == "LDX r0, [HL]",
            "offset 0x00 must render as the bare [HL] form")
    require(disasm.disasm(bytes([0x70, 0x58, 0x00]), 0, 1)[0][2] == "ADD SP, 0",
            "signed offset 0x00 must render as 0, not as an empty operand")
    require(disasm.disasm(bytes([0x70, 0x50, 0x80]), 0, 1)[0][2] == "LDX r0, [HL-128]",
            "offset 0x80 is -128, not +128")
    print(f"  operand 0x00 swept on exactly the {len(swept)} operand-taking encodings "
          f"({n_single} single-byte, {len(swept) - n_single} escape): the bytes come from "
          f"the tables, asm reproduces them, disasm re-reads them, and moving any swept "
          f"operand byte off zero changes the instruction")

def test_all_encodings_walk_one_image():

    rows = sweep_images()
    image = b"".join(rows)
    require(len(image) <= CODE_SIZE, f"the all-encodings image is {len(image)} bytes")
    got = b"".join(r[1] for r in disasm.disasm(image))
    require(got == image, f"walking all {len(rows)} encodings reproduces {len(got)} of "
                          f"{len(image)} bytes")
    n = len(disasm.disasm(image))
    require(n == len(rows), f"the walk produced {n} rows for {len(rows)} encodings")
    print(f"  one image of {len(image)} bytes holding all {len(rows)} assigned encodings "
          f"walks back exactly ({n} rows)")

def test_reserved_cannot_re_assemble():

    undef = [op for op in range(256) if op != 0x70 and op not in disasm.SINGLE]
    reserved = [sub for sub in range(256) if sub not in disasm.ESC]
    for op in undef:
        row = disasm.disasm(bytes([op]), 0, 1)[0]
        require("DB" in row[2] and "undefined opcode" in row[2],
                f"undefined opcode {op:#04x} renders as {row[2]!r}")
        msg = refuses(asm, row[2])
        require(msg is not None, f"undefined opcode {op:#04x} disassembles to {row[2]!r} "
                                 f"which assembles into a legal instruction")
    for sub in reserved:
        img = bytes([0x70, sub])
        row = disasm.disasm(img, 0, 1)[0]
        require("reserved subcode" in row[2], f"reserved subcode {sub:#04x} renders as "
                                             f"{row[2]!r}")
        msg = refuses(asm, row[2])
        require(msg is not None, f"reserved subcode {sub:#04x} disassembles to {row[2]!r} "
                                 f"which assembles into a legal instruction")

        require(row[1] == img, f"reserved row for {img.hex()} carries {row[1].hex()}")
    require(set(undef) == set(range(256)) - set(disasm.SINGLE) - {0x70},
            "the undefined-opcode sweep is not the complement of the single-byte table: "
            f"they differ at {byte_list(set(undef) ^ (set(range(256)) - set(disasm.SINGLE) - {0x70}))}")
    require(set(reserved) == set(range(256)) - set(disasm.ESC),
            "the reserved-subcode sweep is not the complement of the escape table: they "
            f"differ at {byte_list(set(reserved) ^ (set(range(256)) - set(disasm.ESC)))}")
    print(f"  all {len(undef)} undefined opcodes and {len(reserved)} reserved subcodes "
          f"decode to DB text that none of the {len(undef) + len(reserved)} re-assembles")

def test_reserved_faults_on_the_machine():

    rng = random.Random(SEEDS["noncanonical"])
    undef = [op for op in range(256) if op != 0x70 and op not in disasm.SINGLE]
    reserved = [sub for sub in range(256) if sub not in disasm.ESC]
    sampled = [(bytes([rng.choice(undef)]), "undefined opcode") for _ in range(20)]
    sampled += [(bytes([0x70, rng.choice(reserved)]), "reserved subcode") for _ in range(20)]
    for img, kind in sampled:
        row = disasm.disasm(img, 0, 1)[0]
        want = "reserved subcode" if kind == "reserved subcode" else "undefined opcode"
        require(want in row[2], f"{img.hex()} as {kind} rendered {row[2]!r}, want it to "
                                f"name {want!r}")
        require(refuses(asm, row[2]) is not None,
                f"a sampled {kind} {img.hex()} re-assembles from {row[2]!r}")
        g = NCP8(img)
        g.r[0] = 7
        before = g.snapshot()
        msg = refuses(g.step)
        require(msg is not None and "MachineError" in msg,
                f"sampled {kind} {img.hex()} did not fault: {msg}")
        require(g.snapshot() == before, f"sampled {kind} {img.hex()} faulted non-atomically")
    for img in (bytes([0x0E]), bytes([0x70, 0x36]), bytes([0x70, 0xFF])):
        g = NCP8(img)
        g.r[0] = 7
        before = g.snapshot()
        msg = refuses(g.step)
        require(msg is not None and "MachineError" in msg,
                f"{img.hex()} did not fault on the reference: {msg}")
        after = g.snapshot()
        require(after == before, f"{img.hex()} faulted but moved state: {before} -> {after}")

    g = NCP8(asm("JNC 0x0003\nOUT r0\nHALT\n"))
    g.run()
    require(g.status == "HALT", "JNC, an assigned encoding next to the reserved gap, "
                                f"did not run: {g.status}")
    print("  the reserved bytes the decoder names do fault on the reference, atomically, "
          "and an adjacent assigned encoding still runs")

def test_non_canonical_don_tcare_bits():

    for op in (0x11, 0x12):
        for v in range(256):
            img = bytes([op, v])
            row = disasm.disasm(img, 0, 1)[0]
            want = f"ADDI {'HL' if op == 0x11 else 'DE'}, r{v & 3}"
            require(row[2] == want, f"{img.hex()} -> {row[2]!r}, want {want!r}")
            require(row[3] == (v > 3), f"{img.hex()} non_canonical={row[3]}, want "
                                       f"{v > 3}")

        canon = [v for v in range(256)
                 if not disasm.disasm(bytes([op, v]), 0, 1)[0][3]]
        require(canon == [0, 1, 2, 3], f"{op:#04x} canonical operand bytes {canon}")
        same = {disasm.disasm(bytes([op, v]), 0, 1)[0][2] for v in range(64)}
        require(len(same) == 4, f"the 64 operand bytes 0x00..0x3F for {op:#04x} render as "
                                f"{len(same)} distinct texts")

        def behaviour(v):
            g = NCP8(bytes([op, v]))
            g.r = [1, 2, 4, 8]
            g.HL = 0x100
            g.DE = 0x100
            try:
                g.step()
            except MachineError as e:
                return f"ERR {e}"
            return (g.HL, g.DE)
        require(behaviour(0x01) == behaviour(0x05) == behaviour(0x41) == behaviour(0x81),
                f"{op:#04x} with different don't-care bits behaves differently")
        require(behaviour(0x00) == behaviour(0x04) == behaviour(0x40) == behaviour(0xFC),
                f"{op:#04x}: operand r0 is not one equivalence class")
        require(behaviour(0x00) != behaviour(0x01), f"{op:#04x}: r0 and r1 behave alike")
        require(behaviour(0x03) != behaviour(0x00), f"{op:#04x}: r3 and r0 behave alike")

    for k in (0x00, 0x0F):
        row = disasm.disasm(bytes([0x70, 0x70, k]), 0, 1)[0]
        require(row[2] == f"EXT {k}" and not row[3], f"EXT {k} -> {row[2]!r} "
                                                     f"non_canonical={row[3]}")
    for k in (0x10, 0x7F, 0xFF):
        row = disasm.disasm(bytes([0x70, 0x70, k]), 0, 1)[0]
        require(row[2] == f"EXT {k}" and row[3], f"EXT {k} -> {row[2]!r} must be flagged, "
                                                 f"the vector table has 16 entries")
        msg = refuses(NCP8(bytes([0x70, 0x70, k])).step)
        require(msg is not None and "MachineError" in msg,
                f"EXT {k} should fault on the machine: {msg}")
    print("  0x11/0x12: 4 canonical operand bytes out of 256 each, the rest reported as "
          "non-canonical and verified to be the same 4 behaviours; EXT k>=16 reported "
          "non-canonical and refused by the machine")

def test_truncated_encodings_reported():

    row = disasm.disasm(bytes([0x09, 0x10]), 0, 1)[0]
    require(row[2] == "<truncated operand>", f"short JMP -> {row[2]!r}")
    require(row[3], "a truncated operand must be flagged non-canonical")
    row = disasm.disasm(bytes([0x70]), 0, 1)[0]
    require(row[2] == "<truncated escape prefix>", f"bare 0x70 -> {row[2]!r}")
    msg = refuses(asm, row[2])
    require(msg is not None, "the truncation text must not assemble")
    r = disasm.decode(bytes([0x70, 0x50]), 0)
    require(not r.assigned and r.size == 2, f"an escape cut mid-operand: {r}")
    print("  truncated operands and a truncated escape prefix are reported and refused "
          "by the assembler")

def test_loader_places_vectors_and_window():
    src = """main:
    LDI r0, 41
    EXT 0
    OUT r0
    HALT
handler:
    ADDI r0, 1
    RET
"""
    r = loader.assemble(src, vectors={0: "handler"}, window=(0x00, 0x08), entry="main")
    require(r.vectors == {0: r.symbols["handler"]}, f"vector table {r.vectors} vs symbols "
                                                    f"{r.symbols}")
    lo = loader.VEC_BASE
    cell = int.from_bytes(r.image[lo:lo + 2], "little")
    require(cell == r.symbols["handler"], f"CODE[0x0F00] holds {cell:#06x}, want "
                                          f"{r.symbols['handler']:#06x}")
    require(r.image[loader.WINDOW_LO_CELL] == 0x00 and r.image[loader.WINDOW_HI_CELL] == 0x08,
            "window bounds not at 0x0F20/0x0F21")
    require(len(r.image) >= 0x0F22, f"image with tables is only {len(r.image)} bytes")
    g = r.to_machine()
    g.r[0] = 41
    g.run()
    require(g.out == bytes([42]) and g.status == "HALT",
            f"the loader-built image does not run: {g.status} out={g.out!r} {g.trace[-1]}")
    require(r.entry == 0, f"entry {r.entry} should be main at 0")
    require("vector" in r.summary() and "window" in r.summary(),
            f"the report does not say what was placed:\n{r.summary()}")
    require(loader.describe(r) == r.summary(), "describe() is not the load report")
    require(r.content_extent == 10 and r.needed == 0x0F22,
            f"content {r.content_extent} bytes, needing {r.needed}")
    require(r.origins[loader.VEC_BASE].startswith("vector 0 -> 0x0007"),
            f"the origin of the vector cell is {r.origins[loader.VEC_BASE]!r}")
    require(r.origins[loader.WINDOW_LO_CELL].startswith("window lo -> "),
            f"the origin of the window cell is {r.origins[loader.WINDOW_LO_CELL]!r}")
    require(r.entry_explicit is True and loader.assemble("  HALT\n").entry_explicit is False,
            "the result does not say whether the entry was given or defaulted")
    print("  vectors={0:'handler'} lands at CODE[0x0F00] and window=(0,8) at 0x0F20/21; "
          "the image runs and the trap returns 42")

def test_loader_refuses_unreadable_abi():

    src = "  EXT 0\n  HALT\n  .org 0x10\nh:\n  ADDI r0, 1\n  RET\n"

    msg = refuses(loader.assemble, src, vectors={0: "h"}, image=loader.VEC_BASE + 1)
    require(msg is not None and "LoaderError" in msg, f"short image accepted: {msg}")
    for frag in ("vector 0", "0x0F00", "3841", "unregistered"):
        require(frag in msg, f"the refusal message lacks {frag!r}: {msg}")

    ok = loader.assemble(src, vectors={0: "h"}, image=loader.MIN_IMAGE_FOR_VECTORS)
    require(ok.image[loader.VEC_BASE:loader.VEC_BASE + 2] == b"\x10\x00",
            "the accepted image does not carry the vector")

    msg = refuses(loader.assemble, "  HALT\n", window=(0x00, 0x08), image=0x0F21)
    require(msg is not None and "outside window" in msg, f"short window image accepted: "
                                                         f"{msg}")
    for frag in ("0x0F20", "0x0F21", "3874"):
        require(frag in msg, f"the window refusal lacks {frag!r}: {msg}")
    require(loader.assemble("  HALT\n", window=(0x00, 0x08), image=0x0F22).length == 0x0F22,
            "0x0F22 must be enough for the window bounds")

    msg = refuses(loader.assemble, "  HALT\n", vectors={16: 0x10})
    require(msg and "0..15" in msg, f"vector index 16 accepted: {msg}")
    msg = refuses(loader.assemble, "  HALT\n", vectors={-1: 0x10})
    require(msg and "0..15" in msg, f"vector index -1 accepted: {msg}")
    msg = refuses(loader.assemble, "  HALT\n", vectors={"zero": 0x10})
    require(msg and "int" in msg, f"non-int vector index accepted: {msg}")
    msg = refuses(loader.assemble, "  HALT\n", vectors={0: 0})
    require(msg and "not registered" in msg, f"a vector at address 0 accepted: {msg}")
    msg = refuses(loader.assemble, "  HALT\n", vectors={0: 0x2000}, image=0x0F22)
    require(msg and "past the end" in msg, f"a vector past the image accepted: {msg}")
    msg = refuses(loader.assemble, "  HALT\n", vectors={0: "nothere"})
    require(msg and "undefined symbol" in msg, f"an undefined vector label accepted: {msg}")

    msg = refuses(loader.assemble, "  HALT\n", window=(0x00, 0x100), image=0x0F22)
    require(msg and "0..255" in msg, f"a window bound wider than one byte accepted: {msg}")
    msg = refuses(loader.assemble, "  HALT\n", window=(0x20, 0x10), image=0x0F22)
    require(msg and "reversed" in msg, f"reversed window bounds accepted: {msg}")
    msg = refuses(loader.assemble, "  HALT\n", window=(0x00, 0x00), image=0x0F22)
    require(msg is None, f"a zero-width window must stay legal (it is the fail-closed "
                         f"default the tests use): {msg}")

    msg = refuses(loader.assemble, "  .org 0x0F00\n  .word 0xDEAD\n", vectors={0: 0x10},
                  image=0x0F22)
    require(msg and "overwrite" in msg, f"a vector over placed data accepted: {msg}")
    print("  8 unreadable/contradictory ABI declarations refused at load time with the "
          "cell, the value and the length in the message; the two minimal lengths "
          "(0x0F02, 0x0F22) accepted")

def test_loader_abi_needed_matches_reference():

    require(loader.abi_needed(vectors={0: 0x10}) == 0x0F02, "EXT 0 needs 0x0F02")
    require(loader.abi_needed(vectors={15: 0x10}) == 0x0F20, "EXT 15 needs 0x0F20")
    require(loader.abi_needed(window=(0, 8)) == 0x0F22, "a window needs 0x0F22")
    require(loader.abi_needed() == 1, "no tables need nothing")

    src = "  EXT 0\n  HALT\n  .org 0x10\nh:\n  ADDI r0, 1\n  RET\n"
    short = loader.assemble(src, image=0x0F01).image
    g = NCP8(short)
    msg = refuses(g.step)

    require(msg and "EXT handler" in msg, f"at 0x0F01 the reference was expected to fail "
                                           f"the dispatch: {msg}")
    good = loader.assemble(src, vectors={0: "h"}, image=0x0F02).image
    g = NCP8(good)
    g.r[0] = 41
    g.run()
    require(g.r[0] == 42, f"at 0x0F02 the same vector dispatches: {g.snapshot()}")
    print("  the declared minimum lengths are the reference's real thresholds (0x0F01 "
          "fails, 0x0F02 dispatches)")

def test_loader_boundaries_both_sides():

    def accepted(**kw):
        src = kw.pop("src", "  HALT\n")
        return loader.assemble(src, **kw)

    def refused(msg_must, **kw):
        src = kw.pop("src", "  HALT\n")
        msg = refuses(loader.assemble, src, **kw)
        require(msg is not None, f"accepted what the loader should refuse: {kw}")
        for frag in msg_must:
            require(frag in msg, f"{kw} refused with {msg!r}, expected to name {frag!r}")

    vec_src = "  EXT 15\n  HALT\n  .org 0x20\nh:\n  ADDI r0, 1\n  RET\n"
    r = accepted(src=vec_src, vectors={15: "h"}, image=0x0F20)
    require(int.from_bytes(r.image[0x0F1E:0x0F20], "little") == r.symbols["h"],
            "vector 15 is not the last two bytes of the table")
    refused(["vector index", "is 16", "0..15"], vectors={16: 0x10})

    refused(["vector 15", "0x0F1E", "0x0F1F"], src=vec_src, vectors={15: "h"},
            image=0x0F1F)

    refused(["vector 0", "0x0F00", "unregistered"], vectors={0: 0x10}, image=0x0F01)
    accepted(vectors={0: 0x10}, image=0x0F02)
    refused(["0x0F20", "0x0F21"], window=(0x00, 0x08), image=0x0F21)
    accepted(window=(0x00, 0x08), image=0x0F22)
    refused(["image length", "1..4096"], image=0)
    accepted(image=1)
    accepted(image=CODE_SIZE)
    refused(["image length", "4097", "outside 1..4096"], image=CODE_SIZE + 1)

    here = "  .org 0x10\n  HALT\n"
    accepted(src=here, image=0x11)
    refused(["too short", "0x0010", "needs at least 17 bytes"], src=here,
                    image=0x10)

    accepted(src="  .org 0x0FFF\n  .byte 7\n")
    refused(["outside CODE", "0x1000"], src="  .org 0x0FFF\n  .byte 7, 8\n")
    refused(["outside CODE", "0x1000"], src="  .org 0x1000\n  HALT\n")
    accepted(src="  HALT\n  .org 0x1000\n")
    refused([".org target", "0..4096"], src="  .org 0x1001\n  HALT\n")

    accepted(src="  .org 0x0EFF\n  .byte 7\n", vectors={0: 0x10}, image=0x0F22)
    refused(["vector 0", "overwrite"], src="  .org 0x0F00\n  .byte 7\n",
            vectors={0: 0x10}, image=0x0F22)
    refused(["vector 0", "overwrite"], src="  .org 0x0F01\n  .byte 7\n",
            vectors={0: 0x10}, image=0x0F22)
    accepted(src="  .org 0x0F1F\n  .byte 7\n", window=(0x00, 0x08), image=0x0F22)
    refused(["window lo", "overwrite"], src="  .org 0x0F20\n  .byte 7\n",
            window=(0x00, 0x08), image=0x0F22)
    refused(["window hi", "overwrite"], src="  .org 0x0F21\n  .byte 7\n",
            window=(0x00, 0x08), image=0x0F22)

    accepted(vectors={0: 0x0F01}, image=0x0F02)
    refused(["vector 0", "past the end", "0x0F02"], vectors={0: 0x0F02}, image=0x0F02)
    refused(["vector 0", "not registered"], vectors={0: 0})

    accepted(entry=0x0F01, image=0x0F02)
    refused(["entry", "past the end", "0x0F02"], entry=0x0F02, image=0x0F02)

    accepted(window=(0x08, 0x08))
    accepted(window=(0x00, 0xFF), image=0x0F22)
    refused(["reversed", "0x08", "0x07"], window=(0x08, 0x07))
    refused(["window hi", "0..255"], window=(0x00, 0x100))
    refused(["window lo", "0..255"], window=(0x100, 0xFF))

    stc = loader.assemble("  LDI r0, 0xEE\n  LDI HL, 0x00FE\n  STC [HL], r0\n  HALT\n",
                          window=(0x00, 0xFF), image=0x0F22).image
    g = NCP8(stc)
    g.run()
    require(g.status == "HALT" and g.code[0x00FE] == 0xEE,
            f"the last address inside the window was not writable: {g.trace[-1]}")
    g2 = NCP8(loader.assemble("  LDI r0, 0xEE\n  LDI HL, 0x00FF\n  STC [HL], r0\n  HALT\n",
                              window=(0x00, 0xFF), image=0x0F22).image)
    msg = refuses(g2.run)

    require(msg and "STC" in msg and "0xff" in msg and g2.code[0x00FF] == 0x00,
            f"the first address past the widest window was written: {msg}")
    print("  every loader boundary checked from both sides: vector index 15/16, the "
          "0x0F01/0x0F02 and 0x0F21/0x0F22 length floors, content against the length "
          "asked for, .org at the last byte of CODE, placed data against the ABI cells, "
          "vector targets, entry, and window order/width; the byte-wide window bound "
          "cannot reach the tables, and the machine's 0xFE/0xFF edge says so")

VEC, WLO, WHI = 0x0F00, 0x0F20, 0x0F21

def hand_place_vector(code, k, addr):

    b = bytearray(code.ljust(VEC + 2 * (k + 1), b"\x00"))
    b[VEC + 2 * k:VEC + 2 * k + 2] = (addr & 0xFFFF).to_bytes(2, "little")
    return bytes(b)

def hand_with_vec(code, table):

    b = bytearray(bytes(code).ljust(max(VEC + 2 * (k + 1) for k in table), b"\x00"))
    for k, addr in table.items():
        b[VEC + 2 * k:VEC + 2 * k + 2] = (addr & 0xFFFF).to_bytes(2, "little")
    return bytes(b)

def hand_code_esc(sub, vec=0, pc=0, tail=(0xAA, 0x55)):

    b = bytearray(max(0x0F20, pc + 4))
    b[pc], b[pc + 1], b[pc + 2], b[pc + 3] = 0x70, sub, tail[0], tail[1]
    if vec:
        for k in range(16):
            b[VEC + 2 * k:VEC + 2 * k + 2] = (vec & 0xFFFF).to_bytes(2, "little")
    return bytes(b)

def hand_build_code(head, vec0=None):

    b = bytearray(bytes(head).ljust(WHI + 2, b"\x00"))
    if vec0 is not None:
        b[VEC:VEC + 2] = (vec0 & 0xFFFF).to_bytes(2, "little")
    return bytes(b)

def hand_selfmod(window_cells):

    src = """    JMP main
sub:
    LDI r0, 7
    RET
main:
    LDI HL, 0x04
    LDI r0, 42
    STC [HL], r0
    CALL sub
    OUT r0
    HALT
"""
    base = asm(src)
    b = bytearray(base.ljust(window_cells, b"\x00"))
    return b, base

def first_difference(want, got):
    for i in range(max(len(want), len(got))):
        a = want[i] if i < len(want) else None
        b = got[i] if i < len(got) else None
        if a != b:
            return i, a, b
    return None

LOADER_CASES = (
    ("test_isa_v2.test_esc_and_trap: EXT 0 -> handler at 0x10",
     lambda: hand_place_vector(bytearray(bytes([0x70, 0x70, 0x00, 0x00]).ljust(0x10, b"\x00"))
                               + bytes([0xD4 | 0, 1, 0x08]), 0, 0x10),
     lambda: loader.assemble("  EXT 0\n  HALT\n  .org 0x10\nhandler:\n  ADDI r0, 1\n"
                             "  RET\n", vectors={0: "handler"}, image=0x0F02)),
    ("test_isa_v2.test_esc_and_trap: nested EXT 0 -> EXT 1",
     lambda: (lambda h0, h1: hand_place_vector(
         hand_place_vector(bytearray(bytes([0x70, 0x70, 0x00, 0x00]).ljust(0x10, b"\x00"))
                           + h0 + h1, 0, 0x10), 1, 0x10 + len(h0)))(
         bytes([0x70, 0x70, 0x01, 0xD4 | 0, 1, 0x08]), bytes([0xD4 | 0, 1, 0x08])),
     lambda: loader.assemble("  EXT 0\n  HALT\n  .org 0x10\nh0:\n  EXT 1\n  ADDI r0, 1\n"
                             "  RET\nh1:\n  ADDI r0, 1\n  RET\n",
                             vectors={0: "h0", 1: "h1"}, image=0x0F04)),
    ("test_isa_v2.test_pc_bookkeeping: EXT 0 then LDI r3, 0xAB",
     lambda: hand_place_vector(bytearray(bytes([0x70, 0x70, 0x00, 0xD0 | 3, 0xAB, 0x00])
                                         .ljust(0x10, b"\x00")) + bytes([0xD4 | 0, 1, 0x08]),
                               0, 0x10),
     lambda: loader.assemble("  EXT 0\n  LDI r3, 0xAB\n  HALT\n  .org 0x10\nhandler:\n"
                             "  ADDI r0, 1\n  RET\n", vectors={0: "handler"}, image=0x0F02)),
    ("test_isa_v2.test_esc_and_trap: an unregistered vector (address 0)",
     lambda: hand_place_vector(bytes([0x70, 0x70, 0x00]), 0, 0),
     lambda: loader.assemble("  EXT 0\n  .org 0x0F00\n  .word 0\n", image=0x0F02)),
    ("test_isa_v2.test_selfmod: window cells at WLO+2, bounds (0x00, 0x08)",
     lambda: bytes((lambda b: (b.__setitem__(WLO, 0x00), b.__setitem__(WHI, 0x08), b)[-1])(
         hand_selfmod(WLO + 2)[0])),
     lambda: loader.assemble("""    JMP main
sub:
    LDI r0, 7
    RET
main:
    LDI HL, 0x04
    LDI r0, 42
    STC [HL], r0
    CALL sub
    OUT r0
    HALT
""", window=(0x00, 0x08), image=0x0F22, entry="main")),
    ("test_isa_v2_equivalence.test_selfmod_lockstep: window cells at WHI+1",
     lambda: bytes((lambda b: (b.__setitem__(WLO, 0x10), b.__setitem__(WHI, 0x18), b)[-1])(
         hand_selfmod(WHI + 1)[0])),
     lambda: loader.assemble("""    JMP main
sub:
    LDI r0, 7
    RET
main:
    LDI HL, 0x04
    LDI r0, 42
    STC [HL], r0
    CALL sub
    OUT r0
    HALT
""", window=(0x10, 0x18), image=0x0F22, entry="main")),
    ("test_error_atomicity.build_code: head at 0, vector 0 at 0x40, cells to WHI+2",
     lambda: hand_build_code(bytes([0x70, 0x70, 0x03]), vec0=0x0040),
     lambda: loader.assemble("  EXT 3\n", vectors={0: 0x0040}, image=WHI + 2)),
    ("test_error_atomicity.build_code: no vector registered at all",
     lambda: hand_build_code(bytes([0xD4 | 1, 0xFF])),
     lambda: loader.assemble("  ADDI r1, 0xFF\n", image=WHI + 2)),
    ("test_isa_v2_equivalence._with_vec: EXT nested, table over two vectors",
     lambda: hand_with_vec(bytearray(bytes([0x70, 0x70, 0x00, 0xD0 | 3, 0x7E, 0x00])
                                     .ljust(0x20, b"\x00"))
                           + bytes([0x70, 0x70, 0x01, 0xD4 | 0, 1, 0x08])
                           + bytes([0xD4 | 0, 5, 0x08]),
                           {0: 0x20, 1: 0x20 + len(bytes([0x70, 0x70, 0x01, 0xD4 | 0, 1, 0x08]))}),
     lambda: loader.assemble("  EXT 0\n  LDI r3, 0x7E\n  HALT\n  .org 0x20\nh0:\n  EXT 1\n"
                             "  ADDI r0, 1\n  RET\nh1:\n  ADDI r0, 5\n  RET\n",
                             vectors={0: "h0", 1: "h1"}, image=0x0F04)),
    ("test_isa_v2_equivalence._code_esc: a reserved subcode at pc=0, no vector",
     lambda: hand_code_esc(0x36, vec=0),
     lambda: loader.assemble("  .byte 0x70, 0x36, 0xAA, 0x55\n", image=0x0F20)),
    ("test_isa_v2_equivalence._code_esc: LDX at pc=256 with all 16 vectors registered",
     lambda: hand_code_esc(0x50, vec=0x0040, pc=256),
     lambda: loader.assemble("  .org 256\n  .byte 0x70, 0x50, 0xAA, 0x55\n",
                             vectors={k: 0x0040 for k in range(16)}, image=0x0F20)),
    ("test_batched_execution.ext_program: the image that file actually builds",
     lambda: (lambda b: (b.__setitem__(slice(VEC, VEC + 2), (0x20).to_bytes(2, "little")),
                         b.__setitem__(slice(VEC + 2, VEC + 4),
                                       (0x26).to_bytes(2, "little")), bytes(b))[-1])(
         bytearray(bytes([0x70, 0x70, 0x00, 0xF8 | 0, 0xD0 | 3, 0x7E, 0x00])
                   .ljust(0x20, b"\x00"))
         + bytes([0x70, 0x70, 0x01, 0xD4 | 0, 1, 0x08])
         + bytes([0xD4 | 0, 5, 0x08])),
     lambda: loader.assemble("  EXT 0\n  OUT r0\n  LDI r3, 0x7E\n  HALT\n  .org 0x20\n"
                             "h0:\n  EXT 1\n  ADDI r0, 1\n  RET\nh1:\n  ADDI r0, 5\n"
                             "  RET\n  .org 0x29\n  .word 0x0020, 0x0026\n", image=45)),
)

def test_loader_reproduces_hand_patched_images():

    lines = []
    mismatches = []
    for name, hand_fn, loader_fn in LOADER_CASES:
        want = hand_fn()
        got = loader_fn().image
        note = f"{len(want)}B == {len(got)}B"
        if want != got:
            d = first_difference(want, got)
            mismatches.append((name, d))
            note = f"DIFF at 0x{d[0]:04X}: hand {d[1]} loader {d[2]} (len {len(want)} vs " \
                   f"{len(got)})"
        lines.append(f"  {'OK  ' if want == got else 'FAIL'} {name}  [{note}]")
    require(not mismatches, "loader did not reproduce these hand-patched images:\n"
                            + "\n".join(lines))
    require(len(LOADER_CASES) == 12, f"{len(LOADER_CASES)} equivalence cases")
    print(f"loader reproduces all {len(LOADER_CASES)} hand-patched images byte for byte:")
    for ln in lines:
        print(ln)

def test_hand_patched_batched_image_is_broken_and_loader_says_so():

    broken = LOADER_CASES[11][1]()
    fixed = loader.assemble("  EXT 0\n  OUT r0\n  LDI r3, 0x7E\n  HALT\n  .org 0x20\n"
                            "h0:\n  EXT 1\n  ADDI r0, 1\n  RET\nh1:\n  ADDI r0, 5\n  RET\n",
                            vectors={0: 0x20, 1: 0x26}).image
    g = NCP8(broken)
    msg = refuses(g.step)
    require(msg is not None and "MachineError" in msg,
            f"the hand image was expected to fault, got {msg}")
    require("EXT handler" in msg, f"unexpected fault text for the hand image: {msg}")
    require(len(broken) == 45 and VEC >= len(broken),
            "the hand image is not the short one this test describes")
    require(broken[0x29:0x2D] == bytes([0x20, 0x00, 0x26, 0x00]),
            f"the misplaced table is not at 0x0029: {broken[0x29:].hex()}")
    require(fixed[VEC:VEC + 2] == b"\x20\x00", "the loader's vector table is not at 0x0F00")
    h = NCP8(fixed)
    h.run()
    require(h.status == "HALT" and h.r[0] == 6 and h.r[3] == 0x7E and h.SP == 4096,
            f"the loader-built equivalent does not run the nested chain: {h.snapshot()}")
    print("  the hand image faults as 'handler 0 unregistered' at tick 0 while the "
          "loader's placement of the same program runs the nested chain (r0=6, stack "
          "restored) - TOOLCHAIN_NOTES.md records the file to fix")

SHAPE_ONLY_HOLES = ("STC nonsense, r0", "ADD HL, r0", "XCHG DE, HL", "HALT r0",
                    "STC [HL]", "LDI")

SYMBOL_ONLY_HOLES = ("x:\nx:\nHALT",)

def test_loader_directives():
    r = loader.assemble(""".equ SLOT, 0x0040
.ascii "Hi\\0"
.org 0x20
data:
  .byte 1, 2, 0b11, low(SLOT), high(SLOT)
  .word SLOT, SLOT*2, main
  .ascii "ab"
main:
  LDI HL, SLOT
  HALT
""", image=0x40)
    require(r.image[0:3] == b"Hi\x00", f".ascii: {r.image[:4].hex()}")
    require(r.image[0x20:0x25] == bytes([1, 2, 3, 0x40, 0x00]),
            f".byte with expressions: {r.image[0x20:0x25].hex()}")
    require(int.from_bytes(r.image[0x25:0x27], "little") == 0x40, ".word little-endian")
    require(int.from_bytes(r.image[0x27:0x29], "little") == 0x80, ".word expression")
    require(int.from_bytes(r.image[0x29:0x2B], "little") == r.symbols["main"],
            ".word of a forward label reference")
    require(r.image[0x2B:0x2D] == b"ab", ".ascii after .word")
    require(r.symbols == {"SLOT": 0x40, "data": 0x20, "main": 0x2D},
            f"symbol table {r.symbols}")
    require(r.entry == r.symbols["main"], f"entry should default to main, got {r.entry}")
    require(r.image[0x2D:0x30] == asm("LDI HL, 0x0040"), f"instruction after data: "
                                                        f"{r.image[0x2D:0x30].hex()}")
    print("  .equ/.org/.byte/.word/.ascii and labels place every byte where declared; "
          "expressions, low()/high() and forward label references resolve")

def test_loader_refuses_bad_source():

    cases = (
        ("JMP nowhere\nHALT", "undefined symbol", "nowhere"),
        ("  .equ A, B\n  HALT\n", "undefined symbol", "B"),
        ("  .equ A, 1\n  .equ A, 2\n  HALT\n", "defined twice", "A"),
        ("x:\nx:\nHALT\n", "defined twice", "x"),
        ("  .org 0x10\n  HALT\n  .org 0x10\n  NOP\n", "overlaps", None),
        ("  .byte 256\n", "0..255", None),
        ("  .word 0x10000\n", "0..65535", None),
        ("  .byte 1,\n", "empty expression", None),
        ("  .org 0x2000\n  HALT\n", "outside 0..4096", None),
        ("  HALT\n  .badthing 3\n", "unknown directive", "badthing"),
        ("  LDI r0, 1/2\n", "/", None),
        ("  LDI r0, (1\n", "expected", None),
        (".ascii no quotes\n", "double-quoted", None),
        ('.ascii "unsure\\x"\n', "unknown escape", None),
        ("  .org 0x0F00\n  .byte 1\n  .org 0x0E00\n  .byte 2\n", None, None),

        ("  MOV r5, r0\n", "invalid register operand", "r5"),
        ("  LDI r0, 300\n", "0..255", None),
        ("  JMP 0x10000\n", "0..65535", None),
        ("  LDX r0, [DE+1]\n", "[HL", None),
        ("  ADD SP, 128\n", "-128..127", None),

        ("  STC nonsense, r0\n", "invalid memory operand", "nonsense"),
        ("  ADD HL, r0\n", "matches no assigned encoding", "ADD"),
        ("  XCHG DE, HL\n", "invalid operand (must read", "position 1"),
        ("  HALT r0\n", "matches no assigned encoding", "HALT"),
        ("  STC [HL]\n", "matches no assigned encoding", None),
        ("  LDI\n", "matches no assigned encoding", None),
        ("main:\n  LDI r0, main\n", "is a label", "main"),
        ("main:\n  EXT main\n", "is a label", "main"),
        ("main:\n  LDX r0, [HL+main]\n", "is a label", "main"),
        ("  HALT\n  JMP 0x0F00 + 2\n", None, None),
    )
    for src, frag, extra in cases:
        msg = refuses(loader.assemble, src)
        if frag is None:

            require(msg is None, f"refused a legal source: {src!r} -> {msg}")
            continue
        require(msg is not None, f"assembled without an error: {src!r}")
        if frag:
            require(frag in msg, f"{src!r} refused with {msg!r}, expected to mention "
                                 f"{frag!r}")
        if extra:
            require(extra in msg, f"{src!r} refused with {msg!r}, expected to name "
                                  f"{extra!r}")
        require("line" in msg, f"{src!r} refused without naming the source line: {msg}")

    lab = loader.assemble("  JMP done\nLDI r0, 1\ndone:\nHALT\n")
    require(lab.image[:6] == asm("JMP done\nLDI r0, 1\ndone:\nHALT"),
            "the loader's first six bytes differ from asm's for the same program")

    ex = loader.assemble("base:\n  HALT\n  JMP base+2\n")
    require(ex.image[1:4] == bytes([0x09, 0x02, 0x00]), f"address expression: "
                                                       f"{ex.image[1:4].hex()}")

    let_through = {}
    for s, frag, _extra in cases:
        if not frag:
            continue
        out = refuses(asm, s.strip())
        if out is None or not out.startswith("AssemblyError"):
            let_through[s.strip()] = out
    require(sorted(let_through) == sorted(SHAPE_ONLY_HOLES + SYMBOL_ONLY_HOLES),
            f"the sources the loader refuses and asm does not are now "
            f"{sorted(let_through)}, expected "
            f"{sorted(SHAPE_ONLY_HOLES + SYMBOL_ONLY_HOLES)}")
    silent = sorted(s for s, out in let_through.items() if out is None
                    and s in SHAPE_ONLY_HOLES)
    unnamed = sorted(s for s, out in let_through.items() if out)
    require(len(silent) == 4 and len(unnamed) == 2
            and len(let_through) == len(SHAPE_ONLY_HOLES) + len(SYMBOL_ONLY_HOLES),
            f"{len(silent)} of them asm encodes into another instruction and "
            f"{len(unnamed)} it fails on with an error that names nothing")
    refused_by_asm = sum(1 for s, frag, _x in cases if frag
                         and (refuses(asm, s.strip()) or "").startswith("AssemblyError"))
    named = sum(1 for _s, frag, _x in cases if frag)
    print(f"  {named} bad sources refused with the line named and the {len(cases) - named} "
          f"legal neighbours beside them accepted; {refused_by_asm} of the refusals are "
          f"asm's own, and the {len(silent)} the shape table adds are ones asm encodes as "
          f"something else ({', '.join(silent)}) or fails on unnamed "
          f"({', '.join(unnamed)})")

def test_loader_default_image_length():

    require(loader.assemble("HALT\n").length == 1, "a bare HALT is 1 byte")
    require(loader.assemble("HALT\nHALT\nHALT\n").length == 4, "3 bytes rounds to 4")
    require(loader.assemble("HALT\n", vectors={0: 0x10}).length == CODE_SIZE,
            "with a vector declared the default must clear 0x0F22")
    require(loader.assemble("HALT\n", window=(0, 8)).length == CODE_SIZE,
            "with a window declared the default must clear 0x0F22")
    big = loader.assemble("  .org 0x100\n  HALT\n")
    require(big.length == 0x200, f"content to 0x0101 gives {big.length}")
    msg = refuses(loader.assemble, "  .org 0x1001\n  HALT\n")
    require(msg and "outside 0..4096" in msg, f".org past CODE_SIZE accepted: {msg}")
    msg = refuses(loader.assemble, "  .org 0x0FFF\n  .byte 1, 2, 3\n")
    require(msg and "outside CODE" in msg, f"content past CODE_SIZE accepted: {msg}")
    print("  default length is the next power of two over the content, and 0x1000 "
          "whenever a vector or window is declared")

def test_profile_totals_match_the_machine():

    cases = []

    a = loader.assemble("  LDI r0, 3\nloop:\n  SUBI r0, 1\n  JNZ loop\n  OUT r0\n  HALT\n",
                        image=64).image
    cases.append(("halts", a, dict(tick_budget=1000)))

    b = loader.assemble("  LDI r0, 5\n  NOP\n  NOP\n  DIV r1, r2\n  HALT\n", image=64).image
    cases.append(("faults mid-run", b, dict(tick_budget=1000)))

    c = loader.assemble("loop:\n  DJNZ r0, loop\n  HALT\n", image=64).image
    cases.append(("budget exhausted", c, dict(tick_budget=97)))

    d = loader.assemble("  LDI r0, 41\n  EXT 0\n  OUT r0\n  HALT\nhandler:\n"
                        "  ADDI r0, 1\n  RET\n", vectors={0: "handler"}).image
    cases.append(("through the trap vector", d, dict(tick_budget=1000)))

    e = loader.assemble("loop:\n  OUT r0\n  JMP loop\n", image=64).image
    cases.append(("output capacity", e, dict(tick_budget=200_000)))

    f = loader.assemble("  JMP 0x0030\n", image=0x30).image
    cases.append(("PC off the image", f, dict(tick_budget=1000)))
    for name, image, kw in cases:
        p = profile.run(image, **kw)
        g = NCP8(image, **kw)
        refuses(g.run)
        machine_ticks = g.tick
        require(p.total_ticks == machine_ticks,
                f"[{name}] profile says {p.total_ticks} ticks, the machine says "
                f"{machine_ticks}")
        require(sum(p.by_codepoint.values()) == p.total_ticks,
                f"[{name}] code-point histogram sums to "
                f"{sum(p.by_codepoint.values())}, not {p.total_ticks}")
        require(sum(p.by_pc.values()) == p.total_ticks,
                f"[{name}] PC histogram sums to {sum(p.by_pc.values())}, not "
                f"{p.total_ticks}")
        p.check()
        require(p.out == bytes(g.out), f"[{name}] output differs from the machine's")
        print(f"  {name:22s} ticks={p.total_ticks:6d} steps={p.steps:6d} "
              f"codepoints={p.distinct_codepoints:3d} pcs={p.distinct_pcs:3d} "
              f"stopped={p.stopped}")

    p = profile.run(b, tick_budget=1000)
    require(len(p.faults) == 1 and p.faults[0][0] == 3, f"fault at {p.faults}")
    require(p.stopped == "fault", f"stopped={p.stopped}")
    require(p.total_ticks == 3, f"ticks before the divide-by-zero: {p.total_ticks}")
    p = profile.run(c, tick_budget=97)
    require(p.overruns == 1 and p.total_ticks == 97, f"overrun accounting: {p}")
    require(p.by_codepoint == {0x6C: 97}, f"DJNZ should own all 97 ticks: {p.by_codepoint}")
    p = profile.run(e, tick_budget=200_000)
    require(p.stopped == "fault" and len(p.faults) == 1, f"output overflow: {p.faults}")

    require(p.by_codepoint[0xF8] == 8192, f"OUT should commit exactly OUT_CAP emits: "
                                          f"{p.by_codepoint}")
    require(p.total_ticks == 8192 * 2, f"ticks around the overflow: {p.total_ticks}")
    p = profile.run(f, tick_budget=1000)
    require(p.stopped == "fault" and p.by_codepoint == {0x09: 1}, f"off-image jump: "
                                                                  f"{p.by_codepoint} "
                                                                  f"{p.faults}")
    print(f"  all {len(cases)} programs: histograms sum exactly to the machine's tick "
          f"count, with faults and the overrun accounted separately")

def test_profile_rejects_source_text():
    msg = refuses(profile.run, "HALT\n")
    require(msg and "image" in msg, f"profile accepted a bare source string: {msg}")
    msg = refuses(profile.run, b"")
    require(msg and "bytes" in msg, f"profile accepted an empty image: {msg}")
    msg = refuses(profile.run, b"\x00" * (CODE_SIZE + 1))
    require(msg and "CODE_SIZE" in msg, f"profile accepted an oversized image: {msg}")
    print("  the profiler refuses source text, an empty image and an oversized one")

def test_profile_reports_every_number_it_carries():

    r = loader.assemble(FLOW_SRC, image=64)
    p = profile.run_result(r, tick_budget=1000)
    require((p.total_ticks, p.steps, p.budget, p.length, p.out) == (11, 11, 1000, 64, b""),
            f"ticks {p.total_ticks}, steps {p.steps}, budget {p.budget}, "
            f"length {p.length}, out {p.out!r}")
    require(p.status == "HALT" and p.stopped == "halt", f"{p.status} / {p.stopped}")
    require(p.distinct_codepoints == 7 and p.distinct_pcs == 7,
            f"{p.distinct_codepoints} code points, {p.distinct_pcs} PCs")
    require(p.by_codepoint == {0xD0: 1, 0xD8: 3, 0x0B: 3, 0x0E: 1, 0xC4: 1, 0x08: 1,
                               0x00: 1}, f"code points {p.by_codepoint}")
    require(p.by_pc == {0x00: 1, 0x02: 3, 0x04: 3, 0x07: 1, 0x0A: 1, 0x0B: 1, 0x0C: 1},
            f"PCs {p.by_pc}")
    require(p.top(2) == [(0x0B, 3), (0xD8, 3)], f"hottest code points {p.top(2)}")
    require(p.hot_pcs(2) == [(0x02, 3), (0x04, 3)], f"hottest addresses {p.hot_pcs(2)}")
    require(p.mnemonic(0x0B) == "JNZ" and p.mnemonic(0x7050) == "LDX",
            f"{p.mnemonic(0x0B)} / {p.mnemonic(0x7050)}")
    require(p.mnemonic(0x70FF) == "0x70 FF" and p.mnemonic(0x99) == "SUB",
            f"{p.mnemonic(0x70FF)} / {p.mnemonic(0x99)}")
    require(len(p.rows) == p.total_ticks, f"{len(p.rows)} rows for {p.total_ticks} ticks")
    require([row[0] for row in p.rows] == list(range(p.total_ticks)),
            "the row list is not one row per committed tick, in tick order")
    require([row[1] for row in p.rows] == [f["pc"] for f in debug.record(r.image).frames],
            "the profiler's per-tick PCs disagree with the debugger's frame log")
    require(all(row[4] and not row[5] for row in p.rows),
            f"a row of an all-assigned, all-canonical program claims otherwise: {p.rows}")
    capped = profile.run(r.image, tick_budget=1000, max_rows=5)
    require(len(capped.rows) == 5 and capped.by_codepoint == p.by_codepoint
            and capped.total_ticks == p.total_ticks,
            f"max_rows changed the counts: {len(capped.rows)} rows, "
            f"{capped.total_ticks} ticks")
    lines = p.report().splitlines()
    require(lines[0] == f"ticks: 11  (steps attempted 11, faults 0, overrun 0)  "
                        f"stopped: halt  machine status: HALT  tick budget: 1000",
            f"the accounting line reads: {lines[0]!r}")
    require(lines[1] == "code point sums: 11  PC sums: 11  (must equal ticks: 11)",
            f"{lines[1]!r}")
    tampered = profile.run(r.image, tick_budget=1000)
    tampered.by_pc[0x1234] = 5
    msg = refuses(tampered.report)
    require(msg and "ProfileError" in msg, f"report() printed its claim about an "
                                           f"unbalanced profile: {msg}")
    sized = profile.run(r.image, data=bytes([0, 0, 0, 7]), tick_budget=1000)
    require(sized.total_ticks == p.total_ticks and sized.check(),
            f"a DATA image changed the run: {sized.total_ticks} against {p.total_ticks}")
    msg = refuses(profile.run, r.image, data=bytes(DATA_SIZE + 1))
    require(msg and "DATA_SIZE" in msg, f"an oversized DATA image was accepted: {msg}")
    msg = refuses(profile.run, r.image, tick_budget=True)
    require(msg and "tick_budget" in msg, f"a bool tick_budget was accepted: {msg}")
    msg = refuses(profile.run_result, loader.assemble("  HALT\nmain:\n  HALT\n"),
                  tick_budget=10)
    require(msg and "entry 0x0001" in msg and "PC 0" in msg,
            f"run_result profiled an image whose declared entry is not the boot PC: {msg}")
    require(profile.compare(p, capped)[0] == profile.compare(p)[0]
            and profile.compare(p)[0] == (11, 7, 7, "HALT", 0, 0),
            f"compare says {profile.compare(p, capped)}")
    print(f"  every Profile number is pinned: {p.total_ticks} ticks in {len(p.rows)} "
          f"rows, budget/length/out, top() and hot_pcs() by name, max_rows shortening "
          f"the trace only, and report() refusing to print an unbalanced account")

def test_profile_bills_the_instruction_that_ran():

    r = loader.assemble("""main:
  LDI r0, 5
  LDI r1, 0xF8
  LDI HL, target
  STC [HL], r1
target:
  NOP
  HALT
""", window=(0x00, 0x20), image=0x0F22, entry="main")
    p = profile.run_result(r, tick_budget=1000)
    require(p.out == bytes([5]), f"the profiled run emitted {p.out!r}; the patched byte "
                                 f"is OUT r0 and r0 holds 5")
    here = [row for row in p.rows if row[1] == r.symbols["target"]]
    require(len(here) == 1 and here[0][2] == 0xF8 and here[0][3] == "OUT r0",
            f"the tick at 0x{r.symbols['target']:04X}, where a NOP was overwritten by an "
            f"OUT r0, was billed as {here}")
    require(p.by_codepoint == {0xD0: 1, 0xD1: 1, 0x0F: 1, 0x7081: 1, 0xF8: 1, 0x00: 1},
            f"code points {p.by_codepoint}")
    p.check()
    print("  a NOP overwritten into OUT r0 is billed to OUT r0, and the emitted byte "
          f"{p.out!r} is in the profile it came from")

FLOW_SRC = """main:
  LDI r0, 3
loop:
  SUBI r0, 1
  JNZ loop
  CALL sub
  HALT
sub:
  MOV r1, r0
  RET
"""
FLOW_MAIN = [0x00, 0x02, 0x04, 0x02, 0x04, 0x02, 0x04, 0x07, 0x0B, 0x0C, 0x0A]

def test_debugger_control_flow():

    r = loader.assemble(FLOW_SRC, image=64)
    require(r.symbols == {"main": 0, "loop": 2, "sub": 0x0B}, f"symbols {r.symbols}")
    t = debug.record(r.image)
    require(t.pcs() == FLOW_MAIN, f"executed {t.pcs()}, expected {FLOW_MAIN}")
    require(debug.stop_pcs(t) == {0x00, 0x02, 0x04, 0x07, 0x0A, 0x0B, 0x0C},
            f"the debugger stopped at {sorted(debug.stop_pcs(t))}, not the addresses the "
            f"program can be walked through by hand")
    require(t.end_state["r"][0] == 0 and t.end_state["r"][1] == 0, f"final regs "
                                                                   f"{t.end_state}")
    require(t.status == "HALT", f"status {t.status}")
    t.replay()
    print(f"  the recorded PC sequence is exactly the {len(FLOW_MAIN)} expected stops "
          f"({' '.join(hex(p) for p in sorted(set(FLOW_MAIN)))}), and replay is exact")

def test_debugger_breakpoints_and_watchpoints():
    r = loader.assemble(FLOW_SRC, image=64)
    d = debug.Debug(r.image, symbols=r.symbols)
    d.break_at("loop")
    frames = d.run(max_steps=100)
    require(d.stopped == "breakpoint", f"stopped={d.stopped}")
    require([f.pc for f in frames] == [0x00] and d.PC == 0x02,
            f"the first leg ran {[f.pc for f in frames]} and stopped at {d.PC:#x}, "
            f"expected one frame then PC=loop")

    frames = d.run(max_steps=100)
    require([f.pc for f in frames] == [0x02, 0x04] and d.PC == 0x02,
            f"the second leg ran {[f.pc for f in frames]}, expected [0x02, 0x04]")
    d.drop_breakpoint("loop")
    d.break_at("sub")
    frames = d.run(max_steps=100)

    require([f.pc for f in frames] == [0x02, 0x04, 0x02, 0x04, 0x07],
            f"the third leg ran {[f.pc for f in frames]}")
    require(d.PC == 0x0B and d.stopped == "breakpoint",
            f"expected to stop at sub (0x0B), got {d.PC:#x} ({d.stopped})")
    frames = d.run(max_steps=100)
    require([f.pc for f in frames] == [0x0B, 0x0C, 0x0A] and d.stopped == "halt",
            f"the last leg ran {[f.pc for f in frames]} and stopped {d.stopped}")
    all_pcs = [f.pc for f in d.frames]
    require(all_pcs == FLOW_MAIN, f"the debugger's own frame log is {all_pcs}, expected "
                                  f"{FLOW_MAIN}")

    msg = refuses(d.drop_breakpoint, "nosuch")
    require(msg and "nosuch" in msg, f"dropping an unknown breakpoint passed silently: {msg}")
    msg = refuses(d.watch, "alsosuch")
    require(msg and "alsosuch" in msg, f"watching an unknown symbol passed silently: {msg}")

    w = loader.assemble("  LDI r0, 7\n  LDI HL, 3\n  MOV [HL], r0\n  MOV [HL], r0\n"
                        "  HALT\n", image=64)
    d = debug.Debug(w.image)
    d.watch(3)
    got = d.run()
    require(d.stopped == "watchpoint" and len(got) == 3,
            f"watchpoint stopped after {len(got)} frames ({d.stopped})")
    require(got[-1].writes == [(3, 7)], f"the store seen is {got[-1].writes}")
    d.run()
    require(d.stopped == "watchpoint", f"a second store of the same value must still fire "
                                       f"the watchpoint, got {d.stopped}")
    require(d.frames[-1].pc == 6 and d.frames[-1].writes == [(3, 7)],
            f"the repeat store frame is {d.frames[-1]!r}")

    d = debug.Debug(w.image)
    last = d.run_until(lambda dd: dd.register(0) == 7)
    require(last.pc == 0, f"condition held at {last.pc:#x}")
    msg = refuses(lambda: debug.Debug(w.image).run_until(lambda dd: False, max_steps=20))
    require(msg and "never held" in msg, f"a condition that never holds passed: {msg}")
    msg = refuses(lambda: debug.Debug(w.image).run_until(lambda dd: dd.PC == 0x7F))
    require(msg and "HALT" in msg, f"run_until should say the machine stopped first: {msg}")
    print("  breakpoints stop on the next visit to an address, watchpoints fire on a "
          "store (including one that writes the same value), and run_until refuses when "
          "the condition never holds")

def test_capped_recording_replays_exactly():

    loop = loader.assemble("  .org 0x100\nstart:\n  ADDI r0, 1\n  JMP start\n").image
    for cap in (1, 2, 37, 500):
        t = debug.run_trajectory(loop, PC=0x100, max_steps=cap)
        require(len(t.frames) == cap, f"cap {cap} recorded {len(t.frames)} frames")
        require(t.stopped == "max_steps", f"cap {cap} reported stopped={t.stopped!r}")
        try:
            t.replay()
        except Exception as e:
            raise AssertionError(f"capped recording at {cap} frames refused to replay: {e}")
    fin = loader.assemble("  .org 0x100\n  LDI r0, 9\n  MOV [HL], r0\n  HALT\n").image
    t = debug.record(fin, PC=0x100)
    require(t.stopped == "halt", f"a halting run reported stopped={t.stopped!r}")
    require(debug.replay(debug.to_dict(t)).status == t.status,
            "a recording through to_dict did not replay")
    print(f"  recordings capped at 1, 2, 37 and 500 steps each replay exactly, and "
          f"the stop reason is recorded ({t.stopped!r} for a halting run)")

def test_debugger_replay_is_exact():

    progs = []
    progs.append(("counts and halts", loader.assemble(FLOW_SRC, image=64).image, {}))
    progs.append(("trap through the vector table",
                  loader.assemble("  LDI r0, 41\n  EXT 0\n  OUT r0\n  HALT\nhandler:\n"
                                  "  ADDI r0, 1\n  RET\n", vectors={0: "handler"}).image,
                  {}))
    mod = loader.assemble("""main:
  JMP step2
sub:
  LDI r0, 7
  RET
step2:
  LDI HL, sub+1
  LDI r0, 42
  STC [HL], r0
  CALL sub
  OUT r0
  HALT
""", window=(0x00, 0x08), image=0x0F22, entry="main")
    progs.append(("self-modifying store", mod.image, {}))
    progs.append(("consumes inputs", loader.assemble("  IN r0\n  IN r1\n  OUT r0\n"
                                                     "  OUT r1\n  IN r2\n  HALT\n",
                                                     image=32).image,
                  dict(inputs=b"\x11\x22")))
    progs.append(("faults on divide by zero",
                  loader.assemble("  LDI r0, 5\n  DIV r1, r2\n  HALT\n", image=32).image,
                  {}))
    progs.append(("spends the tick budget",
                  loader.assemble("loop:\n  JMP loop\n", image=32).image,
                  dict(tick_budget=11)))
    for name, image, kw in progs:
        t = debug.record(image, **kw)
        require(len(t) >= 1, f"[{name}] recorded no frames")
        t.replay()
        fresh = debug.record(image, **kw)
        require(debug.to_dict(t) == debug.to_dict(fresh),
                f"[{name}] two recordings of the same program differ")
        if name == "self-modifying store":

            require(t.end_code[4] == 42 and t.image[4] == 7,
                    f"the self-modifying case did not modify the code image: "
                    f"{t.image[:8].hex()} -> {t.end_code[:8].hex()}")
            require(t.out == bytes([42]), f"self-modification output {t.out!r}")
        if name == "consumes inputs":
            require(t.out == b"\x11\x22", f"input echo {t.out!r}")
        print(f"  {name:28s} {len(t):3d} frames replayed exactly (out {len(t.out)}B, "
              f"DATA {sum(1 for x, y in zip(t.data, t.end_data) if x != y)} cells "
              f"changed, CODE {sum(1 for x, y in zip(t.end_code, image) if x != y)} "
              f"changed)")

    t = debug.record(loader.assemble(FLOW_SRC, image=64).image)
    t.frames[3]["post"] = dict(t.frames[3]["post"], r=[9, 9, 9, 9])
    msg = refuses(t.replay)
    require(msg and "ReplayError" in msg and "frame 3" in msg, f"a tampered frame was "
                                                              f"accepted: {msg}")
    bad = debug.record(loader.assemble(FLOW_SRC, image=64).image)
    bad.frames = bad.frames[:-1]
    msg = refuses(bad.replay)
    require(msg and "frames" in msg, f"a truncated recording was accepted: {msg}")
    print("  a tampered frame and a dropped frame are both caught by replay, so the "
          "exactness claim is a check and not a tautology")

def test_debugger_reports_atomicity():

    image = loader.assemble("  LDI r0, 5\n  LDI HL, 3\n  MOV [HL], r0\n  DIV r1, r2\n"
                            "  HALT\n", image=64).image
    d = debug.Debug(image)
    d.run(max_steps=3)
    before = (d.state(), bytes(d.m.data))
    f = d.step()
    require(f.raised is not None and "DIV" in f.raised, f"expected the divide to fault: "
                                                        f"{f.raised}")
    require(f.post is None and f.writes == [], f"a faulting frame committed something: "
                                              f"{f.post} {f.writes}")
    require(d.state() == before[0], "the faulting step changed the visible state")
    require(bytes(d.m.data) == before[1], "the faulting step changed DATA")
    require(f.tick == 3 and d.state()["tick"] == 3, "tick moved on an atomic error")
    print("  the faulting frame carries no writes and no post-state, and the machine is "
          "bit-identical across it")

def test_debugger_records_what_the_machine_did():

    acc = loader.assemble("  LDI HL, 3\n  MOV r0, [HL]\n  ADDI r0, 1\n  MOV [HL], r0\n"
                          "  HALT\n", image=64)
    t = debug.record(acc.image)
    dirty = sum(1 for v in t.data if v)
    require(t.data == bytes(DATA_SIZE),
            f"the recording is supposed to store the DATA it started from, but {dirty} of "
            f"its cells are non-zero: replay would resume from the end of the run it is "
            f"checking")
    require(t.end_data[3] == 1 and t.data[3] == 0, "the accumulating store is not in the "
                                                   "recording's end state")
    require([f["writes"] for f in t.frames] == [[], [], [], [(3, 1)], []],
            f"the store log is {t.frames}")
    t.replay()
    debug.replay(debug.to_dict(t))
    msg = refuses(debug.replay, 5)
    require(msg and "Trajectory" in msg, f"replay accepted a frame as a recording: {msg}")
    msg = refuses(debug.replay, {"image": b"\x00"})
    require(msg and "DebugError" in msg and "data" in msg,
            f"replay took a dict that is not a serialized recording: {msg}")

    mod = loader.assemble("""main:
  CALL sub
  LDI r1, 42
  LDI HL, sub+1
  STC [HL], r1
  CALL sub
  HALT
sub:
  LDI r0, 7
  RET
""", window=(0x00, 0x20), image=0x0F22, entry="main")
    t = debug.record(mod.image)
    here = [f for f in t.frames if f["pc"] == mod.symbols["sub"]]
    require([f["code"] for f in here] == [bytes([0xD0, 7]), bytes([0xD0, 42])],
            f"the two visits to sub ran {[(f['code'].hex(), f['text']) for f in here]}, "
            f"expected the loaded byte then the patched byte")
    require([f["text"] for f in here] == ["LDI r0, 0x07", "LDI r0, 0x2A"],
            f"sub disassembles as {[f['text'] for f in here]}")
    require(t.end_code[mod.symbols["sub"] + 1] == 42 and t.image[mod.symbols["sub"]
                                                                + 1] == 7,
            "the recording's load image and its end CODE are not both kept")
    t.replay()

    off = loader.assemble("  JMP 0x0030\n", image=0x30).image
    d = debug.Debug(off)
    fr = d.run()
    require(d.stopped == "fault" and [f.committed() for f in fr] == [True, False],
            f"a PC past the end of the image ran {[f.committed() for f in fr]} "
            f"({d.stopped})")
    require(fr[1].codepoint == debug.PC_OUTSIDE_IMAGE and fr[1].code == b""
            and fr[1].post is None and fr[1].raised is not None and not fr[1].assigned,
            f"the off-image frame is {fr[1]!r}")
    require(fr[1].text == "<PC outside the image>", f"off-image text {fr[1].text!r}")
    require(fr[1].state_view() == fr[1].pre, "a frame with no post-state must show its pre")
    require(debug.PC_OUTSIDE_IMAGE < 0, "the sentinel code point must not be an encoding")
    debug.record(off).replay()

    d = debug.Debug(loader.assemble("loop:\n  JMP loop\n", image=32).image, tick_budget=3)
    fr = d.run()
    require(d.stopped == "overrun" and [f.committed() for f in fr] == [True, True, True,
                                                                      False],
            f"a spent tick budget left {[f.committed() for f in fr]} ({d.stopped})")
    require(fr[-1].post is not None and fr[-1].post["tick"] == fr[-2].post["tick"] == 3
            and fr[-1].raised is None, f"the overrun frame is {fr[-1]!r}")
    debug.record(loader.assemble("loop:\n  JMP loop\n", image=32).image,
                 tick_budget=3).replay()
    d = debug.Debug(loader.assemble("loop:\n  JMP loop\n", image=32).image)
    fr = d.run(max_steps=4)
    require(d.stopped == "max_steps" and len(fr) == 4 and d.PC == 0,
            f"run(max_steps=4) on an endless loop stopped {d.stopped} after {len(fr)}")
    msg = refuses(d.run, max_steps=0)
    require(msg and "max_steps" in msg, f"max_steps=0 was accepted: {msg}")
    print(f"  a recording replays from the state it started in (DATA {DATA_SIZE} zero "
          f"cells), shows the patched bytes it actually executed, and marks the two "
          f"ticks that commit nothing")

def test_validation_survives_python_O():

    root = os.path.dirname(os.path.abspath(__file__))
    script = r"""
import sys
sys.path.insert(0, sys.argv[1])
import loader, profile, debug
from golden_sim import NCP8
n = 0
for fn, args, kw in (
    (loader.assemble, ("JMP nowhere\nHALT",), {}),
    (loader.assemble, ("  MOV r5, r0\n",), {}),
    (loader.assemble, ("  STC nonsense, r0\n",), {}),
    (loader.assemble, ("  .byte 256\n",), {}),
    (loader.assemble, ("  .equ A, 1\n  .equ A, 2\n",), {}),
    (loader.assemble, ("HALT\n",), {"vectors": {16: 0x10}}),
    (loader.assemble, ("HALT\n",), {"vectors": {0: 0x10}, "image": 0x0F01}),
    (loader.assemble, ("HALT\n",), {"window": (0, 8), "image": 0x0F21}),
    (loader.assemble, ("  .org 0x0F01\n  .byte 7\n",),
     {"vectors": {0: 0x10}, "image": 0x0F22}),
    (loader.assemble, ("  .org 0x0F20\n  .byte 7\n",),
     {"window": (0, 8), "image": 0x0F22}),
    (profile.run, ("HALT\n",), {}),
    (profile.run, (b"",), {}),
    (debug.Debug, ("HALT\n",), {}),
    (debug.Debug, (b"\x00\x00",), {"tick_budget": -1}),
    (debug.Debug, (b"\x00\x00",), {"PC": 99}),
    (debug.run_trajectory, (b"\x00\x00",), {"max_steps": 0}),
):
    try:
        fn(*args, **kw)
    except Exception as e:
        if type(e).__name__ in ("LoaderError", "ProfileError", "DebugError"):
            n += 1
        else:
            print("WRONG ERROR TYPE", type(e).__name__, e)
t = debug.record(b"\xd0\x05\x00", max_steps=10)
t.frames[0]["post"] = dict(t.frames[0]["post"], r=[1, 2, 3, 4])
try:
    t.replay()
except Exception as e:
    n += 1 if type(e).__name__ == "ReplayError" else 0
    if type(e).__name__ != "ReplayError":
        print("WRONG REPLAY ERROR", type(e).__name__, e)
print(n)
"""
    outs = []
    for flags in ([], ["-O"]):
        got = subprocess.run([sys.executable] + flags + ["-c", script, root],
                             capture_output=True, text=True)
        require(got.returncode == 0, got.stderr)
        require("WRONG" not in got.stdout, got.stdout)
        outs.append(got.stdout.strip())
    require(outs == ["17", "17"], f"guard counts differ between modes: {outs}")
    print("  17 input guards across loader/profile/debug fire identically under python "
          "and python -O, and a tampered replay still raises")

def test_no_path_hacks_and_no_silent_except():

    here = os.path.dirname(os.path.abspath(__file__))
    tmp_marker = "/" + "tmp"
    for mod in ("loader.py", "disasm.py", "profile.py", "debug.py", "test_toolchain.py"):
        text = open(os.path.join(here, mod)).read()
        require(tmp_marker not in text, f"{mod} mentions an absolute temp path")
        require("sys.path.insert" not in text or mod == "test_toolchain.py",
                f"{mod} manipulates sys.path")
        require("except" + ":" not in text, f"{mod} has a bare except")
        require("except Exception:\n        pass" not in text, f"{mod} swallows errors")
        if mod != "test_toolchain.py":

            bad = [ln for ln in text.splitlines()
                   if ln.strip().startswith("assert ") and not ln.strip().startswith("assert not")]
            require(not bad, f"{mod} validates with bare assert (gone under -O): {bad[:3]}")

    got = subprocess.run([sys.executable, "-c",
                          "import loader, disasm, profile, debug; print('ok')"],
                         capture_output=True, text=True, cwd="/",
                         env={**os.environ, "PYTHONPATH": here})
    require(got.returncode == 0 and got.stdout.strip() == "ok",
            f"the modules do not import as siblings: {got.stderr}")
    print("  loader/disasm/profile/debug import as siblings from anywhere, contain no "
          "absolute temp-path hack, no bare except and no bare-assert validation")

def test_shape_table_agrees_with_the_decoder():

    checked = 0
    for name, shapes in sorted(loader.SHAPES.items()):
        for entry in shapes:
            _n, specs, size, cps, tmpl = entry
            rendered = []
            for spec in specs:
                kind, fixed = spec
                if kind == "fixed":
                    rendered.append(fixed)
                elif kind == "reg":
                    rendered.append("r1")
                elif kind == "mem":
                    rendered.append(fixed)
                elif kind == "frame":
                    rendered.append("[HL+3]")
                elif kind == "addr16":
                    rendered.append("0x1234")
                elif kind == "soff":
                    rendered.append("-4")
                elif kind == "imm8":
                    rendered.append("3" if name == "EXT" else "0x55")
                else:
                    raise AssertionError(f"unhandled kind {kind}")
            text = f"{name} {', '.join(rendered)}" if rendered else name
            img = asm(text)
            require(len(img) == size, f"{text!r}: asm gave {len(img)} bytes, the shape "
                                      f"table says {size} ({tmpl!r})")
            row = disasm.decode(img, 0)
            require(row.size == size and row.assigned, f"{text!r} decoded as {row}")
            require(row.codepoint in cps, f"{text!r} decoded to 0x{row.codepoint:04X}, "
                                          f"not one of {[hex(c) for c in cps]} from "
                                          f"{tmpl!r}")
            checked += 1
    require(len(loader.SHAPES) >= 40, f"only {len(loader.SHAPES)} mnemonics in the shape "
                                      f"table")
    print(f"  all {checked} canonical operand shapes assemble to a code point the "
          f"decoder assigns them ({len(loader.SHAPES)} mnemonics)")

def test_loader_and_asm_agree_on_every_instruction():

    EXAMPLE = {"reg": "r2", "mem": "[DE]", "frame": "[HL]", "addr16": "0x0F05",
               "soff": "0", "imm8": "0x02"}
    shapes = sum(len(v) for v in loader.SHAPES.values())
    for name, entries in sorted(loader.SHAPES.items()):
        for entry in entries:
            _n, specs, _size, _cps, _tmpl = entry
            args = [f if k in ("fixed", "mem") else EXAMPLE[k] for k, f in specs]
            if name == "EXT":
                args = ["0"]
            text = f"{name} {', '.join(args)}" if args else name
            want = asm(text)
            got = loader.assemble(f"  {text}\n", image=max(len(want), 4)).image[:len(want)]
            require(got == want, f"{text!r}: loader placed {got.hex()}, asm emits "
                                 f"{want.hex()}")
    print(f"  {shapes} canonical shapes place asm's own bytes, operand for operand")

def main():
    print("NCP-8 toolchain acceptance (CPU only, no GPU, no circuits):")
    tests = [
        test_round_trip_all_encodings,
        test_zero_operand_swept,
        test_all_encodings_walk_one_image,
        test_reserved_cannot_re_assemble,
        test_reserved_faults_on_the_machine,
        test_non_canonical_don_tcare_bits,
        test_truncated_encodings_reported,
        test_loader_places_vectors_and_window,
        test_loader_refuses_unreadable_abi,
        test_loader_abi_needed_matches_reference,
        test_loader_boundaries_both_sides,
        test_loader_reproduces_hand_patched_images,
        test_hand_patched_batched_image_is_broken_and_loader_says_so,
        test_loader_directives,
        test_loader_refuses_bad_source,
        test_loader_default_image_length,
        test_shape_table_agrees_with_the_decoder,
        test_loader_and_asm_agree_on_every_instruction,
        test_profile_totals_match_the_machine,
        test_profile_rejects_source_text,
        test_profile_reports_every_number_it_carries,
        test_profile_bills_the_instruction_that_ran,
        test_debugger_control_flow,
        test_debugger_breakpoints_and_watchpoints,
        test_capped_recording_replays_exactly,
        test_debugger_replay_is_exact,
        test_debugger_reports_atomicity,
        test_debugger_records_what_the_machine_did,
        test_validation_survives_python_O,
        test_no_path_hacks_and_no_silent_except,
    ]
    for t in tests:
        t()
    print(f"toolchain acceptance: all {len(tests)} checks passed")

if __name__ == "__main__":
    main()