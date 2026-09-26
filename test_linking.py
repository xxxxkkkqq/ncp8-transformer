"""Linking acceptance: several sources, one image, byte for byte.

`loader.link` places every unit and then resolves every name against the union of the symbol
tables, so a program split across units must come out exactly like the same program written as
one unit with hand-placed addresses -- not "equivalently", but byte for byte, holes included. A
name no unit defines, a name two units define at different addresses, two units placing content
on one byte, and a resolved value the operand cannot hold are each refused, never silently
zeroed, chosen between, shifted or truncated. The bytes a link produces are then run beside the
bytes the single-unit form produces, on the same output stream, terminal status and `DATA`.

Run: python3 test_linking.py
"""
import loader
from golden_sim import MachineError

IMAGE = 0x1000

BYTE_CASES = (
    {
        "name": "backward-jmp links to the hand-placed image",
        "units": [".org 0x100\nentry:\n  JMP helper\n  HALT\n",
                  ".org 0x200\nhelper:\n  LDI r0, 42\n  JMP entry\n"],
        "one": ".org 0x100\nentry:\n  JMP helper\n  HALT\n.org 0x200\nhelper:\n"
               "  LDI r0, 42\n  JMP entry\n",
        "windows": [(0x0100, b"\x09\x00\x02\x00"), (0x0200, b"\xd0\x2a\x09\x00\x01")],
    },
    {
        "name": "word-cross-ref links to the hand-placed image",
        "units": [".org 0x100\nentry:\n  .word table\n  .word table + 2\n",
                  ".org 0x200\ntable:\n  .byte 1, 2, 3, 4\n"],
        "one": ".org 0x100\nentry:\n  .word table\n  .word table + 2\n"
               ".org 0x200\ntable:\n  .byte 1, 2, 3, 4\n",
        "windows": [(0x0100, b"\x00\x02\x02\x02"), (0x0200, b"\x01\x02\x03\x04")],
    },
    {
        "name": "equ-across-units links to the hand-placed image",
        "units": [".equ MARK, 7\n", ".org 0x100\nentry:\n  LDI r0, MARK\n  HALT\n"],
        "one": ".equ MARK, 7\n.org 0x100\nentry:\n  LDI r0, MARK\n  HALT\n",
        "windows": [(0x0100, b"\xd0\x07\x00")],
    },
    {
        "name": "a name defined later in another unit resolves",
        "units": [".org 0x100\nentry:\n  JMP tail\n  HALT\n",
                  ".org 0x200\nmid:\n  NOP\n  JMP tail\n",
                  ".org 0x300\ntail:\n  LDI r0, 1\n  JMP entry\n"],
        "one": ".org 0x100\nentry:\n  JMP tail\n  HALT\n.org 0x200\nmid:\n  NOP\n"
               "  JMP tail\n.org 0x300\ntail:\n  LDI r0, 1\n  JMP entry\n",
        "windows": [(0x0100, b"\x09\x00\x03\x00"), (0x0200, b"\x01\x09\x00\x03"),
                    (0x0300, b"\xd0\x01\x09\x00\x01")],
    },
    {
        "name": "a .equ from one unit is read by two others",
        "units": [".equ SHIFT, 3\n", ".org 0x100\nentry:\n  LDI r0, SHIFT\n  HALT\n",
                  ".org 0x200\nother:\n  ADDI r1, SHIFT * 2\n"],
        "one": ".equ SHIFT, 3\n.org 0x100\nentry:\n  LDI r0, SHIFT\n  HALT\n"
               ".org 0x200\nother:\n  ADDI r1, 6\n",
        "windows": [(0x0100, b"\xd0\x03\x00"), (0x0200, b"\xd5\x06")],
    },
    {
        "name": "one name defined by two units at one value is one definition",
        "units": [".equ K, 5\n.org 0x100\na:\n  LDI r0, K\n",
                  ".equ K, 5\n.org 0x200\nb:\n  ADDI r1, K\n"],
        "one": ".equ K, 5\n.org 0x100\na:\n  LDI r0, 5\n.org 0x200\nb:\n  ADDI r1, 5\n",
        "windows": [(0x0100, b"\xd0\x05"), (0x0200, b"\xd5\x05")],
    },
)

REFUSALS = (
    {
        "name": "a label resolved across units still cannot sit in an 8-bit operand",
        "units": [".org 0x100\nentry:\n  LDI r0, helper\n", ".org 0x200\nhelper:\n  HALT\n"],
        "frags": ("i8 slot", "helper", "label"),
        "reason": "the operand's kind: an address symbol is not a numeric value",
    },
    {
        "name": "two units placing content on one byte are refused",
        "units": [".org 0x100\nentry:\n  LDI r0, 1\n  LDI r1, 2\n",
                  ".org 0x103\nother:\n  HALT\n"],
        "frags": ("overlap", "unit 0", "unit 1", "0x0103"),
        "reason": "the two units and the colliding range",
    },
    {
        "name": "one name defined by two units at two addresses is refused",
        "units": [".org 0x100\nshared:\n  HALT\n", ".org 0x200\nshared:\n  LDI r0, 1\n"],
        "frags": ("defined twice", "shared", "unit 0", "unit 1"),
        "reason": "the name and the two addresses it was defined at",
    },
    {
        "name": "a value resolved across units that the operand cannot hold is refused, "
                "not truncated",
        "units": [".org 0x100\nentry:\n  LDI r0, BIG\n", ".equ BIG, 300\n"],
        "frags": ("BIG", "300", "0..255"),
        "reason": "the width of the operand's own field",
    },
    {
        "name": "an unresolved cross-unit name is refused, never read as 0",
        "units": [".org 0x100\nentry:\n  JMP nowhere\n  HALT\n",
                  ".org 0x200\nhelper:\n  LDI r0, 1\n"],
        "frags": ("undefined symbol", "nowhere"),
        "reason": "the name no unit defines",
    },
    {
        "name": "a placement value cannot read a unit that is not placed yet",
        "units": [".org LATE\n  HALT\n", ".equ LATE, 0x100\nlate:\n  HALT\n"],
        "frags": ("cannot refer to", "LATE", "unit 1"),
        "reason": "the unit that defines it and the fact it is not placed yet",
    },
    {
        "name": "an empty unit list is refused",
        "units": [],
        "frags": ("at least one source",),
        "reason": "an empty list places no content, so there is nothing to link",
    },
    {
        "name": "a unit that is not source text is refused",
        "units": ["main:\n  HALT\n", 7],
        "frags": ("unit 1 must be a source string",),
        "reason": "the unit's position and the type it was handed",
    },
    {
        "name": "a list of source text, not one source string, is what links",
        "units": "main:\n  HALT\n",
        "frags": ("must be a list of source strings",),
        "reason": "one string is a program, not a list of units",
    },
    {
        "name": "a name defined as a label in one unit and a constant in another is refused",
        "units": [".equ K, 5\n.org 0x100\na:\n  LDI r0, K\n", ".org 0x200\nK:\n  HALT\n"],
        "frags": ("defined twice", "K"),
        "reason": "the same value under two kinds is two definitions",
    },
)

RUN_UNITS = (
    "main:\n  LDI r0, 40\n  LDI r2, 3\nloop:\n  ADDI r0, 1\n  EXT 0\n  OUT r0\n"
    "  SUBI r2, 1\n  JNZ loop\n  LDI HL, SLOT\n  STX [HL], r0\n  HALT\n",
    ".equ SLOT, 4\n.org 0x100\nhandler:\n  ADDI r0, 1\n  RET\n",
)
RUN_ONE = (".equ SLOT, 4\nmain:\n  LDI r0, 40\n  LDI r2, 3\nloop:\n  ADDI r0, 1\n"
           "  EXT 0\n  OUT r0\n  SUBI r2, 1\n  JNZ loop\n  LDI HL, SLOT\n  STX [HL], r0\n"
           "  HALT\n.org 0x100\nhandler:\n  ADDI r0, 1\n  RET\n")
RUN_KW = {"image": 0x400, "vectors": {0: "handler"}, "entry": "main"}

DECL_UNITS = [".org 0x0000\nmain:\n  LDI HL, 0x0100\n  LDI r0, 1\n  STC [HL], r0\n  HALT\n",
              ".equ SPAN, 0x0110\n.org 0x0100\nhandler:\n  RET\n  .org 0x010F\n"
              "  .byte 0\n"]
DECL_ONE = DECL_UNITS[0] + DECL_UNITS[1]
DECL_KW = {"image": 0x400, "vectors": {0: "handler"}, "window": (0x0100, "SPAN"),
           "entry": "main"}

ROWS = (tuple(c["name"] for c in BYTE_CASES) + tuple(c["name"] for c in REFUSALS) + (
    "one unit links to exactly what assemble gives the same text",
    "entry names a label in another unit and is reported",
    "the symbol table reports every unit's labels",
    "a vector and a window declared across units reach the machine",
    "placement leaves a hole and the link does not compact it",
    "the linked bytes run: same output stream",
    "the linked bytes run: same terminal status",
    "the linked bytes run: same DATA",
    "that run is a real program, not an empty compare",
))

FAILS = []
WALKED = []

def check(name, condition, detail=""):

    WALKED.append(name)
    if condition:
        print(f"  ok   {name}")
        return True
    FAILS.append(f"{name}: {detail}")
    print(f"  FAIL {name}  {detail}")
    return False

def outcome(fn):

    try:
        return fn()
    except Exception as exc:
        return False, f"the probe raised {type(exc).__name__}: {exc}"

def link_of(case):
    return loader.link(case["units"], image=IMAGE)

def byte_case_row(case):

    linked = link_of(case)
    hand = loader.assemble(case["one"], image=IMAGE)
    if bytes(linked.image) != bytes(hand.image):
        at = next(i for i in range(len(hand.image))
                  if linked.image[i] != hand.image[i])
        return False, (f"byte 0x{at:04X} is {linked.image[at]:02X}, the hand-placed "
                       f"single unit builds {hand.image[at]:02X}")
    for addr, want in case["windows"]:
        got = bytes(linked.image[addr:addr + len(want)])
        if got != want:
            return False, (f"0x{addr:04X}..0x{addr + len(want) - 1:04X} is {got.hex()}, "
                           f"the encoding table gives {want.hex()}")
    empty = [i for i in range(0x100) if linked.image[i]]
    if empty:
        return False, f"bytes were placed below 0x0100 where no unit placed any: {empty[:4]}"
    return True, ""

def refusal_row(case):

    try:
        result = loader.link(case["units"], image=IMAGE)
    except loader.LoaderError as exc:
        msg = str(exc)
        missing = [f for f in case["frags"] if f not in msg]
        if missing:
            return False, (f"refused, but the message lacks {missing} -- the row is about "
                           f"{case['reason']}: {msg!r}")
        return True, ""
    except Exception as exc:
        return False, f"raised {type(exc).__name__} instead of LoaderError: {exc}"
    return False, (f"LINKED {result.length} bytes; the row refuses for {case['reason']}, "
                   f"got image {bytes(result.image[:8]).hex()} with symbols "
                   f"{sorted(result.symbols)}")

def run_once(result):

    machine = result.to_machine(tick_budget=200)
    try:
        machine.run()
    except MachineError:
        pass
    return (machine.status, machine.fault_reason, bytes(machine.out), bytes(machine.data))

def symmetry_row():

    src = "main:\n  LDI r0, 3\nloop:\n  SUBI r0, 1\n  JNZ loop\n  HALT\n"
    one = loader.assemble(src, image=64, entry="main")
    linked = loader.link([src], image=64, entry="main")
    for field, a, b in (("image", bytes(one.image), bytes(linked.image)),
                        ("symbols", one.symbols, linked.symbols),
                        ("entry", one.entry, linked.entry),
                        ("report", one.report, linked.report)):
        if a != b:
            return False, f"the {field} differs: link gives {b!r}, assemble gives {a!r}"
    return True, ""

def entry_row():

    units = [".org 0x0000\nmain:\n  LDI r0, 1\n  OUT r0\n  HALT\n",
             ".org 0x100\nhelper:\n  LDI r0, 9\n  OUT r0\n  HALT\n"]
    linked = loader.link(units, image=0x200, entry="helper")
    if linked.entry != 0x100:
        return False, f"entry is 0x{linked.entry:04X}, `helper` is placed at 0x0100"
    if linked.entry_explicit is not True:
        return False, "the load does not report the entry as given"
    hand = loader.assemble(units[0] + units[1], image=0x200, entry="helper")
    if bytes(linked.image) != bytes(hand.image):
        return False, "the program with a cross-unit entry differs from the hand-placed one"
    ran = run_once(linked)
    if ran[2] != b"\x09" or ran[0] != "HALT":
        return False, (f"the linked image booted at its entry ends {ran[0]} emitting "
                       f"{ran[2]!r}, not the helper's 9")
    ran0 = run_once(loader.link(units, image=0x200))
    if ran0[2] != b"\x01" or ran0[0] != "HALT":
        return False, (f"the same image without the entry boots at 0 and ends "
                       f"{ran0[0]} emitting {ran0[2]!r}, not main's 1")
    return True, ""

def symbols_row():

    units = [".equ K, 5\n", ".org 0x100\na:\n  NOP\n", ".org 0x200\nb:\n  NOP\nc:\n  NOP\n"]
    linked = loader.link(units, image=0x400)
    want = {"K": 5, "a": 0x100, "b": 0x200, "c": 0x201}
    if linked.symbols != want:
        return False, f"the symbol table is {linked.symbols}, want {want}"
    return True, ""

def declarations_row():

    linked = loader.link(DECL_UNITS, **DECL_KW)
    hand = loader.assemble(DECL_ONE, **DECL_KW)
    if bytes(linked.image) != bytes(hand.image):
        return False, "the declared program's bytes differ from the hand-placed load"
    if linked.vectors != hand.vectors or linked.window != hand.window:
        return False, (f"vectors {linked.vectors} / window {linked.window}, the "
                       f"single-unit load declares {hand.vectors} / {hand.window}")
    block = linked.config()
    if block.codelen != hand.config().codelen:
        return False, (f"the block declares CODELEN {block.codelen} where the "
                       f"single-unit load declares {hand.config().codelen}")
    if block.codelen != linked.content_extent:
        return False, (f"CODELEN {block.codelen} is not the {linked.content_extent} bytes "
                       "the units placed")
    if block.vector(0) != 0x100:
        return False, f"vector 0 dispatches to 0x{block.vector(0):04X}, `handler` is 0x0100"
    if block.winlo != 0x0100 or block.winhi != 0x0110:
        return False, (f"the window is [0x{block.winlo:04X}, 0x{block.winhi:04X}), named "
                       "from a constant in the other unit")
    return True, ""

def hole_row():

    units = BYTE_CASES[0]["units"]
    linked = loader.link(units, image=IMAGE)
    if set(linked.image[:0x100]) != {0}:
        return False, "bytes were placed at 0x0000..0x00FF, where no unit put any"
    if set(linked.image[0x104:0x200]) != {0}:
        return False, "the gap between the two units was filled in"
    if linked.content_extent != 0x205 or linked.image[0x200] != 0xD0:
        return False, (f"content reaches 0x{linked.content_extent:04X} and 0x0200 holds "
                       f"{linked.image[0x200]:02X}; the last unit ends at 0x0204 holding "
                       "0xD0")
    if linked.length != IMAGE:
        return False, f"the image is {linked.length} bytes, the load asked for {IMAGE}"
    return True, ""

def main():
    print("linking acceptance (a linked program is the same bytes as a hand-placed one):")
    for case in BYTE_CASES:
        check(case["name"], *outcome(lambda c=case: byte_case_row(c)))
    for case in REFUSALS:
        check(case["name"], *outcome(lambda c=case: refusal_row(c)))
    check("one unit links to exactly what assemble gives the same text",
          *outcome(symmetry_row))
    check("entry names a label in another unit and is reported", *outcome(entry_row))
    check("the symbol table reports every unit's labels", *outcome(symbols_row))
    check("a vector and a window declared across units reach the machine",
          *outcome(declarations_row))
    check("placement leaves a hole and the link does not compact it", *outcome(hole_row))

    linked = loader.link(RUN_UNITS, **RUN_KW)
    hand = loader.assemble(RUN_ONE, **RUN_KW)
    ran_link, ran_hand = run_once(linked), run_once(hand)
    check("the linked bytes run: same output stream",
          ran_link[2] == ran_hand[2] and bool(ran_link[2]),
          f"the linked image emits {ran_link[2]!r}, the hand-placed one {ran_hand[2]!r}")
    check("the linked bytes run: same terminal status",
          ran_link[:2] == ran_hand[:2],
          f"the linked image ends {ran_link[0]} ({ran_link[1]}), the hand-placed one "
          f"ends {ran_hand[0]} ({ran_hand[1]})")
    check("the linked bytes run: same DATA", ran_link[3] == ran_hand[3],
          f"the linked image leaves {sorted(i for i, b in enumerate(ran_link[3]) if b)} "
          f"written, the hand-placed one "
          f"{sorted(i for i, b in enumerate(ran_hand[3]) if b)}")
    check("that run is a real program, not an empty compare",
          bytes(linked.image) == bytes(hand.image) and len(ran_link[2]) == 3
          and ran_link[0] == "HALT" and any(ran_link[3]),
          f"out {ran_link[2]!r} status {ran_link[0]} written cells "
          f"{[i for i, b in enumerate(ran_link[3]) if b][:8]} images equal "
          f"{bytes(linked.image) == bytes(hand.image)}")

    walked = set(WALKED)
    print(f"\n  {len(walked)} rows walked: {len(BYTE_CASES)} linked-against-hand-placed "
          f"byte comparisons ({sum(len(c['windows']) for c in BYTE_CASES)} byte windows), "
          f"{len(REFUSALS)} refusals each named by its reason, {len(RUN_UNITS)} units run "
          "on the reference machine beside their single-unit form, and "
          f"{len(linked.image)} bytes compared per run")
    missing = [r for r in ROWS if r not in walked]
    extra = [r for r in walked if r not in ROWS]
    if missing or extra:
        FAILS.append(f"the walk is not the acceptance: missing {missing}, unexpected {extra}")
        print(f"  FAIL the row list: missing {len(missing)}, unexpected {len(extra)}")
    print(f"linking acceptance: {len(FAILS)} failure(s)")
    for f in FAILS:
        print("  !!", f)
    return 1 if FAILS else 0

if __name__ == "__main__":
    raise SystemExit(main())