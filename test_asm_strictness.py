"""Assembler strictness acceptance: no silent wrong code, no bare failures.

The two-pass assembler must keep resolving forward label references and numeric
literals in every supported base, and refuse everything else with a hard error
naming the offending symbol or instruction, the value and the source line. An
undefined symbol used to assemble to address 0, an unsupported expression was
silently zeroed, and an out-of-range immediate raised a bare ValueError without
any location; in an automated pipeline those are silent wrong programs. The
successful encodings are compared against hand-computed byte strings, so the
strictness work is shown not to have changed any working case.
"""
import random

from golden_sim import AssemblyError, MachineError, asm


MUST_FAIL = (
    ("JMP looop\nHALT", 1, ("undefined symbol", "looop")),
    ("NOP\nNOP\nJMP looop", 3, ("undefined symbol", "looop")),
    ("LDI HL, nolabel", 1, ("undefined symbol", "nolabel")),
    ("lab:\nHALT\nLDI HL, lab+2", 3, ("unsupported operand expression", "lab+2")),
    ("lab:\nLDI HL, lab*2", 2, ("unsupported operand expression", "lab*2")),
    ("LDI r0, 300", 1, ("LDI", "300", "0..255")),
    ("ADDI r0, -1", 1, ("ADDI", "-1", "0..255")),
    ("SUBI r0, 256", 1, ("SUBI", "256", "0..255")),
    ("ADCI r0, 1000", 1, ("ADCI", "1000", "0..255")),
    ("JMP 0x10000", 1, ("JMP", "65536", "0..65535")),
    ("LDI HL, 70000", 1, ("LDI", "70000", "0..65535")),
    ("EXT 300", 1, ("EXT", "300", "0..255")),
    ("lab:\nLDI r0, lab", 2, ("label", "lab", "8-bit immediate")),
    ("FOO r0, r1", 1, ("unknown instruction", "FOO")),
    ("JMP 0xZZ", 1, ("0xZZ",)),
    ("; a comment line\n\nNOP\n\nJMP oops", 5, ("undefined symbol", "oops")),
)


MUST_PASS = (
    ("forward jump", "JMP done\nLDI r0, 1\ndone:\nHALT",
     bytes([0x09, 0x05, 0x00, 0xD0, 0x01, 0x00])),
    ("forward address load", "LDI HL, target\ntarget:\nHALT",
     bytes([0x0F, 0x03, 0x00, 0x00])),
    ("backward jump", "loop:\nLDI r0, 1\nSUBI r0, 1\nJNZ loop\nHALT",
     bytes([0xD0, 0x01, 0xD8, 0x01, 0x0B, 0x00, 0x00, 0x00])),
    ("forward DJNZ", "LDI r2, 3\nDJNZ r2, out\nNOP\nout:\nHALT",
     bytes([0xD0 | 2, 0x03, 0x6C | 2, 0x06, 0x00, 0x01, 0x00])),
    ("literal bases", "LDI r0, 0x0F\nADDI r0, 0b1010\nSUBI r0, 0o17\nADCI r0, 255",
     bytes([0xD0, 0x0F, 0xD4, 0x0A, 0xD8, 0x0F, 0xDC, 0xFF])),
    ("address literals in bases", "JMP 0x0010\nJMP 16\nJMP 0b10000\nJMP 0o20",
     bytes([0x09, 0x10, 0x00]) * 4),
    ("comments and blank lines", "; header\n\nNOP ; inline\n\nHALT",
     bytes([0x01, 0x00])),
)


def test_must_fail():
    for src, line, frags in MUST_FAIL:
        try:
            code = asm(src)
        except AssemblyError as e:
            text = str(e)
            assert e.line == line, (src, "wrong line number", e.line, text)
            assert f"line {line}" in text, (src, "message lacks the line", text)
            for frag in frags:
                assert frag in text, (src, "message lacks fragment", frag, text)
        else:
            raise AssertionError((src, "assembled without an error", code.hex()))
    print(f"  assembler refuses {len(MUST_FAIL)} bad inputs "
          f"(undefined symbol / unsupported expression / out-of-range immediate / unknown mnemonic), "
          f"each naming the symbol or instruction and the source line")


def test_must_pass():
    for name, src, want in MUST_PASS:
        got = asm(src)
        assert got == want, (name, got.hex(), want.hex())


    labelled = "LDI r0, 3\nloop:\nSUBI r0, 1\nJNZ loop\nCALL sub\nHALT\nsub:\nRET\n"
    literal = "LDI r0, 3\nSUBI r0, 1\nJNZ 2\nCALL 11\nHALT\nRET\n"
    want = bytes([0xD0, 0x03, 0xD8, 0x01, 0x0B, 0x02, 0x00,
                  0x0E, 0x0B, 0x00, 0x00, 0x08])
    assert asm(labelled) == want == asm(literal), (asm(labelled).hex(), want.hex())
    print(f"  assembler keeps {len(MUST_PASS) + 2} working cases byte-exact "
          f"(forward/backward labels, DJNZ, 0x/0b/0o/decimal literals, comments)")


def test_error_type():


    try:
        asm("JMP looop")
    except MachineError as e:
        assert isinstance(e, AssemblyError) and e.line == 1, (type(e), getattr(e, "line", None))
    else:
        raise AssertionError("no error raised")

    rng = random.Random(11)
    for _ in range(50):
        name = "lbl%d" % rng.randrange(1000)
        src = f"JMP {name}\nHALT"
        try:
            asm(src)
        except AssemblyError as e:
            assert name in str(e), (src, str(e))
        else:
            raise AssertionError((src, "undefined symbol assembled silently"))
    print("  errors are AssemblyError(MachineError) with the line number; "
          "50 generated undefined symbols all refused")


if __name__ == "__main__":
    print("assembler strictness:")
    test_must_fail()
    test_must_pass()
    test_error_type()
    print("assembler strictness: all passed")