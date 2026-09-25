# NCP-8 instruction set

## 1. Machine state

| field | size | notes |
|---|---|---|
| `r` (`r0`-`r3`) | 8 bit each | general registers, published as one four-element list |
| `HL`, `DE` | 16 bit each | address pointers |
| `SP` | 16 bit | stack pointer, starts at 4096 and grows down; legal values are `[0, 4096]` |
| `PC` | 16 bit | program counter |
| `C`, `Z` | 1 bit each | carry and zero flags |
| `CODE` | 4096 bytes | program memory; readable anywhere, written only by `STC`, and only inside the declared window (see 5) |
| `DATA` | 4096 bytes | data memory and stack |
| input | byte stream | `IN`, cursor `ipos` |
| output | byte stream, capacity 8192 | `OUT`; see 2.3 |
| `tick` | counter | bounded by a tick budget |
| `status` | 2 bit | `0` running, `1` halted, `2` tick budget exhausted, `3` error; a terminal status is sticky (see 2.1) |
| `fault_reason` | 8 bit | which rule the machine stopped on; `0` means no fault |
| `fault_addr` | 16 bit | the address of the instruction that faulted |

The fields from `r` to `fault_addr` are what a machine publishes as its visible state, and they are the fields every cross-implementation comparison is made over.

`CODE_SIZE = DATA_SIZE = 4096` and `OUT_CAP = 8192` are declared once and shared by all
three implementations. All three sizes are required to be positive powers of two, and the
requirement is checked at import rather than assumed: the tensor and kernel paths contain
a write at `address & (SIZE - 1)` to keep a machine's store inside its own buffer, and
that mask only confines a write when the size is a power of two.

## 2. Error contract

An error tick is atomic: it sets `status = 3` and changes nothing else. This
covers undefined opcodes, reserved subcodes, instruction fetch past the end of
`CODE`, data access outside `DATA`, stack underflow/overflow, division or modulo
by zero, an unregistered trap vector, a self-modification write outside the
declared window, and a `SP` write that would leave `[0, 4096]`. The tick counter
advances only on a successful tick.

A 16-bit memory access checks both of its bytes before either one is read or
written, so a violating tick cannot commit half a word. The same rule applies to
`PUSHW`/`POPW`, which occupy two stack slots.

An error tick records *why* and *where*, in two machine fields that are part of the
state: `fault_reason` holds the cause and `fault_addr` holds the address of the
instruction whose execution faulted. `fault_addr` is the program counter at the start of
the tick, not the value left after the instruction's own fetches advanced it, so it names
the instruction a reader can look up rather than a byte inside it.

Two properties hold, in both directions, and both are checked on every path:
`fault_reason != 0` exactly when `status = 3`, and at most one cause is recorded per tick.
The list of causes lives in one place, `isa_table.FAULT_CAUSES`, numbered densely from
`0` = no fault; this document does not restate it, because a second copy of a table is how
two sources of truth start disagreeing. `test_spec_conformance.py` checks the table against
the machine.

### 2.1 Terminal status is sticky

Once `status` is `1`, `2` or `3` it never changes again, and the machine's other fields
are frozen with it. Stepping a stopped machine is a **no-op, not an error**: `step()`
returns having committed nothing, and `run()` returns immediately. This is what makes it
safe for a driver to call `step()` without first asking whether the machine is still
running, and it is the behaviour the batched executor depends on, because a batch keeps
stepping machines that halted on different ticks.

### 2.2 The tick budget is checked before the instruction

The budget is tested at the start of a tick, so the tick that exhausts it executes
nothing: `status` becomes `2`, `PC` stays where it was, and `tick` does not advance. The
same rule holds through `step()` as through `run()`; a budget is a property of the
machine, not of one particular driver.

A tick that raises is rolled back rather than left half-applied, including for exceptions
that are not machine errors: `PC` is restored, so a failed `step()` cannot advance the
program counter on its own.

### 2.3 Output capacity is an error, not a truncation

A machine may produce at most `OUT_CAP = 8192` bytes. The attempt to produce byte number
`8193` is an atomic error tick (`status = 3`, and the stream stays at 8192 bytes), rather
than a run that "succeeds" with a silently shortened output. In the batched path the
capacity is per machine: one machine overflowing cannot truncate or extend its neighbour's
stream.

### 2.4 State can only be installed through a validating constructor

`load_state`/`check_state` refuse a value outside a field's declared width and name the
field, the index and the offending value. This is enforced on every path, including under
`python -O`: validation that lives in a bare `assert` disappears with the flag, and a
silently unvalidated state constructor would make the bit-for-bit comparisons in the test
suites depend on how the interpreter was started.


## 3. Registers and flags

Flags are only modified by instructions that declare it; moves, loads, pushes,
pops and pointer increments leave them untouched. `GETF` returns flags packed as
`Z | (C << 1)`.

## 4. Instruction encoding

Fields do not overlap. `f = (r << 2) | s` with `r`, `s` in 0-3.

Which spellings the assembler accepts is stated once, in `isa_forms.FORMS`: one entry per
mnemonic, each a list of operand shapes, and a line is legal only when one of its mnemonic's
shapes matches it with every argument consumed. Both front ends consult that table -- the one
that turns text into bytes, and the loader that decides whether a line is legal before it
evaluates an operand's expression -- so the two cannot disagree about which instructions
exist. Addressing modes are operand kinds of their own, never variants folded into a register
kind. The table is checked against the assignments below: `python3 isa_forms.py` proves that
every assigned code point is claimed by exactly one shape, that every shape is backed by an
assigned code point, that operand bytes account for each shape's declared length, and that the
matcher accepts each shape's own spelling with no ambiguity. A refusal quotes the forms the
table gives for that mnemonic rather than inventing one.

### 4.1 Control and 16-bit operands (0x00-0x1F)

| code | mnemonic | effect | flags |
|---|---|---|---|
| 0x00 | HALT | stop | - |
| 0x01 | NOP | no operation | - |
| 0x02 / 0x03 | INC HL / DEC HL | `HL += 1` / `HL -= 1` | untouched |
| 0x04 | INC DE | `DE += 1` | untouched |
| 0x05 | CLC | `C = 0` | C |
| 0x06 / 0x07 | OUTM / OUTDE | output `DATA[HL]` / `DATA[DE]`, then increment the pointer | - |
| 0x08 | RET | pop 2 bytes (high first) into PC | - |
| 0x09-0x0D | JMP / JZ / JNZ / JC / JNC a16 | absolute conditional and unconditional jumps | - |
| 0x0E | CALL a16 | push next PC (low first), jump | - |
| 0x0F / 0x10 | LDI HL / LDI DE, i16 | load a 16-bit address | untouched |
| 0x11 / 0x12 | ADDI HL / ADDI DE, r | add an 8-bit register to a pointer | untouched |
| 0x13 | JPHL | `PC = HL` (indirect jump) | - |
| 0x14+r | GETPC r | `r =` address of this instruction | untouched |
| 0x18+r | GETSP r | `r = SP & 0xFF` | untouched |
| 0x1C+r | GETF r | `r = Z \| (C << 1)` | untouched |

### 4.2 Bitwise and multiply (0x20-0x5F)

| code | mnemonic | effect | flags |
|---|---|---|---|
| 0x20+f | AND r,s | `r &= s` | Z |
| 0x30+f | OR r,s | `r \|= s` | Z |
| 0x40+f | XOR r,s | `r ^= s` | Z |
| 0x50+f | MUL r,s | `r = (r * s) & 0xFF`, `C = 1 if (r * s) > 255` | C,Z |

`MUL` leaves the low byte in `r` and sets `C` to a single "the high byte is
non-zero" bit; the high byte itself is not retained. `MULH` (escape subcode
`0x90`+f) returns that high byte, so an 8x8 product is one `MUL` plus one `MULH`,
and wider products are assembled with the `ADC` chain.

### 4.3 Single-register operations (0x60-0x6F)

| code | mnemonic | effect | flags |
|---|---|---|---|
| 0x60+r | SHL r | `C = msb`, `r <<= 1` | C,Z |
| 0x64+r | SHR r | `C = lsb`, `r >>= 1` | C,Z |
| 0x68+r | TST r | `Z = (r == 0)` | Z |
| 0x6C+r | DJNZ r, a16 | `r -= 1`; jump if `r != 0` | untouched |

`DJNZ` intentionally leaves the flags alone so it can terminate a loop without
disturbing a carry chain.

### 4.4 Register-to-register (0x80-0xCF) and register-to-immediate (0xD0-0xDF)

| code | mnemonic | effect | flags |
|---|---|---|---|
| 0x80+f | ADD r,s | `r += s` | C,Z |
| 0x90+f | SUB r,s | `r -= s` | C,Z |
| 0xA0+f | ADC r,s | `r += s + C` | C,Z |
| 0xB0+f | SBB r,s | `r -= s + C` | C,Z |
| 0xC0+f | MOV r,s | `r = s` | untouched |
| 0xD0+r | LDI r, i8 | `r = i` | untouched |
| 0xD4+r | ADDI r, i8 | `r += i` | C,Z |
| 0xD8+r | SUBI r, i8 | `r -= i` | C,Z |
| 0xDC+r | ADCI r, i8 | `r += i + C` | C,Z |

### 4.5 Memory, stack and IO (0xE0-0xFF)

| code | mnemonic | effect | flags |
|---|---|---|---|
| 0xE0+r | MOV r, [HL] | `r = DATA[HL]` | untouched |
| 0xE4+r | MOV [HL], r | `DATA[HL] = r` | untouched |
| 0xE8+r | MOV r, [DE] | `r = DATA[DE]` | untouched |
| 0xEC+r | MOV [DE], r | `DATA[DE] = r` | untouched |
| 0xF0+r | PUSH r | `SP -= 1`, `DATA[SP] = r` | - |
| 0xF4+r | POP r | `r = DATA[SP]`, `SP += 1` | - |
| 0xF8+r | OUT r | output one byte | - |
| 0xFC+r | IN r | read one byte; on end of input `r = 0` and `C = 1` | C |

### 4.6 Escape prefix (0x70 + subcode)

The first byte `0x70` is a prefix: the following byte selects the operation, and
instruction length is counted from the prefix byte.

| subcode | mnemonic | effect | flags |
|---|---|---|---|
| 0x00+f | DIV r,s | `r = r / s` (integer); `s == 0` is an error | Z |
| 0x10+f | MOD r,s | `r = r % s`; `s == 0` is an error | Z |
| 0x20+f | CMP r,s | sets `Z = (r == s)`, `C = (r < s)`, writes no register | C,Z |
| 0x30 / 0x31 | MOVW HL, DE / MOVW DE, HL | copy between the two 16-bit pointers | untouched |
| 0x32 / 0x33 | MOVW HL, SP / MOVW DE, SP | copy `SP` into a pointer | untouched |
| 0x34 / 0x35 | MOVW SP, HL / MOVW SP, DE | `SP =` pointer; an out-of-range pointer is an error | untouched |
| 0x38 / 0x39 | PUSHW HL / PUSHW DE | `SP -= 2`, `DATA[SP] = low byte`, `DATA[SP+1] = high byte` | untouched |
| 0x3A / 0x3B | POPW HL / POPW DE | read the pair at `SP` low-first, then `SP += 2` | untouched |
| 0x3C / 0x3D | STW [HL], DE / STW [DE], HL | store 16 bits little-endian | untouched |
| 0x3E / 0x3F | LDW DE, [HL] / LDW HL, [DE] | load 16 bits little-endian | untouched |
| 0x40+r | NOT r | `r = ~r` | Z |
| 0x44+r | NEG r | `r = (-r) & 0xFF` | C,Z |
| 0x48+r | ROL r | rotate left through carry (9-bit rotation: C is the ninth bit) | C,Z |
| 0x4C+r | ROR r | rotate right through carry | C,Z |
| 0x50+r, i8 | LDX r, [HL+i8] | `r = DATA[(HL + i8) & 0xFFFF]`, `i8` sign-extended to 16 bits | untouched |
| 0x54+r, i8 | STX [HL+i8], r | `DATA[(HL + i8) & 0xFFFF] = r`, `i8` sign-extended | untouched |
| 0x58, i8 | ADD SP, i8 | `SP = (SP + i8) & 0xFFFF`, `i8` sign-extended; result must stay in `[0, 4096]` | untouched |
| 0x60 | ADD HL, DE | 16-bit pointer addition | C |
| 0x61 | SUB HL, DE | 16-bit pointer subtraction | C |
| 0x62 | XCHG HL, DE | swap the pointer pair | untouched |
| 0x70 k | EXT k | push return address, then jump to `vector[k]` | - |
| 0x80+r | STC [HL], r | `CODE[HL] = r`, allowed only inside the declared window | - |
| 0x84+r | LDC r, [HL] | `r = CODE[HL]` | untouched |
| 0x90+f | MULH r, s | `r = (r * s) >> 8`, the high byte of the widening product | Z |

Three subcode groups take a trailing immediate byte and are therefore 3 bytes
long: `LDX`, `STX`, `ADD SP` (and, as before, `EXT k`). Every other escape
instruction is 2 bytes: prefix plus subcode.

`PUSHW`/`POPW` store the pair low byte first, at the lower address. `CALL`,
`EXT` and `RET` push a return address low byte first as well, which places the
low byte at the *higher* address because the stack grows down; the two pair
conventions are therefore not interchangeable, and `PUSHW` pairs only with
`POPW`.

Every subcode not listed above is reserved and raises an atomic error, as does
any single-byte opcode not listed in 4.1-4.5.

### 4.7 Encodings that carry more bits than they use

Two assigned encodings have an operand byte with bits the decoder does not read, and the
disassembler marks them non-canonical: `ADDI HL, r` and `ADDI DE, r` take the register index
from the low two bits of their operand byte, so 252 of its 256 values each spell an instruction
that another of those bytes already spells.

For `ADDI HL` and `ADDI DE` the remaining bits are don't-care: `11 04` and `11 00` add the same
register, set the same flags and advance the same way. That is measured on all four
implementations for every operand value that leaves the rendered instruction unchanged, not
assumed, and executing such an instruction leaves the byte as it was loaded -- the machine reads
`CODE`, it does not canonicalise it.

`EXT k` is not like them, and no value of its operand byte is non-canonical: all eight bits
select a vector, so each of the 256 bytes means something different. What a `k` outside the
declared table means is a fault at run time (`TRAP_UNREG`), not a different register, and every
one of those bytes is spellable on both front ends. Whether a vector is registered is a property
of the configuration a machine was loaded under, so no load-time assembler can decide it, and an
encoding the machine runs is not a must-be-zero pattern.

Programs are compared between implementations by the bytes in `CODE` and what each tick commits,
so two images that differ only in the unread bits of an alias are different images; narrowing an
alias to its canonical form is a toolchain decision, and no execution path makes it.

## 5. Load-time configuration

The bounds that describe a machine rather than a program are handed in beside the
image, by `isa_table.MachineConfig`. No cell of `CODE` holds any of them, so a program
cannot read one, widen one or move one: `STC` writes program bytes and nothing else,
and the set of addresses it may write is decided before the first tick. Constructing
the block validates it once, and no instruction can reach the block itself - the state
a step publishes holds no configuration field, and the state installer takes no
constraint.

| field | accepted values | what it bounds |
|---|---|---|
| `codelen` | 0..65535 | how many `CODE` bytes are the program |
| `winlo`, `winhi` | 0..65535, supplied together or not at all | the self-modification window `[winlo, winhi)` |
| `vec` | at most 16 entries, each 0..65535 | trap entry points; `0` means unregistered |
| `tickbudget` | 0..2^62 | ticks before `OVERRUN` |
| `outcap` | a power of two, from 1 up to 32768 | output bytes before the capacity fault; a non-power-of-two is refused because the store masks the write index |
| `nbanks` | 1..65535 | carried, not yet consulted by any instruction (memory banks) |
| `tdlim` | 0..255 | carried, not yet consulted by any instruction (trap depth) |

A field left as `None` is *absent*, and absence has one meaning per field: `codelen`
is the length of the loaded image, `tickbudget` and `outcap` are the constructor
arguments, the window is the empty span (no `STC` writes anything) and the vector table
is all-zero (no `EXT` dispatches anywhere). A reversed window (`winhi < winlo`) is
refused at construction and is never read as the empty window, because that would make
a typo in one bound indistinguishable from switching self-modification off. Declaring a
bound in the block and also moving the matching constructor argument off its default to
a different number is a load-time refusal (`resolve_constraint`) rather than a
precedence rule; an argument sitting at its default is not a second declaration.

`equivalent_to_default()` reports a block that declares nothing,
`config_difference(before, after)` names the fields that moved between two blocks -
which is how the acceptance states that no tick of a program changed the machine's own
constraints - and `as_dict()` / `from_dict()` are how a block travels as data, which is
what lets a training record and a debugger recording hold the machine they describe.

### 5.1 What a run cannot change

Two independent facts can refuse an `STC`, and each names its own cause: the declared
span (`WINDOW`, with the message stating that both bounds are load-time configuration)
and the loaded content (`CODE_OOB`, for an address past the code that was placed). A
case where only one of them excludes the address is therefore distinguishable from a
case where both do, and the pair census below is judged by that decision rather than by
reading a byte back at an address that may be part of the program itself.

The addresses these values used to occupy - a 16-bit trap vector table at
`CODE[0x0F00 + 2k]` and two 8-bit window bound cells at `CODE[0x0F20]` and
`CODE[0x0F21]` - are ordinary program memory now. A write there commits, and commits
*harmlessly*: what `EXT` dispatches to and what span `STC` obeys come from the block, so
the bytes a program leaves at those addresses decide nothing. This is the property the
sweep states, and it is the reason the earlier protection - a window bound too narrow to
reach the tables - was not a design: the reachability of the machine's own limits was a
consequence of an 8-bit field's width, and widening that width would have moved the
limits inside the memory the machine writes.

Coverage of the census, since 16-bit bounds make the admissible pair set too large to
state as a total: every one of the 65536 byte-wide `(winlo, winhi)` pairs, every pair
whose bounds straddle one of the former table addresses, and every pair on a 257-address
stride across the whole span, each run aiming a write at the pair's own endpoints and
at the former table addresses and leaving every declared field exactly where the load
put it. The sweep prints the number of runs it made; the byte-wide set is covered in
full, and the two circuits are run over the region-straddling pairs.
census is checked to be able to fire: making the upper bound inclusive instead of
exclusive is rejected at the first pair it examines.

### 5.2 Loaded length is not the address space

Every bound the machine checks is the length of what was loaded, not the 4096-byte
address space: a fetch past the content faults rather than reading zeros, `codelen` is
how far the placed content reaches rather than how long the buffer holding it is, and a
declared vector target must lie inside the content. Bytes past the end of the content
are padding, so neither a trap target nor a fetch can land on them and call it code.

A declaration costs no bytes: because neither the vector table nor the window bounds are
written into the image, there is no minimum image length for a program that traps or
self-modifies, and an image too short to hold a declared handler is refused at load
rather than run as an unregistered trap.

### 5.3 Where execution starts

Every implementation boots at `CODE[0]`. The loader's `entry=` and its `main` symbol
report where a load's `main` is, and that address is not a start address: a program
whose entry is elsewhere begins with a jump to it. A start address the machine obeys
would have to be configuration, like the fields above, and is not yet.

## 6. Invariants

1. Flags change only through instructions that declare it.
2. `status` leaves `0` only through `HALT`, an exhausted tick budget, or an error
   tick, and never returns: a stopped machine commits nothing further (see 2.1).
3. Out-of-range access raises; nothing wraps silently, and no write leaves the
   machine's own buffer.
4. State is installed only through a validating constructor, and the validation
   holds under `python -O` as well as normally (see 2.4). The complete state travels
   through `record_state` / `install_state`, which carry every field of section 1 plus
   both memory images, both byte streams and the configuration block: see 6.1.
5. All three implementations agree tick by tick, on every field, from a common
   start: same program, same input, same initial state, and the state, memories
   and output stream are compared after each tick, on the error paths as well as
   the successful ones. Agreement under a *shared* start is what is claimed and
   what is tested.

### 6.1 A run can be captured and put back at an instruction boundary

Every path publishes the pair `record_state()` / `install_state(record)`: the reference
machine, the tensor circuit, the kernel circuit, and the resident batch - whose two entry
points name a row and write that row only.

A record is a dict with one entry per component, and the component list has one owner,
`isa_table.RECORD_COMPONENTS`: every field of section 1's state table, the two memory
images `CODE` and `DATA` each at their full width, the output stream `out`, the input
stream `inputs` the machine consumed its `ipos` from, and the configuration block
`block` the machine was loaded under. The output cursor is not a component: the stream
is, and a record that carried a second copy of its length could have the two disagree. Nothing is left at a reset value behind the caller's back, so a machine that has
emitted bytes, consumed input, rewritten `CODE` or stopped comes back as the same machine.

`install_state` refuses, before writing anything, a record that:

* is not that mapping of components, or is missing one, or carries one that is not a
  component;
* carries an image that is not this machine's `CODE_SIZE` / `DATA_SIZE` bytes;
* carries an output stream longer than this machine's output capacity;
* carries an input stream that is not the one this machine was given;
* carries a configuration block that is not the one this machine runs under;
* holds a field outside its declared width, or a `fault_reason` / `status` pairing the
  state table does not allow (see 2.1).

One tick is one whole instruction on every path, so every record is taken at an
instruction boundary and none needs a marker saying so. A refusal writes nothing: a
rejected record leaves the machine exactly as it was. A record
taken on one path installs on every other path, and the capturing machine and the one that
received the record then advance identically - which is measured at every tick a program
survives, and across paths.

What the pair does not do is carry a machine onto a *different* machine: the
configuration, the input stream and the image widths are components of the record and are
checked against the receiver, not applied to it. `load_state` stays the boot entry point
for a fresh machine - it installs the eight values its signature names, and no `status`,
stream or image - which is why `install_state` is the resumption path.


