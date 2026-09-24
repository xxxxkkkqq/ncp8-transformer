# NCP-8 instruction set

## 1. Machine state

| field | size | notes |
|---|---|---|
| `r0`-`r3` | 8 bit each | general registers |
| `HL`, `DE` | 16 bit each | address pointers |
| `SP` | 16 bit | stack pointer, starts at 4096 and grows down; legal values are `[0, 4096]` |
| `PC` | 16 bit | program counter |
| `C`, `Z` | 1 bit each | carry and zero flags |
| `CODE` | 4096 bytes | program memory; readable anywhere, written only by `STC`, and only inside the declared window (see 5) |
| `DATA` | 4096 bytes | data memory and stack |
| input | byte stream | `IN`, cursor `ipos` |
| output | byte stream, capacity 8192 | `OUT`; see 2.3 |
| `tick` | counter | bounded by a tick budget |

Status: `0` running, `1` halted, `2` tick budget exhausted, `3` error.

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

## 5. Fixed code-region tables

| address | contents |
|---|---|
| `CODE[0x0F00 + 2k]` | 16-bit little-endian entry point for trap `EXT k`, `k` in 0-15; a zero entry means unregistered |
| `CODE[0x0F20]`, `CODE[0x0F21]` | the self-modification window's lower bound (inclusive) and upper bound (exclusive), as two independent 8-bit bytes |

If `WLO >= WHI` the window is empty and every `STC` raises. `EXT` with `k >= 16`
or with a zero vector raises. Both pushes follow the `CALL` convention (low byte
first), so handlers may nest and return with `RET`.

### 5.1 Why the machine cannot reach these tables

`STC` writes `CODE`. It is kept out of the tables above by **two declared limits
that happen to coincide**, and readers must not confuse this with `CODE` being
write-protected, because it is not:

* the window bounds are 8-bit, so the widest expressible window is `[0x00, 0xFF]`
  and the highest address `STC` can ever reach is `0xFE`;
* the tables above start at `0x0F00`, far outside that reach.

An exhaustive sweep of all 65536 declarable `(WLO, WHI)` pairs confirms no `STC`
lands in `[0x0F00, 0x0F22)`.

**Consequence for anyone widening this.** Because the protection comes from the
bound *width* and not from read-only-ness, enlarging the window to 16-bit bounds
without further change would let `STC` reach the trap vector table and the window
bound cells themselves, so the machine could widen its own window to all of `CODE`.
Any change to the bound width must therefore be a change to *where the tables live*.
There are two ways to make that safe, and they are not equivalent: declare a
protected region that the datapath refuses regardless of the window, or keep the
constraint tables out of the addressable image so that no window can reach them. The
first still leaves the machine's own limits inside the memory the machine writes; the
second removes the reachability. This implementation currently relies on the width
coincidence described above, which is neither of the two, and is stated here rather
than presented as a design.

### 5.2 Loaded length is not the address space

`CODE` is a 4096-byte address space, but every bound the machine checks is the
**length of the loaded image**, not 4096. So an image must be long enough to
contain the tables it is supposed to have:

* `EXT k` needs the image to reach `0x0F02`, otherwise the vector reads as zero and
  the trap raises `handler k unregistered` even though the caller believes it wrote
  one;
* `STC` and the window declaration need the image to reach `0x0F22`, otherwise the
  bounds read as zero, the window is empty, and `STC` raises `outside window`
  reporting the range `[0x0, 0x0)`.

Both failure messages describe the *symptom*, not this cause. A builder placing
these tables must pad the image to at least `0x0F22` bytes.

## 6. Invariants

1. Flags change only through instructions that declare it.
2. `status` leaves `0` only through `HALT`, an exhausted tick budget, or an error
   tick, and never returns: a stopped machine commits nothing further (see 2.1).
3. Out-of-range access raises; nothing wraps silently, and no write leaves the
   machine's own buffer.
4. State is installed only through a validating constructor, and the validation
   holds under `python -O` as well as normally (see 2.4). What that constructor
   covers is the register file, the pointers, `SP`, the flags and `tick` - and
   nothing else: see 6.1.
5. All three implementations agree tick by tick, on every field, from a common
   start: same program, same input, same initial state, and the state, memories
   and output stream are compared after each tick, on the error paths as well as
   the successful ones. Agreement under a *shared* start is what is claimed and
   what is tested.

### 6.1 A run cannot be resumed mid-stream

`load_state` on all three implementations takes the registers, `HL`, `DE`, `PC`,
`SP`, `C`, `Z` and `tick` - and nothing else. It accepts no output stream, no input
cursor and no `status`, so a machine that has already emitted bytes, consumed input,
halted or faulted cannot be restored through it: the entry point named "load state"
carries a subset, and the rest silently returns to its initial value. This is a
limitation of the interface rather than a numeric discrepancy - given a complete
state the datapaths do agree, and the batched executor's `set_state` really does
accept all eleven scalar fields - but the asymmetry between the two entry points is
itself the reason the property cannot be stated as "a trace can be replayed".

Two things are true in its place. The suites compare implementations tick-by-tick from
a shared start (invariant 5), and the debugger can re-run a recording from that
recording's own beginning and demand an exact match on every frame, both memories and
the output stream. Neither is resumption from an arbitrary point.


