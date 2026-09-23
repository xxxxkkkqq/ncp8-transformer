# NCP-8 instruction set

## 1. Machine state

| field | size | notes |
|---|---|---|
| `r0`-`r3` | 8 bit each | general registers |
| `HL`, `DE` | 16 bit each | address pointers |
| `SP` | 16 bit | stack pointer, starts at 4096 and grows down |
| `PC` | 16 bit | program counter |
| `C`, `Z` | 1 bit each | carry and zero flags |
| `CODE` | 4096 bytes | program memory, read-only to the machine |
| `DATA` | 4096 bytes | data memory and stack |
| input / output | byte streams | `IN` / `OUT` |
| `tick` | counter | bounded by a tick budget |

Status: `0` running, `1` halted, `2` tick budget exhausted, `3` error.

## 2. Error contract

An error tick is atomic: it sets `status = 3` and changes nothing else. This
covers undefined opcodes, reserved subcodes, instruction fetch past the end of
`CODE`, data access outside `DATA`, stack underflow/overflow, division or modulo
by zero, an unregistered trap vector, and a self-modification write outside the
declared window. The tick counter advances only on a successful tick.

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
non-zero" bit; the high byte itself is not retained, so a widening product is
built with the shift/rotate family and the `ADC` chain rather than from `C`.

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
| 0x40+r | NOT r | `r = ~r` | Z |
| 0x44+r | NEG r | `r = (-r) & 0xFF` | C,Z |
| 0x48+r | ROL r | rotate left through carry (9-bit rotation: C is the ninth bit) | C,Z |
| 0x4C+r | ROR r | rotate right through carry | C,Z |
| 0x60 | ADD HL, DE | 16-bit pointer addition | C |
| 0x61 | SUB HL, DE | 16-bit pointer subtraction | C |
| 0x62 | XCHG HL, DE | swap the pointer pair | untouched |
| 0x70 k | EXT k | push return address, then jump to `vector[k]` | - |
| 0x80+r | STC [HL], r | `CODE[HL] = r`, allowed only inside the declared window | - |
| 0x84+r | LDC r, [HL] | `r = CODE[HL]` | untouched |

Every subcode not listed above is reserved and raises an atomic error, as does
any single-byte opcode not listed in 4.1-4.5.

## 5. Fixed code-region tables

Both tables live in `CODE`, which the machine cannot write, so they can only be
established at load time.

| address | contents |
|---|---|
| `CODE[0x0F00 + 2k]` | 16-bit little-endian entry point for trap `EXT k`, `k` in 0-15; a zero entry means unregistered |
| `CODE[0x0F20]`, `CODE[0x0F21]` | lower and upper bound of the self-modification window; the upper bound is exclusive |

If `WLO >= WHI` the window is empty and every `STC` raises. `EXT` with `k >= 16`
or with a zero vector raises. Both pushes follow the `CALL` convention (low byte
first), so handlers may nest and return with `RET`.

## 6. Invariants

1. Flags change only through instructions that declare it.
2. Halt happens only through `HALT` or by exhausting the tick budget.
3. Out-of-range access raises; nothing wraps silently.
4. A trace recorded from any implementation can be replayed to reproduce the
   state of the reference simulator exactly.
