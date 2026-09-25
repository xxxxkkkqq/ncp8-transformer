# ncp8 transformer

NCP-8: a small machine specified in integer bit-planes.

Three implementations of one ISA, required to agree bit-for-bit:

| file | implementation |
|---|---|
| `golden_sim.py` | reference simulator, pure Python integers, no floating point |
| `circuit_torch.py` | datapath built from one-hot gated rows over a decode ROM |
| `circuit_triton.py` | the same cycle fused into a single Triton kernel |

They share one specification and differ in how the transition is computed. They
are not independent derivations of it, so agreement between them is evidence
about transcription and about numeric determinism, not proof that the
specification is unambiguous; `ISA.md` is the normative text.

Acceptance is recomputation, never inspection: every implementation is compared
against the reference per tick, field by field (registers, pointers, stack
pointer, PC, flags, memory, output stream, tick counter, status), including
error paths. See `ISA.md` for the instruction set.

## Running

```
pip install torch triton          # triton only needed for circuit_triton.py
python3 isa_table.py                  # the decode table against the live ROMs and dispatch
python3 test_isa_v2.py                # instruction semantics (reference only)
python3 test_arithmetic_bounds.py     # bit-width / radix bounds, carry chains
python3 test_form_domain.py           # every assigned encoding is spellable on both front ends
python3 banks.py                        # what a group of machines may do to each other's DATA
python3 test_banks.py                   # page ownership, the selector, and what stays unreachable
python3 test_linking.py                 # several sources, one image, byte for byte
python3 test_debug_resume.py            # continue a session from a record, not a replay
python3 test_asm_strictness.py        # assembler must refuse, never mis-encode
python3 isa_forms.py                  # one accepted-form table, checked against the encodings
python3 test_circuit_equivalence.py   # both circuits vs reference, all 256 opcodes
python3 test_isa_v2_equivalence.py    # escape subcode space + program lockstep
python3 test_error_atomicity.py       # every bound case, on the reference too
python3 test_fault_registers.py       # why and where a tick stopped the machine
python3 test_state_contract.py        # the state contract every path must honour
python3 test_state_record.py          # capture a machine mid-run and put it back
python3 test_spec_conformance.py      # the ISA is stated once: table vs all three
python3 test_recursion.py             # multiply, nested CALL/RET, stack overflow
python3 test_batched_execution.py     # batched/resident executor vs reference
python3 test_config_block.py        # load-time bounds, same machine on all four paths
python3 test_training_records.py      # a label needs four-path agreement and a re-run
python3 mini_interpreter.py           # a 16-opcode interpreter implemented in NCP-8
python3 selfread.py                   # programs that read their own PC/SP/flags
python3 test_toolchain.py             # loader, disassembler, profiler, debugger
python3 loader.py                     # directives, symbols, entry, and what it refuses
python3 disasm.py                     # disassembly, with non-canonical detection
python3 debug.py                      # breakpoints, watchpoints, replay, checkpoint/resume
python3 profiler.py                   # per-instruction profile of a run
python3 programs.py                   # the pinned programs, each re-assembled and checked
```

The reference-only suites run on CPU. The circuit suites need CUDA.

## Toolchain

The machine is specified in integers, so reading it needs no emulation of anything
except itself. Four modules, each usable on its own:

| module | what it does |
|---|---|
| `disasm.py` | decodes one instruction from an image, and reports an encoding that is legal but not canonical rather than folding it |
| `loader.py` | source text to a placed image: directives, expressions, symbols, and the load-time declarations the machine reads but cannot write |
| `profile.py` | committed ticks attributed by code point and by PC, with faults and budget overruns accounted separately |
| `debug.py` | breakpoints, watchpoints, per-tick frames, and `replay()`, which re-runs a recording and demands an exact match |

Every one of these refuses to invent behaviour: an unassigned encoding, an ambiguous
operand form, or a collision between two load-time declarations is an error naming what
was asked for and why it cannot be honoured. `test_toolchain.py` checks the round trip
`assemble -> disassemble -> assemble` byte-for-byte over every assigned encoding, and
checks that a tampered debugging frame is caught.

## Properties the test suite pins down

* **Determinism.** The reference simulator uses integers only; no floating point
  appears anywhere in the datapath, so a program's result is a function of its
  input, not of numeric precision.
* **Atomic errors.** A violating tick writes `status = 3` and nothing else: no
  register, memory, flag or PC update. Verified separately for undefined
  opcodes, out-of-range accesses, stack bounds, division by zero, writes outside
  the self-modification window, and a program producing more output than the
  machine can hold.
* **A bounded, validated state.** `status` is sticky once terminal and stepping a
  stopped machine commits nothing; the tick budget is enforced before the
  instruction it stops, identically through `step()` and `run()`; state can only be
  installed through a constructor that rejects an out-of-width field, and that
  rejection survives `python -O`. See `ISA.md` section 2.
* **Exactness across implementations.** Three implementations of the same
  specification, differing in how the transition is computed, agree bit-for-bit
  on every opcode, every escape subcode and on multi-thousand-tick program runs.
  Agreement is checked per tick and per field, on the error paths as well as the
  successful ones; `test_state_contract.py` exists because the fields that were
  *not* compared are exactly where the divergences turned out to be.

## Extension mechanism

The encoding reserves an escape prefix (`0x70`) followed by a subcode byte, so
new instructions can be added without renumbering existing ones; codes that are
not assigned are reserved and raise an atomic error. Three levels of extension
are available:

1. **New opcodes** occupy subcode slots in the escape space.
2. **User-defined instructions** dispatch through `EXT k`: the machine pushes a
   return address and jumps to the entry point stored in the vector table, which is
   populated at load time and lives in a region of `CODE` that `STC` cannot address
   (item 3). A handler is an ordinary program, so it can be verified by running it.
3. **Controlled self-modification** through `STC [HL], r`, restricted to a window
   declared at load time. An undeclared window is zero-width, which disables
   self-modification entirely. Writes outside the window raise an atomic error.
   The window bounds are two 8-bit bytes, so the highest address `STC` can reach
   at all is `0xFE`, which is what keeps it out of the trap vector table at
   `0x0F00`. See `ISA.md` section 5.1 before widening those bounds.

## Status

All three implementations cover the full instruction set, including the escape
space, the user-defined-instruction trap and the self-modification window. No
opcode is skipped anywhere: the equivalence suite enumerates all 256 opcodes
against each circuit, and the 256 escape subcodes plus their program-level
lockstep are enumerated separately for each implementation.

## License

Apache License 2.0 (see `LICENSE`).
