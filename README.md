# ncp8 transformer

NCP-8: a small machine specified in integer bit-planes.

Three independent implementations of one ISA, required to agree bit-for-bit:

| file | implementation |
|---|---|
| `golden_sim.py` | reference simulator, pure Python integers, no floating point |
| `circuit_torch.py` | datapath built from one-hot gated rows over a decode ROM |
| `circuit_triton.py` | the same cycle fused into a single Triton kernel |

Acceptance is recomputation, never inspection: every implementation is compared
against the reference per tick, field by field (registers, pointers, stack
pointer, PC, flags, memory, output stream, tick counter, status), including
error paths. See `ISA.md` for the instruction set.

## Running

```
pip install torch triton          # triton only needed for circuit_triton.py
python3 test_isa_v2.py            # instruction semantics (reference only)
python3 test_arithmetic_bounds.py # bit-width / radix bounds, carry chains
python3 test_circuit_equivalence.py   # both circuits vs reference, all 256 opcodes
python3 test_isa_v2_equivalence.py    # escape subcode space + program lockstep
python3 test_recursion.py         # multiply, nested CALL/RET, stack overflow
python3 mini_interpreter.py       # a 16-opcode interpreter implemented in NCP-8
python3 selfread.py               # programs that read their own PC/SP/flags
```

The reference-only suites run on CPU. The circuit suites need CUDA.

## Properties the test suite pins down

* **Determinism.** The reference simulator uses integers only; no floating point
  appears anywhere in the datapath, so a program's result is a function of its
  input, not of numeric precision.
* **Atomic errors.** A violating tick writes `status = 3` and nothing else: no
  register, memory, flag or PC update. Verified separately for undefined
  opcodes, out-of-range accesses, stack bounds, division by zero and writes
  outside the self-modification window.
* **Exactness across implementations.** Three implementations of the same
  specification, differing in how the transition is computed, agree bit-for-bit
  on every opcode, every escape subcode and on multi-thousand-tick program runs.

## Extension mechanism

The encoding reserves an escape prefix (`0x70`) followed by a subcode byte, so
new instructions can be added without renumbering existing ones; codes that are
not assigned are reserved and raise an atomic error. Three levels of extension
are available:

1. **New opcodes** occupy subcode slots in the escape space.
2. **User-defined instructions** dispatch through `EXT k`: the machine pushes a
   return address and jumps to the entry point stored in the vector table, which
   lives in the read-only code region and is populated at load time. A handler
   is an ordinary program, so it can be verified by running it.
3. **Controlled self-modification** through `STC [HL], r`, restricted to a
   window declared in the read-only code region, so the machine cannot widen its
   own window. An undeclared window is zero-width, which disables
   self-modification entirely. Writes outside the window raise an atomic error.

## Status

* The reference simulator and the tensor circuit implement the full instruction
  set including the escape space, the trap and the self-modification window.
* The Triton kernel currently implements the pre-extension instruction set; the
  opcodes it does not yet cover are listed explicitly in
  `test_circuit_equivalence.py` and skipped there rather than silently accepted.

## License

Apache License 2.0 (see `LICENSE`).
