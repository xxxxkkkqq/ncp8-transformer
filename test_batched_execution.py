"""Acceptance for the resident batched executor (circuit_triton.TritonBatch).

The resident path runs B machines with the tick loop inside the kernel, so a whole
program run costs one launch instead of one launch per tick. Every case is a
recomputation against golden_sim.py, machine by machine: output bytes, status,
tick count, the whole DATA image and the whole CODE image.

  1. single-step equivalence through the batch path with B = 1 over the full
     opcode enumeration (256 opcodes x 6 states) and the full escape subcode
     enumeration (256 subcodes x 2 vector-table states x 4 states), i.e. the same
     comparison the per-tick path is accepted with;
  2. the same opcode enumeration packed 64 rows to a launch, which checks that
     the rows of one launch stay independent;
  3. an 83-machine batch of random programs (halting, erroring, escape-space,
     input-output, and the v3.0 frame-pointer function) with random inputs and
     random initial states;
  4. a 64-machine batch whose members finish at very different ticks, stepped in
     chunks and compared with the reference after every chunk, so "some halted,
     some still running" is covered directly;
  5. a 1024-machine batch, the resident path against the per-tick path, and
     run_resident() against run();
  6. a measured benchmark of both paths on one workload.

Run: python3 test_batched_execution.py            (all of the above)
     python3 test_batched_execution.py --bench    (benchmark only)
"""
import random
import sys
import time

import torch

import programs
from circuit_triton import CODE_SIZE, DATA_SIZE, TritonBatch, TritonCircuit, run_batch
from golden_sim import NCP8, MachineError, asm
from test_state_contract import assert_widths

STATUS = {"RUNNING": 0, "HALT": 1, "OVERRUN": 2, "ERR": 3}
VEC = 0x0F00
WLO, WHI = 0x0F20, 0x0F21
STATE_LEN = 14

def golden_view(g):
    return dict(r=list(g.r), HL=g.HL, DE=g.DE, SP=g.SP, PC=g.PC, C=g.C, Z=g.Z,
                ipos=g.ipos, oplen=len(g.out), tick=g.tick, status=STATUS[g.status])

def golden_machine(code, data=b"", inputs=b"", row=None, budget=200_000):

    g = NCP8(code, data=data, inputs=inputs, tick_budget=budget)
    if row is not None:
        g.load_state(row[0:4], row[4], row[5], row[7], row[8], row[9], row[12], PC=row[6])
        g.ipos = row[10]
        assert row[11] == 0 and row[13] == 0, "a fresh machine starts with no output"
        assert_widths(golden_view(g), ("golden machine", row))
    return g

def golden_step(g):

    pre, pre_data = golden_view(g), list(g.data)
    pre_code, pre_out = bytes(g.code), bytes(g.out)
    try:
        g.step()
        assert_widths(golden_view(g), ("reference post-tick", pre["tick"]))
        return False
    except MachineError:
        assert golden_view(g) == pre, ("reference error tick was not atomic", pre, golden_view(g))
        assert list(g.data) == pre_data, "reference modified DATA before raising"
        assert bytes(g.code) == pre_code, "reference modified CODE before raising"
        assert bytes(g.out) == pre_out, "reference wrote output before raising"
        return True

def golden_run(code, data=b"", inputs=b"", row=None, budget=200_000):

    g = golden_machine(code, data, inputs, row, budget)
    raised = False
    while g.status == "RUNNING" and g.tick < g.tb:
        if golden_step(g):
            raised = True
            break
    if raised:
        g.status = "ERR"
    elif g.status == "RUNNING":
        g.status = "OVERRUN"
    return g

def golden_advance(g, steps):

    for _ in range(steps):
        if g.status != "RUNNING":
            break
        if golden_step(g):
            g.status = "ERR"
            break
    return g

def padded(code):
    return bytes(code).ljust(CODE_SIZE, b"\x00")

def row_of(r0, r1, r2, r3, HL, DE, PC, SP, C, Z, ipos=0, oplen=0, tick=0, status=0):
    return [r0, r1, r2, r3, HL, DE, PC, SP, C, Z, ipos, oplen, tick, status]

def push_row(batch, i, row):

    batch.set_state(i, r=row[0:4], HL=row[4], DE=row[5], PC=row[6], SP=row[7],
                    C=row[8], Z=row[9], ipos=row[10], oplen=row[11], tick=row[12],
                    status=row[13])

def op_case(op, seed):

    rng = random.Random(seed)
    code = bytes([op, rng.randrange(256), rng.randrange(256)])
    data = bytes(rng.randrange(256) for _ in range(4096))
    inputs = bytes(rng.randrange(256) for _ in range(3))
    R = [rng.randrange(256) for _ in range(4)]
    HL = rng.choice([rng.randrange(4096), rng.randrange(4096, 4400)])
    DE = rng.choice([rng.randrange(4096), rng.randrange(4096, 4400)])
    SP = rng.choice([0, 1, 2, 3, rng.randrange(16, 4093), 4095, 4096])
    C, Z = rng.randrange(2), rng.randrange(2)
    tick = rng.randrange(100)
    return code, data, inputs, row_of(R[0], R[1], R[2], R[3], HL, DE, 0, SP, C, Z, tick=tick)

def esc_case(sub, vec, seed):

    rng = random.Random(seed * 977 + sub + (1 << 20 if vec else 0))
    b = bytearray(WLO)
    b[0], b[1], b[2], b[3] = 0x70, sub, 0xAA, 0x55
    if vec:
        for k in range(16):
            b[VEC + 2 * k: VEC + 2 * k + 2] = (vec & 0xFFFF).to_bytes(2, "little")
    code = bytes(b)
    data = bytes(rng.randrange(256) for _ in range(4096))
    R = [rng.randrange(256) for _ in range(4)]
    if seed % 4 == 3:
        HL, DE = (rng.choice([0, 1, 4093, 4094, 4095, 4096, 65535]),
                  rng.choice([0, 1, 4093, 4094, 4095, 4096, 65535]))
    else:
        HL, DE = rng.randrange(4094), rng.randrange(4094)
    SP = rng.choice([0, 1, 2, 3, rng.randrange(16, 4093), 4095, 4096])
    C, Z = rng.randrange(2), rng.randrange(2)
    return code, data, b"", row_of(R[0], R[1], R[2], R[3], HL, DE, 0, SP, C, Z,
                                   tick=rng.randrange(100))

DELAY_SRC = """
  LDI r0, {j}
  LDI r1, {k}
outer:
  LDI r2, {k}
inner:
  SUBI r2, 1
  JNZ inner
  SUBI r0, 1
  JNZ outer
  OUT r1
  HALT
"""

def delay_prog(j, k):

    return asm(DELAY_SRC.format(j=j, k=k))

DIVMOD = asm("""
  LDI r0, 200
  LDI r1, 7
  DIV r0, r1
  OUT r0
  MOV r2, r0
  MOD r2, r1
  OUT r2
  CMP r0, r2
  LDI r3, 0
  JZ eq
  LDI r3, 1
eq:
  LDI r1, 3
  NEG r1
  NOT r1
  ROL r1
  ROR r1
  MUL r2, r1
  OUT r2
  LDI HL, 100
  LDI DE, 5
  ADD HL, DE
  SUB HL, DE
  XCHG
  OUT r3
  HALT
""")

ECHO = asm("""
  LDI r3, 6
loop:
  IN r0
  OUT r0
  SUBI r3, 1
  JNZ loop
  HALT
""")

INFINITE = asm("""
loop:
  NOP
  JMP loop
""")

ERR_DIV0 = asm("LDI r0, 9\nLDI r1, 0\nDIV r0, r1\nOUT r0\nHALT")
ERR_RESERVED = bytes([0x70, 0x63, 0x00])
ERR_OPCODE = bytes([0x7F, 0x00])
ERR_DATA_OOB = asm("LDI HL, 6000\nMOV r0, [HL]\nOUT r0\nHALT")
ERR_STACK = asm("recurse:\nCALL recurse\n")

def stc_program(wlo=0x00, whi=0x08):

    src = """
      JMP main
    sub:
      LDI r0, 7
      RET
    main:
      LDI HL, 0x04
      LDI r0, 42
      STC [HL], r0
      CALL sub
      OUT r0
      LDC r0, [HL]
      OUT r0
      HALT
    """
    b = bytearray(asm(src).ljust(WHI + 1, b"\x00"))
    b[WLO], b[WHI] = wlo, whi
    return bytes(b)

def ext_program():

    main = bytes([0x70, 0x70, 0x00, 0xF8 | 0, 0xD0 | 3, 0x7E, 0x00])
    h0 = bytes([0x70, 0x70, 0x01, 0xD4 | 0, 1, 0x08])
    h1 = bytes([0xD4 | 0, 5, 0x08])
    b = bytearray(bytes(main).ljust(0x20, b"\x00")) + h0 + h1
    for k, addr in ((0, 0x20), (1, 0x20 + len(h0))):
        b[VEC + 2 * k:VEC + 2 * k + 2] = addr.to_bytes(2, "little")
    return bytes(b)

def check_solo_step(batch, code, data, inputs, row, kind):

    batch.set_program(0, code, data, inputs)
    push_row(batch, 0, row)
    g = golden_machine(code, data, inputs, row)
    pre, pre_data, pre_code = golden_view(g), list(g.data), padded(g.code)
    raised = golden_step(g)
    batch.step(1)
    got = batch.snapshot(0)
    assert_widths(got, ("batch solo", kind))
    if raised:
        assert got["status"] == 3, (kind, "expected ERR", got)
        for k in pre:
            if k == "status":
                continue
            assert pre[k] == got[k], (kind, "error tick not atomic", k, pre[k], got[k])
        assert batch.data(0) == pre_data, (kind, "DATA was modified by an error tick")
        assert bytes(batch.code(0)) == pre_code, (kind, "CODE was modified by an error tick")
        assert batch.out(0) == bytes(g.out), (kind, "out")
        return "err"
    assert golden_view(g) == got, (kind, golden_view(g), got)
    assert batch.data(0) == list(g.data), (kind, "DATA")
    assert bytes(batch.code(0)) == padded(g.code), (kind, "CODE")
    assert batch.out(0) == bytes(g.out), (kind, "out")
    return "ok"

def test_opcode_enumeration_solo():
    batch = TritonBatch(1, max_in=3)
    tot = {"ok": 0, "err": 0}
    for op in range(256):
        for seed in range(6):
            tot[check_solo_step(batch, *op_case(op, seed), kind=("op", op, seed))] += 1
    torch.cuda.synchronize()
    print(f"[batch B=1] opcode single step: {256 * 6} cases match "
          f"(ok {tot['ok']} + error {tot['err']})")
    return tot

def test_escape_enumeration_solo():
    batch = TritonBatch(1, max_in=3)
    tot = {"ok": 0, "err": 0}
    for sub in range(256):
        for vec in (0, 0x1234):
            for seed in range(4):
                tot[check_solo_step(batch, *esc_case(sub, vec, seed),
                                    kind=("esc", sub, vec, seed))] += 1
    torch.cuda.synchronize()
    print(f"[batch B=1] escape subcode single step: {2048} cases match "
          f"(ok {tot['ok']} + error {tot['err']})")
    return tot

def test_opcode_enumeration_packed(width=64):

    cases = [op_case(op, seed) for op in range(256) for seed in range(6)]
    batch = TritonBatch(width, max_in=3)
    tot = {"ok": 0, "err": 0}
    launches = 0
    for base in range(0, len(cases), width):
        chunk = cases[base:base + width]
        mine = []
        for i, (code, data, inputs, row) in enumerate(chunk):
            batch.set_program(i, code, data, inputs)
            push_row(batch, i, row)
            mine.append(golden_machine(code, data, inputs, row))
        batch.step(1)
        launches += 1
        for i, g in enumerate(mine):
            pre, pre_data = golden_view(g), list(g.data)
            raised = golden_step(g)
            got = batch.snapshot(i)
            assert_widths(got, ("packed", base + i))
            if raised:
                assert got["status"] == 3, (base + i, "expected ERR", got)
                for k in pre:
                    if k == "status":
                        continue
                    assert pre[k] == got[k], (base + i, "error tick not atomic", k, pre[k], got[k])
                assert batch.data(i) == pre_data, (base + i, "DATA was modified")
                tot["err"] += 1
            else:
                assert golden_view(g) == got, (base + i, golden_view(g), got)
                assert batch.data(i) == list(g.data), (base + i, "DATA")
                tot["ok"] += 1
        torch.cuda.synchronize()
    print(f"[batch B={width}] opcode single step packed: {len(cases)} cases in {launches} "
          f"launches match (ok {tot['ok']} + error {tot['err']})")
    return tot

def build_batch(specs, **kw):

    max_in = max(1, max(len(inp) for _, _, _, inp, _, _ in specs))
    batch = TritonBatch(len(specs), max_in=max_in, **kw)
    for i, (kind, code, data, inputs, row, budget) in enumerate(specs):
        batch.set_program(i, code, data, inputs)
        push_row(batch, i, row)
        batch.set_budget(i, budget)
    return batch

def compare_batch(tag, batch, specs, res=None):

    if res is None:
        res = batch.run()
    assert len(res.outs) == len(specs)
    counts = {}
    for i, (kind, code, data, inputs, row, budget) in enumerate(specs):
        g = golden_run(code, data, inputs, row, budget)
        want_status = STATUS[g.status]
        got_status, got_tick = res.status[i], res.ticks[i]
        assert_widths(batch.snapshot(i), (tag, kind, i, "batch snapshot"))
        assert res.outs[i] == bytes(g.out), (tag, kind, i, "out", res.outs[i], bytes(g.out))
        assert got_status == want_status, (tag, kind, i, "status", got_status, want_status)
        assert got_tick == g.tick, (tag, kind, i, "tick", got_tick, g.tick)
        assert res.oplens[i] == len(g.out), (tag, kind, i, "oplen")
        assert batch.data(i) == list(g.data), (tag, kind, i, "DATA")
        assert bytes(batch.code(i)) == padded(g.code), (tag, kind, i, "CODE")
        counts[want_status] = counts.get(want_status, 0) + 1
    return counts, res

def random_state(rng, pc=0, sp=None, hl=None, de=None, tick=None):
    return row_of(rng.randrange(256), rng.randrange(256), rng.randrange(256),
                  rng.randrange(256),
                  rng.randrange(4096) if hl is None else hl,
                  rng.randrange(4096) if de is None else de,
                  pc,
                  4096 if sp is None else sp,
                  rng.randrange(2), rng.randrange(2),
                  tick=rng.randrange(64) if tick is None else tick)

def mixed_batch_specs(rng):

    specs = []
    for i in range(24):
        j, k = rng.randrange(1, 60), rng.randrange(1, 40)
        specs.append(("halt", delay_prog(j, k), bytes(4096), b"", random_state(rng), 200_000))
    for i in range(6):
        specs.append(("escape-divmod", DIVMOD, bytes(4096), b"", random_state(rng), 200_000))
    for wlo, whi in ((0x00, 0x08), (0x00, 0x08), (0x10, 0x18)):
        specs.append(("escape-stc", stc_program(wlo, whi), bytes(4096), b"", random_state(rng), 200_000))
    for i in range(3):
        specs.append(("escape-ext", ext_program(), bytes(4096), b"", random_state(rng), 200_000))
    for i in range(6):
        n = rng.randrange(0, 9)
        inputs = bytes(rng.randrange(256) for _ in range(n))
        specs.append(("io-echo", ECHO, bytes(4096), inputs, random_state(rng), 200_000))
    for i in range(3):
        specs.append(("overrun", INFINITE, bytes(4096), b"", random_state(rng), 200))
    specs.append(("err-div0", ERR_DIV0, bytes(4096), b"", random_state(rng), 200_000))
    specs.append(("err-reserved", ERR_RESERVED, bytes(4096), b"", random_state(rng), 200_000))
    specs.append(("err-opcode", ERR_OPCODE, bytes(4096), b"", random_state(rng), 200_000))
    specs.append(("err-data-oob", ERR_DATA_OOB, bytes(4096), b"", random_state(rng), 200_000))
    for i in range(4):
        specs.append(("err-stack", ERR_STACK, bytes(4096), b"", random_state(rng, sp=4096), 6000))
    for i in range(12):
        kind, code, data = rng.choice([
            ("mul", programs.MUL, bytes([rng.randrange(256), rng.randrange(256)])),
            ("fib", programs.FIB, bytes([rng.randrange(14)])),
            ("sumrec", programs.SUMREC, bytes([rng.randrange(64)])),
            ("longadd", programs.LONG_ADD, bytes([8] + [rng.randrange(256) for _ in range(24)])),
        ])
        specs.append((kind, code, data, b"", random_state(rng, tick=rng.randrange(8)), 200_000))
    for i in range(6):
        a = rng.randrange(65536)
        data = bytes([a & 0xFF, a >> 8] + [rng.randrange(256) for _ in range(4)])
        specs.append(("v3-frame", programs.FRAME_MUL, data, b"",
                      random_state(rng), 200_000))
    for i in range(12):
        n = rng.randrange(1, 24)
        code = bytes(rng.randrange(256) for _ in range(n))
        specs.append(("raw", code, bytes(4096), b"", random_state(rng, pc=0), rng.randrange(80, 400)))
    return specs

def test_mixed_batch():
    rng = random.Random(20260923)
    specs = mixed_batch_specs(rng)
    batch = build_batch(specs)
    counts, res = compare_batch("mixed", batch, specs)
    assert len(specs) >= 64, len(specs)
    halted, errored = counts.get(1, 0), counts.get(3, 0)
    overran = counts.get(2, 0)
    assert halted >= 24 and errored >= 8 and overran >= 3, counts
    for i, (kind, code, data, inputs, row, budget) in enumerate(specs):
        if kind == "overrun":
            assert res.status[i] == 2 and res.ticks[i] == budget, (i, res.status[i], res.ticks[i])
        if kind in ("halt", "escape-divmod", "escape-stc", "escape-ext", "io-echo", "v3-frame"):
            assert res.status[i] in (1, 3), (kind, i, res.status[i])
        if kind == "v3-frame":

            assert res.status[i] == 1 and res.oplens[i] == 4, (i, res.status[i], res.oplens[i])
        if kind.startswith("err-"):
            assert res.status[i] == 3, (kind, i, res.status[i])
    print(f"[batch B={len(specs)}] mixed random batch: all {len(specs)} machines match the "
          f"reference (halt {halted} + error {errored} + overrun {overran}, "
          f"inputs and initial states random)")
    return batch, specs, res

def varied_finish_specs():

    specs = []
    for i in range(64):
        j = 1 << (i % 8)
        k = 1 << (i // 8)
        specs.append(("delay", delay_prog(j, k), bytes(64), b"",
                      random_state(random.Random(i), tick=0), 200_000))
    return specs

def test_varied_finish_and_incremental():
    specs = varied_finish_specs()
    batch = build_batch(specs)
    refs = [golden_machine(code, data, inputs, row, budget)
            for kind, code, data, inputs, row, budget in specs]

    total = 0
    mixed_chunks = 0
    stopped_before = {}
    for chunk in (1, 9, 40, 200, 800, 2000):
        batch.step(chunk)
        total += chunk
        stopped_now = running_now = 0
        for i, g in enumerate(refs):
            golden_advance(g, chunk)
            got = batch.snapshot(i)
            assert_widths(got, ("step", chunk, i, "batch snapshot"))
            assert golden_view(g) == got, ("step", chunk, i, golden_view(g), got)
            assert batch.data(i) == list(g.data), ("step DATA", chunk, i)
            if got["status"] == 0:
                running_now += 1
            else:
                stopped_now += 1
                if i in stopped_before:
                    assert stopped_before[i] == (got["tick"], got["status"]), \
                        ("stopped machine advanced", chunk, i, stopped_before[i], got)
                else:
                    stopped_before[i] = (got["tick"], got["status"])
        mixed_chunks += int(stopped_now > 0 and running_now > 0)
        torch.cuda.synchronize()
    assert mixed_chunks >= 1, "no chunk had both halted and still-running machines"
    assert len(stopped_before) >= 48, len(stopped_before)

    counts, res = compare_batch("varied", batch, specs)
    assert counts.get(1, 0) >= 48, counts
    ticks = res.ticks
    assert min(ticks) <= 20 and max(ticks) >= 2000, (min(ticks), max(ticks))
    assert max(ticks) >= 50 * min(ticks), (min(ticks), max(ticks))
    print(f"[batch B={len(specs)}] mixed finish times: ticks {min(ticks)}..{max(ticks)} "
          f"in one launch, {mixed_chunks} chunk(s) with halted and running machines at "
          f"once, chunked comparison after 1/10/50/250/1050/3050 ticks, all match "
          f"(halt {counts.get(1, 0)} + error {counts.get(3, 0)})")
    return batch, specs, res

def test_large_batch(n=1024):

    specs = []
    for i in range(n):
        k = (i % 8) + 1
        specs.append(("delay", delay_prog(k, k), bytes(64), b"",
                      random_state(random.Random(1000 + i)), 200_000))
    batch = build_batch(specs)
    counts, res = compare_batch("large", batch, specs)
    assert counts.get(1, 0) == n, counts
    print(f"[batch B={n}] large batch: all {n} machines halt and match the reference "
          f"in one launch")
    return batch, specs, res

def default_row():

    return row_of(0, 0, 0, 0, 0, 0, 0, 4096, 0, 0)

def test_resident_vs_per_tick():

    cases = [
        programs.MUL, programs.FIB, programs.SUMREC, programs.LONG_ADD, DIVMOD,
        stc_program(0x00, 0x08), ext_program(), ECHO,
    ]
    datas = [bytes([200, 30]), bytes([13]), bytes([32]), bytes([8] + [7] * 24),
             bytes(4096), bytes(4096), bytes(4096), bytes([9, 8, 7])]
    inputs = [b""] * 7 + [bytes([1, 2, 3])]
    budgets = [200_000] * len(cases)
    cases += [ERR_DIV0, ERR_OPCODE, INFINITE]
    datas += [bytes(4096), bytes(4096), bytes(4096)]
    inputs += [b"", b"", b""]
    budgets += [200_000, 200_000, 150]

    rows = [default_row() for _ in cases]
    specs = [("case", code, data, inp, row, budget)
             for code, data, inp, budget, row in zip(cases, datas, inputs, budgets, rows)]
    batch = build_batch(specs)
    counts, res = compare_batch("resident-vs-per-tick", batch, specs)

    for i, (code, data, inp, budget) in enumerate(zip(cases, datas, inputs, budgets)):
        old = TritonCircuit(code, data=data, inputs=inp, tick_budget=budget)
        old_out = old.run()
        snap = old.snapshot()
        assert_widths(snap, ("per-tick path", i))
        assert res.outs[i] == old_out, (i, "resident output differs from the per-tick path")
        assert res.status[i] == snap["status"], (i, res.status[i], snap["status"])
        assert res.ticks[i] == snap["tick"], (i, res.ticks[i], snap["tick"])
        assert batch.data(i) == old.DATA.cpu().tolist(), (i, "DATA")
        assert batch.code(i) == old.CODE.cpu().tolist(), (i, "CODE")

    for code, data, inp, budget in zip(cases, datas, inputs, budgets):
        a = TritonCircuit(code, data=data, inputs=inp, tick_budget=budget)
        b = TritonCircuit(code, data=data, inputs=inp, tick_budget=budget)
        assert a.run() == b.run_resident(), "run() and run_resident() disagree"
        assert a.snapshot() == b.snapshot(), (a.snapshot(), b.snapshot())
        assert a.DATA.cpu().tolist() == b.DATA.cpu().tolist()
        assert a.CODE.cpu().tolist() == b.CODE.cpu().tolist()
    print(f"[batch B={len(cases)}] resident path vs per-tick path: identical output, status, "
          f"tick and memory on the same {len(cases)} programs (incl. atomic ERR and "
          f"OVERRUN); run() == run_resident()")

def test_one_shot_run_batch():

    cases = [(programs.MUL, bytes([200, 30]), 200_000),
             (DIVMOD, bytes(4096), 200_000),
             (stc_program(0x00, 0x08), bytes(4096), 200_000),
             (INFINITE, bytes(4096), 250),
             (ERR_DIV0, bytes(4096), 200_000),
             (ERR_OPCODE, bytes(4096), 200_000)]
    rng = random.Random(5)
    rows = [random_state(rng) for _ in cases]
    res = run_batch([c for c, _, _ in cases], [d for _, d, _ in cases],
                    budgets=[b for _, _, b in cases], states=rows)
    for i, (code, data, budget) in enumerate(cases):
        g = golden_run(code, data, b"", rows[i], budget)
        assert_widths(golden_view(g), ("run_batch reference", i))
        assert res.outs[i] == bytes(g.out), (i, res.outs[i], bytes(g.out))
        assert res.status[i] == STATUS[g.status], (i, res.status[i], g.status)
        assert res.ticks[i] == g.tick, (i, res.ticks[i], g.tick)
    assert res.status[3] == 2 and res.ticks[3] == 250, (res.status[3], res.ticks[3])
    print(f"[batch B={len(cases)}] one-shot run_batch(): per-machine budgets, initial "
          f"states and outputs all match the reference (budget exhaustion included)")

BENCH_ROWS = [
    (programs.MUL, bytes([200, 30]), b""),
    (programs.FIB, bytes([13]), b""),
    (programs.SUMREC, bytes([64]), b""),
    (delay_prog(128, 8), bytes(64), b""),
] * 8

def benchmark(reps=5):

    codes = [c for c, _, _ in BENCH_ROWS]
    datas = [d for _, d, _ in BENCH_ROWS]
    inputs = [i for _, _, i in BENCH_ROWS]
    n = len(codes)

    batch = TritonBatch(n)
    for i in range(n):
        batch.set_program(i, codes[i], datas[i], inputs[i])
    images = [torch.tensor(list(d), dtype=torch.int32, device=batch.dev) for d in datas]

    def reset(b, first):

        b.STATE.zero_()
        b.STATE[:, 7] = DATA_SIZE
        b.DATA.zero_()
        for i in (range(len(images)) if first is None else (first,)):
            if images[i].numel():
                b.DATA[i, :images[i].numel()] = images[i]

    batch.run()
    torch.cuda.synchronize()
    best_batch, res = float("inf"), None
    for _ in range(reps):
        reset(batch, None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = batch.run()
        best_batch = min(best_batch, time.perf_counter() - t0)
    total_ticks = sum(res.ticks)

    solo = TritonBatch(1)
    solo.set_program(0, codes[0], datas[0], inputs[0])
    solo.run()
    torch.cuda.synchronize()
    best_solo, solo_res = float("inf"), None
    for _ in range(reps):
        reset(solo, 0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        solo_res = solo.run()
        best_solo = min(best_solo, time.perf_counter() - t0)
    solo_ticks = solo_res.ticks[0]

    best_old, old_ticks, out = float("inf"), 0, None
    for _ in range(reps):
        machines = [TritonCircuit(codes[i], data=datas[i], inputs=inputs[i]) for i in range(n)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = [m.run() for m in machines]
        best_old = min(best_old, time.perf_counter() - t0)
        old_ticks = sum(int(m.tick.item()) for m in machines)
        del machines
    assert old_ticks == total_ticks, (old_ticks, total_ticks)
    assert out[0] == res.outs[0], "the two paths disagree on the workload output"

    print()
    print(f"benchmark: {n} machines, {total_ticks} ticks total (same workload both times), "
          f"best of {reps} runs, fresh state per run")
    print(f"  old per-tick path : {n} sequential machines, {total_ticks} launches, "
          f"{best_old * 1e3:9.3f} ms -> {total_ticks / best_old:14,.0f} ticks/s")
    print(f"  resident batch    : one launch for all {n} machines, "
          f"{best_batch * 1e3:9.3f} ms -> {total_ticks / best_batch:14,.0f} ticks/s "
          f"({best_old / best_batch:.0f}x)")
    print(f"  resident B=1      : {solo_ticks} ticks in {best_solo * 1e3:9.3f} ms "
          f"-> {solo_ticks / best_solo:14,.0f} ticks/s (single machine, one launch; "
          f"run_resident() wraps this path)")
    return best_old, best_batch, total_ticks

def run_all():
    print("resident batched executor acceptance:")
    test_opcode_enumeration_solo()
    test_escape_enumeration_solo()
    test_opcode_enumeration_packed()
    test_mixed_batch()
    test_varied_finish_and_incremental()
    test_large_batch()
    test_resident_vs_per_tick()
    test_one_shot_run_batch()
    print("batched execution: all passed")

if __name__ == "__main__":
    if "--bench" in sys.argv:
        benchmark()
    elif "--no-bench" in sys.argv:
        run_all()
    else:
        run_all()
        benchmark()