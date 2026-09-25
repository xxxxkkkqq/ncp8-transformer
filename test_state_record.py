"""State-record acceptance: what a record carries, and what an install restores.

`record_state()` publishes the whole state of a machine at an instruction boundary and
`install_state()` restores one, on the reference simulator, the tensor circuit, the
Triton circuit and every row of a resident batch. The pair is the route from a recorded
rollout back to a running machine, so what is checked is that nothing about the machine
is left behind: the output stream, the input cursor and the stream it cursor-ed through,
both memory images at full width, the terminal status with the cause that produced it,
and the configuration the record was taken under.

Checks:
  R1 the published pair on all four datapaths: one component list, one representation,
     and it covers every field of every path's visible state;
  R2 every resume point of every bundled program continues identically to the
     reference's own unbroken run, in all four datapaths;
  R3 record -> install -> record is the identity, on a path and into any other path;
  R4 a machine that stopped installs as a machine that stopped and stays stopped, and a
     stopped machine whose cause does not pair with its status is refused;
  R5 the refusals, each naming the value it refused: a missing component, a state taken
     inside an instruction, a record from a differently configured machine, a stream over
     capacity, an input stream that is not the one the record consumed, an out-of-width
     field; and a refused record writes nothing at all;
  R6 installing one row of a resident batch leaves every other row's state, CODE, DATA,
     output buffer, instruction bound and input stream exactly as they were, and reaches
     no image loader;
  R7 the pair is host-side: no instruction path and no step path reaches it, it stores
     nothing on the machine, a record's shape does not change with the configuration it
     was taken under, and every field a path makes visible is carried by a record;
  R8 each probe program cut at every tick it survives, restored on all four paths and
     run to the end, with its final state, both memories and its whole output stream
     compared against the unbroken run; and the cost of arriving at a given tick by
     install rather than by replay.

Run: python3 test_state_record.py
     NCP8_RESUME_CAP=40 python3 test_state_record.py   # to bound the resume census
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import time

import programs
import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import CODE_SIZE, DATA_SIZE, OUT_CAP, TritonBatch, TritonCircuit
from golden_sim import MachineError, NCP8, STATUS_CODE, asm
from test_batched_execution import ECHO, INFINITE, padded
from test_state_contract import VIEW_FIELDS, circuit_view, ref_view

TICK_BUDGET = 200_000
CONT = 4
BATCH_ROWS = 16
RESUME_CAP = int(os.environ.get("NCP8_RESUME_CAP", "0"))

FAILS = []
COUNTS = {"resume points": 0, "resumes compared": 0, "cross installs": 0,
          "refusals": 0}

def check(name, cond, detail=""):
    if not cond:
        FAILS.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")
    return bool(cond)

def refuse(fn, *args, **kw):

    try:
        fn(*args, **kw)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None

BUNDLED = [
    ("longadd", programs.LONG_ADD,
     bytes([8]) + bytes(range(1, 9)) + bytes(range(9, 17)) + bytes(8), b"",
     None, None, None),
    ("fib", programs.FIB, bytes([13]) + bytes(12), b"", None, None, None),
    ("sumrec", programs.SUMREC, bytes([40]) + bytes(55), b"", None, None, None),
    ("frame-mul", programs.FRAME_MUL,
     bytes([0x34, 0x12]) + bytes([3, 4, 5, 6]), b"", None, None, None),
    ("mul", programs.MUL, bytes([200, 30]), b"", None, None, None),
    ("nested", programs.NESTED, bytes([7]) + bytes(15), b"", None, None, None),
    ("stack-overflow", programs.OVERFLOW, bytes(16), b"", None, None, None),
]

EMIT = asm("loop:\n  LDI r0, 65\n  OUT r0\n  JMP loop")

SELFMOD = asm("""
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
""")

EXTRA = [
    ("io-echo", ECHO, bytes(64), bytes([9, 8, 7, 6, 5, 4, 3, 2]), None, None, None),
    ("overrun", INFINITE, bytes(64), b"", 300, None, None),
    ("selfmod", SELFMOD, bytes(64), b"", None,
     ISA.MachineConfig(winlo=0x00, winhi=0x08, codelen=len(SELFMOD)), None),
    ("out-cap", EMIT, bytes(8), b"", 200, None, 8),
]

CASES = [(n, c, d, i, b or TICK_BUDGET, cfg, cap or OUT_CAP)
         for n, c, d, i, b, cfg, cap in BUNDLED + [e for e in EXTRA]]
BUNDLED_NAMES = {c[0] for c in BUNDLED}

class Path:
    name = "?"

    def build(self, case, rows=1):
        raise NotImplementedError

    def record(self, h):
        raise NotImplementedError

    def install(self, h, snap):
        raise NotImplementedError

    def step(self, h):
        raise NotImplementedError

    def scalars(self, h):

        raise NotImplementedError

    def stream(self, h):

        raise NotImplementedError

    def images(self, h):

        raise NotImplementedError

    def region(self, h):

        raise NotImplementedError

    def tick_view(self, h):
        return dict(self.scalars(h), out=self.stream(h))

    def view(self, h):
        code, data = self.images(h)
        return dict(self.tick_view(h), CODE=code, DATA=data)

class _Ref(Path):
    name = "reference"

    def build(self, case, rows=1):
        _n, code, data, inputs, budget, config, cap = case
        return (NCP8(code, data=data, inputs=inputs, tick_budget=budget,
                     config=config, out_cap=cap), 0)

    def record(self, h):
        return h[0].record_state()

    def install(self, h, snap):
        h[0].install_state(snap)

    def step(self, h):
        try:
            h[0].step()
        except MachineError:
            pass

    def scalars(self, h):
        return ref_view(h[0])

    def stream(self, h):
        return bytes(h[0].out)

    def images(self, h):
        return padded(h[0].code), bytes(h[0].data)

    def region(self, h):
        return len(h[0].code), h[0].codelen

class _Tensor(Path):
    name = "torch"
    ctor = TorchCircuit

    def build(self, case, rows=1):
        _n, code, data, inputs, budget, config, cap = case
        return (self.ctor(code, data=data, inputs=inputs, tick_budget=budget,
                          config=config, out_cap=cap), 0)

    def record(self, h):
        return h[0].record_state()

    def install(self, h, snap):
        h[0].install_state(snap)

    def step(self, h):
        h[0].step()

    def scalars(self, h):
        return circuit_view(h[0])

    def stream(self, h):
        return h[0].out()

    def _image(self, t):
        return bytes(int(v) for v in t.cpu().tolist())

    def images(self, h):
        return self._image(h[0].CODE), self._image(h[0].DATA)

    def region(self, h):
        return h[0].CODE.numel(), h[0].codelen

class _Triton(_Tensor):
    name = "triton"
    ctor = TritonCircuit

class _Batch(Path):

    name = "batch"

    def build(self, case, rows=1):
        _n, code, data, inputs, budget, config, cap = case

        if config is not None and config.codelen is not None:
            config = ISA.MachineConfig(**{n: getattr(config, n) for n in
                                          config.__slots__ if n != "codelen"})
        b = TritonBatch(rows, max_in=max(1, len(inputs)), tick_budget=budget,
                        out_cap=cap, config=config)
        for i in range(rows):
            b.set_program(i, code, data, inputs)
        return (b, 0)

    def record(self, h):
        return h[0].record_state(h[1])

    def install(self, h, snap):
        h[0].install_state(h[1], snap)

    def step(self, h):
        h[0].step(1)

    def scalars(self, h):
        return h[0].snapshot(h[1])

    def stream(self, h):
        return h[0].out(h[1])

    def images(self, h):
        b, i = h
        return (padded(bytes(int(v) & 0xFF for v in b.code(i))),
                bytes(int(v) & 0xFF for v in b.data(i)))

    def region(self, h):
        b, i = h
        return int(b.CODE.shape[1]), int(b.CODELENS[i].item())

PATHS = [_Ref(), _Tensor(), _Triton(), _Batch()]

def r1_surface():
    print("R1 the published pair, on every datapath")
    case = CASES[4]
    for p in PATHS:
        h = p.build(case)
        m = h[0]
        for has in ("record_state", "install_state"):
            check(f"R1 {p.name}.{has} exists", callable(getattr(m, has, None)),
                  f"{p.name} publishes no {has}")
        params = list(inspect.signature(getattr(m, "record_state")).parameters)
        want = ["i"] if p.name == "batch" else []
        check(f"R1 {p.name}.record_state signature", params == want,
              f"got {params}, want {want}")
        params = list(inspect.signature(getattr(m, "install_state")).parameters)
        want = ["i", "snap"] if p.name == "batch" else ["snap"]
        check(f"R1 {p.name}.install_state signature", params == want,
              f"got {params}, want {want}")
        snap = p.record(h)
        check(f"R1 {p.name} record components", tuple(snap) == tuple(ISA.RECORD_COMPONENTS),
              f"got {tuple(snap)}")

        missing = [f for f in VIEW_FIELDS if f not in snap and f != "oplen"]
        check(f"R1 {p.name} record covers its visible state", not missing,
              f"snapshot fields {missing} are carried by no record")
        he = p.build(next(c for c in CASES if c[0] == "io-echo"))
        for _ in range(5):
            p.step(he)
        cursor, live = p.scalars(he)["oplen"], len(p.stream(he))
        check(f"R1 {p.name} carries its output cursor as the stream it counts",
              cursor == live == len(p.record(he)["out"]) > 0,
              f"{p.name}: view says {cursor}, its stream holds {live} byte(s), and a "
              f"record of the same machine carries {len(p.record(he)['out'])}")
        raw = h[0].snapshot(h[1]) if p.name == "batch" else h[0].snapshot()
        extra = sorted(set(raw) - set(snap) - {"oplen"})
        check(f"R1 {p.name} publishes no visible field outside its record", not extra,
              f"{extra} are in snapshot() and in no record component")
        check(f"R1 {p.name} record state is the state table",
              tuple(n for n in snap if n in ISA.STATE_FIELD_NAMES) == \
              tuple(ISA.STATE_FIELD_NAMES),
              f"got {tuple(n for n in snap if n in ISA.STATE_FIELD_NAMES)}")
        check(f"R1 {p.name} record images are byte regions of full width",
              isinstance(snap["CODE"], bytes) and len(snap["CODE"]) == CODE_SIZE
              and isinstance(snap["DATA"], bytes) and len(snap["DATA"]) == DATA_SIZE,
              f"CODE {type(snap['CODE']).__name__}({len(snap['CODE'])})")
        check(f"R1 {p.name} record streams are bytes",
              isinstance(snap["out"], bytes) and isinstance(snap["inputs"], bytes),
              f"{type(snap['out']).__name__}/{type(snap['inputs']).__name__}")
        check(f"R1 {p.name} record block is the declared configuration fields",
              set(snap["block"]) <= set(ISA.MachineConfig.__slots__)
              and isinstance(snap["block"], dict), f"{sorted(snap['block'])}")

    for a, b in ((PATHS[0], PATHS[1]), (PATHS[1], PATHS[3]), (PATHS[2], PATHS[0])):
        ra = a.record(a.build(case))
        rb = b.record(b.build(case))
        check(f"R1 {a.name} and {b.name} record the same machine identically",
              ra == rb,
              f"components differ: {sorted(n for n in ra if ra[n] != rb[n])}")

    base = set(PATHS[0].build(case)[0].snapshot())
    for p in PATHS:
        h = p.build(case)
        keys = set(h[0].snapshot(h[1]) if p.name == "batch" else h[0].snapshot())
        want = base | (set() if p.name == "reference" else {"oplen"})
        check(f"R1 {p.name} publishes the reference's fields plus its output cursor",
              keys == want, f"differs by {sorted(keys ^ want)}")

    carried = {p.name: len(p.record(p.build(case))["CODE"]) for p in PATHS}
    for p in PATHS:
        h = p.build(case)
        cells, bound = p.region(h)
        snap = p.record(h)
        check(f"R1 {p.name} holds the CODE region its record carries",
              cells == len(snap["CODE"]) == CODE_SIZE,
              f"{p.name} holds {cells} cells and records {len(snap['CODE'])}")
        check(f"R1 {p.name} bounds its program by its own count, not by the region",
              bound < cells and snap["block"]["codelen"] == bound,
              f"codelen {bound} in a {cells}-cell region, and the record's block says "
              f"{snap['block']['codelen']}")
        p.install(h, snap)
        shape, restamped = p.region(h), p.record(h)
        check(f"R1 {p.name} is the same machine after installing its own record",
              shape == (cells, bound) and restamped == snap,
              f"shape {shape} against {(cells, bound)}, record differs: "
              f"{sorted(n for n in snap if snap[n] != restamped[n])}")
    check("R1 every path records the same CODE width",
          set(carried.values()) == {CODE_SIZE}, str(carried))
    print(f"  CODE widths: held and carried {sorted(set(carried.values()))}, one region "
          f"per machine on {len(PATHS)} paths")

def history(case):

    _n, code, data, inputs, budget, config, cap = case
    g = NCP8(code, data=data, inputs=inputs, tick_budget=budget, config=config,
             out_cap=cap)
    p = PATHS[0]
    h = (g, 0)
    recs, views = [p.record(h)], [p.view(h)]
    guard = 0
    while g.status == "RUNNING" and guard <= budget + 2:
        p.step(h)
        recs.append(p.record(h))
        views.append(p.view(h))
        guard += 1
    return recs, views

def resume_points(n):

    if not RESUME_CAP or n + 1 <= RESUME_CAP:
        return list(range(n + 1)), "every"
    stride = -(-(n + 1) // RESUME_CAP)
    pts = list(range(0, n + 1, stride))
    if pts[-1] != n:
        pts.append(n)
    return pts, f"stride {stride}"

def stream_diff(a, b):

    n = min(len(a), len(b))
    i = next((i for i in range(n) if a[i] != b[i]), n)
    return (f" | out differs at byte {i}: {a[i:i + 8]!r} vs {b[i:i + 8]!r} "
            f"(lengths {len(a)} vs {len(b)})")

def compare_run(p, h, k, views, tag, ticks=CONT):

    last = len(views) - 1
    for j in range(1, ticks + 1):
        p.step(h)
        at = min(k + j, last)
        got, want = p.tick_view(h), views[at]
        diffs = [f for f in VIEW_FIELDS if got[f] != want[f]]
        out = "" if got["out"] == want["out"] else stream_diff(got["out"], want["out"])
        COUNTS["resumes compared"] += 1
        if not check(f"R2 {tag}/{p.name} resume {k} tick {at}", not diffs and not out,
                     f"{[(f, got[f], want[f]) for f in diffs if f != 'r']}"
                     f" r {got['r']} vs {want['r']}{out}"):
            return False
    got, want = p.view(h), views[min(k + ticks, last)]
    return check(f"R2 {tag}/{p.name} resume {k} images",
                 got["CODE"] == want["CODE"] and got["DATA"] == want["DATA"],
                 f"CODE differs {got['CODE'] != want['CODE']}, DATA differs "
                 f"{got['DATA'] != want['DATA']}")

def r2_resume():
    print(f"R2 every resume point of every bundled program, {len(PATHS)} datapaths, "
          f"{CONT} continuation ticks")
    for case in CASES:
        name = case[0]
        recs, views = history(case)
        pts, how = resume_points(len(recs) - 1)
        tag = "bundled" if name in BUNDLED_NAMES else "extra"
        print(f"  {name:14s} {len(recs) - 1:6d} ticks, {len(pts):5d} resume points "
              f"({how})")
        COUNTS["resume points"] += len(pts)
        COUNTS[tag] = COUNTS.get(tag, 0) + len(pts)
        for p in PATHS:
            if p.name == "batch":
                r2_batch(p, case, pts, recs, views, tag)
                continue
            for k in pts:
                h = p.build(case)
                p.install(h, recs[k])
                compare_run(p, h, k, views, tag)

def r2_batch(p, case, pts, recs, views, tag):

    _n, code, data, inputs, budget, config, cap = case
    if config is not None and config.codelen is not None:
        config = ISA.MachineConfig(**{f: getattr(config, f) for f in config.__slots__
                                      if f != "codelen"})
    last = len(views) - 1
    for start in range(0, len(pts), BATCH_ROWS):
        group = pts[start:start + BATCH_ROWS]
        b = TritonBatch(len(group), max_in=max(1, len(inputs)), tick_budget=budget,
                        out_cap=cap, config=config)
        for i in range(len(group)):
            b.set_program(i, code, data, inputs)
        for i, k in enumerate(group):
            b.install_state(i, recs[k])
        for j in range(1, CONT + 1):
            b.step(1)
            for i, k in enumerate(group):
                at = min(k + j, last)
                got, want = dict(b.snapshot(i), out=b.out(i)), views[at]
                diffs = [f for f in VIEW_FIELDS if got[f] != want[f]]
                out = "" if got["out"] == want["out"] else stream_diff(got["out"],
                                                                        want["out"])
                COUNTS["resumes compared"] += 1
                check(f"R2 {tag}/batch resume {k} tick {at}", not diffs and not out,
                      f"{[(f, got[f], want[f]) for f in diffs]}{out}")
        for i, k in enumerate(group):
            at = min(k + CONT, last)
            got, want = p.view((b, i)), views[at]
            check(f"R2 {tag}/batch resume {k} images",
                  got["CODE"] == want["CODE"] and got["DATA"] == want["DATA"],
                  f"CODE differs {got['CODE'] != want['CODE']}, DATA differs "
                  f"{got['DATA'] != want['DATA']}")

def r3_identity():
    print("R3 record -> install -> record is the identity on the record")
    n = 0
    for case in CASES:
        recs, _ = history(case)
        for k in {0, len(recs) // 2, len(recs) - 1}:
            for p in PATHS:
                h = p.build(case)
                p.install(h, recs[k])
                again = p.record(h)
                n += 1
                check(f"R3 {p.name} {case[0]} resume {k} identity", again == recs[k],
                      f"components differ: "
                      f"{sorted(x for x in again if again[x] != recs[k][x])}")
    COUNTS["identity"] = n
    print(f"  record/install/record identities checked: {n}")

def r3_cross():
    print("R3 cross products: a record from any path installs into any path")
    pairs = 0
    for case in CASES:
        recs, views = history(case)
        k = min(7, len(recs) - 1)
        src = {}
        for p in PATHS:
            h = p.build(case)
            p.install(h, recs[k])
            src[p.name] = p.record(h)
        for a in PATHS:
            for b in PATHS:
                h = b.build(case)
                b.install(h, src[a.name])
                got = b.record(h)
                check(f"R3 {a.name}->{b.name} {case[0]} record survives",
                      got == src[a.name],
                      f"differs: {sorted(n for n in got if got[n] != src[a.name][n])}")
                check(f"R3 {a.name}->{b.name} {case[0]} continues",
                      compare_run(b, h, k, views, f"{a.name}->{b.name} cross"), "")
                pairs += 1
    COUNTS["cross installs"] = pairs
    print(f"  cross-path installs checked: {pairs} "
          f"({len(PATHS)} sources x {len(PATHS)} targets x {len(CASES)} cases)")

HALT_CASE = ("halt", asm("  LDI r0, 65\n  OUT r0\n  HALT\n  LDI r0, 66\n  OUT r0\n"),
             bytes(8), b"", TICK_BUDGET, None, OUT_CAP)
OVERRUN_CASE = ("overrun", INFINITE, bytes(8), b"", 120, None, OUT_CAP)
ERROR_CASE = ("error", asm("  LDI HL, 6000\n  MOV r0, [HL]\n  HALT"), bytes(8), b"",
              TICK_BUDGET, None, OUT_CAP)
TERMINALS = [HALT_CASE, OVERRUN_CASE, ERROR_CASE]

def run_to_end(case):
    _n, code, data, inputs, budget, config, cap = case
    g = NCP8(code, data=data, inputs=inputs, tick_budget=budget, config=config,
             out_cap=cap)
    while g.status == "RUNNING":
        try:
            g.step()
        except MachineError:
            pass
    return g

def r4_terminal():
    print("R4 a stopped machine installs stopped, stays stopped, and pairs its cause")
    for case in TERMINALS:
        g = run_to_end(case)
        snap = g.record_state()
        check(f"R4 {case[0]} reference is terminal", STATUS_CODE[g.status] != 0,
              g.status)
        for p in PATHS:
            h = p.build(case)
            p.install(h, snap)
            before = p.tick_view(h)
            check(f"R4 {p.name} {case[0]} installs the status",
                  before["status"] == STATUS_CODE[g.status],
                  f"{before['status']} vs {STATUS_CODE[g.status]}")
            check(f"R4 {p.name} {case[0]} installs the cause and its address",
                  before["fault_reason"] == g.fault_reason
                  and before["fault_addr"] == g.fault_addr,
                  f"{before['fault_reason']}@{before['fault_addr']} vs "
                  f"{g.fault_reason}@{g.fault_addr}")
            check(f"R4 {p.name} {case[0]} installs the stream it stopped with",
                  p.stream(h) == bytes(g.out), f"{p.stream(h)!r} vs {bytes(g.out)!r}")
            for _ in range(3):
                p.step(h)
            again = p.tick_view(h)
            check(f"R4 {p.name} {case[0]} ticks on no further step",
                  again == before,
                  f"moved: {sorted(f for f in again if again[f] != before[f])}")
            check(f"R4 {p.name} {case[0]} record identity", p.record(h) == snap,
                  f"differs: {sorted(n for n in snap if snap[n] != p.record(h)[n])}")
    base = run_to_end(HALT_CASE).record_state()
    for what, over in (("status 3 with no cause", dict(status=3, fault_reason=0)),
                       ("a cause while still running", dict(status=0, fault_reason=1)),
                       ("a cause on a halted machine", dict(status=1, fault_reason=9)),
                       ("a cause on an overrun machine",
                        dict(status=2, fault_reason=9))):
        snap = dict(base, **over)
        for p in PATHS:
            h = p.build(HALT_CASE)
            msg = refuse(lambda: p.install(h, snap))
            COUNTS["refusals"] += 1
            check(f"R4 {p.name} refuses {what}", msg is not None and "fault" in msg,
                  repr(msg))
            check(f"R4 {p.name} kept its own status after refusing {what}",
                  p.scalars(h)["status"] == 0, repr(p.scalars(h)["status"]))

CASE5 = ("counts",
         asm("  LDI r0, 65\n  OUT r0\n  IN r1\n  ADDI r2, 1\n  CMP r2, r3\n"
             "  JNZ 0x0000\n  HALT"),
         bytes(32), bytes("abc", "ascii"), 500, None, OUT_CAP)

def fresh5(inputs=None, out_cap=OUT_CAP, budget=None, config=None):
    _n, code, data, _i, b, _c, _cap = CASE5
    return NCP8(code, data=data, inputs=b"abc" if inputs is None else inputs,
                tick_budget=b if budget is None else budget, out_cap=out_cap,
                config=config)

def r5_refusals():
    print("R5 the refusals, each naming the value it refused")
    _n, code, data, inputs, budget, config, cap = CASE5
    g = NCP8(code, data=data, inputs=inputs, tick_budget=budget)
    for _ in range(5):
        g.step()
    snap = g.record_state()
    check("R5 the case emitted bytes and consumed input",
          bytes(g.out) and g.ipos > 0, f"out {bytes(g.out)!r} ipos {g.ipos}")
    check("R5 a good record installs",
          refuse(fresh5().install_state, snap) is None,
          refuse(fresh5().install_state, snap))

    targets = ("reference", "torch", "triton", "batch")
    for name in tuple(snap):
        cut = {k: v for k, v in snap.items() if k != name}
        for t in targets:
            msg = refuse(install_on, t, cut, CASE5)
            COUNTS["refusals"] += 1
            check(f"R5 {t} refuses a record missing {name}",
                  msg is not None and name in (msg or "") and "component" in msg,
                  repr(msg))

    for junk in (None, 7, "record", [snap], 3.5):
        msg = refuse(fresh5().install_state, junk)
        COUNTS["refusals"] += 1
        check(f"R5 refuses a record that is a {type(junk).__name__}", msg is not None,
              repr(msg))
    msg = refuse(fresh5().install_state, dict(snap, extra_component=1))
    check("R5 refuses an unknown component", msg is not None
          and "extra_component" in msg, repr(msg))

    for value in (0, 1, 2, OUT_CAP, -1, False):
        msg = refuse(fresh5().install_state, dict(snap, oplen=value))
        COUNTS["refusals"] += 1
        check(f"R5 refuses a record carrying oplen={value}",
              msg is not None and "oplen" in msg and "not a component" in msg,
              repr(msg))

    long_out = bytes(OUT_CAP + 1)
    msg = refuse(fresh5().install_state, dict(snap, out=long_out))
    check("R5 refuses an output stream over OUT_CAP",
          msg is not None and str(len(long_out)) in msg and str(OUT_CAP) in msg,
          repr(msg))
    small = NCP8(code, data=data, inputs=inputs, tick_budget=budget, out_cap=32)
    small_snap = small.record_state()
    msg = refuse(small.install_state, dict(small_snap, out=bytes(40)))
    check("R5 refuses an output stream over this machine's capacity",
          msg is not None and "40" in msg and "32" in msg, repr(msg))

    for name in ("CODE", "DATA"):
        size = CODE_SIZE if name == "CODE" else DATA_SIZE
        for cut in (snap[name][:10], snap[name] + b"\x00"):
            msg = refuse(fresh5().install_state, dict(snap, **{name: cut}))
            COUNTS["refusals"] += 1
            check(f"R5 refuses a {name} image of {len(cut)} bytes",
                  msg is not None and str(len(cut)) in msg and str(size) in msg,
                  repr(msg))

    other = bytes(b ^ 0x01 for b in inputs)
    msg = refuse(fresh5(inputs=other).install_state, snap)
    check("R5 refuses a different input stream",
          msg is not None and str(snap["ipos"]) in msg, repr(msg))
    msg = refuse(fresh5(inputs=inputs + b"d").install_state, snap)
    check("R5 refuses a longer input stream", msg is not None
          and str(len(inputs)) in msg and str(len(inputs) + 1) in msg, repr(msg))
    msg = refuse(fresh5(inputs=b"ab").install_state, dict(snap, inputs=b"ab", ipos=5))
    check("R5 refuses an ipos past the end of the recorded stream",
          msg is not None and "5" in msg and "2" in msg, repr(msg))
    msg = refuse(fresh5(inputs=b"").install_state, dict(snap, inputs=b""))
    check("R5 refuses a consumed cursor over an empty stream", msg is not None,
          repr(msg))

    for field in ISA.CONFIG_NAMES:
        moved = moved_block(snap["block"], field)
        msg = refuse(fresh5().install_state, dict(snap, block=moved))
        COUNTS["refusals"] += 1
        check(f"R5 refuses a record with a different {field}",
              msg is not None and field.lower() in (msg or "").lower()
              and "configuration" in msg, repr(msg))
    for junk in (None, "config", 5, {"nbanks": 1, "not_a_field": 2}):
        msg = refuse(fresh5().install_state, dict(snap, block=junk))
        COUNTS["refusals"] += 1
        check(f"R5 refuses a block that is a {type(junk).__name__}", msg is not None,
              repr(msg))

    for field, value in (("SP", DATA_SIZE + 1), ("PC", 1 << 16), ("C", 2),
                         ("tick", -1), ("ipos", -1), ("status", 4),
                         ("fault_reason", 256), ("fault_addr", 1 << 16), ("HL", -1),
                         ("MB", 1 << 16)):
        for t in targets:
            msg = refuse(install_on, t, dict(snap, **{field: value}), CASE5)
            COUNTS["refusals"] += 1
            check(f"R5 {t} refuses {field}={value}",
                  msg is not None and field in msg and "outside" in msg, repr(msg))
    for over in (dict(status=3, fault_reason=0), dict(status=0, fault_reason=9)):
        for t in targets:
            msg = refuse(install_on, t, dict(snap, **over), CASE5)
            COUNTS["refusals"] += 1
            check(f"R5 {t} refuses the pairing {over}", msg is not None
                  and "fault" in msg, repr(msg))
    for r in ([0, 0, 0], [0, 0, 0, 0, 0], [300, 0, 0, 0]):
        msg = refuse(fresh5().install_state, dict(snap, r=r))
        check(f"R5 refuses r={r}", msg is not None, repr(msg))

    pristine = NCP8(code, data=data, inputs=inputs, tick_budget=budget)
    before = pristine.record_state()
    for name in tuple(snap):
        refuse(pristine.install_state, {k: v for k, v in snap.items() if k != name})
    for bad in (dict(snap, status=4), dict(snap, out=bytes(OUT_CAP + 1)),
                dict(snap, oplen=3), dict(snap, block=moved_block(snap["block"], "VEC")),
                dict(snap, SP=DATA_SIZE + 1), dict(snap, inputs=b"zzz")):
        refuse(pristine.install_state, bad)
    check("R5 a refused record writes nothing", pristine.record_state() == before,
          "an install that refused still changed the machine")

def install_on(target, snap, case):

    _n, code, data, inputs, budget, config, cap = case
    if target == "reference":
        return NCP8(code, data=data, inputs=inputs, tick_budget=budget,
                    config=config, out_cap=cap).install_state(snap)
    if target == "torch":
        return TorchCircuit(code, data=data, inputs=inputs, tick_budget=budget,
                            config=config, out_cap=cap).install_state(snap)
    if target == "triton":
        return TritonCircuit(code, data=data, inputs=inputs, tick_budget=budget,
                             config=config, out_cap=cap).install_state(snap)
    b = TritonBatch(1, max_in=max(1, len(inputs)), tick_budget=budget, out_cap=cap,
                    config=config)
    b.set_program(0, code, data, inputs)
    return b.install_state(0, snap)

def moved_block(block, field):

    moved = dict(block)
    if field == "VEC":
        moved["vec"] = list(block["vec"])
        moved["vec"][3] = moved["vec"][3] + 1
    elif field == "CODELEN":
        moved["codelen"] = block["codelen"] + 1
    elif field == "WINLO":
        moved["winlo"], moved["winhi"] = 2, 6
    elif field == "WINHI":
        moved["winhi"] = block["winhi"] + 4
    elif field == "NBANKS":
        moved["nbanks"] = block["nbanks"] + 1
    elif field == "TDLIM":
        moved["tdlim"] = block["tdlim"] + 1
    elif field == "TICKBUDGET":
        moved["tickbudget"] = block["tickbudget"] + 1
    elif field == "OUTCAP":
        moved["outcap"] = block["outcap"] // 2
    else:
        raise AssertionError(f"no way to move the declared field {field}")
    return moved

def r6_batch():
    print("R6 installing one row disturbs no other row and loads no image")
    rows, pick = 5, 3
    specs = [(programs.MUL, bytes([200, 30]) + bytes(14), b""),
             (programs.FIB, bytes([9]) + bytes(12), b""),
             (ECHO, bytes(16), bytes([1, 2, 3])),
             (INFINITE, bytes(8), b""),
             (programs.SUMREC, bytes([12]) + bytes(28), b"")]
    b = TritonBatch(rows, max_in=8, tick_budget=TICK_BUDGET)
    for i, (code, data, inputs) in enumerate(specs):
        b.set_program(i, code, data, inputs)
        b.set_state(i, r=[i, 0, 0, 0], HL=i, DE=2 * i, SP=DATA_SIZE - i, tick=i)
    for _ in range(6):
        b.step(1)
    before = [b.record_state(i) for i in range(rows)]
    cells = {name: [getattr(b, name)[i].clone() for i in range(rows)]
             for name in ("CODE", "DATA", "INPUTS", "OUTBUF", "STATE", "CFG")}
    vectors = {name: b.__dict__[name].clone()
               for name in ("CODELENS", "INLENS", "BUDGETS")}

    def explode(*a, **kw):
        raise AssertionError("install_state reached set_program")

    b.set_program = explode
    try:
        b.install_state(pick, before[pick])
    finally:
        del b.set_program
    after = [b.record_state(i) for i in range(rows)]

    check(f"R6 row {pick} holds the record it was given",
          after[pick] == before[pick],
          f"components differ: "
          f"{sorted(n for n in after[pick] if after[pick][n] != before[pick][n])}")
    for i in range(rows):
        if i == pick:
            continue
        check(f"R6 row {i} record untouched", after[i] == before[i],
              f"components differ: "
              f"{sorted(n for n in after[i] if after[i][n] != before[i][n])}")
        for name, saved in cells.items():
            now = getattr(b, name)[i]
            check(f"R6 row {i} {name} untouched", bool((saved[i] == now).all().item()),
                  f"{int((saved[i] != now).sum().item())} cells differ")
        for name, col in vectors.items():
            check(f"R6 row {i} {name} untouched", col[i].item() == b.__dict__[name][i].item(),
                  f"{col[i].item()} -> {b.__dict__[name][i].item()}")

    for i in range(rows):
        if i != pick:
            check(f"R6 row {i} still has its own instruction bound",
                  b.CODELENS[i].item() == len(specs[i][0]),
                  f"{b.CODELENS[i].item()} vs {len(specs[i][0])}")
            check(f"R6 row {i} still has its own input stream",
                  b.INLENS[i].item() == len(specs[i][2]),
                  f"{b.INLENS[i].item()} vs {len(specs[i][2])}")

PAIR = ("install_state", "record_state")

def r7_unreachable():
    print("R7 the pair is host-side: no instruction and no step reaches it")
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("golden_sim.py", "circuit_torch.py", "circuit_triton.py"):
        with open(os.path.join(here, fname), encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), fname)
        hits = []
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            if fn.name in PAIR or fn.name == "install":
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    called = f.attr if isinstance(f, ast.Attribute) else (
                        f.id if isinstance(f, ast.Name) else "")
                    if called in PAIR:
                        hits.append((fn.name, called))
        check(f"R7 {fname} calls the pair from no function", not hits, f"{hits}")
        defined = sorted({n.name for n in ast.walk(tree)
                          if isinstance(n, ast.FunctionDef) and n.name in PAIR})
        check(f"R7 {fname} defines the pair", defined == sorted(PAIR), f"{defined}")
    sel = ({row["alu"] for row in ISA.SINGLE.values()}
           | {row["alu"] for row in ISA.ESCAPE.values()})
    check("R7 no ALU selector names a state installer",
          not [s for s in sel if "state" in s.lower() or "record" in s.lower()],
          f"{sorted(s for s in sel if 'state' in s.lower() or 'record' in s.lower())}")

    g, g2 = NCP8(programs.MUL, data=bytes([3, 4])), NCP8(programs.MUL, data=bytes([3, 4]))
    g.record_state()
    g.install_state(g2.record_state())
    check("R7 the reference stores no record on the machine",
          set(vars(g)) == set(vars(g2)),
          f"{sorted(set(vars(g)) ^ set(vars(g2)))}")
    c = TorchCircuit(programs.MUL, data=bytes([3, 4]))
    check("R7 the tensor path stores no record on the machine",
          set(vars(c)) == set(vars(TorchCircuit(programs.MUL, data=bytes([3, 4])))),
          "record_state or install_state added an attribute")

    declared = ({n.lower() for n in ISA.CONFIG_NAMES}
                | {"out_cap", "tick_budget", "tb", "config", "cfg"})
    leaked = sorted({n.lower() for n in ISA.RECORD_COMPONENTS} & declared)
    check("R7 no record component names a constraint", not leaked, f"{leaked}")
    for p in PATHS:
        h = p.build(CASE5)
        snap = p.record(h)
        leaked = sorted({str(k).lower() for k in snap} & declared)
        check(f"R7 {p.name} record exposes no constraint field", not leaked, f"{leaked}")

    wide = NCP8(asm("  HALT"), tick_budget=1000, out_cap=64)
    narrow = NCP8(asm("  HALT"), tick_budget=1000, out_cap=32)
    msg = refuse(narrow.install_state, wide.record_state())
    check("R7 a record from a wider machine is refused", msg is not None
          and "outcap" in (msg or "").lower(), repr(msg))
    same = NCP8(asm("  HALT"), tick_budget=1000, out_cap=64)
    same.install_state(wide.record_state())
    check("R7 install leaves every bound alone",
          (same.codelen, same.out_cap, same.tb, same.nbanks, same.tdlim)
          == (wide.codelen, wide.out_cap, wide.tb, wide.nbanks, wide.tdlim),
          f"{(same.codelen, same.out_cap, same.tb)} vs "
          f"{(wide.codelen, wide.out_cap, wide.tb)}")

    for p in PATHS:
        h = p.build(CASE5)
        p.step(h)
        extra = sorted(set(p.scalars(h)) - set(p.record(h)) - {"oplen"})
        check(f"R7 {p.name} visible state is carried by its record", not extra,
              f"{extra} are in snapshot() and in no record")

PROBE = [
    ("counts and emits", "loop:\n  LDI r0, 65\n  OUT r0\n  ADDI r1, 1\n  LDI r2, 9\n"
                         "  CMP r1, r2\n  JNZ loop\n  HALT"),
    ("uses the stack", "  LDI r0, 3\n  LDI r1, 5\n  PUSH r0\n  PUSH r1\n  CALL 0x0020\n"
                       "  HALT\n  POP r2\n  POP r3\n  OUT r2\n  OUT r3\n  HALT"),
    ("reads its input", "loop:\n  IN r0\n  OUT r0\n  ADDI r1, 1\n  LDI r2, 4\n"
                        "  CMP r1, r2\n  JNZ loop\n  HALT"),
    ("multiplies", "  LDI r0, 200\n  LDI r1, 30\n  MUL r0, r1\n  OUT r0\n  HALT"),
]
PROBE_INPUTS = b"\x11\x22\x33\x44\x55\x66"
PROBE_DATA = bytes(range(1, 64)) * 4
PROBE_BUDGET = 4096

def probe_case(text):

    code = asm(text)
    return ("probe", code, PROBE_DATA, PROBE_INPUTS, PROBE_BUDGET, None, OUT_CAP)

def r8_probe():
    print("R8 the round trip at every cut, then what reaching tick k costs")
    fields = VIEW_FIELDS + ("out", "CODE", "DATA")
    cuts = 0
    for what, text in PROBE:
        case = probe_case(text)
        _n, code, data, inputs, budget, config, cap = case
        g = NCP8(code, data=data, inputs=inputs, tick_budget=budget, out_cap=cap)
        while g.status == "RUNNING":
            try:
                g.step()
            except MachineError:
                pass
        whole = PATHS[0].view((g, 0))
        recs, _ = history(case)
        for k in range(len(recs)):
            m = NCP8(code, data=data, inputs=inputs, tick_budget=budget, out_cap=cap)
            for _ in range(k):
                try:
                    m.step()
                except MachineError:
                    break
            snap = m.record_state()
            for p in PATHS:
                h = p.build(case)
                p.install(h, snap)
                while p.scalars(h)["status"] == 0:
                    p.step(h)
                got = p.view(h)
                diffs = [f for f in fields if got[f] != whole[f]]
                check(f"R8 {p.name} {what} cut {k} continues to the same end",
                      not diffs, f"{diffs}")
                cuts += 1
    COUNTS["probe round trips"] = cuts
    print(f"  round trips compared to the unbroken run: {cuts}")
    loop = probe_case("  LDI r0, 1\nloop:\n  ADDI r0, 1\n  JMP loop")
    for k in (100, 1000, 10000):
        _n, code, data, inputs, budget, config, cap = loop
        t0 = time.time()
        g = NCP8(code, data=data, inputs=inputs, tick_budget=k + 8, out_cap=cap)
        while g.tick < k and g.status == "RUNNING":
            g.step()
        replay = time.time() - t0
        snap = g.record_state()
        t0 = time.time()
        fork = NCP8(code, data=data, inputs=inputs, tick_budget=k + 8, out_cap=cap)
        fork.install_state(snap)
        install = time.time() - t0
        check(f"R8 a fork at tick {k} arrives with the recorded tick",
              fork.tick == g.tick == k, f"{fork.tick} vs {g.tick}")
        print(f"  tick {k:6d}: replay {replay * 1e3:8.2f} ms   one install "
              f"{install * 1e3:6.3f} ms   ratio {replay / max(install, 1e-9):8.1f}x")

def main():
    t0 = time.time()
    check("R0 the state table's own claims hold", ISA.check_state_table() is True)
    r1_surface()
    r2_resume()
    r3_identity()
    r3_cross()
    r4_terminal()
    r5_refusals()
    r6_batch()
    r7_unreachable()
    r8_probe()
    print(f"\nstate-record acceptance: {len(FAILS)} failure(s) in "
          f"{time.time() - t0:.1f}s")
    for f in FAILS:
        print("  !!", f)
    print(f"counts: {COUNTS}")
    return 1 if FAILS else 0

if __name__ == "__main__":
    sys.exit(main())