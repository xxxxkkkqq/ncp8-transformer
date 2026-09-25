"""Resume acceptance: a debugger session continues from a record, not from a replay.

`Debug.checkpoint()` publishes the machine's own state record and `resume(record)` opens a
session on a machine rebuilt from it, so a rollout can be picked up at any instruction boundary
without stepping the frames that produced it. The comparison is whole-machine: emitted bytes,
consumed input, a rewritten `CODE` image, a fault and its cause, an exhausted budget and
self-modification are each resumed at several boundaries and the remaining frames and final state
are required to match the unbroken run. A record that cannot be installed here -- a foreign
input stream, a different configuration block -- is refused there, and a stopped machine stays
stopped.

Run: python3 test_debug_resume.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import debug
import isa_table as ISA
from debug import Debug, TracingData, _frame_equal, resume, run_trajectory
from golden_sim import CODE_SIZE, asm

CASES = [
    ("counts and emits",
     "loop:\n  LDI r0, 65\n  OUT r0\n  ADDI r1, 1\n  LDI r2, 10\n  CMP r1, r2\n"
     "  JNZ loop\n  HALT", b"", None, None),
    ("reads its input",
     "loop:\n  IN r0\n  OUT r0\n  ADDI r1, 1\n  LDI r2, 5\n  CMP r1, r2\n  JNZ loop\n  HALT",
     bytes(range(1, 7)), None, None),
    ("stack round trip",
     "  LDI r0, 3\n  LDI r1, 5\n  PUSH r0\n  PUSH r1\n  POP r2\n  POP r3\n  OUT r2\n"
     "  OUT r3\n  HALT", b"", None, None),
    ("calls and returns", "  CALL sub\n  OUT r0\n  HALT\nsub:\n  LDI r0, 9\n  RET", b"",
     None, None),
    ("multiplies", "  LDI r0, 200\n  LDI r1, 30\n  MUL r0, r1\n  OUT r0\n  HALT", b"",
     None, None),
    ("data fault", "  LDI HL, 0x2000\n  MOV r0, [HL]\n  HALT", b"", None, None),
    ("undecoded byte", "@7100", b"", None, None),
    ("division by zero", "  LDI r0, 5\n  LDI r1, 0\n  DIV r0, r1\n  HALT", b"", None, None),
    ("self-modifies",
     "  LDI HL, 0x0008\n  LDI r1, 144\n  STC [HL], r1\n  LDI r0, 1\n  LDI r1, 1\n"
     "  ADD r0, r1\n  OUT r0\n  HALT", b"",
     ISA.MachineConfig(winlo=0x0000, winhi=0x000E), None),
    ("fills its output", "loop:\n  LDI r0, 66\n  OUT r0\n  JMP loop", b"", None, 8),
    ("budget exhausted", "loop:\n  ADDI r0, 1\n  JMP loop", b"", None, None),
]

FAILS = []
COUNTS = {"checkpoints": 0, "frames compared": 0}

def check(name, condition, detail=""):
    if not condition:
        FAILS.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")
    return bool(condition)

def same_frame(a, b):

    return _frame_equal(a.as_dict() if hasattr(a, "as_dict") else a,
                        b.as_dict() if hasattr(b, "as_dict") else b)

def refuse(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None

def image_of(src):

    return bytes.fromhex(src[1:]) if src.startswith("@") else asm(src)

def session(case):
    label, src, inputs, config, cap = case
    image = image_of(src)
    block = config
    machine = Debug(image, data=bytes(range(64)) * 2, inputs=inputs, tick_budget=4000,
                    config=block)
    if cap is not None:
        machine = Debug(image, data=bytes(range(64)) * 2, inputs=inputs, tick_budget=4000,
                        config=ISA.MachineConfig(winlo=block.winlo if block else 0,
                                                 winhi=block.winhi if block else 0,
                                                 outcap=cap))
    return machine

CAP = 220

def run_session(case, stride):

    d = session(case)
    marks = []
    while d.m.status == "RUNNING" and len(d.frames) < CAP:
        d.step()
        if len(d.frames) % stride == 0:
            marks.append((len(d.frames), d.checkpoint()))
    if not marks:
        marks.append((len(d.frames), d.checkpoint()))
    return d, marks

def one_case(case):
    label = case[0]
    stride = 5 if label in ("fills its output", "budget exhausted") else 3
    d, marks = run_session(case, stride)
    ran = len(d.frames)
    whole = run_trajectory(image_of(case[1]), data=bytes(range(64)) * 2, inputs=case[2],
                           tick_budget=4000, max_steps=ran, config=d.config)
    check(f"{label}: the debug session and the recorder produce the same run",
          len(whole.frames) == ran, f"{ran} frames stepped, {len(whole.frames)} recorded")

    by_at = dict(marks)
    published = refuse(resume, d.checkpoint())
    check(f"{label}: what a session publishes is something resume can take",
          published is None, (published or "")[:150])
    by_at[ran] = d.checkpoint()
    picks = sorted({marks[0][0], marks[len(marks) // 2][0], marks[-1][0], ran})
    for at in picks:
        record = by_at[at]
        COUNTS["checkpoints"] += 1
        r = resume(record, symbols=d.symbols)
        if ran > at:
            r.run(max_steps=ran - at)
        got, want = r.frames, d.frames[at:]
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if not same_frame(a, b)),
                     None)
        check(f"{label}: resume at frame {at} yields the same frames",
              len(got) == len(want) and first is None,
              f"{len(got)} frames after the checkpoint, {len(want)} remained in the run"
              + (f", first difference at +{first}" if first is not None else ""))
        COUNTS["frames compared"] += len(got)
        end = d.m
        pairs = (("PC", r.m.PC, end.PC), ("status", r.m.status, end.status),
                 ("tick", r.m.tick, end.tick), ("out", bytes(r.m.out), bytes(end.out)),
                 ("DATA", bytes(r.m.data), bytes(end.data)),
                 ("CODE", bytes(r.m.code).ljust(CODE_SIZE, b"\0"),
                  bytes(end.code).ljust(CODE_SIZE, b"\0")))
        where = [f"{n}: {g!r} vs {w!r}" for n, g, w in pairs if g != w]
        check(f"{label}: resume at frame {at} ends on the same machine", not where,
              f"tick {r.m.tick} status {r.m.status}; " + "; ".join(where)[:220])

        check(f"{label}: a resumed machine is the shape the record was taken from",
              len(r.m.code) == len(end.code) == len(record["CODE"]) == CODE_SIZE
              and r.m.codelen == end.codelen == record["block"]["codelen"],
              f"the run's machine holds {len(end.code)} cells with codelen "
              f"{end.codelen}, the resumed one holds {len(r.m.code)} with codelen "
              f"{r.m.codelen}, and the record carries a {len(record['CODE'])}-byte image")

        written = next((a for f in d.frames for a, _v in (f.writes or ())), None)

        if published is None and at < ran:
            again = resume(d.checkpoint(),
                           watchpoints=() if written is None else (written,))
            traced = refuse(again.run, max_steps=8)
            check(f"{label}: a resumed session runs on the buffers it installed",
                  traced is None, (traced or "")[:150])

    by_method = d.resume()
    by_module = resume(d.checkpoint())
    check(f"{label}: the session's resume and the module's are one operation",
          by_method.frames == [] == by_module.frames
          and by_method.m.snapshot() == by_module.m.snapshot(),
          "the two entry points landed on different machines")
    if d.m.status != "RUNNING":

        stopped = resume(d.checkpoint())
        before = stopped.m.snapshot()
        stopped.run(max_steps=4)
        check(f"{label}: a checkpoint of a stopped machine runs no further tick",
              stopped.frames == [] and stopped.m.snapshot() == before
              and stopped.stopped == d.m.status.lower(),
              f"{len(stopped.frames)} frames after resuming a {d.m.status} machine")
    else:
        print(f"  note: {label} was cut at {CAP} frames, so D3 does not apply to it")

def refusals():
    case = CASES[0]
    d = session(case)
    d.run(max_steps=6)
    check("D5 setup: the session took checkpoints", bool(d.checkpoint()), "no record")
    record = d.checkpoint()
    msg = refuse(resume, "not a record")
    check("D5 resume refuses something that is not a checkpoint", msg is not None
          and "record_state" in msg, repr(msg))
    msg = refuse(resume, {k: v for k, v in record.items() if k != "block"})
    check("D5 resume refuses a checkpoint with no configuration block", msg is not None
          and "block" in msg, repr(msg))
    prog = image_of(case[1])
    wide = ISA.MachineConfig(winlo=0x0000, winhi=len(prog), outcap=8)
    other = Debug(image_of(case[1]), tick_budget=4000, config=wide)
    msg = refuse(other.m.install_state, record)
    check("D5 a checkpoint is refused by a machine under a different block", msg is not None
          and ("outcap" in msg.lower() or "config" in msg.lower() or "block" in msg.lower()),
          repr(msg))
    msg = refuse(resume, dict(record, out=b"\x00" * 9000))
    check("D5 a checkpoint over the receiving capacity is refused", msg is not None,
          repr(msg))

    back = resume(record)
    check("D5 the resumed machine's bounds are the record's own block",
          back.m.tb == record["block"]["tickbudget"]
          and back.m.out_cap == record["block"]["outcap"]
          and back.m.codelen == record["block"]["codelen"],
          f"machine {back.m.tb}/{back.m.out_cap}/{back.m.codelen}, record "
          f"{record['block']['tickbudget']}/{record['block']['outcap']}/"
          f"{record['block']['codelen']}")
    COUNTS["refusals"] = COUNTS.get("refusals", 0) + 5

def tracing_survives():

    src = "  LDI HL, 8\n  LDI r0, 0x5A\n  MOV [HL], r0\n  ADDI r0, 1\n  HALT"
    d = Debug(asm(src), data=bytes(range(64)) * 2, tick_budget=64)
    for _ in range(2):
        d.step()
    rec = d.checkpoint()
    check("D6 the checkpoint before the store names no write yet",
          not [a for f in d.frames for a, _v in (f.writes or ())],
          str([f.writes for f in d.frames]))
    opened = refuse(resume, rec, watchpoints=[8])
    check("D6 a session can be opened from what a session publishes",
          opened is None, (opened or "")[:170])
    if opened is not None:
        return
    r = resume(rec, watchpoints=[8])
    stopped = refuse(r.run, max_steps=8)
    check("D6 a resumed session records the stores it makes",
          stopped is None and any(a == 8 for f in r.frames for a, _v in (f.writes or ())),
          (stopped or f"{r.stopped!r} after {len(r.frames)} frames, writes "
           f"{[f.writes for f in r.frames]}")[:170])
    check("D6 a watchpoint on a resumed session stops it at the store it installed",
          stopped is None and r.stopped == "watchpoint",
          (stopped or f"stopped {r.stopped!r}")[:170])

def data_identity_survives():

    src = "  LDI HL, 8\n  LDI r0, 0x5A\n  MOV [HL], r0\n  ADDI r0, 1\n  HALT"
    code = asm(src)
    two = ISA.MachineConfig(nbanks=2)
    ran = []

    def row(name, condition, detail=""):
        ran.append(name)
        return check(name, condition, detail)

    maker = Debug(code, data=bytes(range(64)) * 2, tick_budget=64, config=two)
    maker.step()
    rec = maker.checkpoint()
    row("D7 setup: the record the install carries is one a machine published",
        bool(rec) and bytes(rec["DATA"][:9]) == bytes(maker.m.data[:9]),
        f"record DATA {bytes(rec['DATA'])[:12].hex() if rec else None}")

    d = Debug(code, data=bytes(range(64)) * 2, tick_budget=64, config=two)
    traced = d.m.data
    row("D7 a session's DATA is the object its store log lives on",
        isinstance(traced, TracingData), f"a {type(traced).__name__}")
    refused = refuse(d.m.install_state, rec)
    if row("D7 the record installs", refused is None, (refused or "")[:170]):
        row("D7 installing a record keeps the machine's DATA object identity, so a"
            " session's store log survives the install",
            d.m.data is traced and isinstance(d.m.data, TracingData),
            f"after the install DATA is a {type(d.m.data).__name__}"
            + ("" if d.m.data is traced else ", which is not the object the session traced"))

        d.watch(8)
        stopped = refuse(d.run, max_steps=8)
        written = [a for f in d.frames for a, _v in (f.writes or ())]
        row("D7 the log a session installed still records the stores it makes",
            stopped is None and 8 in written and d.stopped == "watchpoint",
            (stopped or f"stopped {d.stopped!r} after {len(d.frames)} frames, "
             f"writes {written}")[:170])

        foreign = bytearray(64 * 2)
        refused = refuse(d.m.install_banks, [traced, foreign], 0, (1, 1))
        row("D7 a machine that installed a record still owns the page its group names",
            refused is None and d.m.banks[d.m.bank_own] is traced,
            (refused or f"own page is the named one: {d.m.banks[0] is traced}")[:170])
    COUNTS["D7 rows"] = COUNTS.get("D7 rows", 0) + len(ran)

def main():
    print(f"debug-resume acceptance: {len(CASES)} programs")
    tracing_survives()
    data_identity_survives()
    for case in CASES:
        one_case(case)
    refusals()
    print(f"\ndebug-resume acceptance: {len(FAILS)} failure(s)")
    for f in FAILS:
        print("  !!", f)
    print(f"counts: {COUNTS}")
    return 1 if FAILS else 0

if __name__ == "__main__":
    sys.exit(main())