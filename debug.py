"""Debugger for NCP-8: breakpoints, watchpoints, frames and exact replay.

A recording (`Trajectory`) stores the image, the initial `DATA`, the inputs, the
configuration block the run was given, the tick budget it started from, the step cap it
honoured, why it stopped, and one frame per
attempted tick holding the pre-state, the post-state, the `DATA` writes that tick
committed and any raised condition. `replay()` re-runs from the recorded start under the
same cap and demands an exact match on every frame, the output stream, both memories and
the final state, so "we looked inside the machine while it ran" is a checked claim rather
than a description of a print loop.

Breakpoints and watchpoints accept loader symbol names as well as addresses. A faulting
frame carries no writes and no post-state, matching the machine's tick atomicity.

Run: python3 debug.py              (self-check)
"""
from __future__ import annotations

import disasm
import isa_table as ISA
from golden_sim import CODE_SIZE, DATA_SIZE, MachineError, NCP8

class DebugError(Exception):

    pass

class ReplayError(DebugError):

    pass

PC_OUTSIDE_IMAGE = -1

def _require(condition, message):
    if not condition:
        raise DebugError(message)

class TracingData(bytearray):

    __slots__ = ("writes",)

    def __init__(self, *args):
        super().__init__(*args)
        self.writes = []

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.writes.append((key, value if isinstance(value, int) else int(value)))
        else:
            self.writes.append(("slice", key))
        return super().__setitem__(key, value)

    def take_writes(self):

        out = [w for w in self.writes if isinstance(w[0], int)]
        self.writes = [w for w in self.writes if not isinstance(w[0], int)]
        return out

class Frame:

    __slots__ = ("tick", "pc", "code", "codepoint", "text", "assigned",
                 "non_canonical", "pre", "post", "writes", "raised", "out_len")

    def __init__(self, tick, pc, code, codepoint, text, assigned, non_canonical,
                 pre, post, writes, raised, out_len):
        self.tick = tick
        self.pc = pc
        self.code = code
        self.codepoint = codepoint
        self.text = text
        self.assigned = assigned
        self.non_canonical = non_canonical
        self.pre = pre
        self.post = post
        self.writes = writes
        self.raised = raised
        self.out_len = out_len

    def committed(self):

        return self.post is not None and self.post["tick"] == self.tick + 1

    def state_view(self):
        if self.post is None:
            return self.pre
        return self.post

    def __repr__(self):
        w = ",".join(f"{a:#x}={v:#x}" for a, v in self.writes) or "-"
        return (f"<Frame {self.tick} {self.pc:04X} {self.code.hex():<6s} {self.text}"
                f"{'!' if self.raised else ''} writes[{w}]>")

    def as_dict(self):
        return dict(tick=self.tick, pc=self.pc, code=self.code, codepoint=self.codepoint,
                    text=self.text, assigned=self.assigned,
                    non_canonical=self.non_canonical, pre=self.pre, post=self.post,
                    writes=list(self.writes), raised=self.raised, out_len=self.out_len)

class Debug:

    def __init__(self, code, *, data=None, inputs=b"", tick_budget=ISA.TICK_BUDGET_DEFAULT, PC=0,
                 symbols=None, config=None):
        if isinstance(code, str):
            raise DebugError("Debug takes an image, not source text: use "
                             "loader.assemble(src, ...).image")
        image = bytes(code)
        _require(1 <= len(image) <= CODE_SIZE,
                 f"image is {len(image)} bytes, outside 1..{CODE_SIZE}")
        _require(0 <= PC < len(image), f"start PC {PC} is outside the image")
        _require(isinstance(tick_budget, int) and not isinstance(tick_budget, bool)
                 and tick_budget >= 0, f"tick_budget must be an int >= 0, got {tick_budget!r}")
        _require(symbols is None or isinstance(symbols, dict),
                 f"symbols must be a dict of name -> address, got {type(symbols).__name__}")
        _require(config is None or isinstance(config, ISA.MachineConfig),
                 f"config must be an isa_table.MachineConfig, got a "
                 f"{type(config).__name__}: the bounds the machine obeys are validated "
                 f"at load, and a mapping would arrive unchecked")
        self.image = image
        self.config = config
        self.symbols = {} if symbols is None else dict(symbols)
        self.m = NCP8(image, data=data, inputs=inputs, tick_budget=tick_budget,
                      config=config)
        self.m.data = TracingData(self.m.data)
        if PC:
            self.m.PC = PC
        self.breakpoints = set()
        self.watchpoints = set()
        self.frames = []
        self.stopped = "running"

    def break_at(self, addr):
        addr = self._code_addr(addr, "breakpoint")
        self.breakpoints.add(addr)
        return addr

    def watch(self, addr):
        addr = self._data_addr(addr, "watchpoint")
        self.watchpoints.add(addr)
        return addr

    def drop_breakpoint(self, addr):

        self.breakpoints.discard(self._code_addr(addr, "breakpoint"))

    def drop_watchpoint(self, addr):
        self.watchpoints.discard(self._data_addr(addr, "watchpoint"))

    def _addr_of(self, addr, what, limit):
        if isinstance(addr, str):
            _require(addr in self.symbols,
                     f"{what} names {addr!r}, which is not in the symbol table this Debug "
                     f"was given (pass symbols=result.symbols, or an int)")
            addr = self.symbols[addr]
        _require(isinstance(addr, int) and not isinstance(addr, bool),
                 f"{what} address must be an int or a symbol name, got {addr!r}")
        _require(0 <= addr < limit, f"{what} address {addr} is outside the {limit}-byte space")
        return addr

    def _code_addr(self, addr, what):
        return self._addr_of(addr, what, CODE_SIZE)

    def _data_addr(self, addr, what):
        return self._addr_of(addr, what, DATA_SIZE)

    def state(self):

        m = self.m
        s = m.snapshot()
        s["ipos"] = m.ipos
        s["out_len"] = len(m.out)
        return s

    @property
    def PC(self):
        return self.m.PC

    def register(self, i):
        return self.m.r[i]

    def memory(self, addr):
        return self.m.data[addr]

    def output(self):
        return bytes(self.m.out)

    def decode_here(self):
        return disasm.decode_machine(self.m, self.m.PC)

    def running_code(self):

        return bytes(self.m.code)

    def program(self):

        return self.running_code()[:disasm.program_extent(self.m)]

    def step(self):

        m = self.m
        m.data.take_writes()
        pc = m.PC
        tick = m.tick
        code_image = self.program()
        row = disasm.decode_machine(self.m, pc)
        if row is not None:
            code, cp, text = code_image[pc:pc + row.size], row.codepoint, row.text
            assigned, non_can = row.assigned, row.non_canonical
        else:

            code, cp, text = b"", PC_OUTSIDE_IMAGE, "<PC outside the image>"
            assigned, non_can = False, True
        pre = self.state()
        raised = None
        try:
            m.step()
        except MachineError as e:
            raised = str(e)
        post = None if raised is not None else self.state()
        writes = m.data.take_writes()

        if raised is not None and writes:
            raise DebugError(f"the faulting tick at PC 0x{pc:04X} wrote DATA {writes} "
                             f"and then raised, which breaks tick atomicity")
        f = Frame(tick, pc, code, cp, text, assigned, non_can, pre, post, writes,
                  raised, len(m.out))
        self.frames.append(f)
        return f

    def run(self, *, max_steps=10_000):

        _require(isinstance(max_steps, int) and not isinstance(max_steps, bool)
                 and max_steps >= 1, f"max_steps must be an int >= 1, got {max_steps!r}")
        got = []
        for _ in range(max_steps):
            if self.m.status != "RUNNING":
                self.stopped = self.m.status.lower()
                break
            f = self.step()
            got.append(f)
            if f.raised is not None:
                self.stopped = "fault"
                break
            if self.watchpoints and f.writes:
                hit = [a for a, _v in f.writes if a in self.watchpoints]
                if hit:
                    self.stopped = "watchpoint"
                    break

            if self.m.status != "RUNNING":
                self.stopped = self.m.status.lower()
                break
            if self.m.PC in self.breakpoints:
                self.stopped = "breakpoint"
                break
        else:
            self.stopped = "max_steps"
        return got

    def run_until(self, predicate, *, max_steps=10_000, what="condition"):

        _require(callable(predicate), f"run_until({what}) needs a callable, got "
                                       f"{predicate!r}")
        for _ in range(max_steps):
            if self.m.status != "RUNNING":
                break
            self.step()
            if predicate(self):
                self.stopped = what
                return self.frames[-1]
        if self.m.status != "RUNNING":
            raise DebugError(f"run_until({what}) never held: the machine reached status "
                             f"{self.m.status} at tick {self.m.tick} first")
        raise DebugError(f"run_until({what}) never held within {max_steps} steps "
                         f"(PC 0x{self.m.PC:04X}, tick {self.m.tick})")

    def checkpoint(self):

        return self.m.record_state()

    def resume(self, *, breakpoints=(), watchpoints=()):

        return resume(self.checkpoint(), symbols=self.symbols, breakpoints=breakpoints,
                      watchpoints=watchpoints)

    def record(self, *, max_steps=100_000):

        return record(self.image, data=bytes(self.m.data), inputs=self.m.inputs,
                      tick_budget=self.m.tb, PC=self.m.PC, max_steps=max_steps)

def _frame_equal(a, b):

    return (a["tick"] == b["tick"] and a["pc"] == b["pc"] and a["code"] == b["code"]
            and a["codepoint"] == b["codepoint"] and a["text"] == b["text"]
            and a["assigned"] == b["assigned"]
            and a["non_canonical"] == b["non_canonical"]
            and a["pre"] == b["pre"] and a["post"] == b["post"]
            and list(a["writes"]) == list(b["writes"]) and a["raised"] == b["raised"]
            and a["out_len"] == b["out_len"])

class Trajectory:

    __slots__ = ("image", "data", "inputs", "tick_budget", "pc", "frames", "out",
                 "end_data", "end_code", "end_state", "status", "max_steps", "stopped",
                 "config")

    def __init__(self, image, data, inputs, tick_budget, pc, frames, out, end_data,
                 end_code, end_state, status, max_steps, stopped, config=None):
        self.image = bytes(image)
        self.data = bytes(data)
        self.inputs = bytes(inputs)
        self.tick_budget = tick_budget
        self.pc = pc
        self.frames = frames
        self.out = bytes(out)
        self.end_data = bytes(end_data)
        self.end_code = bytes(end_code)
        self.end_state = end_state
        self.status = status
        self.max_steps = max_steps
        self.stopped = stopped
        self.config = config

    def __len__(self):
        return len(self.frames)

    def pcs(self):

        return [f["pc"] for f in self.frames]

    def replay(self):

        got = run_trajectory(self.image, data=self.data, inputs=self.inputs,
                             tick_budget=self.tick_budget, PC=self.pc,
                             max_steps=self.max_steps, config=self.config)
        if len(got.frames) != len(self.frames):
            raise ReplayError(f"replay produced {len(got.frames)} frames, the recording "
                              f"has {len(self.frames)} (both capped at "
                              f"max_steps={self.max_steps}, the recording stopped for "
                              f"{self.stopped!r} and the replay for "
                              f"{got.stopped!r})")
        for i, (a, b) in enumerate(zip(self.frames, got.frames)):
            if not _frame_equal(a, b):
                raise ReplayError(f"frame {i} differs on replay:\n  recorded "
                                  f"{a}\n  replayed {b}")
        if got.out != self.out:
            raise ReplayError(f"replay output differs: {got.out!r} against recorded "
                              f"{self.out!r}")
        if got.end_data != self.end_data:
            diffs = [(i, x, y) for i, (x, y) in enumerate(zip(self.end_data, got.end_data))
                     if x != y]
            raise ReplayError(f"replay DATA differs at {len(diffs)} cells, first few "
                              f"{diffs[:5]}")
        if got.end_code != self.end_code:
            diffs = [(i, x, y) for i, (x, y) in enumerate(zip(self.end_code, got.end_code))
                     if x != y]
            raise ReplayError(f"replay CODE differs at {len(diffs)} cells (self-"
                              f"modification did not repeat), first few {diffs[:5]}")
        if got.end_state != self.end_state:
            raise ReplayError(f"replay end state differs: {got.end_state} against "
                              f"recorded {self.end_state}")
        if got.status != self.status:
            raise ReplayError(f"replay end status {got.status} against recorded "
                              f"{self.status}")
        return self

def run_trajectory(code, *, data=None, inputs=b"", tick_budget=ISA.TICK_BUDGET_DEFAULT, PC=0,
                   max_steps=100_000, config=None):

    if isinstance(code, str):
        raise DebugError("run_trajectory takes an image, not source text")
    _require(isinstance(max_steps, int) and not isinstance(max_steps, bool)
             and max_steps >= 1, f"max_steps must be an int >= 1, got {max_steps!r}")
    image = bytes(code)
    dbg = Debug(image, data=data, inputs=inputs, tick_budget=tick_budget, PC=PC,
                config=config)

    start_data = bytes(dbg.m.data)
    steps = 0
    last = None
    while dbg.m.status == "RUNNING" and steps < max_steps:
        last = dbg.step()
        steps += 1
        if last.raised is not None:
            break
    if last is not None and last.raised is not None:
        stopped = "fault"
    elif steps == max_steps and dbg.m.status == "RUNNING":
        stopped = "max_steps"
    else:
        stopped = str(dbg.m.status).lower()
    frames = [f.as_dict() for f in dbg.frames]
    return Trajectory(image, start_data, dbg.m.inputs, tick_budget, PC, frames,
                      bytes(dbg.m.out), bytes(dbg.m.data), bytes(dbg.m.code),
                      dbg.state(), dbg.m.status, max_steps, stopped, config)

def record(code, *, data=None, inputs=b"", tick_budget=ISA.TICK_BUDGET_DEFAULT, PC=0,
           max_steps=100_000, config=None):

    return run_trajectory(code, data=data, inputs=inputs, tick_budget=tick_budget,
                          PC=PC, max_steps=max_steps, config=config)

def resume(record, *, symbols=None, breakpoints=(), watchpoints=()):

    _require(isinstance(record, dict),
             f"resume takes a record_state() mapping, got a {type(record).__name__}")
    block = record.get("block")
    _require(isinstance(block, dict),
             "resume takes a record that carries its configuration block: without one there "
             "is no way to say which bounds the checkpoint was taken under")
    d = Debug(record["CODE"], data=record["DATA"], inputs=record["inputs"],
              PC=record["PC"], symbols=symbols,
              config=ISA.MachineConfig.from_dict(block))
    d.m.install_state(record)

    d.m.data = TracingData(d.m.data)
    for addr in breakpoints or ():
        d.break_at(addr)
    for addr in watchpoints or ():
        d.watch(addr)
    return d

def replay(traj):

    if isinstance(traj, Trajectory):
        return traj.replay()
    if isinstance(traj, dict):
        missing = [k for k in Trajectory.__slots__ if k not in traj]
        if missing:
            raise DebugError(f"replay takes a Trajectory or the dict `to_dict` wrote; "
                             f"this dict is not one, it has no {', '.join(missing)} "
                             f"(it has {sorted(traj)})")
        t = Trajectory(traj["image"], traj["data"], traj["inputs"], traj["tick_budget"],
                       traj["pc"], traj["frames"], traj["out"], traj["end_data"],
                       traj["end_code"], traj["end_state"], traj["status"],
                       traj["max_steps"], traj["stopped"],
                       ISA.MachineConfig.from_dict(traj["config"]))
        return t.replay()
    raise DebugError(f"replay takes a Trajectory or its dict, got a "
                     f"{type(traj).__name__}")

def to_dict(traj):

    return dict(image=traj.image, data=traj.data, inputs=traj.inputs,
                tick_budget=traj.tick_budget, pc=traj.pc, frames=traj.frames,
                out=traj.out, end_data=traj.end_data, end_code=traj.end_code,
                end_state=traj.end_state, status=traj.status,
                max_steps=traj.max_steps, stopped=traj.stopped,
                config=None if traj.config is None else traj.config.as_dict())

def stop_pcs(traj):

    return {f["pc"] for f in traj.frames}

if __name__ == "__main__":
    import loader

    r = loader.assemble("  LDI r0, 3\nloop:\n  SUBI r0, 1\n  JNZ loop\n  OUT r0\n"
                        "  HALT\n", image=64)
    d = Debug(r.image)
    d.break_at(0x04)
    print(d.run())
    print([repr(f) for f in d.frames])
    record(r.image).replay()
    print("replay exact")