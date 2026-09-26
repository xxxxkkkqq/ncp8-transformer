"""Profiler for NCP-8: committed ticks by code point and by PC.

Counts only ticks that change machine state, so a halted or faulting machine that is
stepped again does not inflate its own profile. Two independent attributions are kept
(`by_codepoint` and `by_pc`) and `check()` demands that each total equal the machine's own
tick count, which turns an uncounted or double-counted tick into a failure instead of a
number that looks plausible.

Faults and tick-budget overruns are accounted separately from executed instructions: the
tick that raises commits nothing, so it must not be charged to the opcode that caused it.

Run: python3 profile.py            (self-check)
"""
from __future__ import annotations

import disasm
import isa_table as ISA
from golden_sim import CODE_SIZE, DATA_SIZE, MachineError, NCP8

class ProfileError(Exception):

    pass

class Profile:

    __slots__ = ("by_codepoint", "by_pc", "rows", "faults", "overruns",
                 "status", "total_ticks", "steps", "out", "budget", "length")

    def __init__(self):
        self.by_codepoint = {}
        self.by_pc = {}
        self.rows = []
        self.faults = []
        self.overruns = 0
        self.status = "RUNNING"
        self.total_ticks = 0
        self.steps = 0
        self.out = b""
        self.budget = 0
        self.length = 0

    def check(self):

        cp = sum(self.by_codepoint.values())
        pcs = sum(self.by_pc.values())
        if cp != self.total_ticks or pcs != self.total_ticks:
            raise ProfileError(f"histogram does not account for the ticks: code point sum "
                               f"{cp}, PC sum {pcs}, machine tick {self.total_ticks}")
        if self.steps != self.total_ticks + len(self.faults) + self.overruns:
            raise ProfileError(f"step accounting is off: {self.steps} steps attempted != "
                               f"{self.total_ticks} ticks + {len(self.faults)} faults + "
                               f"{self.overruns} overrun")
        return True

    @property
    def distinct_codepoints(self):
        return len(self.by_codepoint)

    @property
    def stopped(self):

        if self.faults:
            return "fault"
        if self.overruns:
            return "overrun"
        return self.status.lower()

    @property
    def distinct_pcs(self):
        return len(self.by_pc)

    def mnemonic(self, codepoint):

        if codepoint >= 0x100:
            sub = codepoint & 0xFF
            tmpl = disasm.ESC.get(sub)
            return f"0x70 {sub:02X}" if tmpl is None else tmpl[0].split(" ")[0]
        tmpl = disasm.SINGLE.get(codepoint)
        return f"DB {codepoint:02X}" if tmpl is None else tmpl[0].split(" ")[0]

    def top(self, n=12):

        return sorted(self.by_codepoint.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def hot_pcs(self, n=12):

        return sorted(self.by_pc.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def report(self, n=12):

        self.check()
        L = [f"ticks: {self.total_ticks}  (steps attempted {self.steps},"
             f" faults {len(self.faults)}, overrun {self.overruns})"
             f"  stopped: {self.stopped}  machine status: {self.status}"
             f"  tick budget: {self.budget}"]
        L.append(f"code point sums: {sum(self.by_codepoint.values())}"
                 f"  PC sums: {sum(self.by_pc.values())}"
                 f"  (must equal ticks: {self.total_ticks})")
        L.append("by code point:")
        for cp, t in self.top(n):
            L.append(f"  {t:8d}  {self.mnemonic(cp):8s} 0x{cp:04X}")
        L.append("by PC:")
        for pc, t in self.hot_pcs(n):
            L.append(f"  {t:8d}  PC 0x{pc:04X}")
        for tick, pc, what in self.faults:
            L.append(f"  fault at tick {tick} PC 0x{pc:04X}: {what}")
        return "\n".join(L)

def _require(condition, message):
    if not condition:
        raise ProfileError(message)

def run(code, *, data=None, inputs=b"", tick_budget=ISA.TICK_BUDGET_DEFAULT,
        max_rows=None, config=None):

    if isinstance(code, str):
        raise ProfileError("profile.run takes an image, not source text: use "
                           "loader.assemble(src, ...).image, and pass its "
                           "config() as `config` if the program traps or writes code")
    image = bytes(code)
    _require(1 <= len(image) <= CODE_SIZE,
             f"image is {len(image)} bytes, outside 1..{CODE_SIZE} (CODE_SIZE)")
    if data is not None:
        _require(len(bytes(data)) <= DATA_SIZE,
                 f"data image is {len(bytes(data))} bytes, above DATA_SIZE {DATA_SIZE}")
    _require(isinstance(tick_budget, int) and not isinstance(tick_budget, bool)
             and tick_budget >= 0, f"tick_budget must be an int >= 0, got {tick_budget!r}")
    m = NCP8(image, data=data, inputs=inputs, tick_budget=tick_budget, config=config)
    p = Profile()
    p.budget = tick_budget

    p.length = m.codelen
    while m.status == "RUNNING":
        pc = m.PC

        row = disasm.decode_machine(m, pc)
        if row is None:

            p.steps += 1
            try:
                m.step()
            except MachineError as e:
                p.faults.append((m.tick, pc, str(e)))
            p.status = m.status
            break
        p.steps += 1
        before = m.tick
        try:
            m.step()
        except MachineError as e:
            _require(m.tick == before, f"the reference committed a tick it also raised on: "
                                       f"tick went {before} -> {m.tick}")
            p.faults.append((before, pc, str(e)))
            p.status = m.status
            break
        if m.tick == before:

            _require(m.status != "RUNNING",
                     f"step() committed no tick and left status RUNNING at PC 0x{pc:04X}")
            p.overruns += 1
            p.status = m.status
            break
        _require(m.tick == before + 1, f"tick advanced by {m.tick - before}, not 1")
        p.by_codepoint[row.codepoint] = p.by_codepoint.get(row.codepoint, 0) + 1
        p.by_pc[pc] = p.by_pc.get(pc, 0) + 1
        if max_rows is None or len(p.rows) < max_rows:
            p.rows.append((before, pc, row.codepoint, row.text, row.assigned,
                           row.non_canonical))
    p.status = m.status
    p.total_ticks = m.tick
    p.out = bytes(m.out)
    p.check()
    return p

def run_result(result, **kw):

    config = kw.pop("config", None)
    return run(bytes(result.image), config=result.config() if config is None else config,
               **kw)

def compare(*profiles):

    return [(p.total_ticks, p.distinct_codepoints, p.distinct_pcs, p.status,
             len(p.faults), p.overruns) for p in profiles]

if __name__ == "__main__":
    import loader

    img = loader.assemble("loop:\n  DJNZ r0, loop\n  HALT\n", image=64).image
    pr = run(img, tick_budget=100)
    print(pr.report())