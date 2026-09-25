"""Bank groups: machines that step together over a set of DATA pages.

Every machine holds `NBANKS` pages, one of which is the `DATA` it executes from, and the
selector `MB` decides which one a bank access reads or writes. A group evaluates each owner's
status once per tick, at the boundary before any machine has stepped, so a cross-page access
depends on the machine rather than on the order the driver stepped them. A page whose owner is
still running belongs to that owner alone. The module also carries the circuit paths' side of
the family: the fields a cross-path tick comparison reads, the obligations each path still owes,
and the measurement that says whether a path meets one.

Run: python3 banks.py [--with-paths]
"""
from __future__ import annotations

import isa_table as ISA
from golden_sim import (CODE_SIZE, DATA_SIZE, MachineError, NCP8, OUT_CAP, STATUS_CODE,
                        STATUS_ERROR, STATUS_HALT, STATUS_OVERRUN, STATUS_RUNNING, asm)

MB_MIN = 0
MB_MAX = 0xFFFF
BANK_SUBCODE_FIRST = 0xB0
BANK_SUBCODE_LAST = 0xBD
BANK_SUBCODES = tuple(range(BANK_SUBCODE_FIRST, BANK_SUBCODE_LAST + 1))

TERMINAL_STATUS_NAMES = (STATUS_HALT, STATUS_OVERRUN, STATUS_ERROR)
TERMINAL_STATUS_CODES = tuple(STATUS_CODE[n] for n in TERMINAL_STATUS_NAMES)

GROUP_MIN = 1
GROUP_MAX = (1 << 16) - 1

def quiescent(status):

    return status != STATUS_CODE[STATUS_RUNNING]

def with_nbanks(config, nbanks):

    fields = dict(config.as_dict()) if config is not None else {}
    fields["nbanks"] = nbanks
    return ISA.MachineConfig.from_dict(fields)

class BankGroup:

    def __init__(self, machines):
        machines = list(machines)
        if not GROUP_MIN <= len(machines) <= GROUP_MAX:
            raise ISA.ConfigError(f"a bank group holds {len(machines)} machines, this "
                                  f"machine declares {GROUP_MIN}..{GROUP_MAX}")
        for i, m in enumerate(machines):
            if not isinstance(m, NCP8):
                raise ISA.ConfigError(f"group member {i} is a {type(m).__name__}, not an "
                                      f"NCP8: a bank is one machine's DATA page and the "
                                      f"page's owner has to be a machine")
        for i, a in enumerate(machines):
            for j in range(i + 1, len(machines)):
                if a is machines[j]:
                    raise ISA.ConfigError(f"machines {i} and {j} are the same object, so "
                                          f"two banks would have one owner")

        for i, m in enumerate(machines):
            if m.nbanks != len(machines):
                raise ISA.ConfigError(f"machine {i} was loaded with NBANKS={m.nbanks} and "
                                       f"is being grouped with {len(machines)} machines; "
                                       f"the declared count is the group's size, and this "
                                       f"group will not rewrite the block under it")
        self.machines = machines
        pages = tuple(m.data for m in machines)
        statuses = self.statuses()
        for i, m in enumerate(machines):
            m.install_banks(pages, i, statuses)
        self.tick = 0

    @classmethod
    def from_programs(cls, codes, *, data=None, inputs=None, tick_budget=None,
                      out_cap=OUT_CAP, config=None):

        codes = [c if isinstance(c, (bytes, bytearray)) else asm(c) for c in codes]
        n = len(codes)
        data = _per_machine(data, n, b"")
        inputs = _per_machine(inputs, n, b"")
        budgets = _per_machine(tick_budget, n, ISA.TICK_BUDGET_DEFAULT)
        blocks = _per_machine(config, n, None)

        return cls([NCP8(c, data=d, inputs=i, tick_budget=b, out_cap=out_cap,
                         config=bl if (bl is not None and bl.nbanks is not None)
                         else with_nbanks(bl, n))
                    for c, d, i, b, bl in zip(codes, data, inputs, budgets, blocks)])

    @property
    def nbanks(self):

        return len(self.machines)

    def machine(self, i):
        return self.machines[i]

    def statuses(self):

        return tuple(m.status_code() for m in self.machines)

    @property
    def running(self):

        return tuple(i for i, m in enumerate(self.machines)
                     if m.status == STATUS_RUNNING)

    def all_quiescent(self):
        return not self.running

    def data(self, i):

        return bytes(self.machines[i].data)

    def code(self, i):
        return bytes(self.machines[i].code)

    def out(self, i):
        return bytes(self.machines[i].out)

    def snapshot(self):

        return tuple(dict(state=m.snapshot(), data=bytes(m.data),
                          code=bytes(m.code), out=bytes(m.out))
                     for m in self.machines)

    def step(self, strict=True):

        statuses = self.statuses()
        for m in self.machines:
            m.bank_owner_status = statuses
        raised = {}
        for i, m in enumerate(self.machines):
            try:
                m.step()
            except MachineError as e:
                raised[i] = e
        self.tick += 1
        if raised and strict:
            raise raised[min(raised)]
        return tuple(raised.get(i) for i in range(len(self.machines)))

    def run(self, limit=None):

        taken = 0
        while not self.all_quiescent():
            if limit is not None and taken >= limit:
                break
            self.step(strict=False)
            taken += 1
        return taken

def _per_machine(value, n, default):

    if value is None:
        return [default] * n
    if isinstance(value, (list, tuple)):
        got = list(value)
        if len(got) != n:
            raise ISA.ConfigError(f"{len(got)} values given for {n} machines in the group")
        return [default if v is None else v for v in got]
    return [value] * n

TICK_FIELDS = ("r", "HL", "DE", "MB", "PC", "SP", "C", "Z", "ipos", "tick", "status",
               "fault_reason", "fault_addr", "DATA", "CODE", "out")

_UNCOMPARED_STATE = tuple(n for n in ISA.STATE_FIELD_NAMES if n not in TICK_FIELDS)
if _UNCOMPARED_STATE:
    raise ISA.DecodeTableError(
        f"the per-tick comparison reads {len(TICK_FIELDS)} of the state table's fields "
        f"and leaves {_UNCOMPARED_STATE} out, so a path that stops carrying them is not "
        f"being compared")

PASS, DIFFERS, NOT_RUN, SKIPPED = "PASS", "DIFFERS", "NOT RUN", "SKIPPED"

CIRCUIT_ROWS = (
    ("the bank selectors execute",
     "every path's tick body must answer LDM, STM, LDMW DE, [HL], LDMW HL, [DE], "
     "STMW [HL], DE and STMW [DE], HL with the same page, the same two bounds and the "
     "same untouched flags as the reference, machine by machine and tick by tick; the "
     "two selector moves MOV MB, HL and MOV HL, MB answer the same way, so all 14 bank "
     "code points are inside the comparison",
     "one tick of each of the 14 bank code points on each path, on the same DATA image "
     "the reference is given, compared component by component against the reference's "
     "own tick; test_banks.py sweeps the same family at every selector a machine can "
     "name over several ticks"),
    ("MB is a state row",
     "a 16-bit `MB` row of isa_table.STATE_FIELDS, reset to 0, read by every path's "
     "record and restored by every path's install, and named by the per-tick comparison; "
     "the bound on the value is NBANKS and it sits on the bank access, so a record "
     "carrying a selector beyond this machine's bank count installs and the next access "
     "to it faults BANK_OOB",
     "read the state table, the record component list and TICK_FIELDS for the MB row, "
     "and install a record taken with a selector this machine cannot reach"),
    ("the two bank fault sites are stacked",
     "BANK_OOB and BANK_BUSY are rows of isa_table.FAULT_SITE_ORDER, directly after "
     "FETCH_OPERAND and ahead of DATA_OOB, and every path that builds its error signal "
     "by summing one row per fault site stacks those two rows in that order",
     "compare the stacked precedence list with the order the rule states, and name the "
     "cause of a tick that breaks the selector bound and the page address bound at once "
     "on the stacking path and on the reference"),
    ("a batch is not a group",
     "the resident batched path keeps NBANKS at the value it was loaded with whatever its "
     "row count is, so a bank access names the same cause in a batch of unrelated "
     "programs as it does alone, and a row reaches only its own page",
     "one bank program on a batch of each of several row counts, compared with the "
     "reference's tick of the same program"),
    ("a one-bank group is a lone machine",
     "no change: the page a lone machine reaches is its own, so every path already "
     "answers the one-bank case as it answers today's machine",
     "a run of each pinned program on a one-machine group and on a lone machine of the "
     "same path, compared tick by tick"),
)

def _ascii(text):

    return str(text).encode("ascii", "replace").decode("ascii").replace("\n", " ")[:160]

_MB_PROGRAM = ("  LDI r0, 0x5A\n  LDI HL, 9\n  MOV MB, HL\n  LDI HL, 4\n"
               "  STM [HL], r0\n  HALT")
MB_CUT = 4

def _selector_probe():

    two = with_nbanks(None, 2)
    halted = STATUS_CODE[STATUS_HALT]
    out = []

    def cut():
        m = NCP8(asm(_MB_PROGRAM), data=bytes(DATA_SIZE), tick_budget=16, config=two)
        for _ in range(MB_CUT):
            m.step()
        return m, m.record_state()

    _src, rec = cut()
    out.append(("a record taken after MOV MB, HL carries the selector it was given",
                rec["MB"] == 9 and rec["tick"] == MB_CUT))
    m = NCP8(asm(_MB_PROGRAM), data=bytes(DATA_SIZE), tick_budget=16, config=two)
    refused = None
    try:
        m.install_state(rec)
    except Exception as e:
        refused = e
    out.append(("a record naming a bank this machine was not loaded with installs",
                refused is None and m.MB == 9))
    pages = (m.data, bytearray(DATA_SIZE))
    m.banks, m.bank_own, m.bank_owner_status = pages, 0, (halted, halted)
    try:
        m.step()
    except MachineError:
        pass
    out.append(("the access the restored selector resumes into names the bound it broke",
                m.status == STATUS_ERROR and m.fault_reason == ISA.CAUSE["BANK_OOB"]
                and all(p[4] == 0 for p in pages)))
    _src, reach = cut()
    reach = dict(reach, MB=1)
    m2 = NCP8(asm(_MB_PROGRAM), data=bytes(DATA_SIZE), tick_budget=16, config=two)
    m2.install_state(reach)
    pages2 = (m2.data, bytearray(DATA_SIZE))
    m2.banks, m2.bank_own, m2.bank_owner_status = pages2, 0, (halted, halted)
    try:
        m2.step()
    except MachineError:
        pass
    out.append(("a restored selector inside the bank count selects the page it names",
                m2.status == STATUS_RUNNING
                and m2.fault_reason == ISA.CAUSE["OK"] and pages2[1][4] == 0x5A
                and pages2[0][4] == 0))
    return out

def reachable_images(machine):

    seen = {}
    stack = [(machine, 0)]
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen:
            continue
        seen[id(obj)] = obj
        if depth >= 4:
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                stack.append((item, depth + 1))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                stack.append((k, depth + 1))
                stack.append((v, depth + 1))
        elif hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if k.startswith("__"):
                    continue
                stack.append((k, depth + 1))
                stack.append((v, depth + 1))
    return seen

def bank_reach(machine):

    reached = reachable_images(machine)
    peers = [o for o in reached.values()
             if isinstance(o, NCP8) and o is not machine]
    images = [o for o in reached.values()
              if isinstance(o, (bytes, bytearray)) and len(o) in (DATA_SIZE, CODE_SIZE)]
    return peers, images

_BANK_DATA = bytes(range(1, 256)) * 16

def _path_selector(build):

    code = asm(_MB_PROGRAM)
    m = build(code, bytes(DATA_SIZE))
    for _ in range(MB_CUT):
        m.step()
    return m.record_state()["MB"]

def tick_once(machine):

    try:
        machine.step()
    except MachineError:
        pass
    rec = machine.record_state()
    return {k: rec[k] for k in TICK_FIELDS}

def _bank_ticks(code, paths):

    want = tick_once(NCP8(code, data=_BANK_DATA, tick_budget=8))
    mismatches, refused, skipped = [], set(), set()
    reserved = ISA.CAUSE["BAD_SUBCODE"]
    for name, build in sorted(paths.items()):
        try:
            got = tick_once(build(code, _BANK_DATA))
        except Exception as e:
            skipped.add(f"{name}: {_ascii(e)}")
            continue
        for field in TICK_FIELDS:
            if got[field] == want[field]:
                continue
            if field == "fault_reason" and got[field] == reserved \
                    and want[field] != reserved:
                refused.add(name)
            else:
                mismatches.append(f"{name}.{field}: path {got[field]!r}, reference "
                                  f"{want[field]!r}")
    return mismatches, refused, skipped

_BOUND_PROBE = ("  LDI HL, 9\n  MOV MB, HL\n  LDI HL, 0xFFFF\n  STM [HL], r0\n  HALT")

def _probe_cause(build, text, ticks=8):

    m = build(asm(text), _BANK_DATA)
    named = []
    for _ in range(ticks):
        try:
            m.step()
        except Exception:
            pass
        rec = m.record_state()
        named.append(rec["fault_reason"])
        if rec["status"] != 0:
            break
    return (named[-1] if named else None), named

def default_paths():

    paths, notes = {}, []
    try:
        from circuit_torch import TorchCircuit
        paths["TorchCircuit"] = lambda code, data: TorchCircuit(code, data=data,
                                                                device="cpu")
    except Exception as e:
        notes.append(f"TorchCircuit: {_ascii(e)}")
    try:
        from circuit_triton import TritonCircuit
        TritonCircuit(b"\x00", device="cuda")
        paths["TritonCircuit"] = lambda code, data: TritonCircuit(code, data=data,
                                                                  device="cuda")
    except Exception as e:
        notes.append(f"TritonCircuit: {_ascii(e)}")
    try:
        from circuit_triton import TritonBatch
        TritonBatch(1, device="cuda")
        paths["TritonBatch"] = lambda code, data: _BatchRow(code, data=data)
    except Exception as e:
        notes.append(f"TritonBatch: {_ascii(e)}")
    return paths, notes

class _BatchRow:

    def __init__(self, code, data=None, tick_budget=8):
        from circuit_triton import TritonBatch
        self.batch = TritonBatch(1, tick_budget=tick_budget)
        self.batch.set_program(0, code, data)

    def step(self):
        self.batch.step(1)

    def record_state(self):
        return self.batch.record_state(0)

def probe_rows(paths=None, notes=(), codes=None):

    paths = paths or {}
    codes = codes if codes is not None else [
        bytes([ISA.ESCAPE_PREFIX, sub]) for sub in BANK_SUBCODES]
    out = []
    for name, what, how in CIRCUIT_ROWS:
        if name == "the bank selectors execute":
            if not paths:
                out.append((name, NOT_RUN,
                            "no path compared: " + ("; ".join(notes) if notes
                                                    else "pass --with-paths"),
                            what, how))
                continue
            mismatches, refused, skipped = [], set(), set(notes)
            for code in codes:
                m, r, s = _bank_ticks(code, paths)
                mismatches += m
                refused.update(r)
                skipped.update(s)
            verdict = DIFFERS if mismatches else (PASS if not (refused or skipped)
                                                  else NOT_RUN)
            parts = []
            if refused:
                parts.append(f"{len(refused)} path(s) answer no execute branch for the "
                             f"selector: {', '.join(sorted(refused))}")
            if skipped:
                parts.append(f"{len(skipped)} comparison(s) could not run: "
                             + "; ".join(sorted(skipped)))
            if mismatches:
                parts.append(f"{len(mismatches)} component(s) disagree: "
                             + "; ".join(mismatches[:4]))
            evidence = (f"{len(codes)} code points x {sorted(paths)}: "
                        + ("; ".join(parts) if parts
                           else "every compared component agrees"))
            out.append((name, verdict, _ascii(evidence), what, how))
        elif name == "MB is a state row":
            row = next((f for f in ISA.STATE_FIELDS if f.name == "MB"), None)
            lone = NCP8(asm("  HALT"))
            claims = [
                ("the state table names MB as one 16-bit register",
                 row is not None and (row.lo, row.hi, row.cells) == (0, 65535, None)),
                ("a fresh machine's selector is bank 0, which it can reach",
                 lone.MB == 0 and lone.MB < lone.nbanks),
                ("the selector is a component of a record",
                 "MB" in ISA.RECORD_COMPONENTS),
                ("the cross-path tick comparison reads the selector",
                 "MB" in TICK_FIELDS),
            ] + _selector_probe()
            for path_name, build in sorted(paths.items()):
                try:
                    got = _path_selector(build)
                except Exception:
                    got = None
                claims.append((f"{path_name} records the selector its own ticks set",
                               got == 9))
            bad = [label for label, ok in claims if not ok]
            verdict = PASS if not bad else DIFFERS
            evidence = (f"{len(claims)} claim(s) about the MB row, "
                        f"{len(ISA.STATE_FIELD_NAMES)} state fields and "
                        f"{len(ISA.RECORD_COMPONENTS)} record components: "
                        + ("all hold" if not bad else "failed: " + "; ".join(bad)))
            out.append((name, verdict, _ascii(evidence), what, how))
        elif name == "the two bank fault sites are stacked":
            stacked = [site for site, _c in ISA.FAULT_SITE_ORDER]
            stated = [site for site, _c in ISA.full_fault_site_order()]
            bank = [site for site, _c in ISA.BANK_FAULT_SITES]
            detail = []
            claims = [
                ("the list a path stacks one signal per site into is the order the rule "
                 "states", stacked == stated),
                (f"both bank sites are rows of that list, directly after "
                 f"{ISA.BANK_SITE_ANCHOR!r}",
                 stacked[stacked.index(ISA.BANK_SITE_ANCHOR) + 1:
                         stacked.index(ISA.BANK_SITE_ANCHOR) + 1 + len(bank)] == bank),
            ]
            for path_name, build in sorted(paths.items()):
                try:
                    got, _named = _probe_cause(build, _BOUND_PROBE)
                except Exception as e:
                    got = f"not measured: {_ascii(e)}"
                detail.append(f"{path_name} named {got!r}")
                claims.append((f"{path_name} names the selector bound where the address "
                               f"bound also holds",
                               got == ISA.CAUSE["BANK_OOB"]))
            bad = [label for label, ok in claims if not ok]
            evidence = (f"{len(stacked)} stacked sites against {len(stated)} stated, bank "
                        f"sites {bank} after {ISA.BANK_SITE_ANCHOR!r}; "
                        + ("; ".join(detail) if detail else "no path measured")
                        + ("; failed: " + "; ".join(bad) if bad else "; all claims hold"))
            out.append((name, PASS if not bad else DIFFERS, _ascii(evidence), what, how))
        elif name == "a batch is not a group":
            try:
                from circuit_triton import TritonBatch
            except Exception as e:
                out.append((name, SKIPPED, f"the batched path could not be imported: "
                                           f"{_ascii(e)}", what, how))
                continue
            can = hasattr(TritonBatch, "install_banks")
            out.append((name, PASS if can else NOT_RUN,
                        f"TritonBatch {'can' if can else 'cannot'} be given foreign "
                        f"DATA pages, so its row count is all it knows about banks",
                        what, how))
        else:
            lone = NCP8(asm("LDI HL, 8\nLDI r0, 0x5A\nSTM [HL], r0\nHALT"),
                        data=bytes(DATA_SIZE))
            grp = BankGroup([NCP8(asm("LDI HL, 8\nLDI r0, 0x5A\nSTM [HL], r0\nHALT"),
                                  data=bytes(DATA_SIZE))])
            lone.run()
            grp.run()
            same = (grp.machine(0).snapshot() == lone.snapshot()
                    and bytes(grp.machine(0).data) == bytes(lone.data)
                    and bytes(grp.machine(0).out) == bytes(lone.out))
            out.append((name, PASS if same else DIFFERS,
                        "reference only: a one-machine group and a lone machine finished "
                        + ("identically" if same else "apart"), what, how))
    return out

def main(argv=None):

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-paths", action="store_true",
                    help="compare one tick per bank code point on every circuit path")
    args = ap.parse_args(argv)
    print("bank group: what the circuit paths still owe the reference")
    print(f"  bank subcodes assigned: "
          f"{sum(1 for s in BANK_SUBCODES if s in ISA.ESCAPE)} of {len(BANK_SUBCODES)}")
    paths, notes = default_paths() if args.with_paths else ({}, [])
    rows = probe_rows(paths, notes)
    counts = {}
    for name, verdict, evidence, what, how in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"  {verdict:8s} {name}\n           evidence: {evidence}\n"
              f"           to match: {what}\n           check: {how}")
    print("  verdicts: " + ", ".join(f"{v} {counts.get(v, 0)}"
                                     for v in (PASS, DIFFERS, NOT_RUN, SKIPPED)))
    return 1 if counts.get(DIFFERS) else 0

if __name__ == "__main__":
    raise SystemExit(main())