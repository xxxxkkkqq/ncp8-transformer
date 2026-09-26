"""Training records: what a sample is, and when its label may be written.

A record names its inputs, the configuration the machine was loaded under, and the
expectation produced by running it; it becomes writable only when the reference and both
circuit paths and the resident batch agree field by field on every tick and a second
execution in a fresh process reproduces the same outcome.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import disasm
import golden_sim as G
import isa_table as ISA

KINDS = ("cpt", "sft", "rl")
FLAVORS = ("reference", "torch", "triton", "batched")
REQUIRED = ("kind", "text", "input", "initial_data", "budget", "config",
            "expected", "machine_commit", "flavor", "compiler_identity", "code")

OPTIONAL = ("agreement", "emission", "execution")

EXPECT_KEYS = ("status", "out", "state_delta", "fault_reason", "reassembles_to")

KIND_EXPECT = {"rl": ("status", "out", "state_delta", "fault_reason"),
               "sft": ("status", "out", "state_delta", "fault_reason"),
               "cpt": ("reassembles_to",)}

COMPARED_FIELDS = tuple(f for f in ("r", "HL", "DE", "MB", "SP", "PC", "C", "Z", "S", "V",
                                    "TDEPTH", "STC_COUNT", "STC_FIRST",
                                    "ipos", "tick", "status", "fault_reason",
                                    "fault_addr")
                        if f != "oplen")

CONFIG_KEYS = tuple(ISA.MachineConfig.__slots__)

IDENTITY_KEYS = ("ncl_version", "emitter_digest", "flags")
DRIVER_SLACK = 8

class RecordError(ValueError):

    def __init__(self, problems, where="record"):
        self.problems = list(problems)
        super().__init__(f"{where} is not writable: " + "; ".join(self.problems))

def is_hex(text):
    return (isinstance(text, str) and len(text) % 2 == 0
            and all(c in "0123456789abcdefABCDEF" for c in text))

def schema_problems(rec):

    problems = []
    for field in REQUIRED:
        if field not in rec:
            problems.append(f"missing field {field!r}")
    if rec.get("kind") not in KINDS:
        problems.append(f"kind {rec.get('kind')!r} is not one of {KINDS}")
    if rec.get("flavor") not in FLAVORS:
        problems.append(f"flavor {rec.get('flavor')!r} is not one of {FLAVORS}")
    if not rec.get("machine_commit"):
        problems.append("no machine_commit, so the record cannot be re-scored against the "
                        "code that labelled it")
    ident = rec.get("compiler_identity")
    if not ident:
        problems.append("no compiler_identity, so a compiler change would silently relabel "
                        "this sample instead of making a new dataset")
    elif not isinstance(ident, dict) or set(ident) != set(IDENTITY_KEYS):

        problems.append("compiler_identity must be exactly the keys "
                        f"{sorted(IDENTITY_KEYS)}, got "
                        f"{sorted(ident) if isinstance(ident, dict) else type(ident).__name__}"
                        ": a missing key cannot be told from an empty one")
    undefined = sorted(set(rec) - set(REQUIRED) - set(OPTIONAL))
    if undefined:
        problems.append(f"the schema defines no field {undefined}: a field no reader in "
                        f"the writer consumes is not part of a record")
    if not isinstance(rec.get("budget"), int) or rec.get("budget", 0) <= 0:
        problems.append("budget must be a positive integer, because 'ran out of ticks' has "
                        "to be distinguishable from 'halted'")
    for field in ("input", "initial_data"):
        if not is_hex(rec.get(field, "")):
            problems.append(f"{field} must be hex text, so a re-scoring run decodes the "
                            f"bytes the label was written from")
    code = rec.get("code")
    if not str(code or "").isascii() or not is_hex(str(code)) or not str(code or ""):
        problems.append("code must be the non-empty hex bytes the record's text compiles "
                        "to, so re-scoring has the program the label was written from")
    cfg = rec.get("config")
    if not isinstance(cfg, dict):
        problems.append("config must be the load-time configuration the label was produced "
                        "under, which nothing else in the record can derive")
    else:
        unknown = sorted(set(cfg) - set(CONFIG_KEYS))
        if unknown:
            problems.append(f"config names constraints outside the documented field set: {unknown}")
    cap = (cfg or {}).get("outcap") if isinstance(cfg, dict) else None
    if not isinstance(cap, int):
        cap = G.OUT_CAP
    exp = rec.get("expected")
    if not isinstance(exp, dict) or not exp:
        problems.append("expected must name at least one of " + "/".join(EXPECT_KEYS))
    else:
        unknown = sorted(set(exp) - set(EXPECT_KEYS))
        if unknown:
            problems.append(f"expected has keys that assert nothing: {unknown}")
        allowed = KIND_EXPECT.get(rec.get("kind"), ())
        misused = sorted(set(exp) - set(allowed))
        if misused:
            problems.append(f"kind {rec.get('kind')!r} cannot assert with {misused}: a "
                            f"continuation sample has no run to score, and an execution "
                            f"sample must not assert only about text")
        if rec.get("kind") == "cpt":
            claim = exp.get("reassembles_to")
            if not claim or not is_hex(claim):
                problems.append("reassembles_to must be the hex bytes the record's text "
                                "compiles to; an empty value asserts nothing")
            elif rec.get("code") and is_hex(str(rec["code"])) and \
                    bytes.fromhex(claim) != bytes.fromhex(rec["code"]):

                problems.append(f"reassembles_to says {claim}, the record's own code is "
                                f"{rec['code']}")
        status = exp.get("status")
        if status == 3 and not exp.get("fault_reason"):
            problems.append("a faulted expectation must name its cause; status 3 alone lets "
                            "any bound violation pass as the intended outcome")
        if status == 2 and "out" not in exp and "state_delta" not in exp:
            problems.append("an overrun expectation with no out or state_delta asserts only "
                            "that the budget ran out")
        out = exp.get("out")
        if out is not None:
            if not is_hex(out):
                problems.append("expected.out must be hex text, for the same reason the "
                                "inputs must be")
            elif len(bytes.fromhex(out)) == cap:
                problems.append(f"expected.out is exactly the run's output capacity "
                                f"({cap} bytes), so a prefix match and a complete answer "
                                f"would be the same test")
    return problems

def encoding_problems(code):

    flagged = [row[0] for row in disasm.disasm(bytes(code), 0, len(code))
               if len(row) > 3 and row[3] is True]
    return [f"encoding at offset {o} is non-canonical, so the corpus would carry programs "
            f"that differ only in bits the datapath ignores" for o in flagged]

def _scalar(v):

    if hasattr(v, "reshape"):
        v = int(v.reshape(-1).tolist()[0])
    if isinstance(v, str):
        return G.STATUS_CODE[v]
    return int(v)

def _row(machine, is_batch, row):
    snap = machine.snapshot(row) if is_batch else machine.snapshot()
    out = {}
    for field in COMPARED_FIELDS:
        if field not in snap:
            out[field] = None
        elif field == "r":
            v = snap[field]
            out[field] = [int(x) for x in (v.reshape(-1).tolist()
                                           if hasattr(v, "reshape") else v)]
        else:
            out[field] = _scalar(snap[field])
    return out

def agreement(code, data=None, inputs=b"", budget=64, config=None,
              tensor_device="cpu"):

    batch = None

    cfg = None
    if config:
        cfg = ISA.MachineConfig.from_dict(config)
    ref = G.NCP8(code, data=data, inputs=inputs, tick_budget=budget, config=cfg)
    paths = [("reference", ref, None)]
    try:
        from circuit_torch import TorchCircuit
        from circuit_triton import TritonBatch, TritonCircuit
    except Exception as exc:
        return None, [f"the circuit paths could not be imported, so four-way agreement "
                      f"did not run: {exc}"]
    paths.append(("torch", TorchCircuit(code, data=data, inputs=inputs, tick_budget=budget,
                                        device=tensor_device, config=cfg), None))
    paths.append(("triton", TritonCircuit(code, data=data, inputs=inputs,
                                          tick_budget=budget, config=cfg), None))

    try:
        batch = TritonBatch(1, tick_budget=budget, max_in=max(1, len(inputs)),
                            config=cfg)
        batch.set_program(0, code, data, inputs)
    except ValueError as exc:
        return None, [f"the batched path could not be loaded with this record: {exc}"]
    paths.append(("batched", batch, 0))

    cap = budget + DRIVER_SLACK
    fails, ticks = [], 0
    for _ in range(cap):
        active = ref.status == G.STATUS_RUNNING
        if active:
            try:
                ref.step()
            except G.MachineError:
                pass
            for _name, machine, row in paths[1:]:
                machine.step(1) if row is not None else machine.step()
        want = _row(ref, False, None)
        out_ref = bytes(ref.out)
        for name, machine, row in paths[1:]:
            got = _row(machine, row is not None, row)
            for field in COMPARED_FIELDS:
                if got[field] != want[field]:
                    fails.append(f"tick {ticks}: {name}.{field} is {got[field]!r}, "
                                 f"reference says {want[field]!r}")
            stream = bytes(machine.out(row) if row is not None else machine.out())
            if stream != out_ref:
                fails.append(f"tick {ticks}: {name}'s output stream is "
                             f"{stream.hex()!r}, reference has {out_ref.hex()!r}")
        ticks += 1
        if not active and all(_status(m, r) != 0 for _n, m, r in paths[1:]):
            break
    else:
        stuck = [n for n, m, r in paths if _status(m, r) == 0]
        fails.append(f"the driver guard of {cap} steps was reached with {stuck} still "
                     f"RUNNING, so the run never produced a label")
    final = _row(ref, False, None)
    return {"ticks_compared": ticks, "final": final, "out": out_ref.hex(),
            "final_tick": final["tick"], "fields": list(COMPARED_FIELDS),
            "tensor_device": tensor_device}, fails

def _status(machine, row):
    snap = machine.snapshot(row) if row is not None else machine.snapshot()
    return _scalar(snap["status"])

def default_compile(text):

    import loader
    loaded = loader.assemble(text)
    return bytes(loaded.image[:loaded.content_extent])

FRONT_END_SOURCES = ("disasm.py", "golden_sim.py", "isa_forms.py", "isa_table.py",
                     "loader.py")

def front_end_digest():

    here = Path(__file__).resolve().parent
    parts = []
    for name in FRONT_END_SOURCES:
        parts.append(name.encode("utf-8") + b"\0" + (here / name).read_bytes() + b"\0")
    return hashlib.sha256(b"".join(parts)).hexdigest()

EMITTERS = [("asm-", default_compile, front_end_digest)]

def register_emitter(prefix, emit, digest=None):

    if any(prefix == seen or prefix.startswith(seen) or seen.startswith(prefix)
           for seen, _, _ in EMITTERS):
        raise RecordError([f"emitter prefix {prefix!r} overlaps a registered prefix"])
    EMITTERS.append((prefix, emit, digest))
    return len(EMITTERS) - 1

def emitter_for(version):

    for prefix, emit, digest in EMITTERS:
        if str(version).startswith(prefix):
            return emit, digest
    return None, None

def emitter_digest(version):

    _, digest = emitter_for(version)
    return None if digest is None else digest()

def emitter_identity(version, flags=()):

    digest = emitter_digest(version)
    if digest is None:
        raise RecordError([f"emitter {version!r} declares no digest, so a record cannot "
                           f"name its producer unambiguously"])
    return {"ncl_version": version, "emitter_digest": digest, "flags": list(flags)}

def emitter_problems(rec, name=None):

    ident = rec.get("compiler_identity") or {}
    version = str(ident.get("ncl_version", ""))
    emit, digest = emitter_for(version)
    if emit is None:
        return [f"no emitter in this tree compiles records tagged {version!r}"]
    where = f"record {name!r}" if name else "record"
    problems = []
    if digest is not None:
        live = digest()
        recorded = str(ident.get("emitter_digest", ""))
        if recorded != live:
            problems.append(
                f"{where} names emitter {version!r} with digest {recorded}, this tree's "
                f"{version!r} front end digests to {live}: the producer recorded in the "
                f"record is not the one reading it, so a change to the compiler would "
                f"relabel this sample silently instead of making a new dataset")
    code = rec.get("code")
    if not is_hex(str(code or "")) or not str(code or ""):
        return problems + [f"code must be the hex bytes the text compiles to, got {code!r}"]
    try:
        rebuilt = bytes(emit(rec["text"]))
    except Exception as exc:
        return problems + [f"the emitter {version} refuses the record's own text: {exc}"]
    if rebuilt.hex() != str(code):
        def short(hexed):
            return hexed if len(hexed) <= 64 else hexed[:64] + f"... ({len(hexed) // 2} bytes)"
        problems.append(f"text compiles to {short(rebuilt.hex())} under {version}, the "
                        f"record's code is {short(str(code))}")
    return problems

def shaped(spec):

    rec = {k: spec[k] for k in spec if k in REQUIRED}
    extras = {k: v for k, v in spec.items() if k not in REQUIRED and k not in OPTIONAL}
    if extras:
        rec["emission"] = extras
    return rec

def label(spec, compile_text=None):

    code = spec.get("code")
    if isinstance(code, str):
        code = bytes.fromhex(code)
    if code is None:
        if compile_text is None:

            version = str((spec.get("compiler_identity") or {}).get("ncl_version", ""))
            compile_text = emitter_for(version)[0] or default_compile
        code = compile_text(spec["text"])
    code = bytes(code)
    data = bytes.fromhex(spec.get("initial_data", "")) or None
    inputs = bytes.fromhex(spec.get("input", ""))
    budget = spec["budget"]
    problems = encoding_problems(code)
    recoded = None
    if spec.get("kind") == "cpt":

        recoded = bytes(default_compile(spec["text"]) if compile_text is None
                        else compile_text(spec["text"]))
        claimed = spec.get("expected", {}).get("reassembles_to")
        if recoded != code:
            problems.append(f"the text compiles to {recoded.hex()}, not to the {code.hex()} "
                            f"the record describes")
        if claimed is not None and bytes.fromhex(claimed) != recoded:
            problems.append(f"reassembles_to says {claimed}, the compiler says "
                            f"{recoded.hex()}")
    meta, fails = agreement(code, data=data, inputs=inputs, budget=budget,
                            config=spec.get("config"))
    problems += fails
    if meta is None:
        raise RecordError(problems or ["agreement did not run"])
    execution = {"status": meta["final"]["status"], "out": meta["out"]}
    if execution["status"] == 3:
        execution["fault_reason"] = ISA.CAUSE_NAME.get(meta["final"]["fault_reason"],
                                                       meta["final"]["fault_reason"])

    expected = {"reassembles_to": recoded.hex()} if recoded is not None else execution
    rec = shaped(spec)
    rec.update({"expected": expected, "machine_commit": machine_commit(),
                "flavor": "reference", "code": code.hex(), "execution": execution})
    rec["agreement"] = {"ticks_compared": meta["ticks_compared"],
                        "compared_fields": meta["fields"],
                        "final_tick": meta["final"]["tick"],
                        "tensor_device": meta["tensor_device"]}
    emission = rec.setdefault("emission", {})
    for field, source in (("source_hash", spec["text"].encode("utf-8")),
                          ("bytes_hash", code)):
        value = hashlib.sha256(source).hexdigest()
        claimed = emission.get(field)
        if claimed is not None and claimed != value:
            problems.append(f"emission {field} says {claimed}, the record's own "
                            f"{'text' if field == 'source_hash' else 'code'} hashes to "
                            f"{value}")
        emission[field] = value
    rec["emission"] = emission
    problems += schema_problems(rec) + emitter_problems(rec)
    if problems:
        raise RecordError(problems)
    return rec

def machine_commit():

    root = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "-C", str(root.parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root.parent), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        return f"unknown:{exc}"
    if not head:
        return "unknown:no-git"
    if dirty:
        digest = hashlib.sha256(dirty.encode("utf-8")).hexdigest()[:12]
        return f"dirty:{head[:12]}+{digest}"
    return head[:12]

def provenance():

    import torch
    import os
    try:
        triton_version = __import__("triton").__version__
    except ImportError:
        triton_version = "absent"
    return {"machine_commit": machine_commit(),
            "torch": torch.__version__,
            "triton": triton_version,
            "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                       else "cpu"),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG",
                                                      "unset")}

REEXECUTE = """
import json, sys
sys.path.insert(0, %(tree)r)
import golden_sim as G
specs = json.loads(open(%(src)r, encoding="utf-8").read())
out = {}
for sp in specs:
    code = bytes.fromhex(sp["code"])
    m = G.NCP8(code, data=bytes.fromhex(sp["initial_data"]) or None,
               inputs=bytes.fromhex(sp["input"]), tick_budget=sp["budget"],
               config=G.ISA.MachineConfig.from_dict(sp.get("config")))
    for _ in range(sp["budget"] + %(slack)d):
        if m.status == G.STATUS_RUNNING:
            try:
                m.step()
            except G.MachineError:
                break
        else:
            break
    out[sp["name"]] = {"out": bytes(m.out).hex(),
                       "status": G.STATUS_CODE[m.status], "tick": m.tick,
                       "fault_reason": getattr(m, "fault_reason", None)}
open(%(dst)r, "w", encoding="utf-8").write(json.dumps(out))
"""

def _sample_name(record, index):

    import hashlib
    material = ":".join((record.get("code", ""), record.get("input", ""),
                         record.get("initial_data", ""), str(record.get("budget", ""))))
    return f"{index}:{hashlib.sha256(material.encode()).hexdigest()[:12]}"

def reexecute(records, name_of=None):

    import os
    import tempfile
    src = Path(tempfile.gettempdir()) / f"records_src_{os.getpid()}.json"
    dst = Path(tempfile.gettempdir()) / f"records_out_{os.getpid()}.json"
    names = [name_of(r) if name_of is not None else _sample_name(r, i)
             for i, r in enumerate(records)]
    payload = [{"name": nm, "code": r["code"],
                "initial_data": r.get("initial_data", ""),
                "input": r.get("input", ""), "budget": r["budget"],
                "config": r.get("config")} for nm, r in zip(names, records)]
    src.write_text(json.dumps(payload), encoding="utf-8")
    script = REEXECUTE % {"tree": str(Path(__file__).resolve().parent),
                          "src": str(src), "dst": str(dst), "slack": DRIVER_SLACK}
    try:
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=3600)
    except Exception as exc:
        return {name_of(r): [f"re-execution did not run: {exc}"] for r in records}
    if proc.returncode or not dst.exists():
        reason = (proc.stderr.strip().splitlines() or ["no output written"])[-1]
        return {name_of(r): [f"re-execution did not run: {reason}"] for r in records}
    got = json.loads(dst.read_text(encoding="utf-8"))
    problems = {}
    for r, key in zip(records, names):
        if key not in got:
            problems[key] = ["the fresh process reported no result for this record"]
            continue

        exp = r["expected"] if r["kind"] != "cpt" else r.get("execution")
        act = got[key]
        diffs = []
        if not isinstance(exp, dict):
            problems[key] = ["a continuation record carries no execution to re-check"]
            continue
        if act["out"] != exp.get("out"):
            diffs.append(f"out: recorded {exp.get('out')!r}, fresh process {act['out']!r}")
        if act["status"] != exp.get("status"):
            diffs.append(f"status: recorded {exp.get('status')}, fresh process "
                         f"{act['status']}")
        recorded_tick = r.get("agreement", {}).get("final_tick")
        if "agreement" in r and recorded_tick is None:
            diffs.append("the record names an agreement block but no final tick, so the "
                         "tick count cannot be re-checked")
        elif recorded_tick is not None and act["tick"] != recorded_tick:
            diffs.append(f"tick: recorded {recorded_tick}, fresh process {act['tick']}")
        if diffs:
            problems[key] = diffs
    for path in (src, dst):
        path.unlink(missing_ok=True)
    return problems

def write_records(path, records, name_of=None):

    records = list(records)
    problems = []
    names = [name_of(r) if name_of is not None else _sample_name(r, i)
             for i, r in enumerate(records)]
    for i, (r, key) in enumerate(zip(records, names)):
        for p in schema_problems(r):
            problems.append(f"record {i}: {p}")
        for p in emitter_problems(r, name=key):
            problems.append(f"record {i}: {p}")
    fresh = reexecute(records, name_of) if name_of is not None \
        else reexecute(records)
    for key, diffs in fresh.items():
        if diffs:
            problems.append(f"record {key!r} did not reproduce in a fresh process: "
                            + "; ".join(diffs))
    if problems:
        raise RecordError(problems, where=f"batch of {len(records)}")
    path = Path(path)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)