"""Acceptance for the training-record contract.

Every refusal has a sample that produces it, and the cross-implementation comparator is
checked by making the reference report a wrong value for one field.
"""
import json
import tempfile
from pathlib import Path

import golden_sim as G
import recordlib as R

FAILS = []

def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)

def refused(name, mutate, want_fragment):

    rec = good_record()
    mutate(rec)
    problems = R.schema_problems(rec)
    hit = [p for p in problems if want_fragment in p]
    check(f"refusal {name}", bool(hit), f"expected a problem containing {want_fragment!r}, "
                                       f"got {problems}")

def good_record():

    return {"kind": "rl", "text": "  HALT\n", "code": "00", "input": "",
            "initial_data": "", "budget": 8, "code": "00",
            "config": {"outcap": G.OUT_CAP}, "expected": {"status": 1, "out": ""},
            "machine_commit": "test-commit", "flavor": "reference",
            "compiler_identity": {"ncl_version": "asm-test", "emitter_digest": "a" * 64, "flags": []}}

def main():

    meta, fails = R.agreement(G.asm("  LDI r0, 65\n  OUT r0\n  HALT"), budget=16)
    check("agreement runs", meta is not None and not fails, str(fails[:2]))
    check("agreement compared ticks", bool(meta) and meta["ticks_compared"] > 0,
          str(meta and meta["ticks_compared"]))
    check("agreement covers the fault fields",
          "fault_reason" in meta["fields"] and "fault_addr" in meta["fields"],
          str(meta["fields"]))

    check("agreement states the tensor device it used",
          meta.get("tensor_device") == "cpu", str(meta.get("tensor_device")))
    prog = G.asm("  LDI r0, 250\nloop:\n  DJNZ r0, loop\n  OUT r0\n  HALT\n")
    from circuit_torch import TorchCircuit
    import torch
    if torch.cuda.is_available():
        other, name = TorchCircuit(prog, tick_budget=300, device="cuda"), \
            "the tensor path answers the same on either device"
    else:
        other, name = G.NCP8(prog, tick_budget=300), \
            "the tensor path on cpu answers as the reference does (no device present)"
    ours = TorchCircuit(prog, tick_budget=300, device="cpu")
    diverged, ticks = None, 0
    for _ in range(250):
        mine, theirs = ours.snapshot(), other.snapshot()
        if mine != theirs:
            diverged = (ticks, {k: (theirs[k], mine[k]) for k in mine if mine[k] != theirs[k]})
            break
        ours.step()
        other.step()
        ticks += 1
    check(name, diverged is None and ticks >= 200,
          f"diverged at tick {diverged}" if diverged else f"{ticks} ticks compared")

    real = G.NCP8.snapshot
    G.NCP8.snapshot = lambda self: {**real(self), "DE": real(self)["DE"] + 1}
    try:
        _meta, lied = R.agreement(G.asm("  HALT\n"), budget=8)
    finally:
        G.NCP8.snapshot = real
    check("a lying reference is caught", any("DE" in f for f in lied), str(lied[:2]))

    refused("field the schema does not define", lambda r: r.update(entry="main"),
            "defines no field")
    refused("zero budget", lambda r: r.update(budget=0), "budget")
    refused("non-hex initial data", lambda r: r.update(initial_data="zz"), "hex text")
    refused("unknown config key", lambda r: r.update(config={"wibble": 1}),
            "outside the documented")
    refused("fault without a cause",
            lambda r: r.update(expected={"status": 3, "out": ""}), "name its cause")
    refused("answer of exactly the capacity",
            lambda r: r.update(expected={"status": 1, "out": "41" * G.OUT_CAP}),
            "output capacity")
    short = good_record()
    short["expected"] = {"status": 1, "out": "41" * (G.OUT_CAP - 1)}
    check("an answer one byte short of the capacity is accepted",
          not R.schema_problems(short), str(R.schema_problems(short)))
    refused("non-hex expected output",
            lambda r: r.update(expected={"status": 1, "out": "b'AB'"}), "hex text")
    refused("label from a path outside the four",
            lambda r: r.update(flavor="model-guess"), "flavor")
    refused("no machine commit", lambda r: r.update(machine_commit=""), "machine_commit")

    listing = "  HALT\n"
    cpt = {**good_record(), "kind": "cpt", "text": listing, "code": "00",
           "expected": {"reassembles_to": "00"}}
    check("cpt asserting its own bytes is accepted", not R.schema_problems(cpt),
          str(R.schema_problems(cpt)))
    for name, mutated, fragment in (
            ("cpt asserting a status", {"status": 1, "out": ""}, "cannot assert"),
            ("cpt with an empty claim", {"reassembles_to": ""}, "asserts nothing"),
            ("cpt claiming other bytes", {"reassembles_to": "ff"},
             "the record's own code is")):
        bad = {**cpt, "expected": mutated}
        hit = [x for x in R.schema_problems(bad) if fragment in x]
        check(f"refusal {name}", bool(hit), str(R.schema_problems(bad)))
    rl_claim = {**good_record(), "kind": "rl", "expected": {"reassembles_to": "00"}}
    check("refusal rl asserting only about text",
          any("cannot assert" in x for x in R.schema_problems(rl_claim)),
          str(R.schema_problems(rl_claim)))

    good_ident = {"ncl_version": "asm-test", "emitter_digest": "b" * 64, "flags": []}
    ok_rec = {**good_record(), "compiler_identity": good_ident}
    check("a compiler-only identity is accepted", not R.schema_problems(ok_rec),
          str(R.schema_problems(ok_rec)))
    for name, ident in (
            ("identity carrying the artifact digest",
             {**good_ident, "bytes_hash": "00" * 32}),
            ("identity carrying a unit-set hash",
             {**good_ident, "unit_set_hash": "11" * 32}),
            ("identity missing a key",
             {"ncl_version": "asm-test", "emitter_digest": "b" * 64}),
            ("identity as a bare string", "ncl0-1.0.0")):
        bad = {**ok_rec, "compiler_identity": ident}
        hit = [x for x in R.schema_problems(bad) if "compiler_identity" in x]
        check(f"refusal {name}", bool(hit), str(R.schema_problems(bad))[:120])

    enc = R.encoding_problems(bytes([0x11, 0xFC]))
    check("non-canonical ADDI HL refused", bool(enc), str(enc))
    check("canonical ADDI HL accepted",
          not R.encoding_problems(bytes([0x11, 0x00])),
          str(R.encoding_problems(bytes([0x11, 0x00]))))

    check("EXT past the vector table is a program, not a spelling",
          not R.encoding_problems(bytes([0x70, 0x70, 0x14])),
          str(R.encoding_problems(bytes([0x70, 0x70, 0x14]))))
    check("canonical HALT accepted", not R.encoding_problems(G.asm("  HALT\n")))

    good = R.label({"kind": "rl", "text": "  LDI r0, 66\n  OUT r0\n  HALT\n",
                    "entry": "main", "input": "", "initial_data": "", "budget": 16,
                    "config": {"outcap": G.OUT_CAP},
                    "compiler_identity": {"ncl_version": "asm-test", "emitter_digest": "a" * 64, "flags": []}})
    check("label built from a program", good["expected"]["out"] == "42",
          str(good["expected"]))
    stream = R.label({"kind": "sft", "text": "loop:\n  IN r0\n  OUT r0\n  ADDI r1, 1\n"
                      "  LDI r2, 4\n  CMP r1, r2\n  JNZ loop\n  HALT\n",
                      "entry": "main", "input": bytes(range(1, 5)).hex(),
                      "initial_data": "", "budget": 40,
                      "config": {"outcap": G.OUT_CAP},
                      "compiler_identity": {"ncl_version": "asm-test", "emitter_digest": "a" * 64, "flags": []}})
    check("a program that reads a multi-byte input stream is labelable",
          stream["expected"]["out"] == "01020304" and stream["kind"] == "sft",
          str(stream["expected"]))

    check("provenance recorded", good["machine_commit"] == R.machine_commit()
          and bool(R.provenance()["machine_commit"]))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corpus.jsonl"
        n = R.write_records(path, [good], name_of=lambda r: "a")
        lines = path.read_text(encoding="utf-8").splitlines()
        check("one record written", n == 1 and len(lines) == 1, str(lines))
        check("the written record re-reads identically",
              json.loads(lines[0])["expected"] == good["expected"])
        doctored = json.loads(json.dumps(good))
        doctored["expected"]["out"] = "ff"
        try:
            R.write_records(path, [doctored], name_of=lambda r: "b")
            check("a doctored label is refused", False, "write_records returned normally")
        except R.RecordError as e:
            check("a doctored label is refused",
                  any("fresh process" in p for p in e.problems), str(e.problems[:2]))
        check("the refused batch left the file alone",
              len(path.read_text(encoding="utf-8").splitlines()) == 1)

    no_code = {k: v for k, v in good.items() if k != "code"}
    check("refusal record without its bytes",
          any("code" in x for x in R.schema_problems(no_code)), str(R.schema_problems(no_code)))
    check("refusal record whose code is not hex",
          any("code" in x for x in R.schema_problems({**good, "code": "zz"})),
          str(R.schema_problems({**good, "code": "zz"})))
    check("emitter refuses a tag no emitter answers to",
          any("no emitter" in x for x in R.emitter_problems(
              {**good, "compiler_identity": {**good["compiler_identity"],
                                             "ncl_version": "brainfuck-1"}})),
          str(R.emitter_problems({**good, "compiler_identity": {
              **good["compiler_identity"], "ncl_version": "brainfuck-1"}})))
    calls = []

    def probe_emit(text):
        calls.append(text)
        return bytes.fromhex(good["code"])

    R.register_emitter("probe-", probe_emit)
    tagged = {**good, "compiler_identity": {**good["compiler_identity"],
                                            "ncl_version": "probe-1.0"}}
    check("a registered emitter rebuilds a record tagged with its prefix",
          not R.emitter_problems(tagged) and calls == [tagged["text"]],
          str(R.emitter_problems(tagged)))
    before = len(calls)
    by_tag = R.label({"kind": "rl", "text": good["text"], "input": "",
                      "initial_data": "", "budget": good["budget"],
                      "config": good["config"],
                      "compiler_identity": {**good["compiler_identity"],
                                            "ncl_version": "probe-tag"}})

    check("a label with no code and no hook compiles through the tagged emitter",
          by_tag["code"] == good["code"] and len(calls) - before == 2
          and set(calls[before:]) == {good["text"]},
          f"{by_tag['code']} vs {good['code']}, {len(calls) - before} calls")
    for collide in ("probe-x", "asm-"):
        try:
            R.register_emitter(collide, probe_emit)
            was_refused = False
        except R.RecordError as e:
            was_refused = "overlaps" in " ".join(e.problems)
        check(f"registering {collide!r} over an existing prefix is refused",
              was_refused, "accepted it")

    swapped = {**good, "text": "  HALT\n"}
    check("emitter catches text that was swapped after labelling",
          any("compiles to" in x for x in R.emitter_problems(swapped)),
          str(R.emitter_problems(swapped)))

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "corpus.jsonl"
        R.write_records(path, [good])
        swapped = json.loads(json.dumps(good))
        swapped["text"] = "  HALT\n"
        try:
            R.write_records(path, [swapped], name_of=lambda r: "s")
            check("write refuses text that its own bytes do not come from", False,
                  "write_records returned")
        except R.RecordError as e:
            check("write refuses text that its own bytes do not come from",
                  any("compiles to" in x for x in e.problems), str(e.problems[:2]))
        check("the doctored-text batch left the corpus at one record",
              len(path.read_text(encoding="utf-8").splitlines()) == 1,
              str(len(path.read_text(encoding="utf-8").splitlines())))

    cont = R.label({"kind": "cpt",
                    "text": "  LDI r0, 7\n  ADDI r0, 1\n  OUT r0\n  HALT\n",
                    "input": "", "initial_data": "", "budget": 24,
                    "config": {"outcap": G.OUT_CAP},
                    "compiler_identity": {"ncl_version": "asm-test",
                                          "emitter_digest": "d" * 64, "flags": []}})
    check("a cpt sample is labelled through the four paths",
          cont["expected"] == {"reassembles_to": cont["code"]}
          and cont["agreement"]["ticks_compared"] > 0
          and cont["execution"]["out"] == "08", str(cont["expected"]))
    check("every kind has a sample that reached four-path agreement in this run",
          {good["kind"], stream["kind"], cont["kind"]} == set(R.KINDS)
          and all(r["agreement"]["ticks_compared"] > 0
                  for r in (good, stream, cont)),
          f"{good['kind']}/{stream['kind']}/{cont['kind']}")
    moved = json.loads(json.dumps(good))
    moved["agreement"]["final_tick"] += 1
    diffs = R.reexecute([moved], name_of=lambda r: "t")
    check("a tick count that does not repeat is caught",
          any("tick:" in x for group in diffs.values() for x in group), str(diffs))
    no_tick = json.loads(json.dumps(good))
    del no_tick["agreement"]["final_tick"]
    check("an agreement block without a final tick is refused",
          any("no final tick" in x for group in R.reexecute([no_tick],
                                                            name_of=lambda r: "n").values()
              for x in group), str(R.reexecute([no_tick], name_of=lambda r: "n")))
    clean = R.reexecute([good], name_of=lambda r: "g")
    check("the unaltered record reproduces its tick count",
          clean == {"g": []}, str(clean))

    with tempfile.TemporaryDirectory() as td:
        cpath = Path(td) / "cpt.jsonl"
        R.write_records(cpath, [cont])
        check("the cpt sample is writable", cpath.read_text(encoding="utf-8").count("\n") == 1,
              cpath.read_text(encoding="utf-8"))

    import hashlib
    check("the record carries a source hash the record's text gives",
          good["emission"]["source_hash"]
          == hashlib.sha256(good["text"].encode("utf-8")).hexdigest(),
          str(good["emission"].get("source_hash")))
    check("the record carries a bytes hash the record's code gives",
          good["emission"]["bytes_hash"]
          == hashlib.sha256(bytes.fromhex(good["code"])).hexdigest(),
          str(good["emission"].get("bytes_hash")))
    check("the producer's extra fields survive as emission, not as schema",
          good["emission"].get("entry") == "main" and "entry" not in good,
          str(sorted(good)))
    try:
        R.label({"kind": "rl", "text": "  HALT\n", "input": "", "initial_data": "",
                 "budget": 8, "config": {"outcap": G.OUT_CAP},
                 "compiler_identity": good["compiler_identity"],
                 "bytes_hash": "f" * 64})
        check("a producer hash that disagrees with the record is refused", False,
              "label returned")
    except R.RecordError as e:
        check("a producer hash that disagrees with the record is refused",
              any("bytes_hash" in x for x in e.problems), str(e.problems[:2]))

    producer_handoff = {k: v for k, v in good.items()
                        if k not in R.OPTIONAL}
    producer_handoff.update({"entry": 0, "unit_set_hash": "ab" * 32})
    folded = R.shaped(producer_handoff)
    check("a producer's extra numbers fold into emission",
          not R.schema_problems(folded)
          and folded["emission"] == {"entry": 0, "unit_set_hash": "ab" * 32},
          str(R.schema_problems(folded)))

    import test_state_contract as T
    check("the record's compared fields are the suite's view minus oplen",
          R.COMPARED_FIELDS == tuple(f for f in T.VIEW_FIELDS if f != "oplen"),
          f"{R.COMPARED_FIELDS} vs {T.VIEW_FIELDS}")

    print(f"\ntraining-record acceptance: {len(FAILS)} failure(s)")
    return 1 if FAILS else 0

if __name__ == "__main__":
    raise SystemExit(main())