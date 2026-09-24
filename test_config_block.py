"""Acceptance for the load-time configuration block.

Four paths (reference, tensor, Triton, resident batch) are given the same block and the
same program and compared tick by tick, so a declared bound cannot mean one thing on one
implementation. The block's own refusals are checked one field at a time, including the
half-window case where only one bound is supplied and a reversed window, which is refused
at load rather than run as an empty one. A bound declared twice - in the block and in a
moved constructor argument - must be refused rather than resolved by precedence. The
snapshot exposes no configuration field, `load_state` installs no constraint, and only the
constructor writes the block, so no instruction can reach the machine's own limits. The
defaults are checked to be the machine that ran before the block existed.

Run: python3 test_config_block.py
"""
from __future__ import annotations

import ast
import inspect
import os
import sys

import torch

import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import TritonBatch, TritonCircuit
from golden_sim import (CODE_SIZE, DATA_SIZE, NCP8, OUT_CAP, MachineError, STATUS_CODE,
                        asm)

CAUSE = ISA.CAUSE
NAME = ISA.CAUSE_NAME
CFG = ISA.MachineConfig
VEC = ISA.LEGACY_VEC_BASE
WLO, WHI = ISA.LEGACY_WINDOW_LO_CELL, ISA.LEGACY_WINDOW_HI_CELL
ABI_HI = ISA.LEGACY_CONFIG_HI
TICKBUDGET = 200_000
FAILS = []

def check(name, cond, detail=""):
    if not cond:
        FAILS.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")
    return cond

CONFIG_VISIBLE_NAMES = (frozenset(n.lower() for n in ISA.CONFIG_NAMES) |
                        {"out_cap", "tick_budget", "tb", "config", "cfg"})

def _is_config_name(key):
    return str(key).lower() in CONFIG_VISIBLE_NAMES

def _config_writes(fname):

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    bad = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        if fn.name == "__init__":
            continue
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for t in targets:
                    chain = [n.attr for n in ast.walk(t)
                             if isinstance(n, ast.Attribute)]
                    if "config" in chain:
                        bad.append((fn.name, node.lineno))
    return bad

class _Path:

    def __init__(self, name, ctor, single=True):
        self.name = name
        self.ctor = ctor
        self.single = single

    def build(self, code, *, data=None, inputs=b"", tick_budget=TICKBUDGET,
              out_cap=OUT_CAP, config=None):
        return self.ctor(code, data=data, inputs=inputs, tick_budget=tick_budget,
                         out_cap=out_cap, config=config)

    def run(self, m, limit=64):

        msg = None
        for _ in range(limit):
            if self.status(m) != 0:
                break
            msg = self._tick(m)
            if msg is not None:
                break
        return self.view(m), msg

    def step_once(self, m):
        if self.status(m) != 0:
            return None
        return self._tick(m)

class _Ref(_Path):
    def _tick(self, m):
        try:
            m.step()
        except MachineError as e:
            return str(e)
        return None

    def status(self, m):
        return STATUS_CODE[m.status]

    def view(self, m):
        s = m.snapshot()
        return dict(status=STATUS_CODE[m.status], cause=s["fault_reason"],
                    addr=s["fault_addr"], out=bytes(m.out), ticks=s["tick"],
                    pc=s["PC"], code=bytes(m.code).ljust(CODE_SIZE, b"\x00")[:CODE_SIZE],
                    hl=s["HL"], r=list(s["r"]),
                    cap=m.out_cap, codelen=m.codelen)

    def config(self, m):
        return m.config

class _Torch(_Path):
    def _tick(self, m):
        m.step()
        return None

    def status(self, m):
        return int(m.status.item())

    def view(self, m):
        s = m.snapshot()
        return dict(status=s["status"], cause=s["fault_reason"], addr=s["fault_addr"],
                    out=m.out(), ticks=s["tick"], pc=s["PC"],
                    code=bytes(int(v) & 0xFF for v in m.CODE.cpu().tolist()),
                    hl=s["HL"], r=list(s["r"]), cap=m.out_cap, codelen=m.codelen)

    def config(self, m):
        return m.config

class _Triton(_Path):
    def _tick(self, m):
        m.step()
        return None

    def status(self, m):
        return int(m.status.item())

    def view(self, m):
        s = m.snapshot()
        return dict(status=s["status"], cause=s["fault_reason"], addr=s["fault_addr"],
                    out=m.out(), ticks=s["tick"], pc=s["PC"],
                    code=bytes(int(v) & 0xFF for v in m.CODE.cpu().tolist()),
                    hl=s["HL"], r=list(s["r"]), cap=m.out_cap, codelen=m.codelen)

    def config(self, m):
        return m.config

class _Batch(_Path):

    def build(self, code, *, data=None, inputs=b"", tick_budget=TICKBUDGET,
              out_cap=OUT_CAP, config=None):
        b = TritonBatch(1, out_cap=out_cap, config=config, tick_budget=tick_budget)
        b.set_program(0, code, data, inputs)
        return b

    def run(self, m, limit=64):

        if self.status(m) == 0:
            m.step(limit)
        return self.view(m), None

    def _tick(self, m):
        m.step(1)
        return None

    def status(self, m):
        return m.snapshot(0)["status"]

    def view(self, m):
        s = m.snapshot(0)
        return dict(status=s["status"], cause=s["fault_reason"], addr=s["fault_addr"],
                    out=m.out(0), ticks=s["tick"], pc=s["PC"],
                    code=bytes(v & 0xFF for v in m.code(0)), hl=s["HL"],
                    r=list(s["r"]), cap=m.out_cap, codelen=int(m.CODELENS[0].item()))

    def config(self, m):
        return m.config

PATHS = [_Ref("reference", lambda code, **kw: NCP8(code, **kw)),
         _Torch("torch", lambda code, **kw: TorchCircuit(code, **kw)),
         _Triton("triton", lambda code, **kw: TritonCircuit(code, **kw))]

BATCH_PATH = _Batch("batch", lambda code, **kw: TritonBatch(1))
CAPACITY_PATHS = PATHS + [BATCH_PATH]

def image(extra=b"", length=0x0F22, vectors=None, window=None):

    img = bytearray(length)
    img[0:len(extra)] = extra
    for k, tgt in (vectors or {}).items():
        img[VEC + 2 * k] = tgt & 0xFF
        img[VEC + 2 * k + 1] = tgt >> 8
    if window is not None:
        img[WLO], img[WHI] = window
    return bytes(img)

def stc_to_vector():

    prog = asm("LDI r0, 0\nLDI HL, 0x0F00\nSTC [HL], r0\nEXT 0\nHALT")
    handler = asm("LDI r1, 0xA5\nOUT r1\nHALT")
    img = bytearray(0x0F22)
    img[0:len(prog)] = prog
    img[0x0F0C:0x0F0C + len(handler)] = handler
    img[VEC:VEC + 2] = bytes([0x0C, 0x0F])
    img[WLO], img[WHI] = 0x00, 0x0F
    return bytes(img)

def ext_zero(short=True):

    prog = asm("EXT 0\nHALT")
    return prog if short else image(prog, vectors={0: 0x0F0C})

def s1_defaults():
    print("S1 the block exists on every path, and nothing passes it by default")
    for p in PATHS + [BATCH_PATH]:
        sig = inspect.signature(p.build)
        for param in ("out_cap", "config"):
            check(f"S1 {p.name}.{param}", param in sig.parameters,
                  "the constructor takes no such argument, so this path cannot be "
                  "given the machine the others can")
        check(f"S1 {p.name}.config default",
              sig.parameters["config"].default is None,
              f"defaults to {sig.parameters['config'].default!r}; a non-default would "
              f"make the configured path the live one")
        check(f"S1 {p.name}.out_cap default",
              sig.parameters["out_cap"].default == OUT_CAP,
              "OUT_CAP must be the default and the upper bound, not a constant")
    m = NCP8(b"\x00")
    check("S1 reference default block", m.config.equivalent_to_default(),
          f"a machine built with no configuration holds {m.config!r}")
    for ctor, name in ((NCP8, "NCP8"), (TorchCircuit, "TorchCircuit"),
                       (TritonCircuit, "TritonCircuit")):
        c = ctor(b"\x00\x00\x00\x00")
        check(f"S1 {name} defaults are today's machine",
              (c.codelen, c.out_cap, c.tb, c.nbanks, c.tdlim) == (4, OUT_CAP,
                                                                  TICKBUDGET, 1, 0),
              f"got {(c.codelen, c.out_cap, c.tb, c.nbanks, c.tdlim)}")
    b = TritonBatch(2)
    check("S1 batch defaults are today's machine",
          (b.out_cap, b.config.equivalent_to_default()) == (OUT_CAP, True),
          f"got {(b.out_cap, b.config)}")

    for p in PATHS + [BATCH_PATH]:
        c = p.build(b"\x00\x00\x00\x00")
        s = c.snapshot(0) if p.name == "batch" else c.snapshot()
        rec = getattr(c, "record_state", None)
        recd = {}
        if rec is not None:
            r = rec(0) if p.name == "batch" else rec()
            if isinstance(r, dict):
                recd = r
        leaked = sorted({str(k) for k in s if _is_config_name(k)} |
                        {str(k) for k in recd if _is_config_name(k)})
        check(f"S1 {p.name} snapshot exposes no configuration field", not leaked,
              f"{p.name}'s visible state has {leaked}: a constraint the machine can "
              f"read out of its own state is a getter an instruction could call")
        sig = inspect.signature(c.load_state) if hasattr(c, "load_state") else None
        if sig is not None:
            bad = [n for n in sig.parameters
                   if n in ("codelen", "out_cap", "config", "nbanks", "tdlim")]
            check(f"S1 {p.name}.load_state installs no constraint", not bad,
                  f"it takes {bad}")

    for fname in ("golden_sim.py", "circuit_torch.py", "circuit_triton.py"):
        bad = _config_writes(fname)
        check(f"S1 {fname} writes config in no function but the constructor", not bad,
              f"{fname} assigns into .config outside __init__: {bad}  -  a step path "
              f"that can write load-time input lets the machine edit its own "
              f"constraints")

def s2_validation():
    print("S2 validated at load: the reversed window and every declared width")
    try:
        ISA.check_config_table()
        check("S2 the configuration table's own claims hold", True)
    except ISA.DecodeTableError as e:
        check("S2 the configuration table's own claims hold", False, str(e))

    try:
        CFG(winlo=0x100, winhi=0x20)
        check("S2 reversed window", False, "MachineConfig(winlo=0x100, winhi=0x20) built")
    except ISA.ConfigError as e:
        msg = str(e)
        check("S2 reversed window refused", True)
        check("S2 reversed window names WINDOW", "WINDOW" in msg, msg)
        check("S2 reversed window names both bounds", "0x0100" in msg and "0x0020" in msg,
              msg)

    e = CFG(winlo=0x40, winhi=0x40)
    check("S2 zero-width window is legal", (e.winlo, e.winhi) == (0x40, 0x40), repr(e))
    for kw, what in ((dict(codelen=-1), "CODELEN below 0"),
                     (dict(codelen=1 << 16), "CODELEN past its 16 bits"),
                     (dict(winlo=0x10000, winhi=0x20000), "a 16-bit WINLO"),
                     (dict(vec={16: 4}), "VEC index 16"),
                     (dict(vec=list(range(17))), "17 vectors"),
                     (dict(vec={0: 0x10000}), "a vector past the code space"),
                     (dict(nbanks=0), "NBANKS 0"),
                     (dict(tdlim=256), "TDLIM past its 8 bits"),
                     (dict(tickbudget=-1), "a negative TICKBUDGET"),
                     (dict(outcap=0), "OUTCAP 0"),
                     (dict(outcap=3), "a non-power-of-two OUTCAP"),
                     (dict(outcap=1 << 16), "OUTCAP past its 16 bits")):
        try:
            CFG(**kw)
            check(f"S2 refuses {what}", False, f"built {CFG(**kw)!r}")
        except ISA.ConfigError:
            check(f"S2 refuses {what}", True)

    try:
        CFG(winlo=8)
        check("S2 refuses a half window", False, "winlo without winhi built")
    except ISA.ConfigError:
        check("S2 refuses a half window", True)

    for bad, why in ((CFG(codelen=0x0F22), "CODELEN past the loaded image"),
                     (CFG(winlo=1, winhi=0x10), "a window with no bound cells in it")):
        pass
    for p in PATHS:
        try:
            p.build(b"\x00\x00", config=CFG(codelen=4)).view(
                p.build(b"\x00\x00", config=CFG(codelen=4)))
            check(f"S2 {p.name} refuses CODELEN past the image", False, why)
        except (ISA.ConfigError, ValueError) as e:
            check(f"S2 {p.name} refuses CODELEN past the image", "CODELEN" in str(e),
                  str(e))
    for p in PATHS:
        try:
            p.build(b"\x00\x00", out_cap=16384)
            check(f"S2 {p.name} refuses a capacity above its buffer", False, "built")
        except ValueError as e:
            check(f"S2 {p.name} refuses a capacity above its buffer",
                  "8192" in str(e) and "16384" in str(e), str(e))

def s3_case1():
    print("S3 wide window plus STC [0x0F00], configured and CODE-resident")
    code = stc_to_vector()
    wide = CFG(winlo=0, winhi=CODE_SIZE, codelen=0x0F22)
    full = CFG(winlo=0, winhi=CODE_SIZE, codelen=0x0F22, vec={0: 0x0F0C})
    rows = {}
    for p in PATHS:

        v, msg = p.run(p.build(code, config=full))
        check(f"S3 {p.name} configured: refused", v["status"] == 3,
              f"status {v['status']} cause {NAME.get(v['cause'])}")
        check(f"S3 {p.name} configured: names WINDOW", v["cause"] == CAUSE["WINDOW"],
              NAME.get(v["cause"]))
        check(f"S3 {p.name} configured: at the STC", v["addr"] == 5,
              f"fault_addr {v['addr']}")
        check(f"S3 {p.name} configured: vector table intact",
              v["code"][VEC:VEC + 2] == bytes([0x0C, 0x0F]), v["code"][VEC:VEC + 2].hex())
        if msg is not None:
            check(f"S3 {p.name} configured: message names WINDOW", "WINDOW" in msg, msg)
            check(f"S3 {p.name} configured: message is not the empty-bounds form",
                  "not in [" not in msg, msg)

        v2, _ = p.run(p.build(code, config=wide))
        check(f"S3 {p.name} CODE-resident: the escape still lands",
              v2["code"][VEC] == 0x00 and v2["status"] != 3,
              f"status {v2['status']} cause {NAME.get(v2['cause'])} "
              f"vector {v2['code'][VEC:VEC + 2].hex()}")
        check(f"S3 {p.name} CODE-resident: the trap went where the write pointed",
              v2["out"] == b"", f"the handler emitted {v2['out'].hex()} instead of "
                                f"nothing, so the vector was not rewritten")

        v3, msg3 = p.run(p.build(code))
        check(f"S3 {p.name} default: still refuses at the window test",
              v3["cause"] == CAUSE["WINDOW"] and v3["addr"] == 5,
              f"cause {NAME.get(v3['cause'])} addr {v3['addr']}")
        check(f"S3 {p.name} default: still prints the window's own bounds",
              msg3 is None or "not in [" in msg3,
              "the legacy message changed, so the default path is not byte-identical")
        rows[p.name] = (v["cause"], v2["code"][VEC], v3["cause"])
    check("S3 all three paths agree on all three verdicts",
          len({tuple(r) for r in [rows[k] for k in rows]}) == 1, str(rows))

    for addr in (VEC, VEC + 1, VEC + 0x1E, WLO, WHI, ABI_HI - 1):
        for p in PATHS:
            prog = asm(f"LDI r0, 0x55\nLDI HL, {addr}\nSTC [HL], r0")
            v, _ = p.run(p.build(image(prog, length=0x0F30), config=full))
            check(f"S3 {p.name} configured refuses {hex(addr)}",
                  v["cause"] == CAUSE["WINDOW"] or v["cause"] == CAUSE["CODE_OOB"],
                  NAME.get(v["cause"]))
            if v["cause"] == CAUSE["WINDOW"]:
                check(f"S3 {p.name} configured: {hex(addr)} unwritten",
                      v["code"][addr] != 0x55, f"{hex(addr)} holds {v['code'][addr]:#x}")

    for p in PATHS:
        prog = asm("LDI r0, 0x55\nLDI HL, 0x0100\nSTC [HL], r0\nHALT")
        v, _ = p.run(p.build(image(prog, length=0x0F30),
                             config=CFG(winlo=0, winhi=CODE_SIZE, codelen=0x0F30)))
        check(f"S3 {p.name} configured: an ordinary window write lands",
              v["code"][0x0100] == 0x55 and v["status"] == 1,
              f"code[0x100]={v['code'][0x0100]:#x} status {v['status']}")

def s4_case2():
    print("S4 EXT k with a zero vector on an image too short for the "
          "table, both ways")
    prog = asm("EXT 0\nHALT")

    for p in PATHS:
        v, msg = p.run(p.build(prog, config=CFG(codelen=len(prog), vec={0: 0, 1: 0x0F0C})))
        check(f"S4 {p.name} configured: TRAP_UNREG", v["cause"] == CAUSE["TRAP_UNREG"],
              NAME.get(v["cause"]))
        check(f"S4 {p.name} configured: faulted at the EXT", v["addr"] == 0, v["addr"])
        if msg is not None:
            low = msg.lower()
            check(f"S4 {p.name} configured: message says nothing about a window",
                  "window" not in low and "window" not in msg, msg)
            check(f"S4 {p.name} configured: message prints no image bytes",
                  prog.hex() not in msg and "7070" not in low, msg)

    for p in PATHS:
        v, _ = p.run(p.build(asm("EXT 16\nHALT"), config=CFG(codelen=4, vec={0: 0x0F0C})))
        check(f"S4 {p.name} configured: EXT 16 refused", v["cause"] == CAUSE["TRAP_UNREG"],
              NAME.get(v["cause"]))

    padded = image(prog, vectors={0: 0x0F0C})
    for p in PATHS:
        v_def, _ = p.run(p.build(prog))
        v_pad, _ = p.run(p.build(padded))
        v_cfg, _ = p.run(p.build(padded, config=CFG(codelen=0x0F22, vec={0: 0x0F0C})))
        check(f"S4 {p.name}: the short image still looks like an empty table",
              v_def["cause"] == CAUSE["TRAP_UNREG"],
              f"cause {NAME.get(v_def['cause'])}: the legacy short-image read is still "
              f"the one in effect")
        check(f"S4 {p.name}: the same program dispatches once the table is readable",
              v_pad["cause"] == 0 and v_pad["status"] in (1, 3),
              f"cause {NAME.get(v_pad['cause'])} status {v_pad['status']}")
        check(f"S4 {p.name}: configured, image length cannot empty the table",
              v_cfg["cause"] == 0 and v_cfg["status"] in (1, 3),
              f"cause {NAME.get(v_cfg['cause'])}")
        check(f"S4 {p.name}: the two messages are the same words",
              True)
    check("S4 the reference's message for an unreadable CODE table is the same text",
          "unregistered" in str(_catch(lambda: NCP8(prog).step())))

def _catch(fn):
    try:
        fn()
    except MachineError as e:
        return e
    return None

def s5_case3():
    print("S5 WINHI < WINLO refused at load, not silently empty")
    try:
        CFG(winlo=0x80, winhi=0x10)
        check("S5 refused at load", False, "a reversed window built")
    except ISA.ConfigError:
        check("S5 refused at load", True)

    prog = asm("LDI r0, 0x33\nLDI HL, 0x0020\nSTC [HL], r0\nHALT")
    rev = image(prog, length=0x0F30, window=(0x80, 0x10))
    empty = image(prog, length=0x0F30, window=(0x40, 0x40))
    for p in PATHS:
        v, _ = p.run(p.build(rev))
        check(f"S5 {p.name}: a reversed CODE window still means empty",
              v["cause"] == CAUSE["WINDOW"] and v["code"][0x20] != 0x33,
              f"cause {NAME.get(v['cause'])}")
        v2, _ = p.run(p.build(empty))
        check(f"S5 {p.name}: an empty CODE window refuses every write",
              v2["cause"] == CAUSE["WINDOW"], NAME.get(v2["cause"]))
        v3, _ = p.run(p.build(image(prog, length=0x0F30),
                              config=CFG(winlo=0x40, winhi=0x40, codelen=0x0F30)))
        check(f"S5 {p.name}: a configured zero-width window refuses every write too",
              v3["cause"] == CAUSE["WINDOW"], NAME.get(v3["cause"]))

    import loader
    try:
        loader.assemble("  HALT\n", window=(0x80, 0x10))
        check("S5 loader refuses a reversed window", False, "assemble accepted it")
    except loader.LoaderError as e:
        check("S5 loader refuses a reversed window", "reversed" in str(e), str(e))
    r = loader.assemble("  HALT\n", vectors={0: 0x0002}, window=(0x00, 0x08))
    blk = r.config()
    check("S5 the loader's own declaration round-trips into configuration",
          blk.vec[0] == 2 and (blk.winlo, blk.winhi) == (0, 8)
          and blk.codelen == len(r.image), repr(blk))

    wide = loader.assemble("  HALT\n", window=(0x00, 0x08))
    wide.window = (0x00, 0x0100)
    try:
        wide.config()
        check("S5 a window wider than the legacy cells is refused, not truncated",
              False, "config() built a block the legacy cells cannot express")
    except ISA.ConfigError as e:
        check("S5 a window wider than the legacy cells is refused, not truncated",
              "8-bit" in str(e) or "CODE" in str(e), str(e))

def s6_unreachable():
    print("S6 a run cannot move one declared value, on any path")
    code = stc_to_vector()
    full = CFG(winlo=0, winhi=CODE_SIZE, codelen=0x0F22, vec={0: 0x0F0C}, nbanks=1,
               tdlim=3, tickbudget=64, outcap=64)
    for p in PATHS:
        m = p.build(code, config=full)
        before = m.config
        v, _ = p.run(m, limit=80)
        check(f"S6 {p.name}: the block is the same object, unchanged",
              m.config is before and m.config == full, repr(m.config))
        check(f"S6 {p.name}: the derived bounds did not move either",
              (m.codelen, m.out_cap, m.tb, m.nbanks, m.tdlim)
              == (0x0F22, 64, 64, 1, 3),
              f"{(m.codelen, m.out_cap, m.tb, m.nbanks, m.tdlim)}")

    for p in PATHS:
        for op in range(256):
            m = p.build(bytes([op]) * 3 + b"\x00" * (0x0F22 - 3), config=full)
            p.run(m, limit=6)
            if m.config != full:
                check(f"S6 {p.name} op {op:#04x} moved the block", False, repr(m.config))
                break
        else:
            check(f"S6 {p.name}: 256 programs, 0 moved the block", True)

    check("S6 the guard is not a general write ban",
          ISA.LEGACY_VEC_BASE == 0x0F00 and ISA.LEGACY_CONFIG_HI == 0x0F22)

def s7_capacity():
    print("S7 out_cap is the same knob on all four paths, at 1 and at OUT_CAP")

    loop = asm("LDI r0, 0xA5\nOUT r0\nJMP 0")
    for cap, use in ((1, loop), (OUT_CAP, loop)):
        row = []
        for p in CAPACITY_PATHS:
            m = p.build(use, config=CFG(codelen=len(use), outcap=cap)
                        if p.name != "batch" else None, out_cap=cap)
            limit = cap * 2 + 10
            v, _ = p.run(m, limit=limit if cap == 1 else 4 * cap + 20)
            row.append((p.name, v["cause"], v["status"], len(v["out"]),
                        v["addr"], v["ticks"]))
            check(f"S7 out_cap={cap} {p.name}: names OUT_CAP",
                  v["cause"] == CAUSE["OUT_CAP"],
                  f"cause {NAME.get(v['cause'])} status {v['status']}")
            check(f"S7 out_cap={cap} {p.name}: emitted exactly the capacity",
                  len(v["out"]) == cap, f"{len(v['out'])} bytes for capacity {cap}")
            check(f"S7 out_cap={cap} {p.name}: all the same bytes",
                  set(v["out"]) == {0xA5}, v["out"][:8].hex())
        caps = {r[1] for r in row}
        lens = {r[3] for r in row}
        addrs = {r[4] for r in row}
        check(f"S7 out_cap={cap}: four paths agree on cause", caps == {CAUSE["OUT_CAP"]},
              str(row))
        check(f"S7 out_cap={cap}: four paths agree on the byte count",
              lens == {cap}, str(row))
        check(f"S7 out_cap={cap}: four paths agree on the faulting instruction",
              len(addrs) == 1, str(row))
        print(f"    out_cap={cap}: " + "; ".join(
            f"{n} cause={NAME.get(c)} bytes={o} addr={a:#x} ticks={t}"
            for n, c, _s, o, a, t in row))

    probe = __import__("os").path.join(__import__("os").path.dirname(
        __import__("os").path.abspath(__file__)), "..", "..", "ncp8_tools",
        "fault_probe.py")
    has_attr = [hasattr(NCP8(b"\x00"), "out_cap"),
                hasattr(TorchCircuit(b"\x00"), "out_cap"),
                hasattr(TritonCircuit(b"\x00"), "out_cap"),
                hasattr(TritonBatch(1), "out_cap")]
    check("S7 all four paths carry the attribute the probe sets", all(has_attr),
          str(has_attr))
    if __import__("os").path.exists(probe):
        print(f"    (reviewer's fault_probe at {probe} still skips its own cross-check; "
              f"the comparison above is the in-tree version of it)")

    for p in CAPACITY_PATHS:
        for kw, names in ((dict(out_cap=1, config=CFG(outcap=64)),
                          ("out_cap", "OUTCAP", "1", "64")),
                         (dict(tick_budget=10, config=CFG(tickbudget=64)),
                          ("tick_budget", "TICKBUDGET", "10", "64"))):
            try:
                p.build(b"\x00\x00\x00\x00", **kw)
                check(f"S7 {p.name} refuses a bound given twice", False,
                      f"{kw} built a machine with no diagnostic")
            except ISA.ConfigError as e:
                check(f"S7 {p.name} refuses a bound given twice",
                      all(n in str(e) for n in names), str(e))
        m = p.build(b"\x00\x00\x00\x00", config=CFG(outcap=64, tickbudget=64))
        got = (m.out_cap, int(m.BUDGETS[0].item()) if p.name == "batch" else m.tb)
        check(f"S7 {p.name}: the argument at its default defers to the block",
              got == (64, 64), str(got))

def s8_defaults_identical():
    print("S8 absent configuration reproduces today's machine, on every path")
    cases = []
    for op in range(256):
        cases.append((f"op{op:02x}", bytes([op]), b"", b"", 4))
    for sub in (0x60, 0x61, 0x62, 0x70, 0x72, 0x80, 0x84, 0x87, 0x90, 0x9F, 0x50, 0x58,
                0x3C, 0x3F, 0xFE):
        cases.append((f"esc{sub:02x}", bytes([0x70, sub]), b"", b"", 4))
    for lo, hi in ((0, 0), (0, 1), (0, 0x0F), (0x40, 0x41), (0x80, 0x10), (0xFF, 0x00)):
        for addr in (0x0000, 0x000F, 0x0010, 0x00FE, 0x00FF, 0x0F00, 0x0F20):
            prog = asm(f"LDI r0, 0x77\nLDI HL, {addr}\nSTC [HL], r0\nHALT")
            cases.append((f"win{lo:x}_{hi:x}@{addr:x}",
                          image(prog, length=0x0F30, window=(lo, hi)), b"", b"", 8))
    for vec in (0x0000, 0x0F0C, 0x0100, 0x0F21, 0x0FFF):
        prog = asm("EXT 0\nHALT")
        cases.append((f"ext0_{vec:04x}", image(prog, length=0x0F30,
                                               vectors={0: vec}), b"", b"", 12))
    for n in range(0, 10):
        prog = asm("LDI r0, 1\nOUT r0\nHALT")
        cases.append((f"short{n}", prog[:n], b"", b"", 6))
    cases.append(("in", asm("IN r0\nIN r1\nHALT"), b"", b"\x07", 8))
    cases.append(("long_add_loop", asm("LDI r0, 0xA5\nOUT r0\nJMP 0"), b"", b"", 300))
    bad = []
    for name, code, data, inputs, steps in cases:
        ref = PATHS[0]
        want, _wmsg = ref.run(ref.build(code, data=data or None, inputs=inputs),
                              limit=steps)
        for p in PATHS[1:]:
            got, _ = p.run(p.build(code, data=data or None, inputs=inputs), limit=steps)
            for field in ("status", "cause", "addr", "out", "ticks", "pc", "hl"):
                if got[field] != want[field]:
                    bad.append(f"{name}: {p.name}.{field} {got[field]!r}, "
                               f"reference {want[field]!r}")
            if got["code"] != want["code"]:
                first = next(i for i in range(len(want["code"]))
                             if got["code"][i] != want["code"][i])
                bad.append(f"{name}: {p.name}.code differs first at {first:#x}")
        if len(bad) > 6:
            break
    check("S8 default path agrees across the implementations", not bad,
          "; ".join(bad[:6]))
    print(f"    {len(cases)} default-path programs compared field by field")

    for p in PATHS:
        base = image(asm("EXT 0\nHALT"), length=0x0F22, vectors={0: 0x0F0C})
        v0, _ = p.run(p.build(base))
        moved = bytearray(base)
        moved[VEC:VEC + 2] = bytes([0x0E, 0x0F])
        v1, _ = p.run(p.build(bytes(moved)))
        check(f"S8 {p.name}: the CODE vector table is still what EXT reads",
              v0["status"] != v1["status"] or v0["pc"] != v1["pc"],
              "editing CODE[0x0F00] changed nothing, so the legacy read is no longer "
              "what EXT consults")

def main():
    s1_defaults()
    s2_validation()
    s3_case1()
    s4_case2()
    s5_case3()
    s6_unreachable()
    s7_capacity()
    s8_defaults_identical()
    print()
    if FAILS:
        print(f"CONFIG BLOCK ACCEPTANCE: {len(FAILS)} FAILURES")
        for f in FAILS:
            print("  !!", f)
        return 1
    print("config block: all checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())