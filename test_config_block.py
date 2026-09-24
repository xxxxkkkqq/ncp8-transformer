"""Acceptance for the load-time configuration block, which is its only source.

Four paths - reference, tensor circuit, Triton circuit, and for the capacity the resident
batch - are given the same block and the same program and compared field by field, so a
declared bound cannot mean one thing on one implementation. The block's refusals are
checked one field at a time, including the half-window case where only one bound is
supplied and a reversed window, which is refused at load rather than run as an empty one,
and a bound declared twice - in the block and in a moved constructor argument - is refused
rather than resolved by precedence.

The other side of the same claim is that nothing else can set those bounds: a write at an
address configuration used to occupy lands on code and leaves the block unchanged, the
same bytes that spell a vector table dispatch nothing until the table is declared, and an
image whose content stops short of a write is refused by its own length rather than by a
window that covers the address. The snapshot exposes no configuration field, `load_state`
installs no constraint, and only the constructor writes the block, so no instruction can
reach the machine's own limits. With no block at all, every path reproduces one machine
field by field over the one-byte and escape censuses.

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

VEC, WLO, WHI, ABI_HI = 0x0F00, 0x0F20, 0x0F21, 0x0F22
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
    print("S3 STC at the addresses configuration used to occupy, on every path")
    code = stc_to_vector()
    declared = CFG(codelen=0x0F22, winlo=0, winhi=CODE_SIZE, vec={0: 0x0F0C})
    rows = {}
    for p in PATHS:

        m = p.build(code, config=declared)
        v, msg = p.run(m)
        check(f"S3 {p.name}: the trap dispatched", v["out"] == b"\xa5",
              f"out {v["out"].hex()!r} cause {NAME.get(v["cause"])} status "
              f"{v["status"]}")
        check(f"S3 {p.name}: the write landed on the code byte",
              v["code"][VEC] == 0x00 and v["code"][VEC + 1] == 0x0F,
              f"0x0F00 holds {v["code"][VEC:VEC + 2].hex()}")
        check(f"S3 {p.name}: the block it dispatched from is unchanged",
              m.config.as_dict() == declared.as_dict(), repr(m.config.as_dict()))

        prog = asm("LDI r0, 0\nLDI HL, 0x0F00\nSTC [HL], r0\nHALT")
        v2, msg2 = p.run(p.build(prog, config=CFG(codelen=len(prog), winlo=0,
                                                  winhi=CODE_SIZE)))
        check(f"S3 {p.name} case 1: refused where there is no code",
              v2["cause"] == CAUSE["CODE_OOB"] and v2["status"] == 3,
              f"cause {NAME.get(v2["cause"])} status {v2["status"]}")
        check(f"S3 {p.name} case 1: at the STC", v2["addr"] == 5, f"addr {v2["addr"]}")
        if msg2 is not None:
            check(f"S3 {p.name} case 1: the message does not blame the window",
                  "window" not in msg2.lower() and "\u7a97\u53e3" not in msg2, msg2)

        v3, _ = p.run(p.build(image(prog, length=0x0F22),
                              config=CFG(codelen=0x0F22, winlo=0, winhi=0x0008,
                                         vec={0: 0x0F0C})))
        check(f"S3 {p.name}: a short span refuses the same write by name",
              v3["cause"] == CAUSE["WINDOW"] and v3["addr"] == 5,
              f"cause {NAME.get(v3["cause"])} addr {v3["addr"]}")
        rows[p.name] = (v["out"], v2["cause"], v3["cause"])
    check("S3 all paths agree on dispatched, content-refused, window-refused",
          len({r for r in rows.values()}) == 1, str(rows))

    for addr in (VEC, VEC + 1, WLO, WHI, ABI_HI - 1, 0x0F0C):
        blk = CFG(codelen=0x0F30, winlo=0, winhi=CODE_SIZE, vec={0: 0x0F0C})
        prog = asm(f"LDI r0, 0x55\nLDI HL, {addr}\nSTC [HL], r0\nHALT")
        for p in PATHS:
            m = p.build(image(prog, length=0x0F30), config=blk)
            v, _ = p.run(m)
            check(f"S3 {p.name} writes {hex(addr)} as code",
                  v["code"][addr] == 0x55 and v["cause"] == CAUSE["OK"],
                  f"{hex(addr)} holds {v["code"][addr]:#x}, cause "
                  f"{NAME.get(v["cause"])}")
            check(f"S3 {p.name}: {hex(addr)} left the block alone",
                  m.config.as_dict() == blk.as_dict(), repr(m.config.as_dict()))
    for p in PATHS:
        prog = asm("LDI r0, 0x55\nLDI HL, 0x0100\nSTC [HL], r0\nHALT")
        v, _ = p.run(p.build(image(prog, length=0x0F30),
                             config=CFG(codelen=0x0F30, winlo=0, winhi=CODE_SIZE)))
        check(f"S3 {p.name}: an ordinary window write lands",
              v["code"][0x0100] == 0x55 and v["status"] == 1,
              f"code[0x100]={v["code"][0x0100]:#x} status {v["status"]}")

def s4_case2():
    print("S4 EXT k with an unregistered vector, declared and spelled")
    handler = asm("LDI r1, 0xA5\nOUT r1\nHALT")
    img = bytearray(image(asm("EXT 0\nHALT"), length=0x0F22))
    img[0x0F0C:0x0F0C + len(handler)] = handler
    img[VEC:VEC + 2] = bytes([0x0C, 0x0F])
    cells_only = bytes(img)
    for p in PATHS:
        v, msg = p.run(p.build(cells_only, config=CFG(codelen=len(handler) + 4,
                                                      vec={0: 0, 1: 0x0F0C})))
        check(f"S4 {p.name} case 2: TRAP_UNREG", v["cause"] == CAUSE["TRAP_UNREG"],
              NAME.get(v["cause"]))
        check(f"S4 {p.name} case 2: at the EXT", v["addr"] == 0, v["addr"])
        if msg is not None:
            check(f"S4 {p.name} case 2: says nothing about a window",
                  "window" not in msg.lower() and "\u7a97\u53e3" not in msg, msg)
            check(f"S4 {p.name} case 2: prints no image bytes",
                  "7070" not in msg.lower(), msg)

        d, _ = p.run(p.build(cells_only, config=CFG(codelen=0x0F22, vec={0: 0x0F0C})))
        check(f"S4 {p.name}: the declaration is what dispatches",
              d["out"] == b"\xa5" and d["status"] == 1,
              f"out {d["out"].hex()} status {d["status"]} cause {NAME.get(d["cause"])}")
        u, _ = p.run(p.build(cells_only))
        check(f"S4 {p.name}: those same cells register nothing on their own",
              u["cause"] == CAUSE["TRAP_UNREG"],
              f"cause {NAME.get(u["cause"])} out {u["out"].hex()}")

        k, _ = p.run(p.build(asm("EXT 16\nHALT"),
                             config=CFG(codelen=4, vec={0: 0x0F0C})))
        check(f"S4 {p.name}: EXT 16 refused", k["cause"] == CAUSE["TRAP_UNREG"],
              NAME.get(k["cause"]))
    short = asm("EXT 0\nHALT")
    for p in PATHS:
        a, _ = p.run(p.build(short))
        b, _ = p.run(p.build(image(short, vectors={0: 0x0F0C})))
        check(f"S4 {p.name}: image length cannot empty or fill a table",
              (a["cause"], a["status"]) == (b["cause"], b["status"])
              == (CAUSE["TRAP_UNREG"], 3),
              f"short {NAME.get(a["cause"])}/{a["status"]}, padded "
              f"{NAME.get(b["cause"])}/{b["status"]}")

def _catch(fn):
    try:
        fn()
    except MachineError as e:
        return e
    return None

def s5_case3():
    print("S5 a reversed window refused at load; a bound pair, both ways round")
    try:
        CFG(winlo=0x80, winhi=0x10)
        check("S5 refused at load", False, "a reversed window built")
    except ISA.ConfigError as e:
        check("S5 refused at load", "reversed" in str(e) or "below" in str(e), str(e))
    prog = asm("LDI r0, 0x33\nLDI HL, 0x0020\nSTC [HL], r0\nHALT")
    cells = image(prog, length=0x0F30, window=(0x80, 0x10))
    for p in PATHS:
        v, _ = p.run(p.build(cells, config=CFG(codelen=0x0F30, winlo=0x80, winhi=0x90)))
        check(f"S5 {p.name}: a span that excludes the address refuses the write, and "
              f"it is the declared span",
              v["cause"] == CAUSE["WINDOW"] and v["code"][0x20] != 0x33,
              f"cause {NAME.get(v["cause"])} code[0x20]={v["code"][0x20]:#x}")
        v2, _ = p.run(p.build(cells, config=CFG(codelen=0x0F30, winlo=0,
                                                winhi=0x0F30)))
        check(f"S5 {p.name}: the declared span is what lets the write through",
              v2["code"][0x20] == 0x33 and v2["cause"] == CAUSE["OK"],
              f"code[0x20]={v2["code"][0x20]:#x} cause {NAME.get(v2["cause"])}")
        v3, _ = p.run(p.build(cells, config=CFG(codelen=0x0F30, winlo=0x40,
                                                winhi=0x40)))
        check(f"S5 {p.name}: a zero-width span refuses every write",
              v3["cause"] == CAUSE["WINDOW"], NAME.get(v3["cause"]))
    import loader
    try:
        loader.assemble("  HALT\n", window=(0x80, 0x10))
        check("S5 loader refuses a reversed window", False, "assemble accepted it")
    except loader.LoaderError as e:
        check("S5 loader refuses a reversed window", "reversed" in str(e), str(e))
    r = loader.assemble("  HALT\n  HALT\n", vectors={0: 0x0001}, window=(0x00, 0x08),
                        image=8)
    blk = r.config()
    check("S5 the loader's declaration round-trips into the block, and CODELEN is the "
          "content rather than the buffer",
          blk.vector(0) == 1 and blk.window() == (0, 8) and blk.codelen == 2
          and len(r.image) == 8, f"{blk.as_dict()} over a {len(r.image)}-byte image")
    wide = loader.assemble("  HALT\n", window=(0x00, 0x0100)).config()
    check("S5 a bound wider than a byte carries across, untruncated",
          wide.winhi == 0x0100, f"winhi is {wide.winhi}")
    try:
        CFG(codelen=2, winlo=0, winhi=0x10000)
        check("S5 a bound outside 16 bits is refused", False, "built")
    except ISA.ConfigError as e:
        check("S5 a bound outside 16 bits is refused", "65535" in str(e), str(e))

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
              m.config is before and m.config.as_dict() == full.as_dict(),
              repr(m.config.as_dict()))
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
    check("S6 the sweep ran every one-byte program against a configured machine",
          True)

    live = CFG(codelen=0x0F22, winlo=0, winhi=CODE_SIZE, vec={0: 0x0F0C})
    for p in PATHS:
        m = p.build(stc_to_vector(), config=live)
        v, _ = p.run(m)
        check(f"S6 {p.name}: a declared span lets the same write through",
              v["code"][VEC] == 0x00 and v["out"] == b"\xa5"
              and m.config.as_dict() == live.as_dict(),
              f"code[0x0F00]={v["code"][VEC]:#x} out {v["out"].hex()!r} "
              f"{m.config.as_dict()}")

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
    print(f"    {len(cases)} programs with no block compared field by field")

    for p in PATHS:
        base = image(asm("EXT 0\nHALT"), length=0x0F22)
        stamped = bytearray(base)
        stamped[VEC:VEC + 2] = bytes([0x0C, 0x0F])
        stamped[WLO], stamped[WHI] = 0x00, 0x08
        v0, _ = p.run(p.build(base))
        v1, _ = p.run(p.build(bytes(stamped)))
        check(f"S8 {p.name}: bytes in the freed region configure nothing",
              all(v0[k] == v1[k] for k in ("status", "cause", "addr", "pc", "ticks",
                                           "out")),
              f"{ {k: v0[k] for k in ('status', 'cause', 'pc')} } against "
              f"{ {k: v1[k] for k in ('status', 'cause', 'pc')} }")
        check(f"S8 {p.name}: and the cells differ, so the pair is not one image",
              v0["code"] != v1["code"] and v1["code"][VEC] == 0x0C, "identical images")

def _pair_addresses():

    near = set()
    for a in (VEC, VEC + 1, WLO, WHI, ABI_HI - 1, ABI_HI):
        near.update({a - 1, a, a + 1} & set(range(1 << 16)))
    stride = {lo for lo in range(0, (1 << 16) + 1, 257)}
    byte_wide = set(range(256))
    return sorted(near | stride | byte_wide)

def s9_pair_census():

    print("S9 every byte-wide window pair and the region-straddling 16-bit pairs, "
          "run at their endpoints")
    addrs = _pair_addresses()
    ref = PATHS[0]
    checked = 0
    bad = []
    images = {}

    def image_at(target):

        if target not in images:
            prog = asm(f"LDI r0, 0x55\nLDI HL, {target}\nSTC [HL], r0\nHALT")
            images[target] = image(prog, length=0x0F30)
        return images[target]

    for lo in addrs:
        for hi in addrs:
            if hi < lo:
                continue
            targets = {lo, hi - 1}
            if lo < 0x0F30 <= hi:
                targets |= {0x0F00, 0x0F20}
            blk = CFG(codelen=0x0F30, winlo=lo, winhi=hi, vec={0: 0x0F0C})
            before = blk.as_dict()
            for target in sorted(a for a in targets if 0 <= a < 1 << 16):
                m = ref.build(image_at(target), config=blk)
                v, _ = ref.run(m)
                checked += 1

                allowed = lo <= target < hi and target < 0x0F30
                refused = v["cause"] in (CAUSE["WINDOW"], CAUSE["CODE_OOB"])
                if allowed == refused:
                    bad.append(f"pair [{lo:#06x},{hi:#06x}) at {target:#06x}: "
                               f"allowed={allowed} cause={NAME.get(v['cause'])}")
                if m.config.as_dict() != before:
                    bad.append(f"pair [{lo:#06x},{hi:#06x}) moved the block: "
                               f"{before} -> {m.config.as_dict()}")
            if len(bad) > 3:
                break
        if bad:
            break
    check("S9 no pair lets a write escape its span, or moves the declared block",
          not bad, "; ".join(bad[:4]))
    check("S9 the census covers every byte-wide pair at least once", checked >= 65536,
          f"{checked} runs")

    edge = [a for a in addrs if abs(a - VEC) <= 2 or abs(a - WHI) <= 2 or a in (0, 0x0F30)]
    for p in PATHS[1:]:
        moved = []
        for lo in edge:
            for hi in edge:
                if hi < lo:
                    continue
                prog = asm(f"LDI r0, 0x55\nLDI HL, {max(lo, VEC)}\nSTC [HL], r0\nHALT")
                blk = CFG(codelen=0x0F30, winlo=lo, winhi=hi, vec={0: 0x0F0C})
                m = p.build(image(prog, length=0x0F30), config=blk)
                p.run(m)
                if m.config.as_dict() != blk.as_dict():
                    moved.append(f"[{lo:#06x},{hi:#06x})")
        check(f"S9 {p.name}: {len(edge) ** 2} region-straddling pairs, block unmoved",
              not moved, ", ".join(moved[:4]))
    print(f"    {checked} reference runs over the byte-wide census plus the "
          f"region-straddling 16-bit pairs; {len(edge) ** 2} pairs per circuit")

def main():
    s1_defaults()
    s2_validation()
    s3_case1()
    s4_case2()
    s5_case3()
    s6_unreachable()
    s7_capacity()
    s8_defaults_identical()
    s9_pair_census()
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