"""Fault-register acceptance: cause, address, and one cause per tick.

Six checks over the two fault registers the machine records when a tick stops it.
Every cause the table can produce is provoked and compared field by field against
the reference on all four paths (reference, tensor, Triton, batched); the cause and
status fields are paired in both directions, since a cause latched while the machine
still runs and a machine stopped without naming why are two different bugs; where two
sites can fire on one instruction the table's order decides the single reported cause;
a cause whose feature has not landed yet is exempted by a check re-derived from the
machine, so it fails the moment that feature arrives rather than quietly staying
exempt; widths are asserted on every tick, not only on the faulting one; and the
faulting tick both raises and records, after which the machine is a sticky terminal
state.

Run: python3 test_fault_registers.py
"""
from __future__ import annotations

import subprocess
import sys

import torch

import circuit_triton
import isa_table as ISA
from circuit_torch import TorchCircuit
from circuit_triton import TritonBatch, TritonCircuit
from golden_sim import (DATA_SIZE, MachineError, NCP8, STATUS_CODE, STATUS_RUNNING,
                        asm)
from test_state_contract import FAULT_WRITES, VIEW_FIELDS, assert_widths, ref_view

CODE_SIZE = 4096
OUT_CAP = 8192
VEC = 0x0F00
WLO, WHI = 0x0F20, 0x0F21
TICK0 = 5

PC_SITES = (0, 256)

CAUSE = ISA.CAUSE

def img(head, pc=0, *, vectors=None, window=None, tail_pad=0, length=None):

    n = max(4, pc + len(head) + 2 + tail_pad)
    if vectors or window:
        n = max(n, WHI + 2)
    if length is not None:
        if length < pc + len(head):
            raise ValueError("the instruction bytes would not fit the image it names")
        n = length
    b = bytearray(n)
    b[pc:pc + len(head)] = bytes(head)
    for k, addr in (vectors or {}).items():
        b[VEC + 2 * k:VEC + 2 * k + 2] = (addr & 0xFFFF).to_bytes(2, "little")
    if window:
        b[WLO], b[WHI] = window
    return bytes(b)

def state(**over):

    s = dict(r=[1, 2, 3, 4], HL=8, DE=8, PC=0, SP=2048, C=0, Z=0, ipos=0, oplen=0,
             tick=TICK0, status=0, fault_reason=0, fault_addr=0, MB=0, TDEPTH=0)
    s.update(over)
    return s

def _c_ok(pc):
    return img(asm("NOP"), pc), state(PC=pc), pc, b"", b""

def _c_bad_opcode(pc):
    return img(bytes([0x71, 0x00]), pc), state(PC=pc), pc, b"", b""

def _c_bad_subcode(pc):
    return img(bytes([0x70, 0x63, 0x00]), pc), state(PC=pc), pc, b"", b""

def _c_fetch_operand(pc):

    return (img(bytes([0x09, 0x00]), pc, length=pc + 2), state(PC=pc), pc, b"", b"")

def _c_fetch_prefix(pc):

    return img(bytes([0x70]), pc, length=pc + 1), state(PC=pc), pc, b"", b""

def _c_data_oob(pc):
    return (img(asm("MOV r0, [HL]"), pc), state(PC=pc, HL=DATA_SIZE), pc, b"", b"")

def _c_stack_underflow(pc):
    return img(asm("POP r0"), pc), state(PC=pc, SP=DATA_SIZE), pc, b"", b""

def _c_stack_underflow_ret(pc):
    return img(asm("RET"), pc), state(PC=pc, SP=DATA_SIZE - 1), pc, b"", b""

def _c_stack_overflow(pc):
    return img(asm("PUSH r0"), pc), state(PC=pc, SP=0), pc, b"", b""

def _c_stack_overflow_call(pc):
    return img(asm("CALL 0x1F00"), pc), state(PC=pc, SP=1), pc, b"", b""

def _c_div_zero(pc):
    return (img(asm("DIV r0, r1"), pc), state(PC=pc, r=[1, 0, 3, 4]), pc, b"", b"")

def _c_mod_zero(pc):
    return (img(asm("MOD r0, r1"), pc), state(PC=pc, r=[1, 0, 3, 4]), pc, b"", b"")

def _c_code_oob(pc):
    image = img(asm("LDC r0, [HL]"), pc)
    return image, state(PC=pc, HL=len(image)), pc, b"", b""

def _c_window(pc):

    image = img(asm("STC [HL], r0"), pc, window=(0x10, 0x18), tail_pad=0x18)
    return image, state(PC=pc, HL=0x04), pc, b"", b""

def _c_trap_unreg(pc):
    return img(asm("EXT 0"), pc), state(PC=pc), pc, b"", b""

def _c_trap_k_oob(pc):

    return img(bytes([0x70, 0x70, 16]), pc), state(PC=pc), pc, b"", b""

def _c_sp_pair_oob(pc):
    return img(asm("MOVW SP, HL"), pc), state(PC=pc, HL=DATA_SIZE + 1), pc, b"", b""

def _c_sp_add_oob(pc):
    return img(asm("ADD SP, 1"), pc), state(PC=pc, SP=DATA_SIZE), pc, b"", b""

def _c_ldx_wrap(pc):
    return (img(asm("LDX r0, [HL-1]"), pc), state(PC=pc, HL=0), pc, b"", b"")

def _c_stw_second_byte(pc):
    return (img(asm("STW [HL], DE"), pc), state(PC=pc, HL=DATA_SIZE - 1), pc, b"", b"")

def _c_out_cap(pc):
    return (img(asm("OUT r0"), pc), state(PC=pc, oplen=OUT_CAP), pc, b"", b"")

def _c_bank_oob(pc, mb=9, access="STM [HL], r0"):
    return (img(asm(access), pc), state(PC=pc, MB=mb), pc, b"", b"")

def _c_bank_oob_ldm(pc):
    return _c_bank_oob(pc, mb=255, access="LDM r0, [HL]")

def _c_bank_addr_oob(pc):

    return (img(asm("STM [HL], r0"), pc), state(PC=pc, HL=DATA_SIZE), pc, b"", b"")

def _c_bank_pair_addr_oob(pc):
    return (img(asm("STMW [HL], DE"), pc), state(PC=pc, HL=DATA_SIZE - 1), pc, b"", b"")

def _c_pc_illegal_jmp(pc):
    return img(asm("JMP 0x0F00"), pc), state(PC=pc), pc, b"", b""

def _c_pc_illegal_call(pc):
    return img(asm("CALL 0x0F00"), pc), state(PC=pc), pc, b"", b""

def _c_pc_illegal_ret(pc):

    return (img(asm("RET"), pc), state(PC=pc), pc,
            b"\x00" * 2048 + b"\x0f\x00", b"")

def _c_pc_illegal_jphl(pc):
    return img(asm("JPHL"), pc), state(PC=pc, HL=0x0F00), pc, b"", b""

def _c_pc_illegal_jz(pc):
    return img(asm("JZ 0x0F00"), pc), state(PC=pc, Z=1), pc, b"", b""

def _c_pc_illegal_djnz(pc):

    return (img(asm("DJNZ r0, 0x0F00"), pc), state(PC=pc, r=[2, 2, 3, 4]), pc,
            b"", b"")

def _c_pc_illegal_js(pc):

    return img(asm("JS 4"), pc), state(PC=pc, S=1), pc, b"", b""

def _c_pc_illegal_ext(pc):

    return (img(asm("EXT 0"), pc), state(PC=pc), pc, b"", b"",
            ISA.MachineConfig(vec={0: 0x0F00}))

def _c_pc_illegal_call_hl(pc):

    return img(asm("CALL HL"), pc), state(PC=pc, HL=0x0F00), pc, b"", b""

def _c_pc_illegal_trapret(pc):

    data = bytearray(DATA_SIZE)
    data[2048:2052] = b"\x0f\x00\x24\xa5"
    return img(asm("TRAPRET"), pc), state(PC=pc, TDEPTH=1), pc, bytes(data), b""

def _c_trap_depth(pc):

    return (img(asm("EXT 0"), pc), state(PC=pc, TDEPTH=1), pc, b"", b"",
            ISA.MachineConfig(vec={0: 0x0F10}, tdlim=1))

def _c_trap_unbalanced(pc):

    return img(asm("TRAPRET"), pc), state(PC=pc), pc, b"", b""

def _c_trap_frame(pc):

    data = bytearray(DATA_SIZE)
    data[2048:2052] = b"\x0f\x10\x00\x00"
    return img(asm("TRAPRET"), pc), state(PC=pc, TDEPTH=1), pc, bytes(data), b""

def _p_fetch_before_trap(pc):

    return (img(bytes([0x70, 0x70]), pc, length=pc + 2), state(PC=pc), pc, b"", b"")

def _p_fetch_before_stack(pc):

    return (img(bytes([0x0E, 0x00]), pc, length=pc + 2), state(PC=pc, SP=0), pc,
            b"", b"")

def _p_trap_before_stack(pc):

    return img(asm("EXT 0"), pc), state(PC=pc, SP=1), pc, b"", b""

def _p_code_oob_before_window(pc):

    image = img(asm("STC [HL], r0"), pc, window=(0x10, 0x18), tail_pad=0x18)
    return image, state(PC=pc, HL=len(image)), pc, b"", b""

def _p_data_oob_before_out_cap(pc):

    return (img(asm("OUTM"), pc), state(PC=pc, HL=DATA_SIZE, oplen=OUT_CAP), pc,
            b"", b"")

def _bank_selector_case(mb):

    def build(pc):
        return (img(asm("STM [HL], r0"), pc), state(PC=pc, MB=mb), pc, b"", b"")
    return build

def _p_bank_oob_before_data_oob(pc):

    return (img(asm("STM [HL], r0"), pc), state(PC=pc, MB=9, HL=DATA_SIZE), pc,
            b"", b"")

BANK_SELECTORS = (0, 1, 2, 63, 255, 4096, 65535)

FOREIGN_ADDR = 8

def _foreign_owner_running_names_busy():

    code = asm("  LDI HL, 1\n  MOV MB, HL\n  LDI HL, 8\n  STM [HL], r0\n  HALT")
    g = NCP8(code, data=bytes(DATA_SIZE), tick_budget=8,
             config=ISA.MachineConfig(nbanks=2))
    pages = (g.data, bytearray(DATA_SIZE))
    running = STATUS_CODE[STATUS_RUNNING]
    g.banks, g.bank_own, g.bank_owner_status = pages, 0, (running, running)
    for _ in range(6):
        try:
            g.step()
        except MachineError:
            break
    return (g.fault_reason == CAUSE["BANK_BUSY"] and g.status == "ERROR"
            and pages[1][FOREIGN_ADDR] == 0)

def _bank_busy_absent():

    for runner in (run_torch, run_triton, run_batch):
        for mb in BANK_SELECTORS:
            _raised, view, _out = runner(_bank_selector_case(mb), 0)
            if view["fault_reason"] == CAUSE["BANK_BUSY"]:
                return False
            want = CAUSE["OK"] if mb == 0 else CAUSE["BANK_OOB"]
            if view["fault_reason"] != want:
                return False
    if not hasattr(NCP8, "install_banks"):
        return False
    if any(hasattr(cls, "install_banks")
           for cls in (TorchCircuit, TritonCircuit, TritonBatch)):
        return False
    return _foreign_owner_running_names_busy()

def _p_trap_k_before_stack(pc):

    return img(bytes([0x70, 0x70, 16]), pc), state(PC=pc, SP=0), pc, b"", b""

def _p_stack_before_pc_target(pc):

    return img(asm("CALL 0x0F00"), pc), state(PC=pc, SP=1), pc, b"", b""

def _p_pop_before_pc_target(pc):

    return (img(asm("RET"), pc), state(PC=pc, SP=DATA_SIZE - 1), pc,
            b"\x00" * (DATA_SIZE - 1) + b"\x0f", b"")

def _p_trap_k_before_pc_target(pc):

    return (img(bytes([0x70, 0x70, 16]), pc), state(PC=pc), pc, b"", b"",
            ISA.MachineConfig(vec={15: 0x0F00}))

def _p_fetch_before_pc_target(pc):

    return (img(bytes([0x09, 0x40]), pc, length=pc + 2), state(PC=pc), pc, b"", b"")

def _p_trap_unreg_before_depth(pc):

    return (img(asm("EXT 0"), pc), state(PC=pc), pc, b"", b"",
            ISA.MachineConfig(tdlim=0))

def _p_depth_before_stack(pc):

    return (img(asm("EXT 0"), pc), state(PC=pc, SP=1), pc, b"", b"",
            ISA.MachineConfig(vec={0: 0x0F10}, tdlim=0))

def _p_depth_before_pc_target(pc):

    return (img(asm("EXT 0"), pc), state(PC=pc), pc, b"", b"",
            ISA.MachineConfig(vec={0: 0x0F00}, tdlim=0))

def _p_unbalanced_before_underflow(pc):

    return (img(asm("TRAPRET"), pc), state(PC=pc, SP=DATA_SIZE - 3), pc, b"", b"")

def _p_underflow_before_frame(pc):

    data = bytearray(DATA_SIZE)
    data[DATA_SIZE - 1] = 0xA5
    return (img(asm("TRAPRET"), pc), state(PC=pc, TDEPTH=1, SP=DATA_SIZE - 3), pc,
            bytes(data), b"")

def _p_frame_before_pc_target(pc):

    data = bytearray(DATA_SIZE)
    data[2048:2052] = b"\x0f\x00\x00\x00"
    return img(asm("TRAPRET"), pc), state(PC=pc, TDEPTH=1), pc, bytes(data), b""

CASES = (
    ("OK", "NOP commits a tick and names no cause", _c_ok),
    ("BAD_OPCODE", "unassigned single-byte opcode 0x71", _c_bad_opcode),
    ("BAD_SUBCODE", "reserved escape subcode 0x70 0x63", _c_bad_subcode),
    ("FETCH_OOB", "JMP with its operand byte past the image end", _c_fetch_operand),
    ("FETCH_OOB", "escape prefix as the last byte of the image", _c_fetch_prefix),
    ("DATA_OOB", "MOV r0, [HL] with HL == DATA_SIZE", _c_data_oob),
    ("DATA_OOB", "STW whose second byte leaves DATA", _c_stw_second_byte),
    ("DATA_OOB", "LDX wrapping below DATA through the frame offset", _c_ldx_wrap),
    ("DATA_OOB", "MOVW SP, HL to a value outside DATA", _c_sp_pair_oob),
    ("DATA_OOB", "ADD SP, i8 to a value outside DATA", _c_sp_add_oob),
    ("STACK_UNDERFLOW", "POP with nothing stored", _c_stack_underflow),
    ("STACK_UNDERFLOW", "RET one slot short of a return address", _c_stack_underflow_ret),
    ("STACK_OVERFLOW", "PUSH at SP == 0", _c_stack_overflow),
    ("STACK_OVERFLOW", "CALL with one slot left", _c_stack_overflow_call),
    ("DIV_ZERO", "DIV r0, r1 with r1 == 0", _c_div_zero),
    ("DIV_ZERO", "MOD r0, r1 with r1 == 0", _c_mod_zero),
    ("CODE_OOB", "LDC at an address past the image", _c_code_oob),
    ("WINDOW", "STC inside the image but outside the declared window", _c_window),
    ("TRAP_UNREG", "EXT 0 with a zero vector entry", _c_trap_unreg),
    ("TRAP_UNREG", "EXT 16, past the 16 handler slots", _c_trap_k_oob),
    ("OUT_CAP", "OUT as byte number OUT_CAP+1", _c_out_cap),
    ("BANK_OOB", "STM [HL], r0 with the selector past this machine's page count",
     _c_bank_oob),
    ("BANK_OOB", "LDM r0, [HL] with the selector at the end of the register's span",
     _c_bank_oob_ldm),
    ("DATA_OOB", "STM [HL], r0 through the page this machine holds, past its last byte",
     _c_bank_addr_oob),
    ("DATA_OOB", "STMW [HL], DE whose second byte leaves the page it holds",
     _c_bank_pair_addr_oob),
    ("PC_ILLEGAL", "JMP to the first address past the image", _c_pc_illegal_jmp),
    ("PC_ILLEGAL", "CALL whose target is past the image", _c_pc_illegal_call),
    ("PC_ILLEGAL", "RET to a return address past the image", _c_pc_illegal_ret),
    ("PC_ILLEGAL", "JPHL with HL past the image", _c_pc_illegal_jphl),
    ("PC_ILLEGAL", "JZ taken past the image", _c_pc_illegal_jz),
    ("PC_ILLEGAL", "DJNZ taken past the image", _c_pc_illegal_djnz),
    ("PC_ILLEGAL", "JS taken past the image's end", _c_pc_illegal_js),
    ("PC_ILLEGAL", "EXT dispatching to a registered handler past the image",
     _c_pc_illegal_ext),
    ("PC_ILLEGAL", "CALL HL whose target register is past the image",
     _c_pc_illegal_call_hl),
    ("PC_ILLEGAL", "TRAPRET restoring a return address past the image",
     _c_pc_illegal_trapret),
    ("TRAP_DEPTH", "EXT 0 with the depth counter at TDLIM (one trap in flight)",
     _c_trap_depth),
    ("TRAP_UNBALANCED", "TRAPRET with no trap in flight", _c_trap_unbalanced),
    ("TRAP_FRAME", "TRAPRET over the stale bytes a CALL-style RET left",
     _c_trap_frame),
)

PAIRS = (

    ("unfetchable immediate + unregistered handler", _p_fetch_before_trap,
     "FETCH_OOB", "TRAP_UNREG"),
    ("unfetchable immediate + no room to push", _p_fetch_before_stack,
     "FETCH_OOB", "STACK_OVERFLOW"),
    ("handler past the vector table + no room to push", _p_trap_k_before_stack,
     "TRAP_UNREG", "STACK_OVERFLOW"),
    ("unregistered handler + no room to push", _p_trap_before_stack,
     "TRAP_UNREG", "STACK_OVERFLOW"),
    ("address past the image + address outside the window", _p_code_oob_before_window,
     "CODE_OOB", "WINDOW"),
    ("DATA address out of range + output stream at capacity", _p_data_oob_before_out_cap,
     "DATA_OOB", "OUT_CAP"),
    ("selector past the page count + address outside the page", _p_bank_oob_before_data_oob,
     "BANK_OOB", "DATA_OOB"),
    ("no room to push + target past the image", _p_stack_before_pc_target,
     "STACK_OVERFLOW", "PC_ILLEGAL"),
    ("nothing stored to pop + target past the image", _p_pop_before_pc_target,
     "STACK_UNDERFLOW", "PC_ILLEGAL"),
    ("handler past the vector table + target past the image",
     _p_trap_k_before_pc_target, "TRAP_UNREG", "PC_ILLEGAL"),
    ("unfetchable operand + target past the image", _p_fetch_before_pc_target,
     "FETCH_OOB", "PC_ILLEGAL"),
    ("no vector registered + depth counter full", _p_trap_unreg_before_depth,
     "TRAP_UNREG", "TRAP_DEPTH"),
    ("depth counter full + no room to push the frame", _p_depth_before_stack,
     "TRAP_DEPTH", "STACK_OVERFLOW"),
    ("depth counter full + handler past the image", _p_depth_before_pc_target,
     "TRAP_DEPTH", "PC_ILLEGAL"),
    ("no trap in flight + nothing stored to pop", _p_unbalanced_before_underflow,
     "TRAP_UNBALANCED", "STACK_UNDERFLOW"),
    ("nothing stored to pop + a plausible stale tag", _p_underflow_before_frame,
     "STACK_UNDERFLOW", "TRAP_FRAME"),
    ("stale tag + target past the image", _p_frame_before_pc_target,
     "TRAP_FRAME", "PC_ILLEGAL"),
)

UNPROVOKABLE_PAIRS = (
    ("FETCH_OOB (code byte) before BAD_SUBCODE",
     "the padded code byte always decodes, so only one of the two can fire"),
)

ABSENT_PROOFS = {
    "BANK_BUSY": ("a circuit path holds one page, its own, so its bank access has no "
                  "foreign owner to wait for; the reference names the cause as soon as a "
                  "group driver hands it a second page",
                  lambda: _bank_busy_absent()),
}

def _install(machine, st, code, data, inputs, budget, cfg=None):

    m = machine(code, data=data, inputs=inputs, tick_budget=budget, config=cfg)
    m.load_state(st["r"], st["HL"], st["DE"], st["SP"], st["C"], st["Z"], st["tick"],
                 PC=st["PC"], S=st.get("S"), V=st.get("V"))
    _set_ipos(m, st["ipos"])
    _set_mb(m, st["MB"])
    _set_tdepth(m, st.get("TDEPTH", 0))
    return m

def _set_mb(m, n):

    if not n:
        return
    if isinstance(m, NCP8):
        m.MB = n
    elif isinstance(m, TorchCircuit):
        m.MB = torch.tensor([n], dtype=torch.int32, device=m.dev)
    else:
        m.S[int(circuit_triton.S_MB)] = n

def _set_ipos(m, n):

    if not n:
        return
    if isinstance(m, NCP8):
        m.ipos = n
    elif isinstance(m, TorchCircuit):
        m.ipos = torch.tensor([n], dtype=torch.int32, device=m.dev)
    else:
        m.S[10] = n

def _set_tdepth(m, n):

    if not n:
        return
    if isinstance(m, NCP8):
        m.TDEPTH = n
    elif isinstance(m, TorchCircuit):
        m.TDEPTH = torch.tensor([n], dtype=torch.int32, device=m.dev)
    else:
        m.S[int(circuit_triton.S_TDEPTH)] = n

def run_reference(case, pc):
    code, st, addr, data, inputs, *rest = case(pc)
    cfg = rest[0] if rest else None
    g = _install(NCP8, st, code, data, inputs, 200_000, cfg)
    if st["oplen"]:
        g.out = bytearray(st["oplen"])
    raised = False
    try:
        g.step()
    except MachineError:
        raised = True
    return raised, ref_view(g), bytes(g.out)

def run_torch(case, pc):
    return _run_circuit(TorchCircuit, case, pc)

def run_triton(case, pc):
    return _run_circuit(TritonCircuit, case, pc)

def _run_circuit(Machine, case, pc):
    code, st, addr, data, inputs, *rest = case(pc)
    cfg = rest[0] if rest else None
    c = _install(Machine, st, code, data, inputs, 200_000, cfg)
    if st["oplen"]:
        _set_oplen(c, st["oplen"])
    c.step()
    return c.snapshot()["status"] == 3, c.snapshot(), c.out()

def _set_oplen(c, n):
    if isinstance(c, TorchCircuit):
        c.oplen = torch.tensor([n], dtype=torch.int32, device=c.dev)
    else:
        c.S[11] = n

def run_batch(case, pc):

    code, st, addr, data, inputs, *rest = case(pc)
    cfg = rest[0] if rest else None
    b = TritonBatch(1, max_in=max(1, len(inputs)), config=cfg)
    b.set_program(0, code, data or b"", inputs or b"")
    b.set_state(0, r=st["r"], HL=st["HL"], DE=st["DE"], PC=st["PC"], SP=st["SP"],
                C=st["C"], Z=st["Z"], S=st.get("S", 0), V=st.get("V", 0),
                ipos=st["ipos"], oplen=st["oplen"],
                tick=st["tick"], MB=st["MB"], TDEPTH=st.get("TDEPTH", 0))
    b.step(1)
    snap = b.snapshot(0)
    return snap["status"] == 3, snap, b.out(0)

IMPLS = (("reference", run_reference), ("torch", run_torch), ("triton", run_triton),
         ("batch", run_batch))

def _cause_text(code):

    name = ISA.CAUSE_NAME.get(code)
    return f"{code} ({name})" if name else f"{code}, which no cause in the table assigns"

def _first_divergence(ref, got):

    for k in VIEW_FIELDS:
        if ref[k] != got[k]:
            return (k, ref[k], got[k])
    return None

def exercise(builders, label, counts, divergences):

    for cause, how, build in builders:
        for pc in PC_SITES:
            rows = {}
            for who, runner in IMPLS:
                raised, view, out = runner(build, pc)
                assert_widths(view, (cause, how, pc, who))
                rows[who] = (raised, view, out)
            r_raised, r_view, r_out = rows["reference"]
            if cause == "OK":
                assert not r_raised and r_view["status"] == 0, (how, pc, r_view)
                assert r_view["fault_reason"] == 0 and r_view["fault_addr"] == 0, (
                    how, pc, "a tick that did not fault recorded a cause", r_view)
            else:
                assert r_raised and r_view["status"] == 3, (how, pc, r_view)
                assert r_view["fault_reason"] == CAUSE[cause], (
                    how, pc, "reference named the wrong cause",
                    _cause_text(r_view["fault_reason"]), "expected", cause)
                assert r_view["fault_addr"] == build(pc)[2], (
                    how, pc, "fault_addr is not the instruction address at tick entry",
                    r_view["fault_addr"], build(pc)[2])
            counts[cause] = counts.get(cause, 0) + 1
            for who, (raised, view, out) in rows.items():
                if who == "reference":
                    continue
                d = _first_divergence(r_view, view)
                assert d is None, (how, pc, who, "differs from the reference", d)
                assert out == r_out, (how, pc, who, "output stream differs", out, r_out)
                if d is not None and cause not in divergences:
                    divergences[cause] = (who, pc, d)

def build_coverage_cases():

    return [(cause, how, build) for cause, how, build in CASES]

def test_c1_cause_table_is_covered_exhaustively():
    counts, divergences = {}, {}
    exercised = set()
    for cause, how, build in build_coverage_cases():
        if cause == "OK":
            continue
        exercised.add(cause)
    reachable = [name for name, _d in ISA.FAULT_CAUSES if name != "OK"
                 and name not in ISA.CAUSES_AWAITING_FEATURE]
    missing = sorted(set(reachable) - exercised)
    assert not missing, ("reachable causes with no case", missing)

    exercise(build_coverage_cases(), "C1", counts, divergences)
    for name, desc in ISA.FAULT_CAUSES:
        awaiting = name in ISA.CAUSES_AWAITING_FEATURE
        got = counts.get(name, 0)
        assert got > 0 or awaiting, (
            f"cause {ISA.CAUSE[name]} {name} ({desc}) was produced by no case and is "
            f"not on the waiting list")
        if awaiting:
            assert got == 0, (f"cause {name} is on the waiting list but a case produced "
                              f"it: the list is stale")
    _print_coverage(counts, divergences)
    return counts

def _print_coverage(counts, divergences):
    print(f"  cause coverage (each case run at {len(PC_SITES)} addresses, every run "
          f"compared on all four paths):")
    print(f"    {'code':>4}  {'name':18s} {'runs':>4}  {'checks':>6}  first divergence")
    for code, (name, desc) in enumerate(ISA.FAULT_CAUSES):
        got = counts.get(name, 0)
        if name in ISA.CAUSES_AWAITING_FEATURE:
            note = f"none yet: {ISA.CAUSES_AWAITING_FEATURE[name]}"
        elif got == 0:
            note = "NO CASE AND NOT EXEMPT"
        else:
            note = "-"
        div = divergences.get(name)
        if div:
            note = f"{div[0]} @ PC {div[1]}: field {div[2][0]} {div[2][2]} vs {div[2][1]}"
        print(f"    {code:4d}  {name:18s} {got:5d}  {got * len(PC_SITES) * 3:6d}  {note}")
    total = sum(counts.values())
    covered = sum(1 for n, _d in ISA.FAULT_CAUSES if counts.get(n))
    print(f"    {covered} of {len(ISA.FAULT_CAUSES)} causes produced by {total} runs "
          f"({total * 3} circuit/batch comparisons against the reference); "
          f"{len(ISA.FAULT_CAUSES) - covered} cannot be reached until their feature lands")

def test_c2_bijection_holds_on_every_tick_of_every_case():

    seen = {"cause-without-status": 0, "status-without-cause": 0, "paired": 0}
    for cause, how, build in build_coverage_cases() + [(p[0], p[0], p[1]) for p in PAIRS]:
        for pc in PC_SITES:
            for who, runner in IMPLS:
                raised, view, _out = runner(build, pc)
                bad = ISA.fault_state_error(view["status"], view["fault_reason"],
                                            view["fault_addr"])
                if bad is None:
                    seen["paired"] += 1
                    continue
                if "recorded while status is" in bad:
                    key = "cause-without-status"
                elif "while fault_reason is 0" in bad:
                    key = "status-without-cause"
                else:
                    key = None
                assert key is not None, (cause, pc, who, bad)
                seen[key] += 1
                raise AssertionError(f"{cause} @ PC {pc} on {who}: {bad}")
    assert seen["cause-without-status"] == 0 and seen["status-without-cause"] == 0, seen
    print(f"  C2 bijection: {seen['paired']} snapshots pair fault_reason with status "
          f"in both directions (cause without status 3: {seen['cause-without-status']}, "
          f"status 3 without cause: {seen['status-without-cause']})")
    return seen

def test_c3_at_most_one_cause_per_tick():

    builders = [(f"{win} before {lose}", build, win, lose)
                for what, build, win, lose in PAIRS]
    for what, build, win, lose in builders:
        for pc in PC_SITES:
            raised, view, _out = run_reference(build, pc)
            assert raised, (what, pc, "the double violation did not fault")
            assert view["fault_reason"] == CAUSE[win], (
                what, pc, "the tick named", ISA.fault_name(view["fault_reason"]),
                "but", win, "outranks", lose, "in isa_table.FAULT_SITE_ORDER")
            for who, runner in IMPLS:
                if who == "reference":
                    continue
                r2, v2, out2 = runner(build, pc)
                assert v2["fault_reason"] == CAUSE[win], (
                    who, what, pc, "named a different cause than the reference",
                    _cause_text(v2["fault_reason"]), "expected", win)
                assert v2["fault_addr"] == view["fault_addr"], (who, what, pc, v2, view)
                assert v2["status"] == 3, (who, what, pc, v2)

    assert len(set(ISA.FAULT_SITE_NAMES)) == len(ISA.FAULT_SITE_NAMES)
    assert ISA.SITE_RANK["TRAP_UNREG"] < ISA.SITE_RANK["STACK_PUSH"], (
        "the trap-vector cause must outrank the stack bound, as golden_sim checks them")
    assert ISA.SITE_RANK["BAD_SUBCODE"] < ISA.SITE_RANK["FETCH_OPERAND"], (
        "decode refusal must outrank the operand bound that follows it")
    assert ISA.SITE_RANK["PC_ILLEGAL"] < ISA.SITE_RANK["FETCH_CODE"], (
        "the branch-target write is judged on its own tick, before the fetch that "
        "would follow the committed PC")
    assert (ISA.SITE_RANK["TRAP_UNREG"] < ISA.SITE_RANK["TRAP_DEPTH"]
            < ISA.SITE_RANK["STACK_PUSH"]), (
        "the EXT chain tests the vector, then the depth counter, then the four-slot "
        "frame push room, in that order")
    assert (ISA.SITE_RANK["TRAP_UNBALANCED"] < ISA.SITE_RANK["STACK_POP"]
            < ISA.SITE_RANK["TRAP_FRAME"]), (
        "the TRAPRET chain tests the depth counter, then the four-slot pop room, "
        "then the frame tag, in that order")
    for what, why in UNPROVOKABLE_PAIRS:
        print(f"    not provokable on this machine: {what} -- {why}")
    print(f"  C3 precedence: {len(builders)} ordering pairs provoked at "
          f"{len(PC_SITES)} addresses each, one cause named per tick in all four paths")
    return len(builders)

def test_c4_absent_causes_are_absent_for_a_reason():

    for name, proof in ISA.CAUSES_AWAITING_FEATURE.items():
        assert name in CAUSE, (name, "not a cause the table assigns")
        what, check = ABSENT_PROOFS[name]
        assert check() is True, (
            f"cause {name} is exempted because {proof}, and {what} was assumed to hold, "
            f"but the machine says otherwise: it is reachable now and needs a case")
    assert set(ABSENT_PROOFS) == set(ISA.CAUSES_AWAITING_FEATURE), (
        set(ABSENT_PROOFS) ^ set(ISA.CAUSES_AWAITING_FEATURE))

    for cause, _how, _build in CASES:
        assert cause not in ISA.CAUSES_AWAITING_FEATURE, (
            cause, "is exercised by a case and should not be exempted")
    for name, (what, _check) in ABSENT_PROOFS.items():
        print(f"    {name}: no case, because {what}")
    print(f"  C4 waiting list: all {len(ABSENT_PROOFS)} exemptions re-derived from the "
          f"machine, each of which fails the moment the feature lands")
    return len(ISA.CAUSES_AWAITING_FEATURE)

def test_c5_the_new_fields_join_the_width_sweep():

    assert "fault_reason" in _width_names() and "fault_addr" in _width_names(), (
        "the width sweep does not carry the fault fields")
    for cause, how, build in build_coverage_cases():
        for pc in PC_SITES:
            for who, runner in IMPLS:
                _raised, view, _out = runner(build, pc)
                for k in ("fault_reason", "fault_addr"):
                    v = view[k]
                    assert isinstance(v, int) and not isinstance(v, bool), (
                        who, cause, pc, k, "is not an integer:", type(v).__name__)
                    assert 0 <= v < (256 if k == "fault_reason" else 1 << 16), (
                        who, cause, pc, k, v)

    c = TorchCircuit(bytes([0x01]))
    assert c.fault_reason.dtype == torch.int32 and c.fault_addr.dtype == torch.int32
    t = TritonCircuit(bytes([0x01]))
    assert t.S.dtype == torch.int32 and int(t.S.numel()) == circuit_triton.STATE_ROWS
    b = TritonBatch(2)
    assert b.STATE.dtype == torch.int32 and b.STATE.shape[1] == circuit_triton.STATE_ROWS
    g = NCP8(bytes([0x71, 0x00]))
    try:
        g.step()
    except MachineError:
        pass
    assert isinstance(g.fault_reason, int) and isinstance(g.fault_addr, int)
    print(f"  C5 widths: fault_reason/fault_addr checked on every tick, all integer "
          f"state on all four paths")

def _width_names():
    import test_state_contract as T
    return set(T.WIDTHS)

def test_c6_reference_records_the_error_status_it_raises_for():

    prog = bytes([0x01, 0x71, 0x00])
    g = NCP8(prog)
    g.step()
    try:
        g.step()
    except MachineError as e:
        assert g.status == "ERROR" and STATUS_CODE[g.status] == 3, g.status
        assert g.fault_reason == CAUSE["BAD_OPCODE"] and g.fault_addr == 1, g.snapshot()
        assert "0x71" in str(e), ("the raise must still name the code point", str(e))
    else:
        raise AssertionError("the undefined opcode tick stopped raising")

    g.step()
    g.step()
    assert g.snapshot()["tick"] == 1 and g.PC == 1, g.snapshot()
    print("  C6 the faulting tick raises *and* records status 3 with its cause; later "
          "steps are the sticky no-op")

def test_c7_pc_illegal_commits_nothing():

    cases = [(how, build) for cause, how, build in CASES if cause == "PC_ILLEGAL"]
    ticks = 0
    for how, build in cases:
        for pc in PC_SITES:
            code, st, addr, data, inputs, *rest = build(pc)
            cfg = rest[0] if rest else None
            want_data = bytes(data).ljust(DATA_SIZE, b"\x00")
            for who, runner in (("reference", _c7_reference), ("torch", _c7_torch),
                                ("triton", _c7_triton), ("batch", _c7_batch)):
                raised, view, out, data_got = runner(build, pc)
                assert raised, (how, pc, who, "the jump tick did not fault")
                assert view["fault_reason"] == CAUSE["PC_ILLEGAL"], (
                    how, pc, who, "named", _cause_text(view["fault_reason"]))
                assert view["fault_addr"] == pc, (how, pc, who, view)
                assert view["tick"] == st["tick"], (
                    how, pc, who, "the faulting tick did not keep its number", view)
                assert view["PC"] == st["PC"], (
                    how, pc, who, "PC did not keep its pre-tick value", view)
                for k in ("r", "HL", "DE", "SP", "MB"):
                    assert view[k] == st[k], (how, pc, who, k, "moved", view[k])
                assert view["ipos"] == st["ipos"], (how, pc, who, "ipos moved", view)
                assert out == b"", (how, pc, who, "the error tick emitted", out)
                assert data_got == want_data, (
                    how, pc, who, "the error tick wrote DATA (a return address?)")
                ticks += 1
    print(f"  C7 branch-target bound: {len(cases)} PC-writing forms x {len(PC_SITES)} "
          f"addresses on all four paths ({ticks} runs): the jump's own tick stops "
          f"with PC_ILLEGAL, fault_addr names the jump, and nothing commits")
    return ticks

def _c7_reference(build, pc):
    code, st, addr, data, inputs, *rest = build(pc)
    cfg = rest[0] if rest else None
    g = _install(NCP8, st, code, data, inputs, 200_000, cfg)
    raised = False
    try:
        g.step()
    except MachineError:
        raised = True
    return raised, ref_view(g), bytes(g.out), bytes(g.data)

def _c7_torch(build, pc):
    code, st, addr, data, inputs, *rest = build(pc)
    cfg = rest[0] if rest else None
    c = _install(TorchCircuit, st, code, data, inputs, 200_000, cfg)
    c.step()
    return (c.snapshot()["status"] == 3, c.snapshot(), c.out(),
            bytes(c.DATA.cpu().tolist()))

def _c7_triton(build, pc):
    code, st, addr, data, inputs, *rest = build(pc)
    cfg = rest[0] if rest else None
    c = _install(TritonCircuit, st, code, data, inputs, 200_000, cfg)
    c.step()
    return (c.snapshot()["status"] == 3, c.snapshot(), c.out(),
            bytes(c.DATA.cpu().tolist()))

def _c7_batch(build, pc):
    code, st, addr, data, inputs, *rest = build(pc)
    cfg = rest[0] if rest else None
    b = TritonBatch(1, max_in=max(1, len(inputs)), config=cfg)
    b.set_program(0, code, data or b"", inputs or b"")
    b.set_state(0, r=st["r"], HL=st["HL"], DE=st["DE"], PC=st["PC"], SP=st["SP"],
                C=st["C"], Z=st["Z"], S=st.get("S", 0), V=st.get("V", 0),
                ipos=st["ipos"], oplen=st["oplen"], tick=st["tick"], MB=st["MB"],
                TDEPTH=st.get("TDEPTH", 0))
    b.step(1)
    snap = b.snapshot(0)
    return snap["status"] == 3, snap, b.out(0), bytes(b.DATA[0].cpu().tolist())

SP0 = 2048
FRAME_FLAGS_SLOT = SP0 - 2

def _drive_all(build, budget=64):

    code, st, addr, data, inputs, *rest = build(0)
    cfg = rest[0] if rest else None
    got = {}
    for who, runner in _DRIVERS.items():
        got[who] = runner(code, st, bytes(data), inputs, cfg, budget)
    head = got["reference"]
    for who, (view, out, datab) in got.items():
        for k in VIEW_FIELDS:
            if k == "tick" or k == "tick":
                continue

        for k in head[0]:
            assert view[k] == head[0][k], (who, k, view[k], head[0][k])
        assert out == head[1], (who, "output stream", out, head[1])
        assert datab == head[2], (who, "DATA image diverged")
    return head

def _driver_reference(code, st, data, inputs, cfg, budget):
    m = NCP8(code, data=data or None, inputs=inputs, tick_budget=budget, config=cfg)
    m.load_state(st["r"], st["HL"], st["DE"], st["SP"], st["C"], st["Z"], st["tick"],
                 PC=st["PC"], S=st.get("S"), V=st.get("V"))
    _set_tdepth(m, st.get("TDEPTH", 0))
    try:
        while m.status == "RUNNING":
            m.step()
    except MachineError:
        pass
    return (ref_view(m), bytes(m.out), bytes(m.data))

def _driver_torch(code, st, data, inputs, cfg, budget):
    c = TorchCircuit(code, data=data or None, inputs=inputs, tick_budget=budget,
                     config=cfg)
    c.load_state(st["r"], st["HL"], st["DE"], st["SP"], st["C"], st["Z"], st["tick"],
                 PC=st["PC"], S=st.get("S"), V=st.get("V"))
    _set_tdepth(c, st.get("TDEPTH", 0))
    while int(c.status.item()) == 0:
        c.step()
    snap = c.snapshot()
    return (snap, c.out(), bytes(int(v) for v in c.DATA.cpu().tolist()))

def _driver_triton(code, st, data, inputs, cfg, budget):
    c = TritonCircuit(code, data=data or None, inputs=inputs, tick_budget=budget,
                      config=cfg)
    c.load_state(st["r"], st["HL"], st["DE"], st["SP"], st["C"], st["Z"], st["tick"],
                 PC=st["PC"], S=st.get("S"), V=st.get("V"))
    _set_tdepth(c, st.get("TDEPTH", 0))
    while int(c.status.item()) == 0:
        c.step()
    snap = c.snapshot()
    return (snap, c.out(), bytes(int(v) for v in c.DATA.cpu().tolist()))

def _driver_batch(code, st, data, inputs, cfg, budget):
    b = TritonBatch(1, max_in=max(1, len(inputs)), config=cfg, tick_budget=budget)
    b.set_program(0, code, data or b"", inputs or b"")
    b.set_state(0, r=st["r"], HL=st["HL"], DE=st["DE"], PC=st["PC"], SP=st["SP"],
                C=st["C"], Z=st["Z"], S=st.get("S", 0), V=st.get("V", 0),
                ipos=st["ipos"], oplen=st["oplen"], tick=st["tick"], MB=st["MB"],
                TDEPTH=st.get("TDEPTH", 0))
    b.run()
    snap = b.snapshot(0)
    return (snap, b.out(0), bytes(int(v) for v in b.DATA[0].cpu().tolist()))

_DRIVERS = {"reference": _driver_reference, "torch": _driver_torch,
            "triton": _driver_triton, "batch": _driver_batch}

def test_c8_trap_protocol_end_to_end():

    prog = asm("LDI r2, 0x80\nADD r2, r2\nEXT 0\nGETF r1\nOUT r1\nHALT")
    handler = asm(f"LDI HL, {FRAME_FLAGS_SLOT}\nLDI r0, 0\nMOV [HL], r0\nTRAPRET")
    for fb in range(16):
        h = handler.replace(asm("LDI r0, 0"), bytes([0xD0 | 0, fb]))
        code = bytearray(prog.ljust(0x10, b"\x00")) + h
        cfg = ISA.MachineConfig(vec={0: 0x10})

        def build(_pc, code=code, cfg=cfg):
            return img(bytes(code), 0), state(), 0, b"", b"", cfg

        view, out, _data = _drive_all(build)
        assert view["status"] == 1 and out == bytes([fb]), (fb, view, out)
        assert view["SP"] == SP0 and view["TDEPTH"] == 0, (fb, view)
        assert (view["Z"], view["C"], view["S"], view["V"]) == (
            fb & 1, (fb >> 1) & 1, (fb >> 2) & 1, (fb >> 3) & 1), (fb, view)
    print(f"  C8a frame round trip: 16 flag bytes stored by the handler, restored "
          f"exactly, on all four paths")

    prog = asm("EXT 0\nLDI r1, 0x77\nOUT r1\nHALT")
    code = bytearray(prog.ljust(0x10, b"\x00")) + asm("EXT 0")
    cfg = ISA.MachineConfig(vec={0: 0x10}, tdlim=2)

    def build(_pc, code=code, cfg=cfg):
        return img(bytes(code), 0), state(), 0, b"", b"", cfg

    view, out, data = _drive_all(build)
    assert view["status"] == 3 and view["fault_reason"] == CAUSE["TRAP_DEPTH"], view
    assert view["fault_addr"] == 0x10 and view["tick"] == TICK0 + 2, view
    assert view["SP"] == SP0 - 8 and view["TDEPTH"] == 2, view
    assert view["r"] == [1, 2, 3, 4] and out == b"", view
    assert data[SP0 - 12:SP0 - 8] == bytes(4), "the refused tick pushed"
    assert data[SP0 - 8:SP0] != bytes(8), "the two frames are gone"
    print("  C8b depth runaway: the third EXT of a self-re-entering handler names "
          "TRAP_DEPTH, two frames intact, nothing pushed")

    prog = asm("LDI HL, 0x0010\nCALL HL\nHALT")
    code = bytearray(prog.ljust(0x10, b"\x00")) + asm("HALT")

    def build(_pc, code=code):
        return img(bytes(code), 0), state(), 0, b"", b""

    view, out, data = _drive_all(build)
    assert view["status"] == 1 and view["SP"] == SP0 - 2, view
    assert data[SP0 - 2:SP0] == bytes([0x00, 0x05]), \
        "the return address is not high-byte-on-top at SP"
    code = bytearray(asm("LDI HL, 0x0010\nCALL HL\nLDI r1, 0x11\nOUT r1\nHALT")
                     .ljust(0x10, b"\x00")) + asm("LDI r0, 0x22\nOUT r0\nRET")

    def build2(_pc, code=code):
        return img(bytes(code), 0), state(), 0, b"", b""

    view, out, _data = _drive_all(build2)
    assert view["status"] == 1 and out == bytes([0x22, 0x11]), (view, out)
    assert view["SP"] == SP0 and view["TDEPTH"] == 0, view
    print("  C8c CALL HL: pushes high-on-top, jumps through HL, RET round-trips")

    prog = asm("LDI HL, 0x0FE0\nMOVW SP, HL\nEXT 0\nTRAPRET\nHALT")
    code = bytearray(prog.ljust(0x10, b"\x00")) + asm("LDI r1, 9\nRET")
    cfg = ISA.MachineConfig(vec={0: 0x10})

    def build(_pc, code=code, cfg=cfg):
        return img(bytes(code), 0), state(), 0, b"", b"", cfg

    view, out, _data = _drive_all(build)
    assert view["status"] == 3 and view["fault_reason"] == CAUSE["TRAP_FRAME"], view
    assert view["fault_addr"] == 8 and view["TDEPTH"] == 1, view
    print("  C8d the 7.3 leak: a handler left by RET is caught by the enclosing "
          "TRAPRET as TRAP_FRAME")

CHECKS = (
    test_c1_cause_table_is_covered_exhaustively,
    test_c2_bijection_holds_on_every_tick_of_every_case,
    test_c3_at_most_one_cause_per_tick,
    test_c4_absent_causes_are_absent_for_a_reason,
    test_c5_the_new_fields_join_the_width_sweep,
    test_c6_reference_records_the_error_status_it_raises_for,
    test_c7_pc_illegal_commits_nothing,
    test_c8_trap_protocol_end_to_end,
)

def run_all():
    print("fault-register acceptance (cause table x boundary sweeps, four paths):")
    for fn in CHECKS:
        fn()
    print(f"fault-register acceptance: all {len(CHECKS)} checks passed")

SABOTAGES = (

    ("a cause recorded without status becoming 3",
     "circuit_torch.py",
     "        new_status = torch.where(err > 0, 3 * torch.ones(1, dtype=i32, device=dev),",
     "        new_status = torch.where(err > 0, 0 * torch.ones(1, dtype=i32, device=dev),",
     "probe_pairing"),
    ("status 3 with the cause left at its reset value",
     "golden_sim.py",
     "            self.fault_reason = reason",
     "            self.fault_reason = CAUSE[\"OK\"]",
     "probe_pairing"),
    ("fault_addr taken after the PC advance",
     "golden_sim.py",
     "            self.fault_addr = pc0",
     "            self.fault_addr = (pc0 + 1) & 0xFFFF",
     "probe_addr"),
    ("two causes permitted to fire in one tick (their codes are summed)",
     "circuit_torch.py",
     "    first = fired * (prior == 0).to(torch.int32)",
     "    first = fired * 0 + fired",
     "probe_precedence"),

    ("one cause code produced by no case at all",
     "test_fault_registers.py",
     '    ("OUT_CAP", "OUT as byte number OUT_CAP+1", _c_out_cap),\n',
     "",
     "probe_coverage"),
)

PROBES = {
    "probe_pairing": "test_c2_bijection_holds_on_every_tick_of_every_case",
    "probe_addr": "test_c1_cause_table_is_covered_exhaustively",
    "probe_precedence": "test_c3_at_most_one_cause_per_tick",
    "probe_coverage": "test_c1_cause_table_is_covered_exhaustively",
}

_PROBE_SRC = """
import sys
sys.path.insert(0, {root!r})
import test_fault_registers as F
try:
    F.{fn}()
except AssertionError as e:
    print("SABOTAGE-CAUGHT:", e)
else:
    print("SABOTAGE-MISSED: the check passed")
"""

def sabotage(root=None):

    import os
    import shutil
    import tempfile
    root = os.path.dirname(os.path.abspath(__file__)) if root is None else root
    for label, fname, old, new, probe in SABOTAGES:
        work = tempfile.mkdtemp(prefix="fault-sabotage-")
        try:
            for keep in sorted(n for n in os.listdir(root) if n.endswith(".py")):
                shutil.copy(os.path.join(root, keep), work)
            src = os.path.join(work, fname)
            text = open(src, encoding="utf-8").read()
            if text.count(old) != 1:
                print(f"[{label}] the sabotage anchor is not unique in {fname}")
                continue
            open(src, "w", encoding="utf-8").write(text.replace(old, new))
            out = subprocess.run(
                [sys.executable, "-c", _PROBE_SRC.format(root=work, fn=PROBES[probe])],
                capture_output=True, text=True, cwd=work)
            printed = (out.stdout or out.stderr).strip().splitlines()
            print(f"\n[{label}]  ({fname}, {probe} -> {PROBES[probe]})")
            for line in printed[-6:]:
                print("   ", line[:400])
        finally:
            shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    if "--sabotage" in sys.argv:
        sabotage()
    else:
        run_all()