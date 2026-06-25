"""Binary patching engine for cubin register re-coloring.

Applies the re-coloring plan to the actual cubin binary:
1. Rename registers in existing instructions (change register field bytes)
2. Insert spill stores (STS) after register definitions
3. Insert reload trampolines (BRA → LDS + renamed_instr + BRA_back) 
4. Update ELF metadata (register count, shared memory size)

The trampoline approach handles the constraint that we can't insert new
instructions into a fixed-size code section. Instead, we:
- Replace the original instruction with a BRA to a NOP slot at section end
- In the NOP slot: LDS reload + original instruction (renamed) + BRA back
- This uses 3-4 NOP slots per trampoline (48-64 bytes)

Shared memory layout for spill slots (strided, per-register):
  Each spilled register gets a 4-byte slot per thread.
  Slot i, thread tid: smem_base + i * SPILL_SLOT_STRIDE + tid * 4
  where SPILL_SLOT_STRIDE = MAX_THREADS_PER_BLOCK * 4.
  Rbase register holds tid * 4 (computed once in prologue).

Instruction encoding uses cubit's native Python bindings:
  cubit.encode(sass, addr) → (lo64, hi64)
This replaces all hand-coded make_* encoders with correct table-driven
encoding for ANY SM 120 instruction.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from typing import Optional

from sasskit.core.cubin import Cubin, KernelInfo
from sasskit.core.decoder import Instruction, RegRef, rename_register
from sasskit.recolor.coloring import ColoringResult, SpillDecision

# Import cubit encoder
try:
    import cubit as _cubit
except ImportError:
    _cubit = None

# ISA database for scheduling-aware stall counts
try:
    from sasskit.core.isa_db import get_db as _get_isa_db
except ImportError:
    _get_isa_db = None


def _isa_def_latency(base_op: str, fallback: int = 4) -> int:
    """Get def_latency for an opcode from the ISA database.

    Returns *fallback* if the database is unavailable or the opcode
    is unknown.
    """
    if _get_isa_db is None:
        return fallback
    try:
        return _get_isa_db().def_latency(base_op)
    except Exception:
        return fallback


# ============================================================================
# Spill system constants
# ============================================================================

MAX_THREADS_PER_BLOCK = 1024
SPILL_SLOT_BYTES = 4       # 32-bit per register
SPILL_SLOT_STRIDE = MAX_THREADS_PER_BLOCK * SPILL_SLOT_BYTES  # 4096


def _compute_spill_stride(kernel: KernelInfo) -> int:
    """Compute the spill slot stride that fits within existing shared memory.

    SM120 CUDA driver does not honor runtime expansion of .nv.shared
    sections — LDS/STS beyond the compile-time size faults.  Spill slots
    must fit within the ORIGINAL shared memory allocation.

    Strategy: reuse the kernel's __shared__ region for spill slots.
    After BAR.SYNC, the kernel's shared data is consumed and the
    region (from nvcc UMOV offset to section end) is available.

    Returns stride in bytes (max_threads * 4, capped to fit).
    """
    # Available = total section size. Spill base = total_smem_base
    # (where kernel user smem ends). If that's == shared_size, no room
    # beyond user smem — put spills INSIDE user smem at a safe offset.
    if kernel.shared_size > 0:
        # Use stride = 1024*4 = 4096 if it fits, otherwise reduce
        available = kernel.shared_size
        stride = min(SPILL_SLOT_STRIDE, available // 4)  # fit at least 4 slots
        return max(stride, 256 * SPILL_SLOT_BYTES)  # min 256 threads
    return SPILL_SLOT_STRIDE


@dataclass
class PatchAction:
    """A single binary modification to apply."""
    code_offset: int      # Offset within kernel text
    description: str      # Human-readable description
    old_bytes: bytes      # Original 16 bytes (for verification/rollback)
    new_instr: int        # New instruction word
    new_ctrl: int         # New control word


@dataclass 
class PatchPlan:
    """Complete plan for all binary modifications."""
    kernel_name: str
    actions: list[PatchAction]
    new_reg_count: int
    new_stack_size: int
    new_smem_size: int
    spill_trampolines: list[TrampolineInfo]
    
    @property
    def summary(self) -> str:
        lines = [
            f"Patch plan for {self.kernel_name}:",
            f"  Register renames: {sum(1 for a in self.actions if 'rename' in a.description)}",
            f"  Trampolines: {len(self.spill_trampolines)}",
            f"  New register count: {self.new_reg_count}",
            f"  New stack size: {self.new_stack_size}",
            f"  New smem size: {self.new_smem_size}",
        ]
        return '\n'.join(lines)


@dataclass
class TrampolineInfo:
    """Info about a spill reload trampoline."""
    original_offset: int     # Where the original instruction was
    trampoline_offset: int   # Where the trampoline starts (NOP slot)
    spill: SpillDecision
    reload_reg: int          # Register used for reload


# ============================================================================
# sm_120 instruction encoders — cubit-backed
# ============================================================================
#
# All encoders call cubit.encode(sass, addr) which returns (lo64, hi64).
# cubit uses table-driven bitfield encoding with 100% roundtrip accuracy
# across all SM 120 instructions.
#
# For instructions where cubit is unavailable (import failed), we keep
# inline fallback constants verified against real RTX 5090 hardware.
# ============================================================================

def _cubit_encode(sass: str, addr: int = 0) -> tuple[int, int]:
    """Encode a SASS instruction via cubit.

    Args:
        sass: Full SASS syntax with semicolon, e.g. "IADD3 R5, PT, PT, R9, R4, R5 ;"
        addr: Instruction address (needed for BRA target computation).

    Returns:
        (instr_word, ctrl_word) — lower and upper 64-bit halves.
    """
    if _cubit is None:
        raise RuntimeError(
            "cubit Python module not available. Install cubit: "
            "pip install -e /path/to/cubit  (or set CUBIT_TABLE and ensure 'import cubit' works)")
    lo, hi = _cubit.encode(sass, addr=addr)
    return lo, hi


def make_nop() -> tuple[int, int]:
    """Encode a NOP instruction."""
    try:
        return _cubit_encode("NOP ;")
    except (RuntimeError, ValueError):
        return 0x0000000000007918, 0x000fc00000000000


def make_nop_stall(stall: int = 15) -> tuple[int, int]:
    """Encode a NOP instruction with a specific stall count.

    cubit encodes the base NOP; we patch the stall bits at ctrl[45:41].
    """
    try:
        lo, hi = _cubit_encode("NOP ;")
        # Clear existing stall and set new one
        hi = (hi & ~(0x1F << 41)) | ((stall & 0x1F) << 41)
        return lo, hi
    except (RuntimeError, ValueError):
        return 0x0000000000007918, 0x000fc00000000000 | ((stall & 0x1F) << 41)


def make_bra(target_offset: int, from_offset: int) -> tuple[int, int]:
    """Encode an unconditional BRA instruction for SM 120.

    cubit handles the non-linear split encoding for BRA targets.
    The addr parameter tells cubit the instruction's own address so it
    can compute the correct PC-relative offset.
    """
    try:
        return _cubit_encode(f"BRA 0x{target_offset:x} ;", addr=from_offset)
    except (RuntimeError, ValueError):
        # Fallback: manual encoding (verified on RTX 5090)
        rel = target_offset - from_offset - 16
        relq = rel // 4
        lo8 = relq & 0xFF
        hi32 = (relq >> 6) & 0xFFFFFFFF
        hi32 = hi32 & 0xFFFFFFFC
        opcode = 0x7947
        if target_offset < from_offset:
            ctrl = 0x000fc0000383ffff
        else:
            ctrl = 0x000fea0003800000
        instr = opcode | (lo8 << 16) | (hi32 << 32)
        return instr, ctrl


def make_s2r_tid(dest_reg: int, stall: int = 0) -> tuple[int, int]:
    """Encode S2R Rd, SR_TID.X."""
    try:
        lo, hi = _cubit_encode(f"S2R R{dest_reg}, SR_TID.X ;")
        if stall:
            hi = (hi & ~(0x1F << 41)) | ((stall & 0x1F) << 41)
        return lo, hi
    except (RuntimeError, ValueError):
        instr = 0x0000000000007919 | ((dest_reg & 0xFF) << 16)
        ctrl = 0x000e6e0000002100 | (stall & 0xF)
        return instr, ctrl


def make_lds_64(dest_reg: int, base_reg: int, offset: int) -> tuple[int, int]:
    """Encode LDS.64 Rd, [Rbase + offset]."""
    try:
        return _cubit_encode(f"LDS.64 R{dest_reg}, [R{base_reg}+0x{offset:x}] ;")
    except (RuntimeError, ValueError):
        instr = (0x7984
                 | ((dest_reg & 0xFF) << 16)
                 | ((base_reg & 0xFF) << 24)
                 | ((offset & 0xFFFF) << 40))
        ctrl = 0x000e220000000a00
        return instr, ctrl


def make_sts_64(base_reg: int, data_reg: int, offset: int) -> tuple[int, int]:
    """Encode STS.64 [Rbase + offset], Rdata:Rdata+1.

    Direct bitfield encoding (cubit STS has offset encoding bug).
    """
    # STS.64 uses the same opcode 0x7388 with .64 modifier in ctrl bits
    instr = (0x7388
             | ((base_reg & 0xFF) << 24)
             | ((data_reg & 0xFF) << 32)
             | ((offset & 0xFFFFFF) << 40))
    ctrl = 0x000fe20000000a00
    return instr, ctrl


def _find_free_wbar_slot(cubin: Cubin, kernel: KernelInfo) -> int:
    """Scan kernel instructions and return a free write barrier slot (0-5).

    SM120 has 6 write barrier slots (0-5).  Kernel instructions claim
    slots via the wbar field in ctrl[48:46].  Trampoline LDS must use
    a slot NOT used by any kernel instruction to avoid scoreboard
    collision (which causes illegal-address crashes on STG).

    Returns the highest free slot, or 5 as fallback.
    """
    used: set[int] = set()
    for i in range(kernel.num_instructions):
        off = i * 16
        _, hi = cubin.read_instruction(kernel, off)
        wbar = (hi >> 46) & 7
        if wbar < 6:
            used.add(wbar)
    # Pick highest free slot (farthest from commonly-used 0,1,2)
    for slot in range(5, -1, -1):
        if slot not in used:
            return slot
    return 5  # fallback — least likely to collide


def make_lds(dest_reg: int, base_reg: int, offset: int,
             wbar: int = 5) -> tuple[int, int]:
    """Encode LDS (32-bit) Rd, [Rbase + offset].

    Uses stall-only scheduling (NO write barrier).  All 6 write barrier
    slots (0-5) are typically occupied by kernel-original instructions;
    reusing any of them causes scoreboard collisions and illegal-memory
    errors.  Instead, we use a conservative stall count of 15 and rely
    on the 2×NOP-stall=15 padding in the caller to wait for the load.

    On SM120 (Blackwell), shared-memory LDS latency is ≤ 30 cycles.
    A stall=15 on the LDS + 2 × NOP stall=15 provides 45 cycles total,
    which is more than sufficient.
    """
    try:
        lo, hi = _cubit_encode(f"LDS R{dest_reg}, [R{base_reg}+0x{offset:x}] ;")
    except (RuntimeError, ValueError):
        lo = (0x7984
              | ((dest_reg & 0xFF) << 16)
              | ((base_reg & 0xFF) << 24)
              | ((offset & 0xFFFF) << 40))
        hi = 0x000e220000000800
    # Use stall=15, NO write barrier (wbar=0 means "no barrier").
    # This avoids colliding with all 6 kernel-original wbar slots.
    SCHED_MASK = 0x1FFFF << 41
    hi = hi & ~SCHED_MASK
    # sched = stall=15, no yield, no wbar, no read/write deps
    sched = 15  # stall=15, all other bits=0 (no wbar, no dep)
    hi = hi | (sched << 41)
    return lo, hi


def make_sts(base_reg: int, data_reg: int, offset: int) -> tuple[int, int]:
    """Encode STS (32-bit) [Rbase + offset], Rdata (shared memory)."""
    instr = (0x7388
             | ((base_reg & 0xFF) << 24)
             | ((data_reg & 0xFF) << 32)
             | ((offset & 0xFFFFFF) << 40))
    ctrl = 0x000fe20000000800
    return instr, ctrl


def make_stl(base_reg: int, data_reg: int, offset: int) -> tuple[int, int]:
    """Encode STL (32-bit) [Rbase + offset], Rdata (local/stack memory).

    STL is identical to STS except the opcode is 0x7387 instead of 0x7388.
    Local memory is per-thread private DRAM (via register spilling stack).
    No shared memory allocation needed; base register is R1 (stack pointer).
    """
    instr = (0x7387
             | ((base_reg & 0xFF) << 24)
             | ((data_reg & 0xFF) << 32)
             | ((offset & 0xFFFFFF) << 40))
    ctrl = 0x000fe20000000800
    return instr, ctrl


def make_ldl(dest_reg: int, base_reg: int, offset: int) -> tuple[int, int]:
    """Encode LDL (32-bit) Rd, [Rbase + offset] (local/stack memory).

    LDL is identical to LDS except the opcode is 0x7983 instead of 0x7984.
    Uses conservative stall=15 (no write barrier needed).
    """
    try:
        lo, hi = _cubit_encode(f"LDL R{dest_reg}, [R{base_reg}+0x{offset:x}] ;")
    except (RuntimeError, ValueError):
        lo = (0x7983
              | ((dest_reg & 0xFF) << 16)
              | ((base_reg & 0xFF) << 24)
              | ((offset & 0xFFFF) << 40))
        hi = 0x000e220000000800
    # Use stall=15, no write barrier
    SCHED_MASK = 0x1FFFF << 41
    hi = hi & ~SCHED_MASK
    hi = hi | (15 << 41)  # stall=15
    return lo, hi


def make_mov(dest_reg: int, src_reg: int) -> tuple[int, int]:
    """Encode MOV Rd, Rs (register-to-register copy)."""
    try:
        return _cubit_encode(f"MOV R{dest_reg}, R{src_reg} ;")
    except (RuntimeError, ValueError):
        instr = (0x7202
                 | ((dest_reg & 0xFF) << 16)
                 | ((src_reg & 0xFF) << 32))
        ctrl = 0x000fe20000000f00
        return instr, ctrl


def make_iadd_carry(dest: int, src1: int, src2: int,
                    pred_out: int = 0) -> tuple[int, int]:
    """Encode IADD Rd, Ppred_out, Ra, Rb — add with carry output."""
    try:
        p_name = f'P{pred_out}' if pred_out < 7 else 'PT'
        return _cubit_encode(f"IADD3 R{dest}, {p_name}, PT, R{src1}, R{src2}, RZ ;")
    except (RuntimeError, ValueError):
        instr = (0x7235
                 | ((dest & 0xFF) << 16)
                 | ((src1 & 0xFF) << 24)
                 | ((src2 & 0xFF) << 32))
        ctrl = 0x000fe20000000000 | 0x07800000 | ((pred_out & 7) << 17)
        return instr, ctrl


def make_iadd_x(dest: int, src1: int, src2: int,
                pred_out: int = 0, pred_in: int = 0) -> tuple[int, int]:
    """Encode IADD.X Rd, Ppred_out, Ra, Rb, Ppred_in — add with carry in/out."""
    try:
        po = f'P{pred_out}' if pred_out < 7 else 'PT'
        pi = f'P{pred_in}' if pred_in < 7 else 'PT'
        return _cubit_encode(f"IADD3.X R{dest}, {po}, PT, R{src1}, R{src2}, RZ, {pi}, PT ;")
    except (RuntimeError, ValueError):
        instr = (0x7235
                 | ((dest & 0xFF) << 16)
                 | ((src1 & 0xFF) << 24)
                 | ((src2 & 0xFF) << 32))
        ctrl = 0x000fe20000000000 | 0x0400 | ((pred_out & 7) << 17) | ((pred_in & 7) << 23)
        return instr, ctrl


def make_shf_l_u32(dest: int, src: int, shift: int) -> tuple[int, int]:
    """Encode SHF.L.U32 Rd, Ra, shift, RZ → Rd = Ra << shift."""
    try:
        return _cubit_encode(f"SHF.L.U32 R{dest}, R{src}, 0x{shift:x}, RZ ;")
    except (RuntimeError, ValueError):
        instr = (0x7819
                 | ((dest & 0xFF) << 16)
                 | ((src & 0xFF) << 24)
                 | ((shift & 0xFF) << 32))
        ctrl = 0x001fd000000006ff
        return instr, ctrl


# ============================================================================
# CGA-aware shared memory addressing (sm_120 Blackwell)
# ============================================================================

# Temporary uniform registers for CGA computation (dead after LEA).
# These MUST NOT collide with UR registers used by the kernel.
# nvcc kernels commonly use UR4/UR5 for address descriptors and
# constant loads — we use UR6/UR7 to avoid collisions.
_CGA_UR_CTAID = 7   # S2UR destination: CGA CTA id
_CGA_UR_BASE  = 6   # UMOV/ULEA destination: CGA base address


def make_s2ur_cgactaid(dest_ur: int) -> tuple[int, int]:
    """Encode S2UR URd, SR_CgaCtaId (read CGA CTA id into uniform register)."""
    try:
        return _cubit_encode(f"S2UR UR{dest_ur}, SR_CgaCtaId ;")
    except (RuntimeError, ValueError):
        instr = 0x79c3 | ((dest_ur & 0xFF) << 16)
        ctrl = 0x000e620000008800
        return instr, ctrl


def make_umov_ur_imm(dest_ur: int, imm: int) -> tuple[int, int]:
    """Encode UMOV URd, <imm32> (load immediate into uniform register)."""
    try:
        return _cubit_encode(f"UMOV UR{dest_ur}, 0x{imm:x} ;")
    except (RuntimeError, ValueError):
        instr = 0x7882 | ((dest_ur & 0xFF) << 16) | ((imm & 0xFFFFFFFF) << 32)
        ctrl = 0x000fce0000000000
        return instr, ctrl


def make_ulea(dest_ur: int, src_ur_a: int, src_ur_b: int, shift: int) -> tuple[int, int]:
    """Encode ULEA URd, URa, URb, shift — compute CGA base address."""
    try:
        return _cubit_encode(
            f"ULEA UR{dest_ur}, UR{src_ur_a}, UR{src_ur_b}, 0x{shift:x} ;")
    except (RuntimeError, ValueError):
        instr = (0x7291
                 | ((dest_ur & 0xFF) << 16)
                 | ((src_ur_a & 0xFF) << 24)
                 | ((src_ur_b & 0xFF) << 32))
        ctrl_lo32 = 0x0f8e00ff | ((shift & 0x1F) << 11)
        ctrl = 0x000fe20000000000 | ctrl_lo32
        return instr, ctrl


def make_lea_r_ur(dest_r: int, src_r: int, src_ur: int, shift: int) -> tuple[int, int]:
    """Encode LEA Rd, Ra, URb, shift — compute per-thread smem address."""
    try:
        return _cubit_encode(
            f"LEA R{dest_r}, R{src_r}, UR{src_ur}, 0x{shift:x} ;")
    except (RuntimeError, ValueError):
        instr = (0x7c11
                 | ((dest_r & 0xFF) << 16)
                 | ((src_r & 0xFF) << 24)
                 | ((src_ur & 0xFF) << 32))
        ctrl_lo32 = 0x0f8e00ff | ((shift & 0x1F) << 11)
        ctrl = 0x000fe20000000000 | ctrl_lo32
        return instr, ctrl


# ============================================================================
# Spill trampoline helpers
# ============================================================================

@dataclass
class SpillSlotInfo:
    """Shared memory slot info for a single spilled register."""
    reg: int             # Register number
    slot_index: int      # 0-based slot index
    smem_offset: int     # Absolute immediate for LDS/STS [Rbase + smem_offset]


def _compute_spill_layout(
    spills: list[SpillDecision],
    existing_smem: int,
    stride: int = SPILL_SLOT_STRIDE,
) -> dict[int, SpillSlotInfo]:
    """Compute shared memory layout for spill slots.

    Each spilled register gets its own strided slot.
    Slots are placed at existing_smem + i * stride.

    Returns {reg_num -> SpillSlotInfo}.
    """
    mapping: dict[int, SpillSlotInfo] = {}
    for i, sp in enumerate(sorted(spills, key=lambda s: s.reg)):
        mapping[sp.reg] = SpillSlotInfo(
            reg=sp.reg,
            slot_index=i,
            smem_offset=existing_smem + i * stride,
        )
    return mapping


def _rename_reg_in_word(word: int, bit_pos: int, old_reg: int, new_reg: int) -> int:
    """Rename a register field in an instruction/control word integer."""
    current = (word >> bit_pos) & 0xFF
    if current != old_reg:
        return word
    word = word & ~(0xFF << bit_pos)
    word = word | ((new_reg & 0xFF) << bit_pos)
    return word


def _find_dead_reg(
    live_colors: set[int],
    exclude: set[int],
    max_color: int,
) -> Optional[int]:
    """Find a dead physical register (any alignment)."""
    for c in range(2, max_color):
        if c not in live_colors and c not in exclude:
            return c
    return None


def _find_dead_consecutive_pair(
    live_colors: set[int],
    exclude: set[int],
    max_color: int,
) -> Optional[int]:
    """Find a dead consecutive register pair R, R+1 with even-aligned base."""
    for c in range(2, max_color - 1, 2):
        if (c not in live_colors and c not in exclude
                and c + 1 not in live_colors and c + 1 not in exclude):
            return c
    return None


def _live_colors_at(
    offset: int,
    live_at: dict[int, set[int]],
    coloring: dict[int, int],
    pair_partners: Optional[dict[int, int]] = None,
) -> set[int]:
    """Compute the set of physical register colors that are live at offset.

    When *pair_partners* is provided, any live register that has a pair
    partner (e.g. R2:R3 for .64 loads) will also mark the partner's
    physical color as live.  This prevents trampolines from accidentally
    clobbering the implicit high half of a 64-bit pair.
    """
    live = live_at.get(offset, set())
    colors: set[int] = set()
    for r in live:
        if r in coloring:
            colors.add(coloring[r])
        # Also block pair partner's color
        if pair_partners and r in pair_partners:
            partner = pair_partners[r]
            if partner in coloring:
                colors.add(coloring[partner])
    return colors


# ============================================================================
# Patching operations
# ============================================================================

def apply_register_renames(cubin: Cubin, kernel: KernelInfo,
                           coloring: dict[int, int],
                           instructions: list[Instruction]) -> list[PatchAction]:
    """Apply register renames based on the coloring result.

    Renames are applied ALL-AT-ONCE per instruction to avoid corruption
    from permutation cycles in the coloring.
    """
    actions: list[PatchAction] = []

    for inst in instructions:
        renames_iw: list[tuple[int, int, int]] = []
        renames_cw: list[tuple[int, int, int]] = []

        for ref in inst.reg_refs:
            if ref.reg_num in coloring:
                new_reg = coloring[ref.reg_num]
                if new_reg == ref.reg_num:
                    continue
                if ref.bit_position < 0:
                    continue
                target = renames_cw if ref.in_ctrl_word else renames_iw
                target.append((ref.bit_position, ref.reg_num, new_reg))

        if not renames_iw and not renames_cw:
            continue

        old_instr, old_ctrl = cubin.read_instruction(kernel, inst.code_offset)
        old_bytes = struct.pack('<QQ', old_instr, old_ctrl)

        new_instr = old_instr
        new_ctrl = old_ctrl

        for bp, old_reg, new_reg in renames_iw:
            if ((new_instr >> bp) & 0xFF) == old_reg:
                new_instr = (new_instr & ~(0xFF << bp)) | ((new_reg & 0xFF) << bp)

        for bp, old_reg, new_reg in renames_cw:
            if ((new_ctrl >> bp) & 0xFF) == old_reg:
                new_ctrl = (new_ctrl & ~(0xFF << bp)) | ((new_reg & 0xFF) << bp)

        if new_instr != old_instr or new_ctrl != old_ctrl:
            cubin.write_instruction(kernel, inst.code_offset, new_instr, new_ctrl)
            actions.append(PatchAction(
                code_offset=inst.code_offset,
                description=f"rename {len(renames_iw)+len(renames_cw)} field(s)",
                old_bytes=old_bytes,
                new_instr=new_instr,
                new_ctrl=new_ctrl,
            ))

    return actions


def create_spill_prologue_local(
    cubin: Cubin,
    kernel: KernelInfo,
    nop_offsets: list[int],
    nop_idx: int,
) -> tuple[list[PatchAction], int]:
    """Minimal prologue for LOCAL MEMORY spills.

    For local-memory spills, R1 (stack pointer) is set up by the kernel's
    own first instruction (LDC R1, c[...]). We only need to:
    1. BRA to the prologue area
    2. Execute the original first instruction (to set R1)
    3. BRA back to the second instruction
    """
    PROLOGUE_SLOTS = 3
    actions: list[PatchAction] = []

    if nop_idx + PROLOGUE_SLOTS > len(nop_offsets):
        raise RuntimeError(
            f"Not enough NOP slots for local-mem spill prologue "
            f"(need {PROLOGUE_SLOTS}, have {len(nop_offsets) - nop_idx})")

    first_off = 0
    orig_instr, orig_ctrl = cubin.read_instruction(kernel, first_off)
    orig_bytes = struct.pack('<QQ', orig_instr, orig_ctrl)

    tramp_start = nop_offsets[nop_idx]

    # 0. Replace first instruction with BRA to prologue
    bra_instr, bra_ctrl = make_bra(tramp_start, first_off)
    cubin.write_instruction(kernel, first_off, bra_instr, bra_ctrl)
    actions.append(PatchAction(
        code_offset=first_off,
        description="BRA to local-mem spill prologue",
        old_bytes=orig_bytes,
        new_instr=bra_instr,
        new_ctrl=bra_ctrl,
    ))

    # [0] Original first instruction (sets R1 = stack pointer)
    cubin.write_instruction(kernel, nop_offsets[nop_idx], orig_instr, orig_ctrl)
    nop_idx += 1

    # [1] NOP stall=15 (wait for R1 to be available)
    nop_s15_i, nop_s15_c = make_nop_stall(15)
    cubin.write_instruction(kernel, nop_offsets[nop_idx], nop_s15_i, nop_s15_c)
    nop_idx += 1

    # [2] BRA back to second instruction
    return_off = first_off + 16
    bra_back, bra_back_ctrl = make_bra(return_off, nop_offsets[nop_idx])
    cubin.write_instruction(kernel, nop_offsets[nop_idx], bra_back, bra_back_ctrl)
    nop_idx += 1

    return actions, nop_idx


def _patch_stack_size(cubin: Cubin, kernel: KernelInfo, new_stack: int) -> None:
    """Patch EIATTR_MAX_STACK_SIZE in ALL relevant ELF sections.

    Patches:
    1. Per-kernel .nv.info.<name>: attr=0x1c (EIATTR_MAX_STACK_SIZE, fmt=0x04)
    2. Global .nv.info: attr=0x11 (EIATTR_STACK?, {kernel_symidx, stack_bytes})
    """
    # 1. Per-kernel .nv.info
    cubin.set_max_stack(kernel, new_stack)

    # 2. Global .nv.info: find all (attr=0x11, size=8, {ki, val}) entries
    #    where ki == kernel's symbol index, and patch the value.
    data = cubin.data
    e_shoff = struct.unpack_from('<Q', data, 0x28)[0]
    e_shnum = struct.unpack_from('<H', data, 0x3C)[0]
    e_shentsize = struct.unpack_from('<H', data, 0x3A)[0]
    e_shstrndx = struct.unpack_from('<H', data, 0x3E)[0]
    shstr_base = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from('<Q', data, shstr_base + 24)[0]

    kernel_symidx = _find_kernel_symidx(cubin, kernel.name)

    for si in range(e_shnum):
        sh_base = e_shoff + si * e_shentsize
        sh_name_idx = struct.unpack_from('<I', data, sh_base)[0]
        name_start = shstr_off + sh_name_idx
        name_end = data.index(0, name_start)
        sec_name = bytes(data[name_start:name_end]).decode('ascii', 'replace')

        if sec_name not in ('.nv.info', '.nv.merc.nv.info'):
            continue

        sec_off = struct.unpack_from('<Q', data, sh_base + 24)[0]
        sec_size = struct.unpack_from('<Q', data, sh_base + 32)[0]

        j = 0
        while j < sec_size:
            off = sec_off + j
            if off + 4 > len(data):
                break
            tag = data[off]
            attr = data[off + 1]
            if tag == 0x04:
                payload_size = struct.unpack_from('<H', data, off + 2)[0]
                if attr == 0x11 and payload_size == 8:
                    symidx = struct.unpack_from('<I', data, off + 4)[0]
                    old_val = struct.unpack_from('<I', data, off + 8)[0]
                    if symidx == kernel_symidx:
                        struct.pack_into('<I', data, off + 8, new_stack)
                        print(f"INFO: Patched stack in {sec_name} "
                              f"at 0x{off:x}: {old_val} → {new_stack}",
                              file=sys.stderr)
                j += 4 + payload_size
            else:
                j += 4


def create_spill_trampolines(
    cubin: Cubin,
    kernel: KernelInfo,
    spills: list[SpillDecision],
    instructions: list[Instruction],
    nop_offsets: list[int],
    nop_idx: int,
    smem_base_reg: int,
    coloring: dict[int, int],
    existing_smem: int,
    live_at: dict[int, set[int]],
    wbar_slot: int = 5,
    stride: int = SPILL_SLOT_STRIDE,
    use_local_mem: bool = False,
    effective_K: int = -1,
    orig_words: Optional[dict[int, tuple[int, int]]] = None,
) -> tuple[list[PatchAction], list[TrampolineInfo], int]:
    """Create spill trampolines using 32-bit LDS/STS (per-register).

    Uses 32-bit (4-byte) loads/stores — only needs ONE dead register
    per spill reference (any alignment).  No pair constraints.

    For each instruction that references a spilled register:
      1. Replace it with BRA to a trampoline in the NOP area
      2. At the trampoline:
         - LDS for each spilled source register
         - Renamed copy of the original instruction
         - STS for each spilled dest register
         - BRA back

    Returns: (actions, trampoline_infos, updated_nop_idx)
    """
    actions: list[PatchAction] = []
    tramp_infos: list[TrampolineInfo] = []

    # Select store/load helpers based on memory type
    if use_local_mem:
        _sts = make_stl
        def _lds(d, b, off, wbar=0): return make_ldl(d, b, off)
    else:
        _sts = make_sts
        _lds = make_lds

    inst_by_offset = {inst.code_offset: inst for inst in instructions}
    spill_set = {s.reg for s in spills}
    spill_layout = _compute_spill_layout(spills, existing_smem, stride=stride)

    # For local memory spills, smem_base_reg=1 (R1) but effective_K is the
    # actual coloring range (target_regs - 3).  Use the caller-provided
    # effective_K if given; otherwise fall back to smem_base_reg (old behavior).
    if effective_K < 0:
        effective_K = smem_base_reg
    exclude_colors: set[int] = {smem_base_reg, 0, 1}

    # Build pair partner map: reg -> partner for ALL .64/WIDE pair registers.
    # This prevents trampolines from using the implicit high half of a
    # 64-bit pair as a temp register (the liveness analysis doesn't track
    # implicit pair partners, so they appear dead but aren't).
    pair_partners: dict[int, int] = {}
    for inst in instructions:
        for ref in inst.reg_refs:
            if ref.is_pair:
                base = ref.reg_num
                high = base + 1
                pair_partners[base] = high
                pair_partners[high] = base

    # Group references by instruction offset
    spill_offsets: set[int] = set()
    for sp in spills:
        for off in sp.use_offsets + sp.def_offsets:
            if off in inst_by_offset:
                spill_offsets.add(off)

    for off in sorted(spill_offsets):
        inst = inst_by_offset[off]

        # Identify spilled registers and whether USE/DEF
        use_regs: list[tuple[int, RegRef]] = []
        def_regs: list[tuple[int, RegRef]] = []
        ref_map: dict[int, RegRef] = {}

        for ref in inst.reg_refs:
            if ref.reg_num in spill_set and ref.bit_position >= 0:
                ref_map[ref.reg_num] = ref
                if ref.is_dest:
                    def_regs.append((ref.reg_num, ref))
                else:
                    use_regs.append((ref.reg_num, ref))

        all_spill_regs = sorted(set(r for r, _ in use_regs + def_regs))
        if not all_spill_regs:
            continue

        # ---- Detect .64 pair relationships for spilled registers ----
        pair_info: dict[int, tuple[int, bool]] = {}
        for reg_num, ref in use_regs + def_regs:
            if ref.is_pair and reg_num not in pair_info:
                partner = reg_num + 1
                pair_info[reg_num] = (partner, partner in spill_set)

        # Count extra LDS/STS needed for spilled partners (Case B)
        n_extra_loads = 0
        n_extra_stores = 0
        for reg_num, (partner, partner_spilled) in pair_info.items():
            if partner_spilled and partner in spill_layout:
                if any(r == reg_num for r, _ in use_regs):
                    n_extra_loads += 1
                if any(r == reg_num for r, _ in def_regs):
                    n_extra_stores += 1

        n_loads = len(use_regs) + n_extra_loads
        n_lds_stall_nops = n_loads * 2
        n_stores = len(def_regs) + n_extra_stores
        n_mov_headroom = len(pair_info) * 7
        n_save_restore_headroom = len(all_spill_regs) * 6
        slots_needed = (n_loads + n_lds_stall_nops + n_mov_headroom
                        + n_save_restore_headroom + 1 + n_stores + 1)

        if nop_idx + slots_needed > len(nop_offsets):
            print(f"WARNING: Not enough NOP slots at 0x{off:04x} "
                  f"(need {slots_needed}, have {len(nop_offsets) - nop_idx})",
                  file=sys.stderr)
            break

        # ---- Allocate dead temp registers (pair-aware) ----
        lc = _live_colors_at(off, live_at, coloring, pair_partners)
        temp_for_reg: dict[int, int] = {}
        temp_used: set[int] = set()
        mov_fixups: dict[int, int] = {}
        ok = True

        for reg_num in all_spill_regs:
            if reg_num in pair_info:
                partner, partner_spilled = pair_info[reg_num]
                if not partner_spilled and partner in coloring:
                    required = coloring[partner] - 1
                    if (required >= 2 and required < effective_K
                            and required % 2 == 0
                            and required not in lc
                            and required not in temp_used
                            and required not in exclude_colors):
                        temp_for_reg[reg_num] = required
                        temp_used.add(required)
                    else:
                        cand = _find_dead_consecutive_pair(
                            lc | temp_used, exclude_colors, effective_K)
                        if cand is not None:
                            temp_for_reg[reg_num] = cand
                            temp_used.add(cand)
                            temp_used.add(cand + 1)
                            mov_fixups[reg_num] = coloring[partner]
                        elif (required >= 2 and required < effective_K
                              and required % 2 == 0):
                            temp_for_reg[reg_num] = required
                            temp_used.add(required)
                            temp_used.add(required + 1)
                            mov_fixups[reg_num] = -1
                        else:
                            cand3 = None
                            for c in range(2, effective_K - 1, 2):
                                if (c not in exclude_colors
                                        and c not in temp_used
                                        and c + 1 not in temp_used):
                                    cand3 = c
                                    break
                            if cand3 is not None:
                                temp_for_reg[reg_num] = cand3
                                temp_used.add(cand3)
                                temp_used.add(cand3 + 1)
                                mov_fixups[reg_num] = -3
                                print(f"INFO: .64 pair steal fallback at "
                                      f"0x{off:04x}: R{reg_num} → "
                                      f"R{cand3}:R{cand3+1} (save/restore)",
                                      file=sys.stderr)
                            else:
                                print(f"WARNING: .64 pair at 0x{off:04x}: "
                                      f"R{reg_num} — all fallbacks exhausted, "
                                      f"skipping", file=sys.stderr)
                                ok = False
                                break
                else:
                    cand = _find_dead_consecutive_pair(
                        lc | temp_used, exclude_colors, effective_K)
                    if cand is None:
                        for c in range(2, effective_K - 1, 2):
                            if (c not in exclude_colors
                                    and c not in temp_used
                                    and c + 1 not in temp_used):
                                cand = c
                                break
                        if cand is None:
                            print(f"WARNING: No consecutive pair at all for "
                                  f"R{reg_num}:R{partner} at 0x{off:04x} — "
                                  f"skipping", file=sys.stderr)
                            ok = False
                            break
                        mov_fixups[reg_num] = -3
                        print(f"INFO: Stealing live pair R{cand}:R{cand+1} "
                              f"for R{reg_num}:R{partner} at 0x{off:04x} "
                              f"(save/restore)", file=sys.stderr)
                    temp_for_reg[reg_num] = cand
                    temp_used.add(cand)
                    temp_used.add(cand + 1)
            else:
                cand = _find_dead_reg(lc | temp_used, exclude_colors, effective_K)
                if cand is None:
                    # No dead register — steal a live one with save/restore.
                    # Pick the lowest live color not in exclude or already used as temp.
                    for c in range(2, effective_K):
                        if c not in exclude_colors and c not in temp_used:
                            cand = c
                            break
                    if cand is None:
                        print(f"WARNING: No register at all for R{reg_num} "
                              f"at 0x{off:04x} — skipping", file=sys.stderr)
                        ok = False
                        break
                    # Mark for save/restore: sentinel -3 in mov_fixups
                    mov_fixups[reg_num] = -3
                    print(f"INFO: Stealing live R{cand} for R{reg_num} "
                          f"at 0x{off:04x} (save/restore)", file=sys.stderr)
                temp_for_reg[reg_num] = cand
                temp_used.add(cand)

        if not ok:
            continue

        # Build partner operation list (Case B)
        partner_ops: list[tuple[int, int, bool]] = []
        for reg_num, (partner, partner_spilled) in pair_info.items():
            if partner_spilled and partner in spill_layout:
                rtemp_partner = temp_for_reg[reg_num] + 1
                if any(r == reg_num for r, _ in use_regs):
                    partner_ops.append((partner, rtemp_partner, False))
                if any(r == reg_num for r, _ in def_regs):
                    partner_ops.append((partner, rtemp_partner, True))

        # ---- Build trampoline ----
        tramp_start = nop_offsets[nop_idx]

        # 0. Replace original with BRA.
        # Read from cubin (post-global-rename, pre-BRA-write).
        orig_instr, orig_ctrl = cubin.read_instruction(kernel, off)
        orig_bytes = struct.pack('<QQ', orig_instr, orig_ctrl)
        bra_instr, bra_ctrl = make_bra(tramp_start, off)
        cubin.write_instruction(kernel, off, bra_instr, bra_ctrl)
        actions.append(PatchAction(
            code_offset=off,
            description=f"BRA to spill trampoline ({len(all_spill_regs)} reg(s))",
            old_bytes=orig_bytes,
            new_instr=bra_instr,
            new_ctrl=bra_ctrl,
        ))

        nop_s15_i, nop_s15_c = make_nop_stall(15)
        scratch_smem = existing_smem + len(spill_layout) * stride

        # 0a. Save stolen live registers BEFORE any LDS (sentinel -3).
        #     When no dead register is available, we steal a live one
        #     and must preserve its value across the trampoline.
        #     For stolen pairs, save both halves.
        for reg_num, fixup_val in mov_fixups.items():
            if fixup_val == -3:
                rtemp = temp_for_reg[reg_num]
                sts_i, sts_c = _sts(smem_base_reg, rtemp, scratch_smem)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        sts_i, sts_c)
                nop_idx += 1
                scratch_smem += stride
                if reg_num in pair_info:
                    sts_i2, sts_c2 = _sts(smem_base_reg, rtemp + 1,
                                          scratch_smem)
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            sts_i2, sts_c2)
                    nop_idx += 1
                    scratch_smem += stride

        # 1. LDS for each USE register
        for reg_num, ref in use_regs:
            rtemp = temp_for_reg[reg_num]
            imm = spill_layout[reg_num].smem_offset
            lds_i, lds_c = _lds(rtemp, smem_base_reg, imm, wbar=wbar_slot)
            cubin.write_instruction(kernel, nop_offsets[nop_idx], lds_i, lds_c)
            nop_idx += 1
            for _ in range(2):
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        nop_s15_i, nop_s15_c)
                nop_idx += 1

        # 1b. LDS for spilled pair partners (Case B)
        for partner_reg, partner_temp, is_store in partner_ops:
            if not is_store:
                imm = spill_layout[partner_reg].smem_offset
                lds_i, lds_c = _lds(partner_temp, smem_base_reg, imm, wbar=wbar_slot)
                cubin.write_instruction(kernel, nop_offsets[nop_idx], lds_i, lds_c)
                nop_idx += 1
                for _ in range(2):
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            nop_s15_i, nop_s15_c)
                    nop_idx += 1

        # 1c. .64 pair fixups (Case A fallbacks)
        for reg_num, fixup_val in mov_fixups.items():
            rtemp = temp_for_reg[reg_num]
            if fixup_val == -2:
                sts_i, sts_c = _sts(smem_base_reg, rtemp + 1, scratch_smem)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        sts_i, sts_c)
                nop_idx += 1
                partner_color = coloring[pair_info[reg_num][0]]
                mov_i, mov_c = make_mov(rtemp + 1, partner_color)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        mov_i, mov_c)
                nop_idx += 1
            elif fixup_val == -1:
                sts_i, sts_c = _sts(smem_base_reg, rtemp, scratch_smem)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        sts_i, sts_c)
                nop_idx += 1
            elif fixup_val >= 0:
                # MOV mode: copy partner's physical value into Rtemp+1
                mov_i, mov_c = make_mov(rtemp + 1, fixup_val)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        mov_i, mov_c)
                nop_idx += 1
            # fixup_val == -3: save/restore handled in steps 1c and 3c

        # 2. Renamed original instruction.
        # Use stall=15 (max) to ensure the preceding LDS has completed
        # before this instruction uses the loaded value.  We do NOT use
        # write barriers (all 6 slots are occupied by the kernel).
        renamed_instr = orig_instr
        renamed_ctrl = orig_ctrl
        # Clear wmask (barrier waits from original code are invalid in
        # trampoline context) and set stall=15 (ensures LDS data is ready).
        # Preserve wbar (needed by DEF STS to wait for LDG/TEX results).
        renamed_ctrl = renamed_ctrl & ~(0x3F << 49)  # clear wmask
        renamed_ctrl = (renamed_ctrl & ~(0x1F << 41)) | (15 << 41)  # stall=15
        # Rename spilled registers to temps.  The instruction already has
        # non-spilled registers globally renamed (from apply_register_renames).
        # For spilled registers: the binary now has coloring[reg] (if reg was
        # in the coloring) or the original number (if not).  We rename from
        # whatever is currently in the binary field to the temp.
        for ref in inst.reg_refs:
            if ref.reg_num not in temp_for_reg or ref.bit_position < 0:
                continue
            rtemp = temp_for_reg[ref.reg_num]
            if ref.in_ctrl_word:
                renamed_ctrl = _rename_reg_in_word(
                    renamed_ctrl, ref.bit_position, ref.reg_num, rtemp)
            else:
                renamed_instr = _rename_reg_in_word(
                    renamed_instr, ref.bit_position, ref.reg_num, rtemp)
        cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                renamed_instr, renamed_ctrl)
        nop_idx += 1

        # 3. STS for each DEF register.
        #    If the renamed instruction sets a write barrier (wbar) — e.g.
        #    LDG, TEX, or other variable-latency ops — the STS must wait
        #    for that barrier before reading the result register.
        renamed_wbar = (renamed_ctrl >> 46) & 0x7
        for reg_num, ref in def_regs:
            rtemp = temp_for_reg[reg_num]
            imm = spill_layout[reg_num].smem_offset
            sts_i, sts_c = _sts(smem_base_reg, rtemp, imm)
            if renamed_wbar < 6:
                sts_c = sts_c | ((1 << renamed_wbar) << 49)  # wmask wait
                sts_c = (sts_c & ~(0x1F << 41)) | (15 << 41)  # stall=15
            cubin.write_instruction(kernel, nop_offsets[nop_idx], sts_i, sts_c)
            nop_idx += 1
            # NOP stall=15 — ensures STS completes before any subsequent
            # LDS from the same slot.  BAR.SYNC is unnecessary here because
            # spill slots are per-thread (each thread has its own address),
            # and a CTA-wide barrier would deadlock with < blockDim threads.
            nop_s15_i, nop_s15_c = make_nop_stall(15)
            cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                    nop_s15_i, nop_s15_c)
            nop_idx += 1

        # 3a. Copy .64 DEF partner value (non-spilled partner → physical reg)
        for reg_num, (partner, partner_spilled) in pair_info.items():
            if (not partner_spilled
                    and partner in coloring
                    and any(r == reg_num for r, _ in def_regs)):
                rtemp = temp_for_reg[reg_num]
                partner_color = coloring[partner]
                if rtemp + 1 != partner_color:
                    mov_i, mov_c = make_mov(partner_color, rtemp + 1)
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            mov_i, mov_c)
                    nop_idx += 1

        # 3b. STS for spilled pair partners (Case B)
        #     MUST happen BEFORE restoring stolen registers — the partner
        #     result sits in rtemp+1, which would be overwritten by restore.
        #     Must also wait on renamed instruction's wbar (LDG.E.64 etc.).
        for partner_reg, partner_temp, is_store in partner_ops:
            if is_store:
                imm = spill_layout[partner_reg].smem_offset
                sts_i, sts_c = _sts(smem_base_reg, partner_temp, imm)
                if renamed_wbar < 6:
                    sts_c = sts_c | ((1 << renamed_wbar) << 49)
                    sts_c = (sts_c & ~(0x1F << 41)) | (15 << 41)
                cubin.write_instruction(kernel, nop_offsets[nop_idx], sts_i, sts_c)
                nop_idx += 1

        # 3c. Restore stolen live registers (sentinel -3)
        #     For stolen pairs, restore both halves.
        #     Use 4 NOP stall=15 (60 cycles) to ensure LDS completes.
        restore_smem = existing_smem + len(spill_layout) * stride
        for reg_num, fixup_val in mov_fixups.items():
            if fixup_val == -3:
                rtemp = temp_for_reg[reg_num]
                lds_i, lds_c = _lds(rtemp, smem_base_reg, restore_smem, wbar=wbar_slot)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        lds_i, lds_c)
                nop_idx += 1
                for _ in range(4):
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            nop_s15_i, nop_s15_c)
                    nop_idx += 1
                restore_smem += stride
                if reg_num in pair_info:
                    lds_i2, lds_c2 = _lds(rtemp + 1, smem_base_reg,
                                          restore_smem, wbar=wbar_slot)
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            lds_i2, lds_c2)
                    nop_idx += 1
                    for _ in range(4):
                        cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                                nop_s15_i, nop_s15_c)
                        nop_idx += 1
                    restore_smem += stride

        # 3d. Restore saved registers (pair fixups -2, -1)
        for reg_num, fixup_val in mov_fixups.items():
            rtemp = temp_for_reg[reg_num]
            if fixup_val == -2:
                is_def = any(r == reg_num for r, _ in def_regs)
                if is_def:
                    partner_color = coloring[pair_info[reg_num][0]]
                    mov_i, mov_c = make_mov(partner_color, rtemp + 1)
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            mov_i, mov_c)
                    nop_idx += 1
                lds_i, lds_c = _lds(rtemp + 1, smem_base_reg, scratch_smem, wbar=wbar_slot)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        lds_i, lds_c)
                nop_idx += 1
                for _ in range(2):
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            nop_s15_i, nop_s15_c)
                    nop_idx += 1
            elif fixup_val == -1:
                lds_i, lds_c = _lds(rtemp, smem_base_reg, scratch_smem, wbar=wbar_slot)
                cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                        lds_i, lds_c)
                nop_idx += 1
                for _ in range(2):
                    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                            nop_s15_i, nop_s15_c)
                    nop_idx += 1

        # 4. NOP stall=15 fence before BRA back (ensure all STS/LDS complete)
        nop_fence_i, nop_fence_c = make_nop_stall(15)
        cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                nop_fence_i, nop_fence_c)
        nop_idx += 1

        # 5. BRA back
        return_off = off + 16
        bra_b, bra_bc = make_bra(return_off, nop_offsets[nop_idx])
        bra_bc = (bra_bc & ~(0x1F << 41)) | (15 << 41)  # stall=15 on BRA
        cubin.write_instruction(kernel, nop_offsets[nop_idx], bra_b, bra_bc)
        nop_idx += 1

        # Record trampoline info for ALL spilled regs at this instruction
        for reg_num in all_spill_regs:
            if reg_num in spill_layout:
                spill_decision = next(
                    (s for s in spills if s.reg == reg_num), spills[0])
                tramp_infos.append(TrampolineInfo(
                    original_offset=off,
                    trampoline_offset=tramp_start,
                    spill=spill_decision,
                    reload_reg=temp_for_reg.get(reg_num, -1),
                ))

    return actions, tramp_infos, nop_idx


def create_spill_prologue(
    cubin: Cubin,
    kernel: KernelInfo,
    nop_offsets: list[int],
    nop_idx: int,
    smem_base_reg: int,
) -> tuple[list[PatchAction], int]:
    """Create prologue to compute Rbase = threadIdx.x * 4.

    Simple prologue:
        S2R Rbase, SR_TID.X      # Rbase = tid
        SHF.L.U32 Rbase, Rbase, 2, RZ  # Rbase = tid * 4

    Spill slot addresses: [Rbase + smem_base_offset + slot_i * stride]

    Returns: (patch_actions, updated_nop_idx)
    """
    PROLOGUE_SLOTS = 9
    actions: list[PatchAction] = []

    if nop_idx + PROLOGUE_SLOTS > len(nop_offsets):
        raise RuntimeError(
            f"Not enough NOP slots for spill prologue "
            f"(need {PROLOGUE_SLOTS}, have {len(nop_offsets) - nop_idx})")

    first_off = 0
    orig_instr, orig_ctrl = cubin.read_instruction(kernel, first_off)
    orig_bytes = struct.pack('<QQ', orig_instr, orig_ctrl)

    tramp_start = nop_offsets[nop_idx]

    # 0. Replace first instruction with BRA to prologue
    bra_instr, bra_ctrl = make_bra(tramp_start, first_off)
    cubin.write_instruction(kernel, first_off, bra_instr, bra_ctrl)
    actions.append(PatchAction(
        code_offset=first_off,
        description="BRA to spill prologue",
        old_bytes=orig_bytes,
        new_instr=bra_instr,
        new_ctrl=bra_ctrl,
    ))

    nop_s15_i, nop_s15_c = make_nop_stall(15)

    # [0] NOP stall=15 (pipeline lead-in)
    cubin.write_instruction(kernel, nop_offsets[nop_idx], nop_s15_i, nop_s15_c)
    nop_idx += 1

    # [6] S2R Rbase, SR_TID.X (thread id)
    s2r_i, s2r_c = make_s2r_tid(smem_base_reg, stall=15)
    cubin.write_instruction(kernel, nop_offsets[nop_idx], s2r_i, s2r_c)
    nop_idx += 1

    # [7-10] NOP stall=15 × 4 (wait for S2R)
    for _ in range(4):
        cubin.write_instruction(kernel, nop_offsets[nop_idx],
                                nop_s15_i, nop_s15_c)
        nop_idx += 1

    # [6] IMAD Rbase, Rbase, 4, RZ → Rbase = tid * 4
    # IMAD with immediate is safe: Rd = Ra * imm + Rc (Rc=RZ → Rd = Ra * 4)
    try:
        imad_i, imad_c = _cubit_encode(
            f"IMAD R{smem_base_reg}, R{smem_base_reg}, 0x4, RZ ;")
    except (RuntimeError, ValueError):
        # Fallback: IMAD Rd, Ra, imm, RZ where imm=4 at bits[47:40]
        imad_i = (0x7824
                  | ((smem_base_reg & 0xFF) << 16)
                  | ((smem_base_reg & 0xFF) << 24)
                  | ((4 & 0xFF) << 40)
                  | 0x800)  # immediate mode bit
        imad_c = 0x000fe20000000f00
    cubin.write_instruction(kernel, nop_offsets[nop_idx], imad_i, imad_c)
    nop_idx += 1

    # [12] Original first instruction
    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                            orig_instr, orig_ctrl)
    nop_idx += 1

    # [13] BRA back to second instruction
    return_off = first_off + 16
    bra_back, bra_back_ctrl = make_bra(return_off, nop_offsets[nop_idx])
    cubin.write_instruction(kernel, nop_offsets[nop_idx],
                            bra_back, bra_back_ctrl)
    nop_idx += 1

    return actions, nop_idx


def apply_patch_plan(cubin: Cubin, kernel_name: str,
                     coloring_result: ColoringResult,
                     instructions: list[Instruction],
                     target_regs: int,
                     blocks: Optional[list] = None) -> PatchPlan:
    """Apply a complete re-coloring patch plan to a cubin.

    This is the main entry point for patching.
    """
    kernel = cubin.get_kernel(kernel_name)

    # Step 1: Apply register renames
    rename_actions = apply_register_renames(
        cubin, kernel, coloring_result.coloring, instructions
    )

    # Step 2: Allocate NOP slots
    nop_offsets: list[int] = []

    # Step 3: Handle spills
    prologue_actions: list[PatchAction] = []
    tramp_actions: list[PatchAction] = []
    tramp_infos: list[TrampolineInfo] = []
    nop_idx = 0
    # Use SHARED MEMORY for spill slots.  Each spilled register gets a
    # strided 4-byte slot: smem[slot_i * stride + tid * 4].
    # The base register (Rbase = tid * 4) is computed once in the prologue.
    n_spills = len(coloring_result.spilled)
    smem_base_reg = target_regs - 3  # highest usable register
    eiattr_regs = target_regs
    if n_spills > 0:
        min_eiattr = smem_base_reg + 6
        eiattr_regs = max(target_regs, min_eiattr)
    stride = _compute_spill_stride(kernel)
    smem_base_offset = kernel.shared_size  # place spill slots AFTER kernel's static smem
    new_stack_bytes = kernel.max_stack
    new_smem = kernel.shared_size
    print(f"INFO: SMEM spill layout: n_spills={n_spills}, stride={stride}, "
          f"smem_base_reg=R{smem_base_reg}",
          file=sys.stderr)

    if coloring_result.spilled:
        total_refs = sum(len(s.use_offsets) + len(s.def_offsets)
                         for s in coloring_result.spilled)
        est_slots = 14 + total_refs * 8 + 20

        new_nop_offsets = cubin.grow_kernel_text(kernel, est_slots)
        nop_offsets = new_nop_offsets

        wbar_slot = _find_free_wbar_slot(cubin, kernel)

        valid_spills = [s for s in coloring_result.spilled
                        if len(s.use_offsets) + len(s.def_offsets) > 0]
        if not valid_spills:
            print("INFO: All spilled registers have 0 references — "
                  "no trampolines needed", file=sys.stderr)
        else:
            if blocks is None:
                print("WARNING: No BasicBlock data — cannot build trampolines. "
                      "Pass 'blocks' to apply_patch_plan().", file=sys.stderr)
            else:
                from sasskit.analysis.liveness import compute_live_at
                live_at = compute_live_at(blocks)

                prologue_actions, nop_idx = create_spill_prologue(
                    cubin, kernel, nop_offsets, nop_idx, smem_base_reg)

                tramp_effective_K = eiattr_regs - 3
                tramp_actions, tramp_infos, nop_idx = create_spill_trampolines(
                    cubin, kernel, valid_spills, instructions,
                    nop_offsets, nop_idx,
                    smem_base_reg,
                    coloring_result.coloring,
                    smem_base_offset, live_at,
                    wbar_slot=wbar_slot,
                    stride=stride,
                    use_local_mem=False,
                    effective_K=tramp_effective_K)

                layout = _compute_spill_layout(valid_spills, smem_base_offset,
                                               stride=stride)
                if layout:
                    max_off = max(info.smem_offset for info in layout.values())
                    print(f"INFO: SMEM spill slots: base={smem_base_offset}, "
                          f"max_offset={max_off}, stride={stride}",
                          file=sys.stderr)

    # Step 4: Update register count in ALL metadata locations.
    if eiattr_regs > target_regs:
        print(f"INFO: Bumping EIATTR from {target_regs} to {eiattr_regs} "
              f"(smem_base=R{smem_base_reg} needs headroom)",
              file=sys.stderr)
    old_rc = kernel.reg_count

    cubin.set_reg_count(kernel, eiattr_regs)
    print(f"INFO: Patched per-kernel reg_count for {kernel.name}: "
          f"{old_rc} → {eiattr_regs}", file=sys.stderr)

    kernel_symidx = _find_kernel_symidx(cubin, kernel.name)
    if kernel_symidx >= 0:
        _patch_eiattr_regcount(cubin, kernel_symidx, eiattr_regs)
    else:
        print(f"WARNING: Could not find symbol '{kernel.name}' "
              f"— global EIATTR_REGCOUNT not patched", file=sys.stderr)

    _patch_maxreg_count(cubin, kernel, eiattr_regs)

    all_actions = prologue_actions + rename_actions + tramp_actions

    return PatchPlan(
        kernel_name=kernel_name,
        actions=all_actions,
        new_reg_count=target_regs,
        new_stack_size=new_stack_bytes,
        new_smem_size=new_smem,
        spill_trampolines=tramp_infos,
    )


def _find_section_index(cubin: Cubin, section_name: str) -> int:
    """Find the ELF section index for a named section."""
    e_shoff = struct.unpack_from('<Q', cubin.data, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin.data, 0x3C)[0]
    e_shentsize = struct.unpack_from('<H', cubin.data, 0x3A)[0]
    e_shstrndx = struct.unpack_from('<H', cubin.data, 0x3E)[0]

    shstr_base = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from('<Q', cubin.data, shstr_base + 24)[0]

    target = section_name.encode() + b'\x00'
    for i in range(e_shnum):
        sh_base = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack_from('<I', cubin.data, sh_base)[0]
        name_start = shstr_off + sh_name_idx
        if name_start + len(target) <= len(cubin.data):
            if bytes(cubin.data[name_start:name_start + len(target)]) == target:
                return i
    return -1


def _find_kernel_symidx(cubin: Cubin, kernel_name: str) -> int:
    """Find the FUNCTION symbol table index for a kernel."""
    e_shoff = struct.unpack_from('<Q', cubin.data, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin.data, 0x3C)[0]
    e_shentsize = struct.unpack_from('<H', cubin.data, 0x3A)[0]
    e_shstrndx = struct.unpack_from('<H', cubin.data, 0x3E)[0]

    shstr_base = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from('<Q', cubin.data, shstr_base + 24)[0]

    symtab_off = symtab_size = symtab_ent = 0
    strtab_off = strtab_size = 0
    for i in range(e_shnum):
        sh_base = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from('<I', cubin.data, sh_base + 4)[0]
        if sh_type == 2:
            symtab_off = struct.unpack_from('<Q', cubin.data, sh_base + 24)[0]
            symtab_size = struct.unpack_from('<Q', cubin.data, sh_base + 32)[0]
            symtab_ent = struct.unpack_from('<Q', cubin.data, sh_base + 56)[0]
            strtab_idx = struct.unpack_from('<I', cubin.data, sh_base + 40)[0]
            st_base = e_shoff + strtab_idx * e_shentsize
            strtab_off = struct.unpack_from('<Q', cubin.data, st_base + 24)[0]
            strtab_size = struct.unpack_from('<Q', cubin.data, st_base + 32)[0]
            break

    if symtab_ent == 0:
        return -1

    target = kernel_name.encode()
    nsym = symtab_size // symtab_ent
    for idx in range(nsym):
        sym_off = symtab_off + idx * symtab_ent
        st_name = struct.unpack_from('<I', cubin.data, sym_off)[0]
        st_info = cubin.data[sym_off + 4]
        if (st_info & 0xF) != 2:
            continue
        name_start = strtab_off + st_name
        if name_start + len(target) + 1 > len(cubin.data):
            continue
        sym_name = bytes(cubin.data[name_start:name_start + len(target)])
        if sym_name == target and cubin.data[name_start + len(target)] == 0:
            return idx
    return -1


def _patch_eiattr_regcount(
    cubin: Cubin,
    text_section_idx: int,
    new_regcount: int,
) -> None:
    """Patch EIATTR_REGCOUNT (attr 0x2f) in global .nv.info sections."""
    e_shoff = struct.unpack_from('<Q', cubin.data, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin.data, 0x3C)[0]
    e_shentsize = struct.unpack_from('<H', cubin.data, 0x3A)[0]
    e_shstrndx = struct.unpack_from('<H', cubin.data, 0x3E)[0]
    shstr_base = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from('<Q', cubin.data, shstr_base + 24)[0]

    for si in range(e_shnum):
        sh_base = e_shoff + si * e_shentsize
        sh_name_idx = struct.unpack_from('<I', cubin.data, sh_base)[0]
        name_start = shstr_off + sh_name_idx
        name_end = cubin.data.index(0, name_start)
        sec_name = bytes(cubin.data[name_start:name_end]).decode('ascii', 'replace')

        if sec_name not in ('.nv.info', '.nv.merc.nv.info'):
            continue

        sec_off = struct.unpack_from('<Q', cubin.data, sh_base + 24)[0]
        sec_size = struct.unpack_from('<Q', cubin.data, sh_base + 32)[0]

        j = 0
        while j < sec_size:
            off = sec_off + j
            if off + 12 > len(cubin.data):
                break
            tag = cubin.data[off]
            attr = cubin.data[off + 1]
            if tag == 0x04:
                payload_size = struct.unpack_from('<H', cubin.data, off + 2)[0]
                if attr == 0x2f and payload_size == 8:
                    symidx = struct.unpack_from('<I', cubin.data, off + 4)[0]
                    old_reg = struct.unpack_from('<I', cubin.data, off + 8)[0]
                    if symidx == text_section_idx:
                        struct.pack_into('<I', cubin.data, off + 8, new_regcount)
                        print(f"INFO: Patched EIATTR_REGCOUNT in {sec_name} "
                              f"at 0x{off:x}: {old_reg} → {new_regcount}",
                              file=sys.stderr)
                j += 4 + payload_size
            else:
                j += 4


def _patch_maxreg_count(
    cubin: Cubin,
    kernel: KernelInfo,
    new_regcount: int,
) -> None:
    """Patch EIATTR_MAXREG_COUNT (attr 0x37) in the per-kernel .nv.info section."""
    if kernel.info_offset is None or kernel.info_size is None:
        return
    data = cubin.data
    i = 0
    while i < kernel.info_size:
        off = kernel.info_offset + i
        if off + 4 > len(data):
            break
        fmt = data[off]
        attr = data[off + 1]
        if fmt == 0x04:
            payload_size = struct.unpack_from('<H', data, off + 2)[0]
            if attr == 0x37 and payload_size >= 4:
                old_val = struct.unpack_from('<I', data, off + 4)[0]
                struct.pack_into('<I', data, off + 4, new_regcount)
                print(f"INFO: Patched EIATTR_MAXREG_COUNT in "
                      f".nv.info.{kernel.name} at 0x{off:x}: "
                      f"{old_val} → {new_regcount}", file=sys.stderr)
            i += 4 + payload_size
        else:
            i += 4


def _patch_smem_size(cubin: Cubin, kernel: KernelInfo, new_size: int) -> None:
    """Patch the shared memory size in the ELF .nv.shared section header."""
    old_size = kernel.shared_size

    e_shoff = struct.unpack_from('<Q', cubin.data, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin.data, 0x3C)[0]
    e_shentsize = struct.unpack_from('<H', cubin.data, 0x3A)[0]
    e_shstrndx = struct.unpack_from('<H', cubin.data, 0x3E)[0]

    shstr_base = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from('<Q', cubin.data, shstr_base + 24)[0]
    shstr_size = struct.unpack_from('<Q', cubin.data, shstr_base + 32)[0]

    target_name = f'.nv.shared.{kernel.name}'.encode() + b'\x00'

    for i in range(e_shnum):
        sh_base = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack_from('<I', cubin.data, sh_base)[0]
        sh_size_field = struct.unpack_from('<Q', cubin.data, sh_base + 32)[0]

        name_start = shstr_off + sh_name_idx
        if name_start + len(target_name) <= len(cubin.data):
            sec_name = bytes(cubin.data[name_start:name_start + len(target_name)])
            if sec_name == target_name and sh_size_field == old_size:
                struct.pack_into('<Q', cubin.data, sh_base + 32, new_size)
                kernel.shared_size = new_size
                print(f"INFO: Patched .nv.shared.{kernel.name} "
                      f"sh_size: {old_size} → {new_size}", file=sys.stderr)
                return

    for i in range(e_shnum):
        sh_base = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from('<I', cubin.data, sh_base + 4)[0]
        sh_size_field = struct.unpack_from('<Q', cubin.data, sh_base + 32)[0]
        if sh_type == 8 and sh_size_field == old_size:
            struct.pack_into('<Q', cubin.data, sh_base + 32, new_size)
            kernel.shared_size = new_size
            print(f"INFO: Patched NOBITS section {i} "
                  f"sh_size: {old_size} → {new_size}", file=sys.stderr)
            return

    print(f"WARNING: Could not patch shared memory size "
          f"({old_size} → {new_size})", file=sys.stderr)


def _expand_reserved_smem(cubin: Cubin, new_size: int) -> None:
    """Expand shared memory by growing .nv.shared.reserved.* section and PHDR.

    On SM120+, all cubins have a ``.nv.shared.reserved.0`` section (typically
    64 bytes) with a corresponding NOBITS program header segment.  Instead of
    creating a new ``.nv.shared.<kernel>`` section (which the CUDA driver's
    ELF loader ignores for CGA shared memory setup), we expand the existing
    reserved section and its program header to cover our spill slots.
    """
    e_shoff = struct.unpack_from('<Q', cubin.data, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin.data, 0x3C)[0]
    e_shentsize = struct.unpack_from('<H', cubin.data, 0x3A)[0]
    e_shstrndx = struct.unpack_from('<H', cubin.data, 0x3E)[0]
    shstr_base = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from('<Q', cubin.data, shstr_base + 24)[0]

    # Find and expand ALL .nv.shared.reserved.* sections (including mercury copies)
    target = b'nv.shared.reserved.'
    expanded_section = False
    for i in range(e_shnum):
        sh_base = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack_from('<I', cubin.data, sh_base)[0]
        name_start = shstr_off + sh_name_idx
        name_end = cubin.data.index(0, name_start)
        sec_name = bytes(cubin.data[name_start:name_end])
        if target in sec_name:
            old_size = struct.unpack_from('<Q', cubin.data, sh_base + 32)[0]
            struct.pack_into('<Q', cubin.data, sh_base + 32, new_size)
            print(f"INFO: Expanded {sec_name.decode()} section: "
                  f"{old_size} → {new_size}", file=sys.stderr)
            expanded_section = True

    if not expanded_section:
        print(f"WARNING: No .nv.shared.reserved section found — "
              f"cannot allocate shared memory for spills", file=sys.stderr)
        return

    # Also expand the NOBITS program header segment
    e_phoff = struct.unpack_from('<Q', cubin.data, 0x20)[0]
    e_phnum = struct.unpack_from('<H', cubin.data, 0x38)[0]
    e_phentsize = struct.unpack_from('<H', cubin.data, 0x36)[0]
    for pi in range(e_phnum):
        ph_base = e_phoff + pi * e_phentsize
        p_filesz = struct.unpack_from('<Q', cubin.data, ph_base + 32)[0]
        p_memsz = struct.unpack_from('<Q', cubin.data, ph_base + 40)[0]
        if p_filesz == 0 and p_memsz > 0:
            final_sz = max(p_memsz, new_size)
            struct.pack_into('<Q', cubin.data, ph_base + 40, final_sz)
            print(f"INFO: Expanded NOBITS PHDR memsz: "
                  f"{p_memsz} → {final_sz}", file=sys.stderr)
            break


# ============================================================================
# Diagnostics for unmatched register fields
# ============================================================================

@dataclass
class UnmatchedRegInfo:
    """Diagnostic info for a register reference that could not be patched."""
    code_offset: int
    reg_num: int
    field_name: str
    bit_position: int
    asm_text: str
    reason: str


def diagnose_unmatched_refs(
    instructions: list[Instruction],
    coloring: dict[int, int],
    spill_set: set[int],
    target_regs: int,
) -> list[UnmatchedRegInfo]:
    """Find register references that would remain >= target_regs after patching."""
    problems: list[UnmatchedRegInfo] = []

    for inst in instructions:
        for ref in inst.reg_refs:
            rn = ref.reg_num
            if rn >= target_regs and rn != 0xFF:
                if ref.bit_position < 0:
                    problems.append(UnmatchedRegInfo(
                        code_offset=inst.code_offset,
                        reg_num=rn,
                        field_name=ref.field_name,
                        bit_position=ref.bit_position,
                        asm_text=inst.asm_text,
                        reason="bit_position=-1: decoder could not locate "
                               "register field in binary encoding",
                    ))
                elif rn in spill_set and rn not in coloring:
                    problems.append(UnmatchedRegInfo(
                        code_offset=inst.code_offset,
                        reg_num=rn,
                        field_name=ref.field_name,
                        bit_position=ref.bit_position,
                        asm_text=inst.asm_text,
                        reason="spilled register with no trampoline "
                               "(instruction not rewritten)",
                    ))
                elif rn not in coloring:
                    problems.append(UnmatchedRegInfo(
                        code_offset=inst.code_offset,
                        reg_num=rn,
                        field_name=ref.field_name,
                        bit_position=ref.bit_position,
                        asm_text=inst.asm_text,
                        reason="register not in coloring map and not spilled",
                    ))

    return problems


def diagnose_encoding_mismatches(
    instructions: list[Instruction],
) -> list[UnmatchedRegInfo]:
    """Find ALL register references where bit_position = -1."""
    problems: list[UnmatchedRegInfo] = []

    for inst in instructions:
        for ref in inst.reg_refs:
            if ref.bit_position < 0:
                iw = inst.instr_word
                cw = inst.ctrl_word
                field_vals = (
                    f"Rd={iw >> 16 & 0xFF}, "
                    f"Ra={iw >> 24 & 0xFF}, "
                    f"Rb={iw >> 32 & 0xFF}, "
                    f"Rc={iw >> 40 & 0xFF}, "
                    f"Rc_ctrl={cw & 0xFF}"
                )
                opcode_hex = f"0x{iw & 0xFFFF:04x}"
                fmt_nibble = (iw >> 12) & 0xF
                problems.append(UnmatchedRegInfo(
                    code_offset=inst.code_offset,
                    reg_num=ref.reg_num,
                    field_name=ref.field_name,
                    bit_position=-1,
                    asm_text=inst.asm_text,
                    reason=(f"opcode={opcode_hex} fmt=0x{fmt_nibble:x} "
                            f"fields=[{field_vals}]"),
                ))

    return problems
