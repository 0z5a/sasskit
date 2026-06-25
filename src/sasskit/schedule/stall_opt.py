"""Stall count optimization for SM120 SASS.

Analyzes register dependencies within each basic block and computes
the minimum legal stall count for each instruction.  ptxas often sets
conservative stalls (up to 15 cycles); reducing them to the minimum
required by actual data dependencies gives free IPC improvement.

The stall for instruction I is:
    stall[I] = max over all RAW deps of (latency - distance)
where:
    latency = producer's def-to-use latency from ISA database
    distance = number of instructions between producer I_p and consumer I
    (clamped to 0 if distance >= latency)

Variable-latency instructions (LDG, LDS, S2R, TEX) use write barriers
managed by the scoreboard — their consumers wait via wmask, not stall.
We don't modify wmask or wbar here; only reduce stall counts.

Control word quirks respected:
- IADD3/IMAD with Rc2=RZ: stall forced to 15 (hi[7:0]=0xFF)
- S2R: hi[15:8] is SR code, not scheduling
- MUFU: hi[15:8] is function code, not scheduling
- NOP/EXIT: fixed ctrl words, skip
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass

from sasskit.core.cubin import Cubin, KernelInfo
from sasskit.core.decoder import Instruction
from sasskit.analysis.cfg import BasicBlock

try:
    from sasskit.core.isa_db import get_db as _get_isa_db
except ImportError:
    _get_isa_db = None


# Instructions where we must NOT modify the stall field
_SKIP_OPCODES = frozenset([
    'NOP', 'EXIT', 'RET', 'BRK', 'CONT', 'BSSY', 'BSYNC',
    'BAR', 'DEPBAR', 'WARPSYNC',
])

# Instructions where hi[7:0] is an opcode extension, not scheduling
_OPEX_OPCODES = frozenset(['S2R', 'CS2R', 'S2UR', 'MUFU'])


def _base_op(opcode: str) -> str:
    """Extract base opcode for ISA DB lookup (strip modifiers)."""
    return opcode.split('.')[0]


def _get_latency(opcode: str) -> tuple[int, bool]:
    """Get (latency, is_variable) for an opcode.

    Returns (4, False) as default for unknown opcodes.
    """
    if _get_isa_db is None:
        return 4, False
    try:
        db = _get_isa_db()
        info = db.get(_base_op(opcode))
        if info is not None:
            return info.def_latency, info.is_variable_latency
    except Exception:
        pass
    return 4, False


def _has_rc2_rz(inst: Instruction) -> bool:
    """Check if instruction uses Rc2=RZ (hi[7:0] = 0xFF).

    IADD3/IMAD with Rc2=RZ have forced stall=15 and cannot be modified.
    """
    return (inst.ctrl_word & 0xFF) == 0xFF


def _current_stall(inst: Instruction) -> int:
    """Read the stall count from the control word.

    SM120 ctrl word: stall = hi[44:41] (4 bits, 0-15), yield = hi[45].
    """
    base = _base_op(inst.opcode)
    if base in _OPEX_OPCODES:
        return 15
    if _has_rc2_rz(inst):
        return 15
    return (inst.ctrl_word >> 41) & 0xF  # 4 bits, NOT 5


@dataclass
class StallResult:
    """Result of stall optimization for one basic block."""
    block_start: int
    original_stalls: list[int]
    optimized_stalls: list[int]
    savings: int

    @property
    def total_original(self) -> int:
        return sum(self.original_stalls)

    @property
    def total_optimized(self) -> int:
        return sum(self.optimized_stalls)


def optimize_block_stalls(
    instructions: list[Instruction],
    block: BasicBlock,
) -> StallResult:
    """Compute minimum legal stall counts for a basic block.

    For each instruction, finds all RAW dependencies (register def→use)
    and computes the minimum stall as:
        min_stall = max(0, latency - distance)
    for all producers that this instruction depends on.

    Variable-latency producers (LDG, LDS, S2R, TEX) are skipped —
    their consumers use write barriers (wmask), not stalls.
    """
    block_insts = [
        inst for inst in instructions
        if block.start_offset <= inst.code_offset <= block.end_offset
    ]

    if not block_insts:
        return StallResult(block.start_offset, [], [], 0)

    n = len(block_insts)
    orig_stalls = [_current_stall(inst) for inst in block_insts]
    opt_stalls = list(orig_stalls)

    last_def: dict[int, tuple[int, int, bool]] = {}
    last_varlatency_idx = -100  # index of last variable-latency instruction

    for i, inst in enumerate(block_insts):
        base = _base_op(inst.opcode)

        # Track variable-latency instructions (LDG, LDS, S2R, TEX)
        _, is_var = _get_latency(inst.opcode)
        if is_var:
            last_varlatency_idx = i

        # Skip instructions with special ctrl word encoding
        if (base in _SKIP_OPCODES or inst.is_nop
                or base in _OPEX_OPCODES or _has_rc2_rz(inst)):
            for r in inst.dest_regs:
                if r != 0xFF:
                    lat, var = _get_latency(inst.opcode)
                    last_def[r] = (i, lat, var)
            continue

        # Compute minimum stall from ALL RAW dependencies.
        # stall[I] = max over each src reg R of:
        #   max(0, producer_latency - (I - producer_index))
        # This is the "before-me" wait: how many cycles must pass
        # between I-1 and I for all of I's source values to be ready.
        min_needed = 0
        has_cross_block_dep = False

        for r in inst.src_regs:
            if r == 0xFF:
                continue
            if r not in last_def:
                has_cross_block_dep = True
                continue
            prod_idx, prod_lat, prod_var = last_def[r]
            if prod_var:
                continue
            # Gap = sum of original stalls between producer and consumer.
            # This is the actual cycle count that passes before I issues
            # (each intervening instruction waits its stall cycles).
            gap = sum(orig_stalls[prod_idx + 1 : i])
            needed = max(0, prod_lat - gap)
            if needed > min_needed:
                min_needed = needed

        near_varlatency = (i - last_varlatency_idx) <= 15 if last_varlatency_idx >= 0 else False
        has_any_src = any(r != 0xFF for r in inst.src_regs)

        if has_cross_block_dep or near_varlatency or inst.is_predicated:
            pass  # keep original
        elif not has_any_src and orig_stalls[i] > 2:
            opt_stalls[i] = max(1, orig_stalls[i] - 1)
        elif min_needed < orig_stalls[i] and not has_any_src:
            opt_stalls[i] = max(min_needed, 1)

        for r in inst.dest_regs:
            if r != 0xFF:
                lat, var = _get_latency(inst.opcode)
                last_def[r] = (i, lat, var)

    savings = sum(max(0, o - n) for o, n in zip(orig_stalls, opt_stalls))

    return StallResult(
        block_start=block.start_offset,
        original_stalls=orig_stalls,
        optimized_stalls=opt_stalls,
        savings=savings,
    )


def apply_stall_optimization(
    cubin: Cubin,
    kernel: KernelInfo,
    instructions: list[Instruction],
    blocks: list[BasicBlock],
    dry_run: bool = False,
) -> dict:
    """Optimize stall counts across all basic blocks.

    Returns stats dict with total savings.
    """
    total_savings = 0
    total_instructions = 0
    modified = 0

    for block in blocks:
        result = optimize_block_stalls(instructions, block)
        total_savings += result.savings
        total_instructions += len(result.original_stalls)

        if dry_run:
            continue

        block_insts = [
            inst for inst in instructions
            if block.start_offset <= inst.code_offset <= block.end_offset
        ]

        for inst, orig, opt in zip(block_insts, result.original_stalls, result.optimized_stalls):
            if opt >= orig:
                continue

            base = _base_op(inst.opcode)
            if base in _SKIP_OPCODES or base in _OPEX_OPCODES:
                continue
            if inst.is_nop or _has_rc2_rz(inst):
                continue

            lo, hi = cubin.read_instruction(kernel, inst.code_offset)
            stall_mask = 0xF << 41  # 4 bits [44:41], preserve yield at bit 45
            new_hi = (hi & ~stall_mask) | ((opt & 0xF) << 41)

            if new_hi != hi:
                cubin.write_instruction(kernel, inst.code_offset, lo, new_hi)
                modified += 1

    return {
        'total_instructions': total_instructions,
        'modified': modified,
        'total_savings_cycles': total_savings,
    }
