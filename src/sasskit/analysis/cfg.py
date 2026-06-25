"""Control-flow graph construction for SASS instruction sequences.

Builds a CFG from decoded instructions by identifying basic block
boundaries at branch targets, instructions following branches, and EXIT.

Branch target resolution uses the SM120 BRA encoding formula derived
from cubit's encoder (cubit/src/encoder.rs ``apply_branch_encoding``):

    Encoding (by assembler):
        rel  = target_addr - bra_addr - 16
        rq   = rel >> 2                        (arithmetic shift)
        bits[23:16] = rq & 0xFF                (low byte)
        bits[63:32] = ((rq >> 8) << 2) | mod   (high, with modifier bits)
        bits[81:64] = sign_ext if rel < 0

    Decoding (by us):
        low_byte  = instr_word[23:16]
        hi_dword  = instr_word[63:32]
        rq_hi     = sign_extend_30(hi_dword >> 2)
        rq        = (rq_hi << 8) | low_byte
        target    = bra_addr + 16 + rq * 4
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from sasskit.core.decoder import Instruction


@dataclass
class BasicBlock:
    """A maximal sequence of instructions with no internal branches."""
    id: int
    start_offset: int
    end_offset: int
    instructions: list[Instruction]
    successors: list[int] = field(default_factory=list)  # Block IDs
    predecessors: list[int] = field(default_factory=list)

    # Dataflow sets
    use_set: set[int] = field(default_factory=set)   # Registers read before written
    def_set: set[int] = field(default_factory=set)   # Registers written
    live_in: set[int] = field(default_factory=set)    # Live at block entry
    live_out: set[int] = field(default_factory=set)   # Live at block exit


@dataclass
class LiveRange:
    """The interval where a register holds a needed value."""
    reg: int
    start: int  # Code offset of definition
    end: int    # Code offset of last use

    def overlaps(self, other: LiveRange) -> bool:
        return self.start < other.end and other.start < self.end


# ============================================================================
# SM120 branch target decoding
# ============================================================================

def decode_bra_target(inst: Instruction) -> Optional[int]:
    """Decode the absolute branch target from an SM120 BRA instruction.

    Uses the encoding formula from cubit/src/encoder.rs.  Works on the
    raw instruction word — no cuobjdump, no asm text parsing.

    Returns None for non-BRA instructions or on decode failure.
    """
    if inst.opcode not in ('BRA', 'BRX', 'BRXU', 'CALL', 'JMP'):
        return None

    lo = inst.instr_word
    addr = inst.code_offset

    low_byte = (lo >> 16) & 0xFF
    hi_dword = (lo >> 32) & 0xFFFFFFFF

    # Drop modifier bits [1:0], recover rq_hi as a 30-bit signed value
    rq_hi_raw = hi_dword >> 2
    if rq_hi_raw & (1 << 29):
        rq_hi = rq_hi_raw - (1 << 30)
    else:
        rq_hi = rq_hi_raw

    rq = (rq_hi << 8) | low_byte
    rel = rq * 4
    target = addr + 16 + rel
    return target


def _is_conditional_branch(inst: Instruction) -> bool:
    """Return True if the branch instruction is conditional.

    On SM120 a BRA is conditional when guarded by a predicate register.
    The cubit decoder sets ``is_predicated=True`` and stores the
    predicate string for standard predicates (@P0, @!P0, etc.).

    Uniform predicates (@UP0, @!UP0) may appear in asm text even when
    ``is_predicated`` is False.

    An unguarded BRA (guard == PT or no guard) is unconditional.
    """
    if inst.is_predicated:
        return True
    # Uniform predicate in asm text
    if re.search(r'@!?UP\d+', inst.asm_text):
        return True
    return False


# ============================================================================
# CFG construction
# ============================================================================

def build_cfg(instructions: list[Instruction]) -> list[BasicBlock]:
    """Build a control-flow graph from a list of decoded instructions.

    Basic block boundaries are placed at:
      - The start of the function
      - Branch target addresses (destinations of BRA instructions)
      - Instructions immediately following a branch or EXIT

    Branch targets are decoded directly from the instruction binary using
    the SM120 BRA encoding formula — no asm text parsing required.
    """
    if not instructions:
        return []

    valid_offsets: set[int] = {inst.code_offset for inst in instructions}

    # ── Resolve branch targets ────────────────────────────────
    branch_targets: dict[int, int] = {}   # source_offset -> target_offset
    block_starts: set[int] = {instructions[0].code_offset}

    for inst in instructions:
        off = inst.code_offset

        if inst.opcode == 'EXIT':
            next_off = off + 16
            if next_off in valid_offsets:
                block_starts.add(next_off)
            continue

        if not inst.is_branch:
            continue

        target = decode_bra_target(inst)
        if target is not None and target in valid_offsets:
            branch_targets[off] = target
            block_starts.add(target)

        # Instruction after the branch starts a new block
        next_off = off + 16
        if next_off in valid_offsets:
            block_starts.add(next_off)

    # ── Create blocks ────────────────────────────────────────
    sorted_starts = sorted(block_starts)

    blocks: list[BasicBlock] = []
    for i, start in enumerate(sorted_starts):
        end = (sorted_starts[i + 1]
               if i + 1 < len(sorted_starts)
               else instructions[-1].code_offset + 16)

        block_instrs = [
            inst for inst in instructions
            if start <= inst.code_offset < end
        ]
        if not block_instrs:
            continue

        blocks.append(BasicBlock(
            id=len(blocks),
            start_offset=start,
            end_offset=end,
            instructions=block_instrs,
        ))

    # ── Build edges ──────────────────────────────────────────
    offset_to_block: dict[int, int] = {
        b.start_offset: b.id for b in blocks
    }

    for block in blocks:
        last = block.instructions[-1]

        if last.opcode == 'EXIT':
            continue  # no successors

        if last.is_branch:
            target = branch_targets.get(last.code_offset)
            if target is not None and target in offset_to_block:
                block.successors.append(offset_to_block[target])

            if _is_conditional_branch(last):
                # Conditional branch: also falls through
                fallthrough = block.end_offset
                if fallthrough in offset_to_block:
                    block.successors.append(offset_to_block[fallthrough])
            # Unconditional BRA: no fallthrough
        else:
            # Non-branch: fall through to next block
            fallthrough = block.end_offset
            if fallthrough in offset_to_block:
                block.successors.append(offset_to_block[fallthrough])

    # ── Build predecessor lists ──────────────────────────────
    for block in blocks:
        for succ_id in block.successors:
            blocks[succ_id].predecessors.append(block.id)

    return blocks
