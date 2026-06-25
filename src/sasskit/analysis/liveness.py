"""Register liveness analysis for SASS instruction sequences.

Computes which registers are live (holding needed values) at each instruction
using standard backward dataflow analysis. This is the foundation for
determining which registers can safely share the same physical register.

Liveness: a register is LIVE at a point if there exists a path from that
point to a USE of the register without an intervening DEFINITION.

Algorithm:
  1. Build a control-flow graph (CFG) from branch instructions
  2. Compute USE and DEF sets for each basic block
  3. Iterate backward dataflow equations until fixed point:
       live_in[B] = USE[B] ∪ (live_out[B] - DEF[B])
       live_out[B] = ∪ live_in[S] for all successors S of B
  4. Extend to per-instruction granularity within blocks
"""

from __future__ import annotations

import re
from typing import Optional

from sasskit.core.decoder import Instruction
from .cfg import BasicBlock


def compute_use_def(blocks: list[BasicBlock]) -> None:
    """Compute USE and DEF sets for each basic block."""
    for block in blocks:
        use: set[int] = set()
        defs: set[int] = set()
        
        for inst in block.instructions:
            # Sources that are used before being defined in this block
            for reg in inst.src_regs:
                if reg not in defs:
                    use.add(reg)
            # Definitions
            for reg in inst.dest_regs:
                defs.add(reg)
        
        block.use_set = use
        block.def_set = defs


def _find_reconvergence_map(blocks: list[BasicBlock],
                            instructions: list[Instruction]) -> dict[int, int]:
    """Build a map: block_id → reconvergence_block_id for BSSY regions.
    
    On SM 120+, BSSY.RECONVERGENT Bn, target marks the start of a divergent
    region, and BSYNC.RECONVERGENT Bn marks the end. The 'target' in BSSY is
    the reconvergence point where all threads rejoin.
    
    For ANY block inside a BSSY region that has multiple successors (conditional
    branch or similar), we can use the reconvergence point's live_in instead of
    the union of all successors. This eliminates false interferences between
    registers in different branch paths.
    
    Returns: dict mapping block_id → reconvergence_block_id.
    """
    # Step 1: Find all BSSY regions
    bssy_regions: list[tuple[int, int]] = []  # (bssy_offset, reconverge_offset)
    for inst in instructions:
        m = re.search(r'BSSY\.\w+\s+B\d+,\s+0x([0-9a-f]+)', inst.asm_text)
        if m:
            bssy_regions.append((inst.code_offset, int(m.group(1), 16)))
    
    # Step 2: Build offset → block mapping
    offset_to_block: dict[int, int] = {}
    for block in blocks:
        offset_to_block[block.start_offset] = block.id
    
    # Step 3: Resolve reconvergence offsets to block IDs
    # BSSY target → BSYNC instruction. The next block starts at reconv+16.
    reconv_block_for_region: dict[int, int] = {}  # bssy_offset → reconv_block_id
    for bssy_off, reconv_off in bssy_regions:
        for candidate in [reconv_off, reconv_off + 16, reconv_off - 16]:
            if candidate in offset_to_block:
                reconv_block_for_region[bssy_off] = offset_to_block[candidate]
                break
    
    # Step 4: For EVERY block with multiple successors, check if it's inside
    # ANY BSSY region and map to the innermost reconvergence block
    reconv_map: dict[int, int] = {}
    
    for block in blocks:
        if len(block.successors) < 2:
            continue
        
        # Use ANY instruction in the block (use start offset) to check coverage
        block_off = block.start_offset
        
        best_bssy = None
        for bssy_off, reconv_off in bssy_regions:
            if bssy_off <= block_off < reconv_off:
                if best_bssy is None or bssy_off > best_bssy:
                    best_bssy = bssy_off
        
        if best_bssy is not None and best_bssy in reconv_block_for_region:
            reconv_map[block.id] = reconv_block_for_region[best_bssy]
    
    return reconv_map


def compute_liveness(blocks: list[BasicBlock], max_iterations: int = 100,
                     instructions: Optional[list[Instruction]] = None) -> None:
    """Compute live_in and live_out for each block using backward dataflow.
    
    Enhanced with BSSY/BSYNC-aware merging: for conditional branches inside
    divergent regions, use reconvergence-point liveness instead of union of
    both branch targets. This eliminates false interferences between registers
    in different branch paths.
    
    Iterates until fixed point:
      live_out[B] = ∪ live_in[S] for all successors S     (standard)
      live_out[B] = live_in[reconvergence]                  (if in BSSY region)
      live_in[B] = USE[B] ∪ (live_out[B] - DEF[B])
    """
    compute_use_def(blocks)
    
    # Build reconvergence map if instructions provided
    reconv_map: dict[int, int] = {}
    if instructions is not None:
        reconv_map = _find_reconvergence_map(blocks, instructions)
    
    changed = True
    iteration = 0
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for block in reversed(blocks):
            # Compute live_out
            if block.id in reconv_map:
                # BSSY-aware: at the branch point, only registers that
                # survive to the reconvergence point need to be live.
                # Registers used exclusively in one branch path don't
                # interfere with registers in the other path.
                reconv_block = blocks[reconv_map[block.id]]
                new_live_out = set(reconv_block.live_in)
            else:
                # Standard: union of all successors' live_in
                new_live_out: set[int] = set()
                for succ_id in block.successors:
                    new_live_out |= blocks[succ_id].live_in
            
            new_live_in = block.use_set | (new_live_out - block.def_set)
            
            if new_live_in != block.live_in or new_live_out != block.live_out:
                changed = True
                block.live_in = new_live_in
                block.live_out = new_live_out


def compute_live_at(blocks: list[BasicBlock],
                    instructions: list[Instruction] | None = None,
                    ) -> dict[int, set[int]]:
    """Compute the set of live registers BEFORE each instruction.
    
    Calls ``compute_liveness`` first (if block-level liveness has not
    been computed yet) to populate ``live_in`` / ``live_out`` on each
    block, then walks backward through each block from ``live_out``
    to produce per-instruction liveness.
    
    The returned set for each offset includes registers that are USED
    by the instruction (src operands) as well as registers that are
    live-through (needed by later instructions but not defined here).
    
    Returns: dict mapping code_offset -> set of live register numbers
             (live BEFORE this instruction executes)
    """
    # Ensure block-level liveness has been computed.
    # A block with an empty use_set AND empty live_in after having
    # instructions is a strong signal that compute_liveness was never
    # called.  Rather than guessing, just always run it — it's cheap
    # and idempotent.
    compute_liveness(blocks, instructions=instructions)

    live_at: dict[int, set[int]] = {}
    
    for block in blocks:
        # Start from live_out and walk backward
        current_live = set(block.live_out)
        
        for inst in reversed(block.instructions):
            # Compute live BEFORE this instruction:
            # live_before = USE ∪ (live_after - DEF)
            live_before = set(current_live)
            for reg in inst.dest_regs:
                live_before.discard(reg)
            for reg in inst.src_regs:
                live_before.add(reg)
            
            live_at[inst.code_offset] = live_before
            
            # For the next (earlier) instruction, current_live = live_before
            current_live = live_before
    
    return live_at


def find_max_pressure(blocks: list[BasicBlock]) -> tuple[int, int, set[int]]:
    """Find the instruction with maximum register pressure.
    
    Returns: (code_offset, pressure, live_set)
    """
    live_at = compute_live_at(blocks)
    
    max_offset = 0
    max_pressure = 0
    max_set: set[int] = set()
    
    for offset, live_set in live_at.items():
        if len(live_set) > max_pressure:
            max_pressure = len(live_set)
            max_offset = offset
            max_set = live_set
    
    return max_offset, max_pressure, max_set


def find_dead_regs_at(blocks: list[BasicBlock], code_offset: int,
                      reg_range: range) -> set[int]:
    """Find registers from reg_range that are dead at a specific offset.
    
    Useful for finding available registers for spill reloads.
    """
    live_at = compute_live_at(blocks)
    live_here = live_at.get(code_offset, set())
    return set(reg_range) - live_here
