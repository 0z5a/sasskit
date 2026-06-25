"""Register re-coloring engine.

Given liveness information, determines whether a kernel's registers can be
re-colored to use fewer physical registers. Uses graph coloring theory:

1. Build an INTERFERENCE GRAPH: registers that are simultaneously live
   cannot share the same physical register (they have an edge).
2. Attempt K-coloring of the interference graph (K = target register count).
3. If K-coloring fails, identify the minimum set of registers to SPILL
   to shared memory.

The spill strategy prioritizes:
- Loop-invariant values (constant across the hot loop)
- Values with few use points (fewer reload instructions needed)
- Values used only in cold sections (prologues, epilogues)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from sasskit.core.decoder import Instruction
from sasskit.analysis.cfg import BasicBlock
from sasskit.analysis.liveness import compute_live_at, compute_liveness

# Optional ILP solver for optimal graph coloring
try:
    from mip import Model, xsum, BINARY, OptimizationStatus
    HAS_MIP = True
except ImportError:
    HAS_MIP = False


@dataclass
class ColoringResult:
    """Result of a register re-coloring attempt."""
    success: bool
    target_regs: int
    coloring: dict[int, int]          # old_reg -> new_reg
    spilled: list[SpillDecision]      # Registers that couldn't be colored
    max_pressure_before: int
    max_pressure_after: int
    smem_base_reg: int = -1           # Physical register for shared memory base
                                      # (-1 = no spills, no base needed)


@dataclass
class SpillDecision:
    """Decision to spill a register to shared memory."""
    reg: int                    # Register to spill
    reason: str                 # Why this register was chosen
    def_offsets: list[int]      # Code offsets where the value is defined
    use_offsets: list[int]      # Code offsets where the value is used
    smem_slot: int              # Shared memory offset for this spill (per-thread)
    reload_reg: int             # Dead register to use for reloads


def greedy_color(graph: dict[int, set[int]], num_colors: int,
                 priority: Optional[dict[int, int]] = None,
                 pair_regs: Optional[set[int]] = None,
                 quad_regs: Optional[set[int]] = None) -> Optional[dict[int, int]]:
    """K-color with alignment: pair_regs=even, quad_regs=4-aligned."""
    if not graph:
        return {}
    if pair_regs is None:
        pair_regs = set()
    if quad_regs is None:
        quad_regs = set()
    
    if priority:
        order = sorted(graph.keys(), key=lambda r: (priority.get(r, 0), r))
    else:
        order = sorted(graph.keys(), key=lambda r: -len(graph[r]))
    
    coloring: dict[int, int] = {}
    
    for reg in order:
        # Find colors used by neighbors
        neighbor_colors = {coloring[n] for n in graph[reg] if n in coloring}
        
        # Alignment: quad_regs need %4==0, pair_regs need %2==0
        if reg in quad_regs:
            max_color = num_colors - 4
            align = 4
        elif reg in pair_regs:
            max_color = num_colors - 2
            align = 2
        else:
            max_color = num_colors - 1
            align = 1
        color = 0
        while color <= max_color:
            if color in neighbor_colors or (align > 1 and color % align != 0):
                color += 1
                continue
            break
        
        if color > max_color:
            return None
        
        coloring[reg] = color
    
    return coloring


# ============================================================================
# DSatur graph coloring with coalesced pair/quad register constraints
# ============================================================================

def _detect_register_groups(
    instructions: list[Instruction],
) -> tuple[set[int], set[int]]:
    """Detect pair and quad register base registers from instruction patterns.

    Returns (pair_bases, quad_bases) where:
      - pair_bases: registers that are .64/.WIDE destinations (need even color)
      - quad_bases: registers that are .128 destinations (need 4-aligned color)
    Overlaps are resolved: quad takes priority over pair.
    """
    pair_bases: set[int] = set()
    quad_bases: set[int] = set()
    for inst in instructions:
        for ref in inst.reg_refs:
            # Collect pair bases from BOTH destinations and sources.
            # Any register used as part of a .64 pair must be even-aligned,
            # whether it's read or written.
            if ref.is_pair:
                pair_bases.add(ref.reg_num)
            if getattr(ref, 'is_quad', False):
                quad_bases.add(ref.reg_num)
        # Fallback: detect .128 from asm text for older decoder output
        if '.128' in inst.asm_text:
            for ref in inst.reg_refs:
                if ref.is_dest:
                    quad_bases.add(ref.reg_num)
    # Remove pair bases that overlap with any quad group member
    for qb in list(quad_bases):
        for i in range(4):
            pair_bases.discard(qb + i)
    # GPU hardware: pair bases MUST be at even register indices.
    # Odd-numbered "pair bases" are decoder artefacts (cubit decoding errors
    # e.g. IMAD.SHL decoded as IMAD.WIDE). Discard them.
    pair_bases = {r for r in pair_bases if r % 2 == 0}
    return pair_bases, quad_bases


def _enforce_group_highs(
    coloring: dict[int, int],
    pair_bases: set[int],
    quad_bases: set[int],
) -> dict[int, int]:
    """Ensure pair highs = base+1, quad members = base+i in the coloring map."""
    result = dict(coloring)
    for base in quad_bases:
        if base in result:
            for i in range(1, 4):
                result[base + i] = result[base] + i
    for base in pair_bases:
        if base in result:
            result[base + 1] = result[base] + 1
    return result


def dsatur_color(
    graph: dict[int, set[int]],
    num_colors: int,
    pair_bases: Optional[set[int]] = None,
    quad_bases: Optional[set[int]] = None,
    forbidden_colors: Optional[set[int]] = None,
) -> Optional[dict[int, int]]:
    """DSatur graph coloring with coalesced pair/quad register constraints.

    DSatur (Degree of Saturation) dynamically picks the uncolored vertex with
    the highest saturation degree (number of distinct colors used by its
    neighbors), breaking ties by vertex degree. This is significantly better
    than static-order greedy for register interference graphs.

    Coalesced groups handle hardware alignment constraints:
      - Quad base R:  R=C, R+1=C+1, R+2=C+2, R+3=C+3  where C%4==0
      - Pair base R:  R=C, R+1=C+1                      where C%2==0
      - Dependent registers (R+i, i>0) are colored when their base is colored.

    Returns coloring dict {reg: color} or None if infeasible.
    """
    if not graph:
        return {}
    if pair_bases is None:
        pair_bases = set()
    if quad_bases is None:
        quad_bases = set()

    # Only consider bases that are actually in the graph
    quad_bases = {r for r in quad_bases if r in graph}
    pair_bases = {r for r in pair_bases if r in graph}

    # Resolve overlaps: quad supersedes pair
    for qb in list(quad_bases):
        for i in range(4):
            pair_bases.discard(qb + i)

    # --- Build allocation groups ---
    # group_of[reg] = (base, offset, group_size, alignment)
    group_of: dict[int, tuple[int, int, int, int]] = {}

    for base in sorted(quad_bases):
        for i in range(4):
            member = base + i
            if member not in group_of:
                group_of[member] = (base, i, 4, 4)

    for base in sorted(pair_bases):
        if base in group_of:
            continue
        high = base + 1
        if high in group_of:
            continue  # high already in another group — treat base as single
        group_of[base] = (base, 0, 2, 2)
        group_of[high] = (base, 1, 2, 2)

    # --- Classify registers into primaries (color directly) and dependents ---
    primaries: set[int] = set()
    dependents: set[int] = set()

    for reg in graph:
        if reg in group_of:
            _, offset, _, _ = group_of[reg]
            (primaries if offset == 0 else dependents).add(reg)
        else:
            primaries.add(reg)

    coloring: dict[int, int] = {}

    # Pin hardware-reserved registers (R0=stack base, R1=stack pointer)
    for hw_reg in (0, 1):
        if hw_reg in primaries:
            coloring[hw_reg] = hw_reg
            primaries.discard(hw_reg)

    # Pin pair bases EARLY to prevent dependent color conflicts.
    # For each pair base, find a color C where both C and C+1 are free
    # (not used by any neighbor of base or its dependent).
    for base in sorted(pair_bases):
        if base in primaries and base not in coloring:
            high = base + 1
            base_nbrs = graph.get(base, set())
            high_nbrs = graph.get(high, set())
            used_by_base = {coloring[n] for n in base_nbrs if n in coloring}
            used_by_high = {coloring[n] for n in high_nbrs if n in coloring}
            # Try identity first (if even), then scan for any valid even pair
            # Pair bases MUST be colored to even registers (GPU hardware constraint).
            cand_even = []
            if base % 2 == 0:
                cand_even.append(base)
            cand_even += [c for c in range(0, num_colors - 1, 2) if c != base]
            for c in cand_even:
                if c not in used_by_base and (c + 1) not in used_by_high and c + 1 < num_colors:
                    # Also check forbidden colors
                    if forbidden_colors and (c in forbidden_colors or (c + 1) in forbidden_colors):
                        continue
                    coloring[base] = c
                    primaries.discard(base)
                    if high in dependents:
                        coloring[high] = c + 1
                        dependents.discard(high)
                    break

    # --- Saturation tracking ---
    # For each primary: set of "blocked base colors" (for ordering heuristic)
    sat: dict[int, set[int]] = {r: set() for r in primaries}

    # Initialize saturation with ALL pre-pinned colors
    for pinned_reg, pinned_color in coloring.items():
        if pinned_reg in graph:
            for nbr in graph[pinned_reg]:
                if nbr in sat:
                    if nbr in group_of:
                        _, nbr_off, _, _ = group_of[nbr]
                        sat[nbr].add(pinned_color - nbr_off)
                    else:
                        sat[nbr].add(pinned_color)
                elif nbr in group_of:
                    nbr_base, nbr_off, _, _ = group_of[nbr]
                    if nbr_base in sat:
                        sat[nbr_base].add(pinned_color - nbr_off)

    def _update_sat(member: int, member_color: int) -> None:
        """Propagate new coloring to saturation sets of uncolored primaries."""
        if member not in graph:
            return
        for nbr in graph[member]:
            if nbr in sat:
                # nbr is an uncolored primary
                if nbr in group_of:
                    _, nbr_off, _, _ = group_of[nbr]
                    sat[nbr].add(member_color - nbr_off)
                else:
                    sat[nbr].add(member_color)
            elif nbr in group_of:
                # nbr is a dependent — propagate to its base
                nbr_base, nbr_off, _, _ = group_of[nbr]
                if nbr_base in sat:
                    sat[nbr_base].add(member_color - nbr_off)

    # --- Main DSatur loop ---
    while primaries:
        # Pick vertex with max saturation, break ties by total group degree.
        # Pair/quad bases get priority bonus to ensure their dependents
        # are colored early — this prevents conflicts where a dependent's
        # forced color (base_color + offset) collides with an earlier
        # assignment to a non-group register.
        best = max(primaries, key=lambda r: (
            len(sat.get(r, set())),
            1 if r in group_of else 0,  # pair/quad bases first
            sum(
                len(graph.get(r + i, set()))
                for i in range(group_of[r][2] if r in group_of else 1)
            ),
            -r,  # deterministic tie-break
        ))

        if best in group_of:
            _, _, gsize, align = group_of[best]
        else:
            gsize, align = 1, 1

        max_base = num_colors - gsize

        # Compute precisely blocked base colors
        blocked: set[int] = set()
        for i in range(gsize):
            member = best + i
            if member in graph:
                for nbr in graph[member]:
                    if nbr in coloring:
                        blocked.add(coloring[nbr] - i)

        # Find the smallest aligned, unblocked, non-forbidden color
        color = 0
        while color <= max_base:
            if color % align != 0 or color in blocked:
                color += 1
                continue
            # Check forbidden: any color in [color, color+gsize-1] forbidden?
            if forbidden_colors and any((color + i) in forbidden_colors
                                        for i in range(gsize)):
                color += align
                continue
            break

        if color > max_base:
            return None  # infeasible

        # Assign colors to the whole group
        coloring[best] = color
        primaries.discard(best)
        sat.pop(best, None)
        _update_sat(best, color)

        for i in range(1, gsize):
            member = best + i
            coloring[member] = color + i
            dependents.discard(member)
            _update_sat(member, color + i)

    # Color any remaining dependents whose bases were already colored
    for reg in list(dependents):
        if reg not in coloring and reg in group_of:
            base, offset, _, _ = group_of[reg]
            if base in coloring:
                coloring[reg] = coloring[base] + offset

    # --- Validate: fix pair-dependent conflicts ---
    # A dependent's forced color (base_color + offset) may conflict with
    # an earlier assignment if DSatur colored the conflicting register
    # before the pair base.  Fix by recoloring the conflicting register.
    changed = True
    while changed:
        changed = False
        for reg, color in list(coloring.items()):
            if reg not in graph:
                continue
            for nbr in graph[reg]:
                if nbr in coloring and coloring[nbr] == color and nbr != reg:
                    # Conflict! Recolor the non-group member (or the one
                    # that's not a forced dependent).
                    victim = nbr
                    if nbr in group_of and group_of[nbr][1] > 0:
                        victim = reg  # nbr is dependent, recolor reg instead
                    if victim in group_of and group_of[victim][1] > 0:
                        continue  # can't recolor a dependent

                    # Find a free color for victim
                    used = {coloring[n] for n in graph.get(victim, set())
                            if n in coloring and n != victim}
                    for c in range(num_colors):
                        if c not in used:
                            coloring[victim] = c
                            changed = True
                            break

    return coloring


def dsatur_color_multi(
    graph: dict[int, set[int]],
    num_colors: int,
    pair_bases: Optional[set[int]] = None,
    quad_bases: Optional[set[int]] = None,
    attempts: int = 50,
    seed: int = 42,
    forbidden_colors: Optional[set[int]] = None,
) -> Optional[dict[int, int]]:
    """Run DSatur multiple times with randomised tie-breaking.

    Returns the best feasible coloring found, or None.
    """
    rng = random.Random(seed)

    best: Optional[dict[int, int]] = None
    best_max = num_colors + 1

    for attempt in range(attempts):
        c = _dsatur_once(graph, num_colors, pair_bases, quad_bases,
                         rng=rng if attempt > 0 else None,
                         forbidden_colors=forbidden_colors)
        if c is not None:
            mx = max(c.values()) + 1
            if mx < best_max:
                best, best_max = c, mx
            if mx <= num_colors:
                return c  # feasible — done

    return best if best is not None and best_max <= num_colors else None


def _dsatur_once(
    graph: dict[int, set[int]],
    num_colors: int,
    pair_bases: Optional[set[int]] = None,
    quad_bases: Optional[set[int]] = None,
    rng: Optional[random.Random] = None,
    forbidden_colors: Optional[set[int]] = None,
) -> Optional[dict[int, int]]:
    """Single DSatur run. Identical to dsatur_color but with optional rng jitter."""
    if not graph:
        return {}
    if pair_bases is None:
        pair_bases = set()
    if quad_bases is None:
        quad_bases = set()

    quad_bases = {r for r in quad_bases if r in graph}
    pair_bases = {r for r in pair_bases if r in graph}
    for qb in list(quad_bases):
        for i in range(4):
            pair_bases.discard(qb + i)

    group_of: dict[int, tuple[int, int, int, int]] = {}
    for base in sorted(quad_bases):
        for i in range(4):
            if base + i not in group_of:
                group_of[base + i] = (base, i, 4, 4)
    for base in sorted(pair_bases):
        if base in group_of:
            continue
        high = base + 1
        if high in group_of:
            continue
        group_of[base] = (base, 0, 2, 2)
        group_of[high] = (base, 1, 2, 2)

    primaries: set[int] = set()
    dependents: set[int] = set()
    for reg in graph:
        if reg in group_of:
            _, offset, _, _ = group_of[reg]
            (primaries if offset == 0 else dependents).add(reg)
        else:
            primaries.add(reg)

    coloring: dict[int, int] = {}

    # Pin hardware-reserved registers (R0, R1) — same as dsatur_color
    for hw_reg in (0, 1):
        if hw_reg in primaries:
            coloring[hw_reg] = hw_reg
            primaries.discard(hw_reg)
            if hw_reg in graph:
                pass  # saturation updated below

    sat: dict[int, set[int]] = {r: set() for r in primaries}

    # Propagate pinned register colors to saturation sets
    for hw_reg in (0, 1):
        if hw_reg in coloring and hw_reg in graph:
            for nbr in graph[hw_reg]:
                if nbr in sat:
                    if nbr in group_of:
                        _, off, _, _ = group_of[nbr]
                        sat[nbr].add(coloring[hw_reg] - off)
                    else:
                        sat[nbr].add(coloring[hw_reg])

    def _update(member, color):
        if member not in graph:
            return
        for nbr in graph[member]:
            if nbr in sat:
                if nbr in group_of:
                    _, off, _, _ = group_of[nbr]
                    sat[nbr].add(color - off)
                else:
                    sat[nbr].add(color)
            elif nbr in group_of:
                b, off, _, _ = group_of[nbr]
                if b in sat:
                    sat[b].add(color - off)

    plist = list(primaries)
    while plist:
        # Sort by saturation desc, degree desc, with optional jitter
        plist.sort(key=lambda r: (
            -len(sat.get(r, set())),
            -sum(len(graph.get(r + i, set()))
                 for i in range(group_of[r][2] if r in group_of else 1)),
            r,
        ))
        # With rng: randomly pick among the top-k tied candidates
        if rng and len(plist) > 1:
            top_sat = len(sat.get(plist[0], set()))
            tied = [r for r in plist if len(sat.get(r, set())) == top_sat]
            best = rng.choice(tied)
        else:
            best = plist[0]

        plist.remove(best)
        primaries.discard(best)

        if best in group_of:
            _, _, gsize, align = group_of[best]
        else:
            gsize, align = 1, 1
        max_base = num_colors - gsize

        blocked: set[int] = set()
        for i in range(gsize):
            m = best + i
            if m in graph:
                for nbr in graph[m]:
                    if nbr in coloring:
                        blocked.add(coloring[nbr] - i)

        color = 0
        while color <= max_base:
            if color % align != 0 or color in blocked:
                color += 1
                continue
            if forbidden_colors and any((color + i) in forbidden_colors
                                        for i in range(gsize)):
                color += align
                continue
            break
        if color > max_base:
            return None

        coloring[best] = color
        sat.pop(best, None)
        _update(best, color)
        for i in range(1, gsize):
            m = best + i
            coloring[m] = color + i
            dependents.discard(m)
            if m in plist:
                plist.remove(m)
            _update(m, color + i)

    for reg in list(dependents):
        if reg not in coloring and reg in group_of:
            b, off, _, _ = group_of[reg]
            if b in coloring:
                coloring[reg] = coloring[b] + off
    return coloring


def ilp_color(
    graph: dict[int, set[int]],
    num_colors: int,
    pair_bases: Optional[set[int]] = None,
    quad_bases: Optional[set[int]] = None,
    time_limit: float = 120.0,
) -> Optional[dict[int, int]]:
    """Optimal graph coloring via Integer Linear Programming (requires ``mip``).

    Formulates as: for each "allocation unit" (single / pair / quad base),
    choose exactly one base color from the feasible set.  No two neighbours
    may share the same physical-register color.

    Returns coloring dict or None.
    """
    if not HAS_MIP:
        return None
    if not graph:
        return {}
    if pair_bases is None:
        pair_bases = set()
    if quad_bases is None:
        quad_bases = set()

    K = num_colors

    # --- groups (same logic as DSatur) ---
    quad_bases = {r for r in quad_bases if r in graph}
    pair_bases = {r for r in pair_bases if r in graph}
    for qb in list(quad_bases):
        for i in range(4):
            pair_bases.discard(qb + i)

    group_of: dict[int, tuple[int, int, int, int]] = {}
    for base in sorted(quad_bases):
        for i in range(4):
            if base + i not in group_of:
                group_of[base + i] = (base, i, 4, 4)
    for base in sorted(pair_bases):
        if base in group_of:
            continue
        high = base + 1
        if high in group_of:
            continue
        group_of[base] = (base, 0, 2, 2)
        group_of[high] = (base, 1, 2, 2)

    # primary registers: bases and singles
    primaries = []
    for reg in sorted(graph):
        if reg in group_of:
            _, off, _, _ = group_of[reg]
            if off == 0:
                primaries.append(reg)
        else:
            primaries.append(reg)

    # reg meta: group_size, alignment
    def _meta(r):
        if r in group_of:
            return group_of[r][2], group_of[r][3]
        return 1, 1

    # --- ILP model ---
    m = Model(sense='MIN')
    m.verbose = 0
    m.threads = 1  # deterministic for reproducible colorings
    if time_limit:
        m.max_seconds = time_limit

    # x[r][c] = 1  iff primary r gets base color c
    x: dict[int, dict[int, any]] = {}
    for r in primaries:
        gs, al = _meta(r)
        feasible_colors = range(0, K - gs + 1, al)
        x[r] = {c: m.add_var(var_type=BINARY, name=f'x_{r}_{c}')
                 for c in feasible_colors}

    # Each primary gets exactly one color
    for r in primaries:
        m += xsum(x[r].values()) == 1

    # Pin hardware-reserved registers to their physical number.
    # R0 and R1 have fixed values set by the CUDA driver at launch.
    for hw_reg in (0, 1):
        if hw_reg in x and hw_reg in x[hw_reg]:
            m += x[hw_reg][hw_reg] == 1

    # Interference: for each pair of primaries that could conflict
    seen_pairs: set[tuple[int, int]] = set()
    for r1 in primaries:
        gs1, _ = _meta(r1)
        for i in range(gs1):
            member = r1 + i
            if member not in graph:
                continue
            for nbr in graph[member]:
                # find nbr's primary
                if nbr in group_of:
                    r2_base, r2_off, _, _ = group_of[nbr]
                else:
                    r2_base, r2_off = nbr, 0
                if r2_base not in x:
                    continue
                pkey = (min(r1, r2_base), max(r1, r2_base), i, r2_off)
                if pkey in seen_pairs:
                    continue
                seen_pairs.add(pkey)
                if r1 == r2_base:
                    continue  # intra-group: handled by consecutive assignment
                # Conflict: color(r1) + i  !=  color(r2_base) + r2_off
                # => for each (c1, c2) in feasible: if c1+i == c2+r2_off, add constraint
                for c1 in x[r1]:
                    needed = c1 + i - r2_off
                    if needed in x[r2_base]:
                        m += x[r1][c1] + x[r2_base][needed] <= 1

    # Minimise max color used (auxiliary variable)
    z = m.add_var(name='z')
    for r in primaries:
        gs, _ = _meta(r)
        for c in x[r]:
            m += z >= (c + gs - 1) * x[r][c]

    m.objective = z

    status = m.optimize()
    if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
        return None

    coloring: dict[int, int] = {}
    for r in primaries:
        for c, var in x[r].items():
            if var.x is not None and var.x > 0.5:
                coloring[r] = c
                gs, _ = _meta(r)
                for j in range(1, gs):
                    coloring[r + j] = c + j
                break

    if max(coloring.values(), default=0) >= K:
        return None
    return coloring


def _spill_reload_cost(reg: int, instructions: list[Instruction]) -> int:
    """Estimate the cost (in cycles) of spilling and reloading a register.

    A register used as a source by expensive (high-latency) instructions
    is more costly to spill because the reload LDS adds to the critical
    path.  Registers used only by cheap ALU ops are cheaper to spill.

    Returns a weighted cost: sum of def_latency for all instructions that
    read this register.  Lower = cheaper to spill.
    """
    try:
        from sasskit.core.isa_db import get_db
        db = get_db()
    except Exception:
        # Fallback: no cost awareness, return 0 (neutral)
        return 0

    cost = 0
    for inst in instructions:
        if reg in inst.src_regs:
            base_op = inst.asm_text.split()[0].split(".")[0]
            info = db.get(base_op)
            # Variable-latency consumers (memory ops) are cheap to add
            # reload latency to — they already wait on scoreboard.
            # Fixed-latency consumers (ALU) are expensive — reload adds
            # to the critical path.
            if not info.is_variable_latency:
                cost += info.def_latency
    return cost


def identify_spill_candidates(instructions: list[Instruction],
                              blocks: list[BasicBlock],
                              target_regs: int,
                              hot_loop_range: Optional[tuple[int, int]] = None
                              ) -> list[SpillDecision]:
    """Identify registers to spill when K-coloring fails.

    Strategy: spill registers with the least hot-loop impact, weighted
    by the cost of reloading (from the ISA database).

    Priority for spilling (most spillable first):
    1. Registers used only outside the hot loop
    2. Registers with low reload cost (used by variable-latency ops)
    3. Registers used <=2 times in the hot loop (cheap to reload)
    4. Loop-invariant registers (defined once, never written in loop)
    """
    live_at = compute_live_at(blocks)

    # Count uses per register, separated by hot/cold
    hot_uses: dict[int, int] = {}
    cold_uses: dict[int, int] = {}
    reg_defs: dict[int, list[int]] = {}
    reg_uses: dict[int, list[int]] = {}

    for inst in instructions:
        in_hot = False
        if hot_loop_range:
            in_hot = hot_loop_range[0] <= inst.code_offset <= hot_loop_range[1]

        for reg in inst.dest_regs:
            reg_defs.setdefault(reg, []).append(inst.code_offset)

        for reg in inst.src_regs:
            reg_uses.setdefault(reg, []).append(inst.code_offset)
            if in_hot:
                hot_uses[reg] = hot_uses.get(reg, 0) + 1
            else:
                cold_uses[reg] = cold_uses.get(reg, 0) + 1

    # Find current max pressure
    max_pressure = max(len(s) for s in live_at.values()) if live_at else 0
    regs_to_spill = max_pressure - target_regs

    if regs_to_spill <= 0:
        return []

    # Pre-compute spill reload cost per register
    reload_cost: dict[int, int] = {}
    all_regs = set()
    for s in live_at.values():
        all_regs |= s
    for reg in all_regs:
        reload_cost[reg] = _spill_reload_cost(reg, instructions)

    # Score: lower = more suitable for spilling.
    # Primary: hot_uses (fewer = better).
    # Secondary: reload_cost (lower = cheaper to spill).
    # Tertiary: total uses (fewer = better).
    # Quaternary: higher register number first.
    candidates = sorted(all_regs, key=lambda r: (
        hot_uses.get(r, 0),       # Fewest hot uses first
        reload_cost.get(r, 0),    # Cheapest reload first
        len(reg_uses.get(r, [])), # Fewest total uses
        -r,                       # Higher register numbers first
    ))
    
    spills: list[SpillDecision] = []
    smem_offset = 0
    
    for reg in candidates:
        if len(spills) >= regs_to_spill:
            break
        
        # Skip registers that are too fundamental to spill (R0, R1, RZ)
        if reg <= 1:
            continue
        
        # Find a dead register for reloads at the use points
        # (we'll determine this more precisely during patching)
        reload_reg = -1  # Placeholder
        
        spills.append(SpillDecision(
            reg=reg,
            reason=f"hot_uses={hot_uses.get(reg, 0)}, total_uses={len(reg_uses.get(reg, []))}",
            def_offsets=reg_defs.get(reg, []),
            use_offsets=reg_uses.get(reg, []),
            smem_slot=smem_offset,
            reload_reg=reload_reg,
        ))
        smem_offset += 8  # 8 bytes per 64-bit register pair
    
    return spills


def plan_recoloring(instructions: list[Instruction],
                    blocks: list[BasicBlock],
                    target_regs: int,
                    hot_loop_range: Optional[tuple[int, int]] = None
                    ) -> ColoringResult:
    """Plan a complete register re-coloring.

    Strategy (in priority order):
    1. DSatur coloring with coalesced pair/quad alignment constraints
    2. If DSatur fails: iterative spill + DSatur retry
    3. Fallback: greedy coloring with spill+retry

    The coloring maps original register numbers -> new register numbers.
    All new registers will be < target_regs.
    """
    from sasskit.analysis.interference import build_interference_graph

    live_at = compute_live_at(blocks)
    max_pressure = max(len(s) for s in live_at.values()) if live_at else 0

    # Build interference graph (pass instructions to enable R0/R1 latency extension)
    graph = build_interference_graph(blocks, instructions=instructions)

    # Detect alignment-constrained register groups
    pair_bases, quad_bases = _detect_register_groups(instructions)

    # ------------------------------------------------------------------
    # SM 120 hardware reserves the top 2 registers from the declared
    # EIATTR_REGCOUNT for internal scoreboard/predicate state.
    # ALL coloring attempts must use (target_regs - 2) as the palette
    # size, even when no spills are needed.
    # ------------------------------------------------------------------
    SM120_HW_RESERVED = 2
    usable_K = target_regs - SM120_HW_RESERVED   # max color = usable_K - 1

    def _validate_coloring(coloring: dict[int, int]) -> bool:
        """Check that no two interfering registers share a color."""
        for reg, color in coloring.items():
            for nbr in graph.get(reg, set()):
                if nbr in coloring and coloring[nbr] == color and nbr != reg:
                    return False
        return True

    # ------------------------------------------------------------------
    # Attempt 1: DSatur (primary algorithm — much better than greedy)
    # ------------------------------------------------------------------
    coloring = dsatur_color(graph, usable_K, pair_bases, quad_bases)

    if coloring is not None and _validate_coloring(coloring):
        coloring = _enforce_group_highs(coloring, pair_bases, quad_bases)
        return ColoringResult(
            success=True,
            target_regs=target_regs,
            coloring=coloring,
            spilled=[],
            max_pressure_before=max_pressure,
            max_pressure_after=max(coloring.values()) + 1 if coloring else 0,
        )

    # ------------------------------------------------------------------
    # Attempt 1b: randomised DSatur (many restarts with jitter)
    # ------------------------------------------------------------------
    coloring = dsatur_color_multi(graph, usable_K, pair_bases, quad_bases,
                                  attempts=200, seed=42)

    if coloring is not None and _validate_coloring(coloring):
        coloring = _enforce_group_highs(coloring, pair_bases, quad_bases)
        return ColoringResult(
            success=True,
            target_regs=target_regs,
            coloring=coloring,
            spilled=[],
            max_pressure_before=max_pressure,
            max_pressure_after=max(coloring.values()) + 1 if coloring else 0,
        )

    # ------------------------------------------------------------------
    # Attempt 1c: ILP (optimal solver — finds global optimum)
    # ------------------------------------------------------------------
    if HAS_MIP:
        coloring = ilp_color(graph, usable_K, pair_bases, quad_bases,
                             time_limit=10.0)
        if coloring is not None and _validate_coloring(coloring):
            coloring = _enforce_group_highs(coloring, pair_bases, quad_bases)
            return ColoringResult(
                success=True,
                target_regs=target_regs,
                coloring=coloring,
                spilled=[],
                max_pressure_before=max_pressure,
                max_pressure_after=max(coloring.values()) + 1 if coloring else 0,
            )

    # ------------------------------------------------------------------
    # Attempt 2: iterative spill + coloring (ILP -> DSatur)
    # ------------------------------------------------------------------
    # Strategy: spill cheapest registers first.  Two tiers of candidates:
    #   Tier 1: "singles" — not part of any pair/quad group (cheapest)
    #   Tier 2: pair bases + their highs (more expensive, but may be needed)
    # Never spill quad bases/members (hardware 4-reg writes can't be avoided)
    # or R0/R1 (fundamental).
    #
    # When spills are present, we reserve the highest physical register
    # (target_regs - 1) as a persistent shared-memory base register for
    # trampoline LDS/STS addressing.  The coloring therefore targets
    # effective_K = target_regs - 1 colours.
    # ------------------------------------------------------------------
    # SM 120 hardware reserves 2 registers above the highest used register
    # for internal scoreboard/predicate state.  To declare EIATTR_REGCOUNT=N,
    # we must ensure max_used_register <= N - 3.  So we color with N - 3
    # colours (0 .. N-4) and place the smem base at colour N - 3.
    effective_K = target_regs - 3      # colours 0 .. target_regs-4
    smem_base_reg = target_regs - 3    # physical register for the base

    rc: dict[int, int] = {}
    for inst in instructions:
        for r in inst.all_regs:
            rc[r] = rc.get(r, 0) + 1

    # R0 and R1 have hardware-fixed values (R1 = stack pointer, set by
    # the CUDA driver at kernel launch).  They MUST keep their physical
    # register numbers.  Add identity constraints for them.
    # Also protect smem_base_reg: this register holds tid*4 for LDS/STS
    # addressing throughout the entire kernel.  If the virtual register
    # with the same number as smem_base_reg were spilled, its trampoline
    # would overwrite smem_base_reg with the spilled value, corrupting
    # all subsequent LDS/STS addresses.
    core_protected: set[int] = quad_bases | {0, 1, smem_base_reg}
    for qb in quad_bases:
        for i in range(4):
            core_protected.add(qb + i)

    # Tier 1: singles (no pair/quad membership)
    pair_all = set()
    for pb in pair_bases:
        pair_all.add(pb)
        pair_all.add(pb + 1)

    tier1 = sorted(
        (r for r in graph if r not in core_protected and r not in pair_all),
        key=lambda r: (rc.get(r, 0), -r),
    )
    # Tier 2: pair bases (spill base + high together), cheapest first
    tier2_bases = sorted(
        pair_bases,
        key=lambda r: rc.get(r, 0) + rc.get(r + 1, 0),
    )

    spills: list[SpillDecision] = []
    spilled_set: set[int] = set()
    smem_offset = 0
    coloring = None

    def _try_color(reduced_graph):
        """Try DSatur on the reduced graph (fast, used inside spill loop).

        Forbidden-color conflicts (non-spilled reg colored to spilled reg's
        physical number) are resolved in post-processing (line ~1369), not here.
        Applying forbidden colors during the spill loop is too restrictive when
        many registers are spilled — it starves the color palette.
        """
        active_pairs = pair_bases - spilled_set
        c = dsatur_color(reduced_graph, effective_K, active_pairs, quad_bases)
        if c is None:
            c = dsatur_color_multi(reduced_graph, effective_K, active_pairs,
                                   quad_bases, attempts=100, seed=42)
        return c

    def _add_spill(reg):
        if reg in core_protected:
            return  # Never spill hardware-fixed or smem_base registers
        rd = [i.code_offset for i in instructions if reg in i.dest_regs]
        ru = [i.code_offset for i in instructions if reg in i.src_regs]
        spills.append(SpillDecision(
            reg=reg, reason=f"spill(uses={rc.get(reg, 0)})",
            def_offsets=rd, use_offsets=ru,
            smem_slot=smem_offset, reload_reg=-1,
        ))
        spilled_set.add(reg)

    # Build a combined candidate list: tier1 first, then tier2 (pairs)
    all_cands: list[list[int]] = [[r] for r in tier1]  # singles
    for pb in tier2_bases:
        all_cands.append([pb, pb + 1])                  # pairs

    for cand_group in all_cands:
        if len(spills) >= 30:
            break
        for reg in cand_group:
            if reg in spilled_set:
                continue
            _add_spill(reg)
        smem_offset += 8 * len(cand_group)

        reduced = {
            r: {n for n in nb if n not in spilled_set}
            for r, nb in graph.items() if r not in spilled_set
        }
        coloring = _try_color(reduced)
        if coloring is not None and _validate_coloring(coloring):
            break

    # ------------------------------------------------------------------
    # Re-insert spilled registers into coloring.
    # Phase A: try global re-insertion (non-conflicting color exists)
    # Phase B: for remaining spills, find a register that is dead at
    #          ALL the spill's use/def instruction points — this avoids
    #          trampolines entirely by doing instruction-level renames.
    # ------------------------------------------------------------------
    true_spills: list[SpillDecision] = []
    if coloring is not None and _validate_coloring(coloring):
        # Phase A: global re-insertion
        # IMPORTANT: process pair/quad BASES first, then singles, then
        # highs.  This ensures pair bases get their even color assigned
        # BEFORE the high is processed.
        remaining_spills: list[SpillDecision] = []

        # Identify which spilled regs are pair/quad bases vs highs vs singles
        spill_bases = [sp for sp in spills if sp.reg in pair_bases or sp.reg in quad_bases]
        spill_singles = [sp for sp in spills
                         if sp.reg not in pair_bases
                         and sp.reg not in quad_bases
                         and not any(sp.reg == pb + i
                                     for pb in pair_bases for i in range(1, 2))
                         and not any(sp.reg == qb + i
                                     for qb in quad_bases for i in range(1, 4))]
        spill_highs = [sp for sp in spills
                       if sp not in spill_bases and sp not in spill_singles]
        ordered_spills = spill_bases + spill_singles + spill_highs

        for sp in ordered_spills:
            reg = sp.reg

            # If this register is a group DEPENDENT (pair high / quad+1/+2/+3)
            # and its base is already colored, just inherit base + offset.
            for pb in pair_bases:
                if reg == pb + 1 and pb in coloring:
                    coloring[reg] = coloring[pb] + 1
                    break
            else:
                for qb in quad_bases:
                    for qi in range(1, 4):
                        if reg == qb + qi and qb in coloring:
                            coloring[reg] = coloring[qb] + qi
                            break
                    else:
                        continue
                    break
                else:
                    pass  # not a dependent with colored base — proceed below
            if reg in coloring:
                continue  # Already handled as dependent

            # If this is a group dependent whose base is NOT yet colored,
            # DEFER it — don't assign a standalone color.
            is_dependent = False
            for pb in pair_bases:
                if reg == pb + 1:
                    is_dependent = True
                    break
            if not is_dependent:
                for qb in quad_bases:
                    for qi in range(1, 4):
                        if reg == qb + qi:
                            is_dependent = True
                            break
                    if is_dependent:
                        break
            if is_dependent:
                remaining_spills.append(sp)
                continue

            # Standard re-insertion for bases and singles
            align = (4 if reg in quad_bases
                     else 2 if reg in pair_bases
                     else 1)
            gsize = (4 if reg in quad_bases
                     else 2 if reg in pair_bases
                     else 1)
            mc = effective_K - gsize

            blocked: set[int] = set()
            for i in range(gsize):
                member = reg + i
                if member in graph:
                    for nbr in graph[member]:
                        if nbr in coloring:
                            blocked.add(coloring[nbr] - i)

            c = 0
            found = False
            # true_spills_set: spilled registers that still need trampolines
            # (can't be colors for other registers)
            final_forbidden = {s for s in spilled_set if s < effective_K}
            while c <= mc:
                if c % align != 0 or c in blocked:
                    c += 1
                    continue
                if final_forbidden and any((c + i) in final_forbidden
                                           for i in range(gsize)):
                    c += align
                    continue
                found = True
                break

            if found:
                coloring[reg] = c
                for i in range(1, gsize):
                    coloring[reg + i] = c + i
            else:
                remaining_spills.append(sp)

        # Phase B: local dead-color analysis (point-wise).
        # For registers where no GLOBAL color is free, check if a color
        # is dead at every instruction that references the register.
        # This works when the spilled register's live range is dominated
        # by regions where some color is unused.
        if remaining_spills:
            live_at = compute_live_at(blocks)

        still_remaining: list[SpillDecision] = []
        for sp in remaining_spills:
            reg = sp.reg

            # Skip dependents whose base is already colored —
            # they'll be fixed by _enforce_group_highs
            is_dep = False
            for pb in pair_bases:
                if reg == pb + 1 and pb in coloring:
                    coloring[reg] = coloring[pb] + 1
                    is_dep = True
                    break
            if is_dep:
                continue

            align = 2 if reg in pair_bases else 1
            gsize = 2 if reg in pair_bases else 1
            # Build feasible set, excluding forbidden colors (spilled regs' physical numbers)
            phase_b_forbidden = {s for s in spilled_set if s < effective_K}
            feasible = set(c for c in range(0, effective_K - gsize + 1, align)
                           if not any((c + i) in phase_b_forbidden for i in range(gsize)))

            # Must check ALL offsets where the register is LIVE,
            # not just where it's referenced. A register is live
            # between its def and its last use — at intermediate
            # instructions it's not referenced but still holds a
            # value that must not be clobbered.
            live_offsets = [
                off for off, ls in live_at.items()
                if reg in ls
            ]
            if not live_offsets:
                continue

            dead_at_all = feasible.copy()
            for off in live_offsets:
                live = live_at.get(off, set())
                live_colors: set[int] = set()
                for r in live:
                    if r in coloring:
                        live_colors.add(coloring[r])
                blocked_bases: set[int] = set()
                for c in feasible:
                    for i in range(gsize):
                        if (c + i) in live_colors:
                            blocked_bases.add(c)
                            break
                dead_at_all -= blocked_bases

            if dead_at_all:
                # Filter out colors that conflict in the interference graph
                # (pair/quad extensions may create interference beyond basic liveness)
                safe = set()
                for c in dead_at_all:
                    ok = True
                    for i in range(gsize):
                        member = reg + i
                        member_color = c + i
                        for nbr in graph.get(member, set()):
                            if nbr in coloring and coloring[nbr] == member_color:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        safe.add(c)
                if safe:
                    choice = min(safe)
                    coloring[reg] = choice
                    for i in range(1, gsize):
                        coloring[reg + i] = choice + i
                else:
                    still_remaining.append(sp)
            else:
                still_remaining.append(sp)

        # Phase C: live-range splitting for truly stubborn registers.
        # Split the register into separate def-use segments and assign
        # each segment a DIFFERENT local color.  The patcher will apply
        # per-instruction renames.  We store these as a mapping
        #   (register, code_offset) -> local_color
        # in the result's coloring (negative-offset keys are per-instr).
        per_instr_coloring: dict[tuple[int, int], int] = {}
        phase_c_forbidden = {s for s in spilled_set if s < effective_K}
        for sp in still_remaining:
            reg = sp.reg
            align = 2 if reg in pair_bases else 1
            gsize = 2 if reg in pair_bases else 1
            feasible = set(c for c in range(0, effective_K - gsize + 1, align)
                           if not any((c + i) in phase_c_forbidden for i in range(gsize)))

            # Build def-use segments
            ref_list = [
                (inst.code_offset,
                 'def' if reg in inst.dest_regs else 'use',
                 inst)
                for inst in instructions if reg in inst.all_regs
            ]
            segments: list[list[tuple[int, str, object]]] = []
            cur: list[tuple[int, str, object]] = []
            for off, kind, inst in ref_list:
                if kind == 'def' and cur:
                    if reg in inst.src_regs:
                        cur.append((off, kind, inst))
                        continue
                    segments.append(cur)
                    cur = []
                cur.append((off, kind, inst))
            if cur:
                segments.append(cur)

            all_ok = True
            for seg in segments:
                # Check ALL offsets where the register is live within
                # this segment's range (first ref offset to last ref
                # offset), not just where it's referenced.
                seg_ref_offsets = [s[0] for s in seg]
                seg_min = min(seg_ref_offsets)
                seg_max = max(seg_ref_offsets)
                seg_live_offsets = [
                    off for off, ls in live_at.items()
                    if reg in ls and seg_min <= off <= seg_max
                ]
                if not seg_live_offsets:
                    seg_live_offsets = seg_ref_offsets

                dead = feasible.copy()
                for off in seg_live_offsets:
                    live = live_at.get(off, set())
                    lc = {coloring[r] for r in live
                          if r in coloring}
                    blocked_b: set[int] = set()
                    for c in feasible:
                        for i in range(gsize):
                            if (c + i) in lc:
                                blocked_b.add(c)
                                break
                    dead -= blocked_b

                if dead:
                    # Filter out colors that conflict in interference graph
                    safe_dead = set()
                    for c in dead:
                        ok = True
                        for ii in range(gsize):
                            member = reg + ii
                            member_color = c + ii
                            for nbr in graph.get(member, set()):
                                if nbr in coloring and coloring[nbr] == member_color:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            safe_dead.add(c)
                    dead = safe_dead
                if dead:
                    c = min(dead)
                    for off, _, _ in seg:
                        per_instr_coloring[(reg, off)] = c
                        if gsize > 1:
                            per_instr_coloring[(reg + 1, off)] = c + 1
                else:
                    all_ok = False
                    break

            if all_ok:
                # Use the most common local color as the "global" entry
                local_colors = [per_instr_coloring[(reg, seg[0][0])]
                                for seg in segments]
                coloring[reg] = local_colors[0]
                for i in range(1, gsize):
                    coloring[reg + i] = local_colors[0] + i
            else:
                true_spills.append(sp)

        coloring = _enforce_group_highs(coloring, pair_bases, quad_bases)
        spills = true_spills

    # Post-processing: ensure no non-spilled register is colored to a spilled
    # register's original physical number.  A spilled register R_s uses
    # trampolines; its physical number R_s appears in the binary for all its
    # def/use instructions.  If another register V is colored to R_s, the
    # trampoline would incorrectly rename V's binary R_s references as well.
    if coloring is not None:
        coloring = _enforce_group_highs(coloring, pair_bases, quad_bases)

    has_spills = coloring is not None and bool(spills)
    return ColoringResult(
        success=coloring is not None,
        target_regs=target_regs,
        coloring=coloring or {},
        spilled=spills if coloring is not None else [],
        max_pressure_before=max_pressure,
        max_pressure_after=(
            max(coloring.values()) + 1 if coloring else max_pressure
        ),
        smem_base_reg=smem_base_reg if has_spills else -1,
    )
