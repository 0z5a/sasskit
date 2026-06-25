"""Fuzz-based recoloring: randomly generate register assignments, patch, test.

Given a cubin and target register count, this module:
1. Performs liveness analysis and builds the interference graph
2. Generates random valid K-colorings of the interference graph
3. Patches the cubin with each random coloring
4. Tests the result on the GPU via the sass_test harness
5. Reports pass/fail statistics and saves passing variants

This is useful for:
- Validating that the patcher handles diverse register mappings
- Finding corner cases in instruction encoding
- Building confidence that the coloring approach is correct
"""

from __future__ import annotations

import copy
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sasskit.core.cubin import Cubin
from sasskit.core.decoder import decode_kernel
from sasskit.analysis.cfg import build_cfg
from sasskit.analysis.liveness import compute_liveness, find_max_pressure
from sasskit.analysis.interference import build_interference_graph
from sasskit.recolor.coloring import (
    greedy_color,
    dsatur_color,
    plan_recoloring,
    ColoringResult,
)


@dataclass
class FuzzResult:
    """Aggregate result from a fuzz campaign."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    crashed: int = 0
    timed_out: int = 0
    patch_errors: int = 0
    elapsed_seconds: float = 0.0
    best_variant: Optional[str] = None   # path to best passing variant

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"Fuzz results: {self.total} variants",
            f"  PASS: {self.passed}  FAIL: {self.failed}  "
            f"CRASH: {self.crashed}  TIMEOUT: {self.timed_out}  "
            f"PATCH_ERR: {self.patch_errors}",
            f"  Pass rate: {self.pass_rate:.1%}",
            f"  Elapsed: {self.elapsed_seconds:.1f}s",
        ]
        if self.best_variant:
            lines.append(f"  Best variant: {self.best_variant}")
        return "\n".join(lines)


def random_valid_coloring(
    graph: dict[int, set[int]],
    num_colors: int,
    fixed: Optional[dict[int, int]] = None,
    seed: Optional[int] = None,
) -> Optional[dict[int, int]]:
    """Generate a random valid K-coloring of an interference graph.

    Uses a randomized greedy approach: process nodes in random order,
    for each node pick a random valid color from the available ones.

    Args:
        graph: interference graph {reg: set(conflicting_regs)}
        num_colors: K — number of colors (target register count)
        fixed: registers with pre-assigned colors (e.g., R0 always → 0)
        seed: random seed for reproducibility

    Returns:
        A coloring dict {old_reg: new_reg}, or None if coloring failed.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    coloring: dict[int, int] = {}
    if fixed:
        coloring.update(fixed)

    # Process nodes in random order
    nodes = list(graph.keys())
    rng.shuffle(nodes)

    for node in nodes:
        if node in coloring:
            continue

        # Find colors used by neighbors
        used = {coloring[nb] for nb in graph[node] if nb in coloring}
        available = [c for c in range(num_colors) if c not in used]

        if not available:
            return None  # coloring failed

        coloring[node] = rng.choice(available)

    return coloring


def run_fuzz(
    cubin_path: str,
    kernel_name: str,
    target_regs: int,
    num_variants: int = 100,
    save_dir: Optional[str] = None,
    sass_test_bin: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> FuzzResult:
    """Run a fuzz campaign on a cubin.

    Args:
        cubin_path: path to the original cubin
        kernel_name: kernel function name
        target_regs: target register count (K)
        num_variants: how many random colorings to try
        save_dir: directory to save passing variants (None = don't save)
        sass_test_bin: path to sass_test binary (auto-detect if None)
        seed: random seed
        verbose: print progress to stdout

    Returns:
        FuzzResult with aggregate statistics.
    """
    from sasskit.core.testing import test_cubin_bytes
    from sasskit.recolor.patcher import apply_patch_plan

    result = FuzzResult()
    t0 = time.time()

    # Load and analyze
    cubin = Cubin.from_file(cubin_path)
    instructions = decode_kernel(cubin, kernel_name)
    blocks = build_cfg(instructions)
    compute_liveness(blocks, instructions=instructions)

    offset, pressure, live_set = find_max_pressure(blocks)
    if verbose:
        print(f"Kernel: {kernel_name}")
        print(f"  Max pressure: {pressure}, target: {target_regs}")

    # Build interference graph
    interference = build_interference_graph(blocks, instructions)

    # Collect all registers in the graph
    all_regs = set(interference.keys())
    for nbrs in interference.values():
        all_regs |= nbrs

    # R0 is always fixed to 0 (it's the zero register / return value)
    fixed = {0: 0}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    rng_seed = seed
    best_time = float("inf")

    for i in range(num_variants):
        result.total += 1

        # Generate a random coloring
        coloring = random_valid_coloring(
            interference, target_regs, fixed=fixed, seed=rng_seed + i,
        )
        if coloring is None:
            result.patch_errors += 1
            if verbose:
                print(f"  [{i+1}/{num_variants}] coloring failed (infeasible)")
            continue

        # Ensure all registers are in the coloring
        for r in all_regs:
            if r not in coloring:
                coloring[r] = r if r < target_regs else 0

        # Create a ColoringResult for the patcher
        cr = ColoringResult(
            success=True,
            target_regs=target_regs,
            coloring=coloring,
            spilled=[],
            max_pressure_before=pressure,
            max_pressure_after=max(
                len({coloring.get(r, r) for r in live_set}),
                0,
            ),
        )

        # Patch
        try:
            cubin_copy = Cubin.from_file(cubin_path)
            instrs_copy = decode_kernel(cubin_copy, kernel_name)
            blocks_copy = build_cfg(instrs_copy)
            compute_liveness(blocks_copy, instructions=instrs_copy)

            apply_patch_plan(
                cubin_copy, kernel_name, cr, instrs_copy,
                target_regs, blocks=blocks_copy,
            )
            patched_bytes = cubin_copy.to_bytes()
        except Exception as e:
            result.patch_errors += 1
            if verbose:
                print(f"  [{i+1}/{num_variants}] patch error: {e}")
            continue

        # Test on GPU
        status, detail = test_cubin_bytes(
            patched_bytes,
            kernel_name=kernel_name,
            sass_test_bin=sass_test_bin,
        )

        if status == "PASS":
            result.passed += 1
            tag = "PASS"
            if save_dir:
                variant_path = os.path.join(save_dir, f"variant_{i:04d}.cubin")
                with open(variant_path, "wb") as f:
                    f.write(patched_bytes)
                # Track best by perf time if available
                import re
                m = re.search(r'per_iter=([\d.]+)ms', detail)
                if m:
                    t = float(m.group(1))
                    if t < best_time:
                        best_time = t
                        result.best_variant = variant_path
                elif result.best_variant is None:
                    result.best_variant = variant_path
        elif status == "FAIL":
            result.failed += 1
            tag = "FAIL"
        elif status == "CRASH":
            result.crashed += 1
            tag = "CRASH"
        elif status == "TIMEOUT":
            result.timed_out += 1
            tag = "TIMEOUT"
        else:
            result.failed += 1
            tag = status

        if verbose:
            renames = sum(1 for old, new in coloring.items() if old != new)
            print(f"  [{i+1}/{num_variants}] {tag} "
                  f"(renames={renames}) {detail}")

    result.elapsed_seconds = time.time() - t0
    return result
