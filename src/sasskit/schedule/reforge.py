"""SASS-to-SASS optimizer via empirical mutation + GPU benchmarking.

Takes a compiled cubin, mutates SASS instructions (reorder, peephole,
stall tuning, register rename), encodes back, and benchmarks on GPU.
Accepts improvements, rejects regressions.  Hill-climbing with
simulated annealing.

Pipeline per iteration:
  1. Pick random mutation (reorder, peephole, stall, rename, substitute)
  2. Apply mutation to instruction list
  3. Encode via cubit → patched cubin
  4. Quick GPU test (crash detection)
  5. If PASS: benchmark throughput
  6. Accept if faster (or SA probability), else revert
"""

from __future__ import annotations

import math
import os
import random
import re
import shutil
import tempfile
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from sasskit.core.cubin import Cubin, KernelInfo
from sasskit.core.decoder import Instruction, decode_kernel
from sasskit.core.isa import build_ctrl

try:
    import cubit as _cubit
except ImportError:
    _cubit = None

try:
    from sasskit.core.isa_db import get_db as _get_isa_db
except ImportError:
    _get_isa_db = None


class MutationType(Enum):
    SWAP_ADJACENT = auto()
    SWAP_LOAD_UP = auto()
    STALL_TWEAK = auto()
    NOP_REMOVE = auto()
    NOP_INSERT = auto()
    # Future: PEEPHOLE, REG_RENAME, INSTR_SUBSTITUTE


@dataclass
class Mutation:
    """A single mutation applied to the instruction stream."""
    kind: MutationType
    block_idx: int
    instr_idx: int
    detail: str = ""


@dataclass
class ReforgeState:
    """Current instructions/CFG and timing; best_time_ms describes the best file."""
    cubin_path: str
    kernel_name: str
    instructions: list[Instruction]
    blocks: list  # BasicBlock list
    best_time_ms: float = float('inf')
    current_time_ms: float = float('inf')
    iteration: int = 0
    accepted: int = 0
    rejected: int = 0
    history: list[tuple[int, str, float]] = field(default_factory=list)
    best_cubin_path: str | None = None
    generation: int = 0


def _is_memory_load(inst: Instruction) -> bool:
    op = inst.opcode.split('.')[0]
    return op in ('LDG', 'LDS', 'LDL', 'LDGSTS', 'LDC', 'LDCU', 'TEX')


def _is_memory_store(inst: Instruction) -> bool:
    op = inst.opcode.split('.')[0]
    return op in ('STG', 'STS', 'STL')


def _is_control(inst: Instruction) -> bool:
    op = inst.opcode.split('.')[0]
    return op in ('BRA', 'EXIT', 'RET', 'BSSY', 'BSYNC', 'BAR',
                  'DEPBAR', 'WARPSYNC', 'BREAK', 'CONT')


def _has_raw_dep(producer: Instruction, consumer: Instruction) -> bool:
    """Check if consumer reads a register written by producer."""
    dests = set(producer.dest_regs)
    srcs = set(consumer.src_regs)
    return bool(dests & srcs)


def _has_war_dep(first: Instruction, second: Instruction) -> bool:
    """Check if second writes a register read by first."""
    srcs = set(first.src_regs)
    dests = set(second.dest_regs)
    return bool(srcs & dests)


def _has_waw_dep(first: Instruction, second: Instruction) -> bool:
    """Check if both write to the same register."""
    d1 = set(first.dest_regs)
    d2 = set(second.dest_regs)
    return bool(d1 & d2)


def _is_discard_move(inst: Instruction) -> bool:
    """Recognize only the audited SM120 MOV RZ, imm32 form.

    A discard destination avoids live-register hazards with instructions outside
    the pair. Live destinations need a scheduling model before admission.
    Encoding: Cubit sm120 MOV_R_II (no modifiers); control layout: build_ctrl.
    """
    if inst.opcode != 'MOV' or inst.is_predicated or inst.predicate or inst.reg_refs:
        return False
    match = re.fullmatch(
        r'MOV RZ, (0x[0-9a-fA-F]+|0|[1-9][0-9]*)'
        r'(?: /\* @sched 0x([0-9a-fA-F]+) \*/)?', inst.asm_text)
    if match is None:
        return False
    if match[2] and int(match[2], 16) != (inst.ctrl_word >> 41) & 0x1FFFF:
        return False
    immediate = int(match[1], 0)
    if not 0 <= immediate <= 0xFFFFFFFF:
        return False
    if inst.instr_word != (immediate << 32) | (255 << 16) | 0x7802:
        return False
    # No reuse, wait mask, barrier assignment, or unexplained modifier bits.
    stall = (inst.ctrl_word >> 41) & 15
    return inst.ctrl_word in (
        build_ctrl(stall=stall, yield_hint=True) | 0xF00,
        build_ctrl(stall=stall, yield_hint=False) | 0xF00,
    )


def _can_swap(inst_a: Instruction, inst_b: Instruction) -> bool:
    """Allow only effect-free moves with identical scheduling controls.

    GPR independence alone does not prove memory, predicate, or surrounding
    latency dependencies. Other instruction forms are deliberately unsupported.
    """
    return (_is_discard_move(inst_a) and _is_discard_move(inst_b)
            and inst_a.ctrl_word == inst_b.ctrl_word)


def _get_block_instructions(instructions: list[Instruction],
                            block) -> list[Instruction]:
    # CFG end_offset is exclusive; the block already owns the exact slice.
    return block.instructions


# ============================================================================
# Mutation generators
# ============================================================================

def gen_swap_adjacent(instructions: list[Instruction],
                      blocks: list) -> Optional[Mutation]:
    """Pick two adjacent non-control instructions and swap if safe."""
    block_idx = random.randrange(len(blocks))
    blk = _get_block_instructions(instructions, blocks[block_idx])
    if len(blk) < 3:
        return None
    idx = random.randrange(1, len(blk) - 1)
    if _can_swap(blk[idx], blk[idx + 1]):
        return Mutation(MutationType.SWAP_ADJACENT, block_idx, idx,
                        f"swap {blk[idx].opcode} ↔ {blk[idx+1].opcode}")
    return None


def gen_swap_load_up(instructions: list[Instruction],
                     blocks: list) -> Optional[Mutation]:
    """Move a memory load instruction earlier in the block."""
    block_idx = random.randrange(len(blocks))
    blk = _get_block_instructions(instructions, blocks[block_idx])
    loads = [(j, i) for j, i in enumerate(blk)
             if _is_memory_load(i) and j > 1]
    if not loads:
        return None
    j, load = random.choice(loads)
    if _can_swap(blk[j - 1], blk[j]):
        return Mutation(MutationType.SWAP_LOAD_UP, block_idx, j,
                        f"hoist {load.opcode} past {blk[j-1].opcode}")
    return None


def gen_stall_tweak(instructions: list[Instruction],
                    blocks: list) -> Optional[Mutation]:
    """Randomly increase or decrease a stall count by 1."""
    block_idx = random.randrange(len(blocks))
    blk = _get_block_instructions(instructions, blocks[block_idx])
    candidates = [j for j, i in enumerate(blk)
                  if not i.is_nop and not _is_control(i)]
    if not candidates:
        return None
    idx = random.choice(candidates)
    direction = random.choice([-1, 1])
    return Mutation(MutationType.STALL_TWEAK, block_idx, idx,
                    f"stall {'+1' if direction > 0 else '-1'} on {blk[idx].opcode}")


def gen_nop_remove(instructions: list[Instruction],
                   blocks: list) -> Optional[Mutation]:
    """Remove a NOP instruction (if surrounded by non-critical code)."""
    block_idx = random.randrange(len(blocks))
    blk = _get_block_instructions(instructions, blocks[block_idx])
    nops = [j for j, i in enumerate(blk) if i.is_nop and 0 < j < len(blk) - 1]
    if not nops:
        return None
    return Mutation(MutationType.NOP_REMOVE, block_idx, random.choice(nops),
                    "remove NOP")


# ============================================================================
# Mutation application
# ============================================================================

def apply_mutation(cubin: Cubin, kernel: KernelInfo,
                   instructions: list[Instruction],
                   blocks: list, mutation: Mutation) -> bool:
    """Apply a mutation to the cubin binary. Returns True if applied."""
    if not 0 <= mutation.block_idx < len(blocks):
        return False
    blk = _get_block_instructions(instructions, blocks[mutation.block_idx])
    idx = mutation.instr_idx
    if not 0 <= idx < len(blk):
        return False

    if mutation.kind in (MutationType.SWAP_ADJACENT, MutationType.SWAP_LOAD_UP):
        first = idx if mutation.kind == MutationType.SWAP_ADJACENT else idx - 1
        if not 0 <= first < len(blk) - 1:
            return False
        a, b = blk[first:first + 2]
        if b.code_offset != a.code_offset + 16 or not _can_swap(a, b):
            return False
        lo_a, hi_a = cubin.read_instruction(kernel, a.code_offset)
        lo_b, hi_b = cubin.read_instruction(kernel, b.code_offset)
        if (lo_a, hi_a) != (a.instr_word, a.ctrl_word) or (
                lo_b, hi_b) != (b.instr_word, b.ctrl_word):
            return False  # The proposal's analysis no longer matches the bytes.
        cubin.write_instruction(kernel, a.code_offset, lo_b, hi_b)
        cubin.write_instruction(kernel, b.code_offset, lo_a, hi_a)
        return True

    elif mutation.kind == MutationType.STALL_TWEAK:
        inst = blk[idx]
        lo, hi = cubin.read_instruction(kernel, inst.code_offset)
        stall = (hi >> 41) & 0xF
        delta = 1 if '+1' in mutation.detail else -1
        new_stall = max(0, min(15, stall + delta))
        if new_stall == stall:
            return False
        hi = (hi & ~(0xF << 41)) | (new_stall << 41)
        cubin.write_instruction(kernel, inst.code_offset, lo, hi)
        return True

    elif mutation.kind == MutationType.NOP_REMOVE:
        inst = blk[idx]
        lo, hi = cubin.read_instruction(kernel, inst.code_offset)
        if not inst.is_nop:
            return False
        if idx + 1 < len(blk):
            next_inst = blk[idx + 1]
            lo_n, hi_n = cubin.read_instruction(kernel, next_inst.code_offset)
            next_stall = (hi_n >> 41) & 0xF
            hi_n = (hi_n & ~(0xF << 41)) | (min(15, next_stall + 1) << 41)
            cubin.write_instruction(kernel, next_inst.code_offset, lo_n, hi_n)
        cubin.write_instruction(kernel, inst.code_offset,
                                0x0000000000007918, 0x000fc00000000000)
        return True

    return False


def revert_mutation(cubin: Cubin, kernel: KernelInfo,
                    instructions: list[Instruction],
                    blocks: list, mutation: Mutation,
                    saved_bytes: dict[int, tuple[int, int]]) -> None:
    """Revert a mutation using saved instruction bytes."""
    for off, (lo, hi) in saved_bytes.items():
        cubin.write_instruction(kernel, off, lo, hi)


# ============================================================================
# GPU benchmarking
# ============================================================================

def gpu_test(cubin_path: str, kernel_name: str,
             blocks: int = 1, threads: int = 256,
             smem: int = 28672, timeout: int = 5) -> bool:
    """Quick crash test. Returns True if kernel survives."""
    from sasskit.core.testing import run_sass_test
    status, _ = run_sass_test(cubin_path, kernel_name, blocks, threads,
                              smem, timeout=timeout)
    return status == "PASS"


def gpu_bench(cubin_path: str, kernel_name: str,
              blocks: int = 1, threads: int = 256,
              smem: int = 28672, iters: int = 3) -> float:
    """Benchmark kernel. Returns ms per iteration."""
    from sasskit.core.testing import run_sass_test
    status, detail = run_sass_test(cubin_path, kernel_name, blocks, threads,
                                   smem, timeout=30)
    if status != "PASS":
        return float('inf')

    result = subprocess.run(
        [Path(__file__).parent.parent.parent.parent / 'harnesses' / 'sass_test',
         cubin_path, '--bench', str(iters), kernel_name,
         str(blocks), str(threads), str(smem)],
        capture_output=True, text=True, timeout=60
    )
    import re
    m = re.search(r'per_iter=([\d.]+)ms', result.stdout)
    if m:
        return float(m.group(1))
    return float('inf')


# ============================================================================
# Main optimization loop
# ============================================================================

MUTATION_GENERATORS = [
    (gen_swap_adjacent, 0.3),
    (gen_swap_load_up, 0.3),
    (gen_stall_tweak, 0.2),
    (gen_nop_remove, 0.2),
]


def _publish_best(source: Path, output: Path) -> str:
    """Publish complete bytes atomically without replacing an existing result."""
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.reforge-',
                                     delete=False) as staging:
        staging_path = Path(staging.name)
    try:
        shutil.copyfile(source, staging_path)
        os.link(staging_path, output)
    finally:
        staging_path.unlink()
    return str(output.absolute())


def reforge(cubin_path: str, kernel_name: str,
            max_iters: int = 200,
            bench_blocks: int = 1, bench_threads: int = 256,
            bench_smem: int = 28672,
            temperature: float = 0.1,
            cooling: float = 0.995,
            seed: int = 42,
            verbose: bool = True, *,
            output_path: str | Path | None = None,
            work_dir: str | Path | None = None) -> ReforgeState:
    """Optimize in a private workspace and return a durable, non-overwritten best.

    ``work_dir`` is the parent of a unique temporary directory. By default,
    results are published in ``./reforge-results/<unique-run>/best.cubin``.
    Explicit output parents must already exist. Input files are never modified.
    """
    output = Path(output_path).absolute() if output_path is not None else None
    if output is not None:
        if output.resolve() == Path(cubin_path).resolve() or (
                output.exists() and output.samefile(cubin_path)):
            raise ValueError('Output must not alias the input cubin')
        if output.exists():
            raise FileExistsError(output)

    with tempfile.TemporaryDirectory(prefix='reforge-', dir=work_dir) as workspace:
        current = Path(workspace) / 'current.cubin'
        best = Path(workspace) / 'best.cubin'
        state = _reforge(cubin_path, kernel_name, max_iters, bench_blocks,
                         bench_threads, bench_smem, temperature, cooling, seed,
                         verbose, tmp_path=str(current), best_path=str(best))
        if not math.isfinite(state.best_time_ms) or state.best_time_ms <= 0:
            raise ValueError('No valid benchmark result to publish')
        if output is None:
            results = Path.cwd() / 'reforge-results'
            results.mkdir(exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix='run-', dir=results)) / 'best.cubin'
        state.best_cubin_path = _publish_best(best, output)
    if verbose:
        print(f"  Saved:    {state.best_cubin_path}", file=sys.stderr)
    return state


def _reforge(cubin_path: str, kernel_name: str,
            max_iters: int = 200,
            bench_blocks: int = 1, bench_threads: int = 256,
            bench_smem: int = 28672,
            temperature: float = 0.1,
            cooling: float = 0.995,
            seed: int = 42,
            verbose: bool = True, *,
            tmp_path: str, best_path: str) -> ReforgeState:
    """Run the SASS-to-SASS optimization loop.

    Args:
        cubin_path: Path to input cubin
        kernel_name: Kernel function name
        max_iters: Maximum mutation iterations
        temperature: Initial SA temperature (0 = pure hill climbing)
        cooling: SA cooling factor per iteration

    Returns:
        ReforgeState with optimization history
    """
    random.seed(seed)

    cubin = Cubin.from_file(cubin_path)
    kernel = cubin.get_kernel(kernel_name)
    instructions = decode_kernel(cubin, kernel_name)

    from sasskit.analysis import build_cfg
    blocks = build_cfg(instructions)

    state = ReforgeState(
        cubin_path=cubin_path,
        kernel_name=kernel_name,
        instructions=instructions,
        blocks=blocks,
    )

    # Baseline benchmark
    cubin.save(tmp_path)
    cubin.save(best_path)

    if verbose:
        print(f"Baseline benchmark...", file=sys.stderr, end=' ', flush=True)
    baseline = gpu_bench(tmp_path, kernel_name, bench_blocks, bench_threads,
                         bench_smem)
    if verbose:
        print(f"{baseline:.4f} ms/iter", file=sys.stderr)

    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError('Baseline benchmark must be finite and positive')

    state.best_time_ms = baseline
    state.current_time_ms = baseline
    temp = temperature

    for iteration in range(max_iters):
        state.iteration = iteration

        # Pick a random mutation type (weighted)
        r = random.random()
        cumulative = 0
        mutation = None
        for gen_fn, weight in MUTATION_GENERATORS:
            cumulative += weight
            if r <= cumulative:
                mutation = gen_fn(instructions, blocks)
                break

        if mutation is None:
            continue

        # Mutate a private candidate: rejection never changes current bytes.
        candidate = Cubin.from_file(tmp_path)
        candidate_kernel = candidate.get_kernel(kernel_name)
        if not apply_mutation(candidate, candidate_kernel, instructions, blocks, mutation):
            continue
        candidate_path = Path(tmp_path).with_name('candidate.cubin')
        candidate.save(candidate_path)

        # The decoder reads assembly from cubin.path, so reload the saved file
        # before constructing the candidate's analysis snapshot.
        candidate = Cubin.from_file(candidate_path)
        candidate_kernel = candidate.get_kernel(kernel_name)
        candidate_instructions = decode_kernel(candidate, kernel_name)
        candidate_blocks = build_cfg(candidate_instructions)

        # Quick crash test
        if not gpu_test(str(candidate_path), kernel_name, bench_blocks, bench_threads,
                        bench_smem):
            state.rejected += 1
            if verbose and iteration % 20 == 0:
                print(f"  [{iteration}] {mutation.detail} → CRASH (reverted)",
                      file=sys.stderr)
            continue

        # Benchmark
        new_time = gpu_bench(str(candidate_path), kernel_name, bench_blocks,
                             bench_threads, bench_smem)

        if not math.isfinite(new_time) or new_time <= 0:
            state.rejected += 1
            temp *= cooling
            continue

        # Accept/reject
        delta = new_time - state.current_time_ms
        accept = False
        if delta < 0:
            accept = True
        elif temp > 0 and delta > 0:
            prob = math.exp(-delta / (temp * state.best_time_ms + 1e-9))
            accept = random.random() < prob

        if accept:
            # Promote bytes and analysis together, including SA non-best accepts.
            os.replace(candidate_path, tmp_path)
            candidate.path = Path(tmp_path)
            cubin, kernel = candidate, candidate_kernel
            instructions, blocks = candidate_instructions, candidate_blocks
            state.instructions, state.blocks = instructions, blocks
            state.generation += 1
            state.current_time_ms = new_time
            state.accepted += 1
            if new_time < state.best_time_ms:
                state.best_time_ms = new_time
                cubin.save(best_path)
            state.history.append((iteration, mutation.detail, new_time))
            if verbose:
                speedup = baseline / new_time if new_time > 0 else 0
                print(f"  [{iteration}] {mutation.detail} → "
                      f"{new_time:.4f} ms ({speedup:.3f}x) ✓",
                      file=sys.stderr)
        else:
            state.rejected += 1

        temp *= cooling

    if verbose:
        speedup = baseline / state.best_time_ms if state.best_time_ms > 0 else 0
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Reforge complete: {max_iters} iterations", file=sys.stderr)
        print(f"  Accepted: {state.accepted}, Rejected: {state.rejected}",
              file=sys.stderr)
        print(f"  Baseline: {baseline:.4f} ms", file=sys.stderr)
        print(f"  Best:     {state.best_time_ms:.4f} ms "
              f"({speedup:.3f}x)", file=sys.stderr)

    return state
