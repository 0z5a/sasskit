"""Hot loop isolation and interface specification.

Given a cubin + kernel + address range, extracts the hot loop and determines
its register interface (live-in, live-out, scratch).  This interface becomes
the contract that any replacement SASS must satisfy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from sasskit.core.cubin import Cubin
from sasskit.core.decoder import decode_kernel, Instruction
from sasskit.analysis import build_cfg, compute_liveness


@dataclass
class ForgeGoals:
    """What we're optimizing for — included in AI prompt as explicit targets."""

    # Performance
    baseline_ns_per_op: float = 0.0     # current best (to beat)
    target_ns_per_op: float = 0.0       # desired target (0 = beat baseline)

    # Register budget trade-off
    prefer_fewer_regs: bool = True       # if equal perf, fewer regs wins
    reg_soft_limit: int = 0              # preferred max (0 = max_registers)

    # Correctness
    reference_result_hex: str = ""       # expected output hex for validation
    iters: int = 1 << 20                 # benchmark iterations

    # Benchmark harness
    harness_type: str = "bench"          # "bench" | "sass_test" | "custom"
    input_data: dict = field(default_factory=dict)  # named input arrays
    output_size_bytes: int = 32          # bytes to read back and compare

    def format_prompt_section(self) -> str:
        lines = ["OPTIMIZATION GOALS:"]
        if self.baseline_ns_per_op:
            lines.append(f"  Current best: {self.baseline_ns_per_op:.1f} ns/op")
        if self.target_ns_per_op:
            lines.append(f"  Target:       {self.target_ns_per_op:.1f} ns/op")
        else:
            lines.append(f"  Target:       beat the current best")
        if self.prefer_fewer_regs:
            limit = self.reg_soft_limit or 0
            hint = f" (prefer ≤ R{limit-1})" if limit else ""
            lines.append(f"  Registers:    fewer is better{hint} — lower regs → higher occupancy")
        if self.reference_result_hex:
            lines.append(f"  Correctness:  output must hash to {self.reference_result_hex[:16]}...")
        return "\n".join(lines)


@dataclass
class HotLoopSpec:
    """Complete specification of an isolated hot loop."""

    name: str
    description: str

    # Cubin source
    cubin_path: str
    kernel_name: str

    # Address range in the .text section (byte offsets)
    start_offset: int
    end_offset: int

    # Register interface
    live_in: dict[str, int]     # name → register number (live at entry)
    live_out: dict[str, int]    # name → register number (must be set at exit)
    scratch: list[int]          # registers free for use inside the loop

    # Constraints
    max_instructions: int       # slot size (how many instructions fit)
    max_registers: int          # register budget (for occupancy target)
    smem_available: int = 0     # bytes of shared memory available for temps
    even_align_regs: list[int] = field(default_factory=list)  # must be even

    # Original code (for reference / correctness checking)
    original_asm: list[str] = field(default_factory=list)

    # Goals (performance targets + benchmark config)
    goals: Optional[ForgeGoals] = None

    # Algorithm context for AI (e.g. C reference code, math description)
    algorithm_description: str = ""     # plain-text description of what it does
    reference_c_code: str = ""          # C/PTX reference implementation

    # Template-based compilation (alternative to cubin patching)
    # When set, ForgeEngine replaces the marked section in template_path
    # with AI-generated SASS and compiles with cubit asm instead of patching.
    template_path: str = ""             # path to .sass template file
    template_begin_marker: str = "// HOT LOOP BEGIN"
    template_end_marker: str = "// HOT LOOP END"
    cubit_dir: str = ""                  # cwd for cubit asm invocation (uses CUBIT_DIR env or cwd)

    @property
    def slot_bytes(self) -> int:
        return self.max_instructions * 16

    @property
    def n_inputs(self) -> int:
        return len(self.live_in)

    @property
    def n_outputs(self) -> int:
        return len(self.live_out)

    @property
    def n_scratch(self) -> int:
        return len(self.scratch)

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HotLoopSpec":
        with open(path) as f:
            d = json.load(f)
        # Convert nested goals dict → ForgeGoals object
        if isinstance(d.get("goals"), dict):
            d["goals"] = ForgeGoals(**d["goals"])
        return cls(**d)

    def format_prompt_section(self) -> str:
        """Format as text for an LLM prompt."""
        lines = [
            f"=== HOT LOOP: {self.name} ===",
            f"Description: {self.description}",
            f"Architecture: sm_120 (NVIDIA Blackwell)",
            "",
        ]

        # Algorithm context (if provided) — truncate to first 80 lines to keep prompt size reasonable
        if self.algorithm_description:
            lines.append("ALGORITHM:")
            algo_lines = self.algorithm_description.strip().split("\n")
            max_algo_lines = 80
            for line in algo_lines[:max_algo_lines]:
                lines.append(f"  {line}")
            if len(algo_lines) > max_algo_lines:
                lines.append(f"  ... ({len(algo_lines) - max_algo_lines} more lines) ...")
            lines.append("")

        lines.append("INPUT REGISTERS (live at loop entry, read-only):")
        for name, reg in sorted(self.live_in.items(), key=lambda x: x[1]):
            lines.append(f"  R{reg:2d}  = {name}")
        lines.append("")
        lines.append("OUTPUT REGISTERS (must contain results at loop exit):")
        for name, reg in sorted(self.live_out.items(), key=lambda x: x[1]):
            lines.append(f"  R{reg:2d}  = {name}")
        lines.append("")
        lines.append(f"SCRATCH REGISTERS (free to clobber): "
                      f"{', '.join(f'R{r}' for r in sorted(self.scratch))}")
        lines.append(f"SLOT: {self.max_instructions} instructions max "
                      f"({self.slot_bytes} bytes)")
        lines.append(f"REGISTER BUDGET: R0–R{self.max_registers - 1} "
                      f"({self.max_registers} total). "
                      f"Fewer registers = better occupancy.")
        if self.even_align_regs:
            lines.append("EVEN-ALIGNMENT REQUIRED for .WIDE/.64 destinations "
                         "(R24, R26, R28, ... max R{self.max_registers-2})")
        lines.append("")

        # Goals
        if self.goals:
            lines.append(self.goals.format_prompt_section())
            lines.append("")

        # Reference C code
        if self.reference_c_code:
            lines.append("REFERENCE IMPLEMENTATION (C/PTX — for algorithmic guidance):")
            lines.append("```c")
            lines.append(self.reference_c_code.strip())
            lines.append("```")
            lines.append("")

        # Original SASS (show first 60 + last 20 lines to keep prompt manageable)
        if self.original_asm:
            label = (f"CURRENT BEST SASS "
                     f"({self.goals.baseline_ns_per_op:.1f} ns/op — beat this):"
                     if self.goals and self.goals.baseline_ns_per_op
                     else "ORIGINAL SASS (reference):")
            lines.append(label)
            show_head = 60
            show_tail = 20
            total = len(self.original_asm)
            if total <= show_head + show_tail:
                for i, asm in enumerate(self.original_asm):
                    lines.append(f"  [{i:3d}] {asm}")
            else:
                for i, asm in enumerate(self.original_asm[:show_head]):
                    lines.append(f"  [{i:3d}] {asm}")
                lines.append(f"  ... ({total - show_head - show_tail} more instructions) ...")
                for i, asm in enumerate(self.original_asm[-show_tail:]):
                    lines.append(f"  [{total - show_tail + i:3d}] {asm}")

        return "\n".join(lines)


def extract_hot_loop(cubin_path: str, kernel_name: str,
                     start_offset: int, end_offset: int,
                     name: str = "hotloop",
                     description: str = "",
                     max_registers: int = 64) -> HotLoopSpec:
    """Extract a hot loop spec from a cubin.

    Analyzes liveness at the loop boundaries to determine
    live-in (inputs), live-out (outputs), and scratch registers.
    """
    cubin = Cubin.from_file(cubin_path)
    instructions = decode_kernel(cubin, kernel_name)

    # Build CFG and compute liveness
    blocks = build_cfg(instructions)
    compute_liveness(blocks, instructions=instructions)

    # Find instructions in the hot loop range
    loop_instrs = [
        inst for inst in instructions
        if start_offset <= inst.code_offset < end_offset
    ]

    if not loop_instrs:
        raise ValueError(
            f"No instructions in range [{start_offset:#x}, {end_offset:#x})"
        )

    # Find blocks containing the hot loop boundaries
    first_off = loop_instrs[0].code_offset
    last_off = loop_instrs[-1].code_offset

    # Get liveness at loop entry: find the block containing first_off
    live_at_entry = set()
    live_at_exit = set()
    for block in blocks:
        if not block.instructions:
            continue
        b_start = block.instructions[0].code_offset
        b_end = block.instructions[-1].code_offset
        if b_start <= first_off <= b_end:
            # Compute per-instruction liveness within this block
            # Walk backwards from block.live_out
            live = set(block.live_out)
            for inst in reversed(block.instructions):
                if inst.code_offset == first_off:
                    live_at_entry = set(live)
                live -= set(inst.dest_regs)
                live |= set(inst.src_regs)
        if b_start <= last_off <= b_end:
            live = set(block.live_out)
            for inst in reversed(block.instructions):
                if inst.code_offset == last_off:
                    # live_at_exit = what's live AFTER this instruction
                    live_at_exit = set(live)
                    break
                live -= set(inst.dest_regs)
                live |= set(inst.src_regs)

    # Registers DEFined inside the loop
    defs_in_loop = set()
    uses_in_loop = set()
    for inst in loop_instrs:
        defs_in_loop.update(inst.dest_regs)
        uses_in_loop.update(inst.src_regs)

    # Remove special registers (RZ=255, PT=7 for predicates, etc.)
    special = {255}  # RZ
    live_at_entry -= special
    live_at_exit -= special
    defs_in_loop -= special
    uses_in_loop -= special

    # Live-in: registers live at entry that are READ in the loop
    live_in_regs = live_at_entry & uses_in_loop

    # Live-out: registers live at exit that were DEFined in the loop
    live_out_regs = live_at_exit & defs_in_loop

    # Scratch: registers used/defined in loop but NOT live-in and NOT live-out
    all_loop_regs = defs_in_loop | uses_in_loop
    scratch_regs = sorted(all_loop_regs - live_in_regs - live_out_regs - special)

    # Also add unused registers up to max_registers as scratch
    used_anywhere = live_in_regs | live_out_regs | set(scratch_regs)
    for r in range(max_registers):
        if r not in used_anywhere and r not in special:
            scratch_regs.append(r)
    scratch_regs = sorted(set(scratch_regs))

    # Find even-aligned registers (used in .64/.WIDE dests)
    even_align = set()
    for inst in loop_instrs:
        for ref in inst.reg_refs:
            if ref.is_dest and ref.is_pair:
                even_align.add(ref.reg_num)

    # Build name maps (auto-generate names like "in_R4", "out_R20")
    in_map = {f"in_R{r}": r for r in sorted(live_in_regs)}
    out_map = {f"out_R{r}": r for r in sorted(live_out_regs)}

    return HotLoopSpec(
        name=name,
        description=description,
        cubin_path=str(cubin_path),
        kernel_name=kernel_name,
        start_offset=start_offset,
        end_offset=end_offset,
        live_in=in_map,
        live_out=out_map,
        scratch=scratch_regs,
        max_instructions=len(loop_instrs),
        max_registers=max_registers,
        even_align_regs=sorted(even_align),
        original_asm=[inst.asm_text for inst in loop_instrs],
    )
