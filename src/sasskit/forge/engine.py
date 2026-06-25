"""Forge engine: AI-driven hot loop optimization loop.

The main loop:
  1. Build prompt from HotLoopSpec + history of previous attempts
  2. Call LLM API (Anthropic Claude) to generate new SASS
  3. Assemble via cubit → binary
  4. Inject into cubin (replace hot loop bytes)
  5. Test: crash detection (sass_test) + correctness + performance
  6. Record result and feed back to AI
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Optional
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from .hotloop import HotLoopSpec
from sasskit.core.isa import Sm120Assembler, AsmInstruction, CTRL_DEFAULT, AssembleError


# ── result types ─────────────────────────────────────────────

@dataclass
class VariantResult:
    """Result of testing one SASS variant."""
    iteration: int
    timestamp: str
    sass_text: list[str]        # the SASS source lines
    n_instructions: int
    max_register: int           # highest register used
    status: str                 # PASS, FAIL, CRASH, TIMEOUT, ASM_ERROR, WRONG
    error_detail: str = ""
    correct: Optional[bool] = None
    elapsed_ms: Optional[float] = None   # kernel execution time
    ns_per_op: Optional[float] = None    # ns per loop iteration
    throughput: Optional[float] = None   # app-specific metric


@dataclass
class ForgeHistory:
    """History of all tested variants."""
    spec_name: str
    results: list[VariantResult] = field(default_factory=list)
    best_idx: Optional[int] = None

    @property
    def best(self) -> Optional[VariantResult]:
        if self.best_idx is not None:
            return self.results[self.best_idx]
        return None

    def add(self, result: VariantResult):
        self.results.append(result)
        # Update best: PASS + correct + better ns_per_op (or fewer regs as tiebreak)
        if result.status == "PASS" and result.correct is not False:
            if self.best is None:
                self.best_idx = len(self.results) - 1
            else:
                b = self.best
                # Primary: ns_per_op (lower is better)
                if result.ns_per_op is not None and b.ns_per_op is not None:
                    if result.ns_per_op < b.ns_per_op:
                        self.best_idx = len(self.results) - 1
                elif result.elapsed_ms is not None and b.elapsed_ms is not None:
                    if result.elapsed_ms < b.elapsed_ms:
                        self.best_idx = len(self.results) - 1
                # Tiebreak: fewer registers
                elif result.max_register < b.max_register:
                    self.best_idx = len(self.results) - 1

    def format_for_prompt(self, last_n: int = 5) -> str:
        """Format recent results for inclusion in LLM prompt."""
        if not self.results:
            return "No previous attempts."

        lines = ["PREVIOUS ATTEMPTS (most recent):"]
        start = max(0, len(self.results) - last_n)
        for i in range(start, len(self.results)):
            r = self.results[i]
            if r.ns_per_op is not None:
                perf = f"{r.ns_per_op:.2f} ns/op"
            elif r.elapsed_ms is not None:
                perf = f"{r.elapsed_ms:.2f}ms"
            else:
                perf = "?"
            corr = "✓" if r.correct else ("✗ WRONG" if r.correct is False else "?")
            is_best = " *** BEST ***" if i == self.best_idx else ""
            lines.append(
                f"  #{r.iteration}: {r.n_instructions} insns, "
                f"max R{r.max_register}, {perf}, "
                f"{r.status} {corr}{is_best}"
            )
            if r.status not in ("PASS",):
                lines.append(f"    → {r.error_detail[:120]}")
        if self.best:
            b = self.best
            perf = f"{b.ns_per_op:.2f} ns/op" if b.ns_per_op else "?"
            lines.append(f"\nCURRENT BEST: #{b.iteration} — "
                         f"{b.n_instructions} insns, max R{b.max_register}, {perf}")
        return "\n".join(lines)

    def save(self, path: str | Path):
        with open(path, "w") as f:
            for r in self.results:
                f.write(json.dumps(asdict(r)) + "\n")

    def load(self, path: str | Path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.add(VariantResult(**d))


# ── prompt builder ───────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert GPU assembly programmer specializing in NVIDIA sm_120 \
(Blackwell) SASS. You write optimal hot loop code for cryptographic field arithmetic.

## SM120 ARCHITECTURE

Single-warp GPU kernel executing in a loop. Performance = ns per loop iteration.
The GPU pipeline issues 1 instruction/cycle when there are no stalls.
Stalls = wasted cycles = slower code.

KEY SM120 FACTS:
- IMAD.WIDE.U32 Rd, Pout, Ra, Rb, Rc → Rd:Rd+1 = Ra*Rb + Rc:Rc+1; Pout=carry. Latency: 6 cycles.
- IMAD.WIDE.U32.X Rd, Pout, Ra, Rb, Rc, Pin → += carry(Pin). EXACT SAME LATENCY.
- IMAD / IMAD.HI: latency 6 cycles.  IADD3 / IADD3.X: latency 4 cycles.  MOV: 4 cycles.
- SEL Rd, Ra, Rb, Ps → Rd = Ps ? Ra : Rb. Used to capture 1-bit overflow from IMAD.WIDE carry.
- RZ = zero register.  PT = always-true predicate.

CRITICAL HARDWARE CONSTRAINT — IMAD.WIDE accumulator:
  IMAD.WIDE.U32 Rd, Pout, Ra, Rb, Rc  →  Rd:Rd+1 = Ra*Rb + Rc:Rc+1; Pout=carry
  *** Rc MUST NOT be Rd+1 (the high output register) — causes ILLEGAL_INSTRUCTION ***
  *** Rc = Rd IS LEGAL (accumulate into the low half of the same register pair) ***

  WRONG:   IMAD.WIDE.U32 R40, P0, R8, R18, R41 ;  ← R41 = Rd+1 = ILLEGAL!
  CORRECT: IMAD.WIDE.U32 R40, P0, R8, R18, RZ  ;  ← RZ as accumulator (no accumulate)
  CORRECT: IMAD.WIDE.U32 R40, P0, R8, R18, R40 ;  ← R40 = Rd, accumulate into pair = OK!
  CORRECT: IMAD.WIDE.U32 R40, P0, R8, R18, R24 ;  ← any other even register = OK

## FUSED MULTIPLY-ACCUMULATE PATTERN (FASTEST APPROACH — BEATS NVCC)

b[0..7] = R16..R23 is CONSTANT throughout the hot loop (same every iteration).
This is a fixed multiplier — exploit this for maximum efficiency.

The OPTIMAL approach for secp256k1 uses fused IMAD.WIDE.U32 with accumulator Rc = Rd:
Each column k (products where i+j=k) can be accumulated DIRECTLY in 1 instruction:

  // Start column k with first product:
  IMAD.WIDE.U32   R24, P0, R8,  R16, RZ   ;  // col0 = a[0]*b[0], P0=carry
  // Add second product to same accumulator:
  IMAD.WIDE.U32   R24, P0, R9,  R16, R24  ;  // col0 += a[1]*b[0] + P0_old; P0=new_carry
  // Use .X to carry from previous overflow:
  IMAD.WIDE.U32.X R26, P1, R8,  R17, R26, P0 ;  // col1 += a[0]*b[1] + carry(P0)
  // Capture overflow bit:
  SEL R28, RZ, 0x1, !P1 ;                    // overflow_col1 = P1 (0 or 1)

FUSED vs IADD3-BASED comparison:
  FUSED:  1 instruction per product → 64 total for 8×8 multiply
  IADD3:  4 instructions per product → 256 total  ← 4× MORE INSTRUCTIONS

nvcc achieves 1495 ns/op with this fused approach (184 total instructions).
Our IADD3 baseline is 3163 ns/op (420 instructions). Beat nvcc with fused approach!

OVERFLOW HANDLING (critical for correctness):
  When accumulating n products in one chain, 1-bit Pout captures each step's overflow.
  For chains with ≤2 products per step: overflow is exactly 1 bit — correct.
  For longer chains, save overflow bits with SEL and add them to the next column:
    SEL R_overflow, RZ, 0x1, !P0   ;  // R_overflow = P0 carry bit (0 or 1)
    // ... later, add R_overflow to column k+2's accumulator ...

secp256k1 REDUCTION (use prime structure!):
  P = 2^256 - 2^32 - 977. After 8×8 multiply (512-bit result):
  result = lo_256 + hi_256 × (2^32 + 977)  (fold using 2^256 ≡ 2^32+977 mod P)
  Then conditional subtract if result ≥ P.

IMAD.WIDE WITH CARRY syntax:
  IMAD.WIDE.U32   Rd, Pout, Ra, Rb, Rc ;         — Rd:Rd+1 = Ra*Rb + Rc:Rc+1; carry→Pout
  IMAD.WIDE.U32.X Rd, Pout, Ra, Rb, Rc, Pin ;    — Rd:Rd+1 = Ra*Rb + Rc:Rc+1 + carry(Pin); carry→Pout

EXACT CARRY CHAIN SYNTAX (cubit sm_120 assembler):
  IADD3  Rd, Pcarry_hi, Pcarry_lo, Ra, Rb, Rc   ← start of chain; Rd = Ra+Rb+Rc; carry→Pcarry_hi:Pcarry_lo
  IADD3.X Rd, Pcarry_hi, Pcarry_lo, Ra, Rb, Rc, Pcarry_in_hi, !Pcarry_in_lo  ← continue; reads carry from Pcarry_in_hi:!Pcarry_in_lo
  
EXAMPLE - 256-bit add across 8 limbs:
  IADD3   R8, P0, PT, R8, R24, RZ ;       ← limb0; carry out → P0
  IADD3.X R9, P0, PT, R9, R25, RZ, P0, !PT ;  ← limb1+carry; carry out → P0
  IADD3.X R10, P0, PT, R10, R26, RZ, P0, !PT ;
  IADD3.X R11, P0, PT, R11, R27, RZ, P0, !PT ;
  IADD3.X R12, P0, PT, R12, R28, RZ, P0, !PT ;
  IADD3.X R13, P0, PT, R13, R29, RZ, P0, !PT ;
  IADD3.X R14, P0, PT, R14, R30, RZ, P0, !PT ;
  IADD3.X R15, PT, PT, R15, R31, RZ, P0, !PT ;  ← last limb: no carry needed

WARNING - WRONG syntax examples (will cause ASM_ERROR):
  IADD3 R24, P0, P1, R8, R9        ← WRONG: missing Rc operand
  IADD3 R24, P0, R8, R9, RZ        ← WRONG: missing second predicate output
  IADD3.X R24, P0, PT, R8, R9, RZ  ← WRONG: missing carry-in predicates
  IADD3.X R24, P0, PT, R8, R9, RZ, P0  ← WRONG: only one carry-in predicate

## HOW TO OPTIMIZE FOR LATENCY

The GPU scheduler WAITS if an instruction reads a register written by a recent instruction
and the latency gap has not yet elapsed. For a single-warp kernel:

  IPC = 1.0 / avg_cycles_per_instruction
  ideal IPC = 1.0  (one new instruction issues every cycle)
  
If IMAD.WIDE has 6-cycle latency and you use its result 2 instructions later,
the scheduler must stall for 4 extra cycles on that consumer instruction.

CRITICAL TIMING RULE: Between any IMAD.WIDE instruction and the first instruction
that reads its result, there must be AT LEAST 5 INDEPENDENT INSTRUCTIONS.
The scheduler encodes stall=6 on IMAD.WIDE when the consumer follows immediately,
but proper interleaving (>=5 independent instructions between IMAD.WIDE and IADD3)
achieves this without stalls and is the optimal approach.

STRATEGY — hide latency with instruction-level parallelism:
  1. Identify independent computation chains (e.g., IMAD.WIDE A×B vs IMAD.WIDE C×D).
  2. Interleave them: issue IMAD.WIDE for chain B while chain A is "in flight".
  3. Don't use chain A's result until ≥5 independent instructions (or 5 NOPs) after IMAD.WIDE.

EXAMPLE — bad (serial, stalls on every consumer):
  IMAD.WIDE.U32 R0, R4, R8, RZ ;   // R0:R1 = a*b  (latency 6)
  IADD3 R2, P0, PT, R0, R1, RZ ;   // ← WRONG: stalls 5 cycles waiting for R0

EXAMPLE — good (5 NOPs between write and read):
  IMAD.WIDE.U32 R0, R4, R8, RZ ;   // R0:R1 = a*b  (latency 6)
  NOP ;  NOP ;  NOP ;  NOP ;  NOP ; // 5 NOPs = 5 independent cycles
  IADD3 R2, P0, PT, R0, R1, RZ ;   // now safe to read R0

EXAMPLE — optimal (interleaved, latency hidden by real work):
  IMAD.WIDE.U32 R0, R4, R8, RZ ;   // T0: chain A — result R0:R1 ready at T6
  IMAD.WIDE.U32 R2, R5, R9, RZ ;   // T1: chain B — result R2:R3 ready at T7
  IMAD.WIDE.U32 R6, R10, R11, RZ ; // T2: chain C — result R6:R7 ready at T8
  IMAD.WIDE.U32 R12, R14, R15, RZ ;// T3: chain D — result R12:R13 ready at T9
  IMAD.WIDE.U32 R20, R16, R17, RZ ;// T4: chain E — result R20:R21 ready at T10
  IMAD.WIDE.U32 R22, R18, R19, RZ ;// T5: chain F — result R22:R23 ready at T11
  IADD3 R30, P0, PT, R0, R1, RZ ;  // T6: reads chain A (R0) — ready! no stall
  IADD3 R31, P0, PT, R2, R3, RZ ;  // T7: reads chain B (R2) — ready! (7 cycles passed)
  IADD3 R32, P0, PT, R6, R7, RZ ;  // T8: reads chain C (R6) — ready!
  // etc. — each consumer reads the result issued 6 instructions earlier

  CRITICAL: Use results in THE SAME ORDER as the IMAD.WIDEs were issued!
  Chain A result R0 → used at T6+, chain B result R2 → used at T7+, etc.
  If you use chain B's result at T6 (only 5 cycles after T1), you get wrong results!

## HOW TO READ NCU PROFILER DATA

When you see profiler data in the prompt:
- "SM throughput %": how busy the SM is overall. Low = stall-bound.
- "IPC": instructions per clock. 0.28 = you're issuing 1 instruction every 3.6 cycles.
- "Avg cycles/instruction": the key metric. 1.0 = ideal. 3.55 = 2.55 wasted cycles/insn.

To reduce avg cycles/instruction:
  → Reorder instructions to maximize distance between a write and its first read.
  → If two chains use the same output registers, they're NOT independent — split them.
  → Minimize total instructions: fewer instructions = fewer latency hazards to schedule around.

{latency_section}

## OUTPUT RULES
1. Output ONLY SASS instructions, one per line, ending with " ;"
2. NO comments, NO blank lines in the code block.
3. Every instruction must be valid SM120 SASS syntax (cubit-compatible).
4. Use only INPUT, OUTPUT, and declared SCRATCH registers.
5. .64/.WIDE destinations must be even-numbered. .128 must be 4-aligned.
6. The loop counter decrement (IADD3 R0, ..., R0, -0x1) must be included exactly as specified.
"""

_LATENCY_FALLBACK = (
    "Instruction latencies (sm_120 Blackwell):\n"
    "  IMAD: 5 cycles, IADD3: 4 cycles, MOV: 2 cycles.\n"
    "  LDG: ~200 cycles (variable), LDS: ~30 cycles (variable).\n"
    "  FFMA: 4 cycles, DADD: 8 cycles."
)


def _get_system_prompt() -> str:
    """Build the system prompt with lazily-loaded latency table."""
    try:
        from sasskit.core.isa_db import get_db
        latency_section = get_db().latency_table_for_prompt()
    except Exception:
        latency_section = _LATENCY_FALLBACK
    return _SYSTEM_PROMPT_TEMPLATE.format(latency_section=latency_section)


# Keep SYSTEM_PROMPT as a module-level name for backward compatibility,
# but make it a lazy property via a descriptor isn't practical for a
# plain string.  Instead, callers should use _get_system_prompt().
# For backward compat, set it to the fallback; build_prompt() below
# uses the dynamic version.
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(latency_section=_LATENCY_FALLBACK)


def _log_iteration(result: "VariantResult", history: "ForgeHistory"):
    """Print a plain-text summary line for a completed iteration (always flushed)."""
    ns = f"{result.ns_per_op:8.2f} ns/op" if result.ns_per_op else " " * 16
    baseline = getattr(history, "_spec_baseline", 0)
    speedup = ""
    if result.ns_per_op and baseline:
        speedup = f"  {baseline/result.ns_per_op:.3f}×"
    best_mark = ""
    if (result.status == "PASS" and history.best
            and history.best.iteration == result.iteration):
        best_mark = "  *** NEW BEST ***"
    detail = f"  [{result.error_detail[:60]}]" if result.status != "PASS" else ""
    print(
        f"[{result.iteration:04d}] {result.status:10s}  "
        f"{result.n_instructions:3d} insns  R{result.max_register:<3d}  "
        f"{ns}{speedup}{best_mark}{detail}",
        flush=True
    )


def build_prompt(spec: HotLoopSpec, history: ForgeHistory,
                 extra_instructions: str = "") -> str:
    """Build the LLM prompt from spec + history."""
    # Separate profiler data from other extra_instructions
    profiler_section = ""
    other_extra = extra_instructions
    if extra_instructions and "NCU PROFILE" in extra_instructions:
        # Split out the NCU block; keep other extras separate
        lines = extra_instructions.splitlines()
        prof_lines, other_lines = [], []
        in_prof = False
        for ln in lines:
            if ln.startswith("NCU PROFILE") or ln.startswith("BASELINE"):
                in_prof = True
            if in_prof:
                prof_lines.append(ln)
            else:
                other_lines.append(ln)
        profiler_section = "\n".join(prof_lines)
        other_extra = "\n".join(other_lines).strip()

    parts = [spec.format_prompt_section()]

    # Profiler data right after spec, before history (AI sees it prominently)
    if profiler_section:
        parts += ["", profiler_section]

    parts += [
        "",
        "VERIFIED INSTRUCTION SYNTAX (cubit sm_120 — EXACT format, no variations):",
        "  IMAD.WIDE.U32 Rd, Ra, Rb, Rc ;   — Rd:Rd+1 = Ra*Rb + Rc  (Rd EVEN; Rc ≠ Rd+1!)",
        "  IMAD.HI.U32   Rd, Ra, Rb, Rc ;   — Rd = high32(Ra*Rb) + Rc",
        "  IMAD.U32      Rd, Ra, Rb, Rc ;   — Rd = low32(Ra*Rb) + Rc",
        "  IADD3   Rd, Pout, PT, Ra, Rb, Rc ;            — Rd=Ra+Rb+Rc; carry→Pout",
        "  IADD3   Rd, Pout, PT, Ra, imm, Rc ;           — with immediate Rb",
        "  IADD3.X Rd, Pout, PT, Ra, Rb, Rc, Pin, !PT ;  — Rd=Ra+Rb+Rc+carry(Pin); carry→Pout",
        "  MOV Rd, Ra ;                      — Rd = Ra",
        "  MOV Rd, imm ;                     — Rd = immediate",
        "  SHF.R.U32.HI  Rd, Ra, imm, Rb ;  — Rd = (Rb:Ra) >> imm  (upper 32 bits)",
        "  SHF.L.U32     Rd, Ra, imm, Rb ;  — funnel shift left",
        "  LOP3.LUT Rd, Ra, Rb, Rc, imm, PT ; — 3-input bitwise: imm is truth table (0xE0=OR, 0xC0=AND, 0xF8=MAJ...)",
        "  SEL  Rd, Ra, Rb, Pn ;             — Rd = Pn ? Ra : Rb",
        "  NOP ;                              — no operation",
        "  ISETP.GE.U32.AND Pd, PT, Ra, Rb, PT ; — Pd = (Ra >= Rb)",
        "",
        "NOTE: Plain 'IADD' does NOT exist — use IADD3 with RZ for 2-operand add:",
        "  IADD3 Rd, PT, PT, Ra, Rb, RZ ;   ← equivalent to Rd = Ra + Rb",
        "",
        history.format_for_prompt(),
        "",
    ]

    if other_extra:
        parts.append(other_extra)
        parts.append("")

    # Targeted closing instruction based on profiler data
    if profiler_section and "cycles/instruction:" in profiler_section:
        # Extract avg latency from profiler text for targeted advice
        import re
        m = re.search(r'Avg cycles/instruction:\s+([\d.]+)', profiler_section)
        avg_lat = float(m.group(1)) if m else 0
        if avg_lat > 3.0:
            closing = (
                f"Generate a NEW implementation that reduces latency stalls "
                f"(currently {avg_lat:.1f} cycles/insn, target <1.5). "
                "Interleave ALL independent IMAD.WIDE chains so each chain's "
                "result is not consumed for ≥8 instructions. "
                "Output a code block with ONLY SASS instructions (one per line, ending with \" ;\")."
            )
        elif avg_lat > 1.5:
            closing = (
                f"Generate a NEW implementation with fewer total instructions "
                f"(latency {avg_lat:.1f} cycles/insn is acceptable, now minimize loop body size). "
                "Output a code block with ONLY SASS instructions (one per line, ending with \" ;\")."
            )
        else:
            closing = (
                "Generate a NEW implementation. Optimize further: fewer instructions, "
                "lower register pressure, or better latency hiding. "
                "Output a code block with ONLY SASS instructions (one per line, ending with \" ;\")."
            )
    else:
        closing = (
            "Generate a NEW implementation. Output a code block with ONLY "
            "SASS instructions (one per line, each ending with \" ;\"). "
            "Aim for fewer instructions AND lower register usage than previous best."
        )
    parts.append(closing)
    return "\n".join(parts)


# ── LLM caller ───────────────────────────────────────────────

def call_anthropic(prompt: str, system: str = None,
                   model: str = "claude-sonnet-4-20250514",
                   max_tokens: int = 8192) -> str:
    """Call Anthropic API and return the response text.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "pip install anthropic  (required for LLM-driven forge loop)"
        )

    if system is None:
        system = _get_system_prompt()
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def parse_sass_from_response(response: str) -> list[str]:
    """Extract SASS instruction lines from LLM response.

    Handles markdown code blocks and bare instruction lists.
    """
    lines = response.strip().split("\n")

    # Try to find a code block
    in_block = False
    sass_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_block:
                break  # end of block
            in_block = True
            continue
        if in_block:
            if stripped and not stripped.startswith("//") and not stripped.startswith("#"):
                # Skip truncated instructions (don't end with ";")
                if stripped.endswith(";"):
                    sass_lines.append(stripped)

    # If no code block found, try to extract lines ending with ";"
    if not sass_lines:
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(";") and not stripped.startswith("//"):
                # Remove leading line numbers like "[  0]" or "0:"
                import re
                cleaned = re.sub(r'^\s*\[?\s*\d+\s*\]?\s*', '', stripped)
                cleaned = re.sub(r'^\d+:\s*', '', cleaned)
                if cleaned:
                    sass_lines.append(cleaned)

    return sass_lines


# ── binary injection ─────────────────────────────────────────

def inject_hot_loop(cubin_path: str, spec: HotLoopSpec,
                    new_bytes: bytes,
                    max_reg: int = 0) -> bytes:
    """Replace the hot loop bytes in a cubin and update regcount.

    Args:
        new_bytes:  assembled hot loop, must equal (end_offset - start_offset) bytes
        max_reg:    highest register number used; cubin regcount set to max(max_reg+1, 32)
                    rounded up to next multiple of 8. 0 = use spec.max_registers.

    Returns the patched cubin as bytes.
    """
    from sasskit.core.cubin import Cubin
    cubin = Cubin.from_file(cubin_path)
    kernel = cubin.get_kernel(spec.kernel_name)

    # Patch instruction bytes
    text_start = kernel.text_offset
    loop_start = text_start + spec.start_offset
    loop_size = spec.end_offset - spec.start_offset

    if len(new_bytes) != loop_size:
        raise ValueError(
            f"New code size ({len(new_bytes)}) != slot size ({loop_size}). "
            f"Pad with NOPs to exactly {loop_size} bytes."
        )

    cubin_data = bytearray(cubin.data)
    cubin_data[loop_start:loop_start + loop_size] = new_bytes

    # Patch EIATTR_REGCOUNT (attr=0x2f, fmt=0x04 SVAL, size=8) in .nv.info sections.
    # This is the actual register count the GPU validates against.
    # Format of each record: [04][2f][08 00][func_ref:4][value:4]
    if max_reg > 0:
        needed = max_reg + 1
    else:
        needed = spec.max_registers
    # Round up to next multiple of 8, minimum 32
    reg_count = max(32, ((needed + 7) // 8) * 8)

    for i in range(len(cubin_data) - 12):
        if cubin_data[i] == 0x04 and cubin_data[i+1] == 0x2f:
            size = int.from_bytes(cubin_data[i+2:i+4], 'little')
            if size == 8:
                cubin_data[i+8:i+12] = reg_count.to_bytes(4, 'little')

    return bytes(cubin_data)


# ── testing ──────────────────────────────────────────────────

def test_cubin(cubin_bytes: bytes, kernel_name: str = "KernelA",
               timeout: int = 30, sass_test_bin: str = None) -> tuple[str, str]:
    """Test a cubin for crashes using sass_test.

    Returns (status, detail) where status is PASS/FAIL/CRASH/TIMEOUT.
    """
    from sasskit.core.testing import test_cubin_bytes
    return test_cubin_bytes(cubin_bytes, kernel_name=kernel_name,
                            timeout=timeout, sass_test_bin=sass_test_bin)


def bench_cubin(cubin_bytes: bytes, spec: "HotLoopSpec",
                bench_bin: str = None,
                timeout: int = 15) -> dict:
    """Benchmark a cubin using bench_harness: ns/op + correctness.

    Returns dict with keys: status, ns_per_op, result_hex, error.
    status: 'PASS' | 'WRONG' | 'CRASH' | 'TIMEOUT' | 'ERROR'
    """
    import re, subprocess, tempfile, os
    from pathlib import Path

    # Locate bench_harness binary
    if bench_bin is None:
        env_harness = os.environ.get('SASSKIT_BENCH_HARNESS')
        candidates = [
            Path(env_harness) if env_harness else None,
            Path(__file__).parent.parent.parent.parent / "harnesses" / "bench_harness",
        ]
        for c in candidates:
            if c and c.exists():
                bench_bin = str(c)
                break
        else:
            return {"status": "ERROR", "error": "bench_harness not found"}

    # Write cubin to temp file
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        tmp = f.name

    try:
        goals = spec.goals
        iters = goals.iters if goals else (1 << 20)
        ref   = goals.reference_result_hex if goals else ""

        args = [bench_bin, tmp, spec.kernel_name, str(iters)]
        if ref:
            args += ["", "", ref]  # empty input_hex, use defaults, provide ref

        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)

        if r.returncode == 0:
            m = re.search(r'ns_per_op=([\d.]+)\s+result=([0-9a-f]+)', r.stdout)
            if m:
                return {"status": "PASS", "ns_per_op": float(m.group(1)),
                        "result_hex": m.group(2), "error": ""}
            return {"status": "ERROR", "error": f"parse fail: {r.stdout[:100]}"}
        elif r.returncode == 1:
            err = r.stderr[:200]
            # Extra diagnosis: if result == DEFAULT_A input, code didn't write R8-R15
            default_a_hex = "0011223355aa55aaeeff0011aabbccdd6677889922334455abcdef0112345678"
            if default_a_hex[:16] in err:
                err += " [R8-R15 unchanged = result never stored into output registers!]"
            return {"status": "WRONG", "ns_per_op": None,
                    "result_hex": "", "error": err}
        elif r.returncode == 3:
            return {"status": "TIMEOUT", "error": r.stderr[:100]}
        else:
            return {"status": "CRASH", "error": r.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": f">{timeout}s"}
    finally:
        os.unlink(tmp)


# ── main forge loop ──────────────────────────────────────────

class ForgeEngine:
    """The main AI-driven optimization loop."""

    def __init__(self, spec: HotLoopSpec,
                 llm_fn: Callable[[str, str], str] = None,
                 sass_test_bin: str = None,
                 bench_bin: str = None,
                 log_path: str = None):
        self.spec = spec
        self.asm = Sm120Assembler()
        self.history = ForgeHistory(spec_name=spec.name)
        self.llm_fn = llm_fn or call_anthropic
        self.sass_test_bin = sass_test_bin
        self.bench_bin = bench_bin
        self.log_path = log_path or f"forge_{spec.name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self._last_patched_bytes: Optional[bytes] = None
        self._profiler_data: Optional[str] = None

    def run(self, max_iterations: int = 0,
            extra_instructions: str = ""):
        """Run the optimization loop.

        Args:
            max_iterations: 0 = unlimited (run until Ctrl+C)
            extra_instructions: extra context passed to each LLM prompt
        """
        from sasskit.forge.live_ui import ForgeUI, console
        from sasskit.forge.profiler import profile_cubin, ProfileResult

        goals = self.spec.goals
        baseline = goals.baseline_ns_per_op if goals else 0
        target   = goals.target_ns_per_op   if goals else 0

        ui = ForgeUI(self.spec.name, baseline_ns=baseline, target_ns=target)
        ui.start()

        # Profile baseline for reference (skip if no cubin_path, e.g. template mode)
        self._profiler_data: Optional[str] = None
        try:
            if self.bench_bin and goals and goals.harness_type == "bench" and self.spec.cubin_path:
                _prof = profile_cubin(
                    self.spec.cubin_path,
                    self.spec.kernel_name,
                    bench_bin=self.bench_bin)
                self._profiler_data = _prof.format_for_prompt("BASELINE NCU PROFILE")
        except Exception:
            pass

        i = 0
        try:
            while max_iterations == 0 or i < max_iterations:
                i += 1
                extra = extra_instructions
                if self._profiler_data:
                    extra = self._profiler_data + "\n\n" + extra

                result = self._one_iteration(i, extra)
                self.history.add(result)

                # Log
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")

                # Print plain-text log line (always visible, no ANSI)
                _log_iteration(result, self.history)

                # Update live UI
                ui.update({
                    "iteration": i,
                    "status": result.status,
                    "ns_per_op": result.ns_per_op,
                    "n_instructions": result.n_instructions,
                    "max_register": result.max_register,
                    "error_detail": result.error_detail,
                })

                # If new best → run profiler and update context for AI
                is_new_best = (
                    self.history.best_idx == len(self.history.results) - 1
                    and result.status == "PASS"
                )
                if is_new_best and self.bench_bin and goals and goals.harness_type == "bench":
                    try:
                        # Profile the new best
                        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
                            # Re-inject to get the best cubin bytes
                            f.write(self._last_patched_bytes or b"")
                            tmp = f.name
                        prof = profile_cubin(tmp, self.spec.kernel_name,
                                             bench_bin=self.bench_bin)
                        import os; os.unlink(tmp)
                        self._profiler_data = prof.format_for_prompt(
                            f"NCU PROFILE (iteration {i}, {result.ns_per_op:.2f} ns/op)"
                        )
                    except Exception:
                        pass

        except KeyboardInterrupt:
            pass
        finally:
            ui.stop()

    def run_manual(self, sass_lines: list[str]) -> VariantResult:
        """Test a manually provided SASS variant."""
        i = len(self.history.results) + 1
        result = self._test_variant(i, sass_lines)
        self.history.add(result)
        self._display_result(result)
        return result

    # ── private ──

    def _one_iteration(self, iteration: int,
                       extra_instructions: str) -> VariantResult:
        """One iteration: prompt → LLM → assemble → inject → test."""

        # 1. Build prompt
        prompt = build_prompt(self.spec, self.history, extra_instructions)

        # 2. Call LLM
        try:
            response = self.llm_fn(prompt, _get_system_prompt())
            sass_lines = parse_sass_from_response(response)
        except Exception as e:
            return VariantResult(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                sass_text=[],
                n_instructions=0,
                max_register=0,
                status="LLM_ERROR",
                error_detail=str(e)[:200],
            )

        if not sass_lines:
            return VariantResult(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                sass_text=[],
                n_instructions=0,
                max_register=0,
                status="PARSE_ERROR",
                error_detail="No SASS instructions found in LLM response",
            )

        return self._test_variant(iteration, sass_lines)

    def _test_variant(self, iteration: int,
                      sass_lines: list[str]) -> VariantResult:
        """Assemble, inject, and test a SASS variant.

        Two modes depending on spec.template_path:
          - template mode: inject sass_lines into .sass template → compile with cubit asm
          - patch mode:    assemble individual instructions → binary-patch into cubin
        """
        if self.spec.template_path:
            return self._test_variant_template(iteration, sass_lines)

        # ── patch mode ──────────────────────────────────────────────────────

        # 3. Assemble
        try:
            instructions = self.asm.assemble_block(
                sass_lines, base_offset=self.spec.start_offset
            )
            # Pad with NOPs to exactly the cubin slot size
            # (end_offset - start_offset gives actual bytes in the cubin)
            actual_slot_bytes = self.spec.end_offset - self.spec.start_offset
            n_slots = actual_slot_bytes // 16
            while len(instructions) < n_slots:
                instructions.append(
                    self.asm.nop(offset=len(instructions) * 16)
                )
            if len(instructions) > n_slots:
                raise AssembleError(
                    f"Too many instructions: {len(instructions)} > {n_slots} "
                    f"(actual slot={actual_slot_bytes} bytes = {n_slots} insns). "
                    f"max_instructions budget in spec ({self.spec.max_instructions}) "
                    f"must be <= {n_slots}."
                )
            new_bytes = self.asm.block_to_bytes(instructions)
            # Hardware constraint checks (catch before GPU run)
            import re as _re
            hw_errors = []
            for idx, line in enumerate(sass_lines):
                # Zero-encoded: cubit silently encodes unknown instructions as 0x0
                b = new_bytes[idx*16:(idx+1)*16]
                if b == bytes(16):
                    hw_errors.append(f"ZERO_ENC #{idx}: {line[:60]}")
                # IMAD.WIDE: Rc must not equal Rd+1 (ILLEGAL_INSTRUCTION on SM120)
                # Format: IMAD.WIDE.U32 Rd, [Pout,] Ra, Rb, Rc [, Pin]
                if 'IMAD.WIDE' in line:
                    all_regs = _re.findall(r'\bR(\d+)\b', line)
                    if len(all_regs) >= 2:
                        rd = int(all_regs[0])  # first R is always Rd
                        # last R before semicolon is Rc (may be followed by Pin)
                        # Find Rc: last non-predicate register (< 255)
                        rc = int(all_regs[-1])
                        # Check if last token before ; is a register (not a pred)
                        last_tok = line.rstrip(' ;').split(',')[-1].strip()
                        if _re.match(r'R\d+$', last_tok):
                            rc_val = int(last_tok[1:])
                        elif len(all_regs) >= 2:
                            rc_val = int(all_regs[-1])
                        else:
                            rc_val = -1
                        if rc_val == rd + 1:
                            hw_errors.append(
                                f"BAD_WIDE #{idx}: Rc=R{rc_val}=Rd+1 ILLEGAL — "
                                f"use Rc=R{rd} or RZ instead: {line[:70]}"
                            )
            if hw_errors:
                raise AssembleError(
                    "Hardware constraint violation(s): " + "; ".join(hw_errors[:3])
                )
        except (AssembleError, Exception) as e:
            return VariantResult(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                sass_text=sass_lines,
                n_instructions=len(sass_lines),
                max_register=0,
                status="ASM_ERROR",
                error_detail=str(e)[:200],
            )

        # Compute max register used
        import re
        max_reg = 0
        for line in sass_lines:
            for m in re.finditer(r'\bR(\d+)\b', line):
                rn = int(m.group(1))
                if rn < 255:
                    max_reg = max(max_reg, rn)

        # 4. Inject into cubin (also patches regcount)
        try:
            patched = inject_hot_loop(self.spec.cubin_path, self.spec,
                                      new_bytes, max_reg=max_reg)
            self._last_patched_bytes = patched  # save for profiler
        except Exception as e:
            return VariantResult(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                sass_text=sass_lines,
                n_instructions=len(sass_lines),
                max_register=max_reg,
                status="INJECT_ERROR",
                error_detail=str(e)[:200],
            )

        # 5. Test — use bench_harness if goals are defined, else sass_test
        if self.spec.goals and self.spec.goals.harness_type == "bench":
            bench = bench_cubin(patched, self.spec, bench_bin=self.bench_bin)
            status = bench["status"]
            detail = bench.get("error", "")
            ns_op  = bench.get("ns_per_op")
            correct = (status == "PASS")
            elapsed = (ns_op * (self.spec.goals.iters / 1e6)) if ns_op else None
        else:
            status, detail = test_cubin(
                patched,
                kernel_name=self.spec.kernel_name,
                sass_test_bin=self.sass_test_bin,
            )
            ns_op = correct = elapsed = None

        return VariantResult(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            sass_text=sass_lines,
            n_instructions=len(sass_lines),
            max_register=max_reg,
            status=status,
            error_detail=detail,
            correct=correct,
            ns_per_op=ns_op,
            elapsed_ms=elapsed,
        )

    def _display_result(self, r: VariantResult):
        colors = {
            "PASS": "\033[32m", "FAIL": "\033[31m", "CRASH": "\033[91m",
            "TIMEOUT": "\033[33m", "ASM_ERROR": "\033[35m",
            "LLM_ERROR": "\033[35m", "PARSE_ERROR": "\033[35m",
            "INJECT_ERROR": "\033[35m",
        }
        c = colors.get(r.status, "")
        reset = "\033[0m" if c else ""
        if r.ns_per_op is not None:
            perf = f"{r.ns_per_op:7.2f} ns/op"
        elif r.elapsed_ms is not None:
            perf = f"{r.elapsed_ms:7.2f}ms"
        else:
            perf = ""
        is_best = " *** BEST ***" if (self.history.best_idx == len(self.history.results) - 1
                             and r.status == "PASS") else ""
        detail = r.error_detail[:50] if r.status != "PASS" else ""
        print(f"[{r.iteration:04d}] {c}{r.status:8s}{reset}  "
              f"{r.n_instructions:3d} insns  R{r.max_register:<3d}  "
              f"{perf:>16s}  {detail}{is_best}")

    def _test_variant_template(self, iteration: int,
                               sass_lines: list[str]) -> VariantResult:
        """Template-compilation mode: inject SASS into .sass file → cubit asm → bench.

        Used when spec.template_path is set (standalone kernel, e.g. MulModP).
        The template has HOT LOOP BEGIN / HOT LOOP END markers that are replaced.
        """
        import re, subprocess
        from pathlib import Path

        spec = self.spec
        n_insns = len(sass_lines)
        goals = spec.goals

        # Max register
        max_reg = 0
        for line in sass_lines:
            for m in re.finditer(r'\bR(\d+)\b', line):
                rn = int(m.group(1))
                if rn < 255:
                    max_reg = max(max_reg, rn)

        # Hardware constraint checks
        hw_errors = []
        for idx, line in enumerate(sass_lines):
            # R62/R63 forbidden as destination (SM120 hardware limit with regcount=64)
            first_token = line.split(',')[0].split()
            if len(first_token) >= 2:
                dest = first_token[-1].strip()
                if re.match(r'R(62|63)$', dest):
                    hw_errors.append(
                        f"FORBIDDEN_DEST #{idx}: R62/R63 cannot be written (SM120 regcount=64 "
                        f"limit). Use R60 or below: {line[:60]}"
                    )

            if 'IMAD.WIDE' in line:
                all_regs = re.findall(r'\bR(\d+)\b', line)
                if len(all_regs) >= 2:
                    rd = int(all_regs[0])
                    last_tok = line.rstrip(' ;').split(',')[-1].strip()
                    rc_val = int(last_tok[1:]) if re.match(r'R\d+$', last_tok) else -1
                    if rc_val == rd + 1:
                        hw_errors.append(
                            f"BAD_WIDE #{idx}: Rc=R{rc_val}=Rd+1 ILLEGAL (use Rc=R{rd} or RZ): "
                            f"{line[:60]}"
                        )
                    # IMAD.WIDE R60 writes R60:R61 (OK). R62 would write R63 (FORBIDDEN).
                    if rd >= 62:
                        hw_errors.append(
                            f"BAD_WIDE_DEST #{idx}: IMAD.WIDE R{rd} forbidden "
                            f"(max IMAD.WIDE dest is R60): {line[:60]}"
                        )
            # P1 is reserved for the loop counter (in template) — AI must not write it
            if 'P1' in line and re.search(r'\bIADD3\b', line):
                m2 = re.match(r'IADD3(?:\.X)?\s+\S+,\s+(\S+),\s+(\S+),', line)
                if m2:
                    if m2.group(1).rstrip(',') == 'P1' or m2.group(2).rstrip(',') == 'P1':
                        hw_errors.append(
                            f"P1_WRITE #{idx}: P1 is reserved for template loop counter — "
                            f"use P0,P2,P3,P4,P5 instead: {line[:60]}"
                        )

        if hw_errors:
            return VariantResult(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                sass_text=sass_lines, n_instructions=n_insns, max_register=max_reg,
                status="ASM_ERROR",
                error_detail="HW: " + "; ".join(hw_errors[:2]),
            )

        # Inject into template
        try:
            template = Path(spec.template_path).read_text()
            begin_idx = template.find(spec.template_begin_marker)
            end_idx   = template.find(spec.template_end_marker)
            if begin_idx < 0 or end_idx < 0:
                raise AssembleError(
                    f"Markers '{spec.template_begin_marker}' / "
                    f"'{spec.template_end_marker}' not found in {spec.template_path}"
                )
            hot_block = "\n".join(f"    {l}" for l in sass_lines)
            new_template = (template[:begin_idx] +
                            spec.template_begin_marker + "\n" +
                            hot_block + "\n" +
                            template[end_idx:])

            tmp_sass = tempfile.NamedTemporaryFile(
                suffix=".sass", delete=False, mode="w",
                prefix=f"{spec.name}_iter{iteration}_"
            )
            tmp_sass.write(new_template)
            tmp_sass.close()
            cubin_path = tmp_sass.name.replace(".sass", ".cubin")

            cubit_dir = spec.cubit_dir or os.environ.get('CUBIT_DIR', '')
            cubit_bin = (str(Path(cubit_dir) / "target" / "release" / "cubit")
                         if cubit_dir else shutil.which('cubit') or 'cubit')
            r = subprocess.run(
                [cubit_bin, "asm", tmp_sass.name, "-o", cubin_path],
                cwd=cubit_dir or None,
                capture_output=True, text=True, timeout=30
            )
            os.unlink(tmp_sass.name)

            if r.returncode != 0:
                return VariantResult(
                    iteration=iteration, timestamp=datetime.now().isoformat(),
                    sass_text=sass_lines, n_instructions=n_insns, max_register=max_reg,
                    status="ASM_ERROR", error_detail=f"cubit: {r.stderr[:150] or r.stdout[:150]}"
                )
            # Check for partially-failed encoding
            m_fail = re.search(r'\((\d+) failed\)', r.stdout)
            if m_fail and int(m_fail.group(1)) > 0:
                try: os.unlink(cubin_path)
                except: pass
                return VariantResult(
                    iteration=iteration, timestamp=datetime.now().isoformat(),
                    sass_text=sass_lines, n_instructions=n_insns, max_register=max_reg,
                    status="ASM_ERROR",
                    error_detail=f"cubit: {m_fail.group(1)} instruction(s) failed to encode"
                )
        except (AssembleError, Exception) as e:
            return VariantResult(
                iteration=iteration, timestamp=datetime.now().isoformat(),
                sass_text=sass_lines, n_instructions=n_insns, max_register=max_reg,
                status="ASM_ERROR", error_detail=str(e)[:200],
            )

        # Test — bench_harness
        self._last_patched_bytes = Path(cubin_path).read_bytes()
        try:
            bench = bench_cubin(self._last_patched_bytes, spec, bench_bin=self.bench_bin)
        finally:
            try: os.unlink(cubin_path)
            except: pass

        status = bench["status"]
        ns_op  = bench.get("ns_per_op")
        detail = bench.get("error", "")

        return VariantResult(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            sass_text=sass_lines,
            n_instructions=n_insns,
            max_register=max_reg,
            status=status,
            error_detail=detail,
            correct=(status == "PASS"),
            ns_per_op=ns_op,
        )

    def _display_summary(self):
        total = len(self.history.results)
        passes = sum(1 for r in self.history.results if r.status == "PASS")
        rate = passes / total * 100 if total else 0
        best = self.history.best
        best_str = (f"#{best.iteration} ({best.n_instructions} insns, R{best.max_register})"
                    if best else "none")
        print(f"  --- {passes}/{total} pass ({rate:.0f}%), best: {best_str} ---")
