#!/usr/bin/env python3
"""Measure SM120 instruction def-to-use latencies on RTX 5090.

Uses Sm120Assembler (cubit-backed) to build SASS kernels, injects into
a stub cubin from nvcc, runs on the GPU.  Zero raw cubit.encode() calls.

Method:
  - Serial chain of N dependent instructions with stall=1 (minimum).
  - Hardware pipeline stalls on RAW dependency → total = N × latency.
  - Timed with S2R SR_CLOCKLO (stall=31 to guarantee result ready).

Usage:
    python3 measure_latencies.py              # all opcodes
    python3 measure_latencies.py --json       # JSON output
    python3 measure_latencies.py --op IADD3   # single opcode
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Auto-discover sm120.json for cubit before any imports
_here = Path(__file__).resolve().parent
for _p in [_here] + list(_here.parents):
    _candidate = _p / "blackwell-isa" / "sm120.json"
    if _candidate.is_file():
        os.environ.setdefault("CUBIT_TABLE", str(_candidate))
        break

sys.path.insert(0, str(_here.parent.parent / "src"))

from cubit.assembler import Assembler, Instruction
from sasskit.core.cubin import Cubin

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
HARNESS = str(_here.parent.parent / "harnesses" / "generic_harness")
STUB_CU = str(_here.parent.parent / "harnesses" / "encoder_test.cu")
CHAIN_LEN = 100
WARMUP = 3
MEASURE = 15


# ── Prologue: verbatim from nvcc's encoder_test stub ─────────
# These match the CUDA kernel ABI for: void f(uint32_t* out, uint32_t n)

PROLOGUE_BINARY = [
    # LDC R1, c[0x0][0x37c]        — stack pointer
    (0x0000df00ff017b82, 0x000e220000000800),
    # S2R R0, SR_TID.X             — thread ID
    (0x0000000000007919, 0x000e640000002100),
    # LDCU.64 UR4, c[0x0][0x358]   — output descriptor
    (0x00006b00ff0477ac, 0x000e620008000a00),
    # LDC.64 R2, c[0x0][0x380]     — output base pointer → R2:R3
    (0x0000e000ff027b82, 0x000e620000000a00),
]


# ── Opcode definitions ─────────────────────────────────────────
# (name, sass_template)
# Template uses R5 as accumulator (serial RAW dependency chain).

OPCODES = [
    # INT ALU
    ("IADD3",      "IADD3 R5, PT, PT, R5, R6, RZ"),
    ("IMAD",       "IMAD R5, R5, R6, R7"),
    ("IMAD_HI",    "IMAD.HI.U32 R5, R5, R6, RZ"),
    ("LOP3",       "LOP3.LUT R5, R5, R6, RZ, 0x96, !PT"),
    ("SHF",        "SHF.L.U32 R5, R5, 0x3, R5"),
    ("PRMT",       "PRMT R5, R5, 0x3210, R5"),
    ("POPC",       "POPC R5, R5"),
    ("FLO",        "FLO.U32 R5, R5"),
    ("BREV",       "BREV R5, R5"),
    ("MOV",        "MOV R5, R5"),
    ("SEL",        "SEL R5, R5, R6, PT"),
    ("SGXT",       "SGXT R5, R5, 0x10"),

    # FP32
    ("FADD",       "FADD R5, R5, 1.0"),
    ("FMUL",       "FMUL R5, R5, R6"),
    ("FFMA",       "FFMA R5, R5, R6, R7"),
    ("MUFU_RCP",   "MUFU.RCP R5, R5"),
    ("MUFU_SQRT",  "MUFU.SQRT R5, R5"),
    ("MUFU_SIN",   "MUFU.SIN R5, R5"),
    ("MUFU_EX2",   "MUFU.EX2 R5, R5"),
    ("MUFU_LG2",   "MUFU.LG2 R5, R5"),

    # FP64 (R4:R5 pair)
    ("DADD",       "DADD R4, R4, 1.0"),
    ("DMUL",       "DMUL R4, R4, R6"),
    ("DFMA",       "DFMA R4, R4, R6, R8"),

    # ── Round 2: additional opcodes ──

    # INT (more)
    # ISETP/FSETP/DSETP write predicates — can't do R5→R5 chain.
    # Skip for now; latency ≈ same as SEL (4c) since same pipe.
    # ("ISETP",      ...),
    ("LEA",        "LEA R5, R5, R6, 0x2"),
    ("IABS",       "IABS R5, R5"),
    ("P2R",        "P2R R5, PR, R5, 0xff"),

    # FP32 (more)
    # ("FSETP" — predicate output, skip)
    ("FMNMX",      "FMNMX R5, R5, R6, PT"),

    # FP16
    ("HFMA2",      "HFMA2 R5, R5, R6, R7"),

    # FP64 (more)
    # ("DSETP" — predicate output, skip)

    # Uniform
    # UMOV uses UR regs, not R5 — would need different chain setup, skip
    # ("UMOV",       "UMOV UR4, UR5"),

    # Special
    ("S2R",        "S2R R5, SR_CLOCKLO"),

    # MUFU remaining
    ("MUFU_LG2",   "MUFU.LG2 R5, R5"),
    ("MUFU_RSQ",   "MUFU.RSQ R5, R5"),
    ("MUFU_COS",   "MUFU.COS R5, R5"),

    # IMAD.WIDE (R4:R5 = 64-bit result)
    ("IMAD_WIDE",  "IMAD.WIDE.U32 R4, R5, R6, R4"),

    # Integer bit ops
    ("BMSK",       "BMSK R5, R5, R6"),
    ("FLO_SH",     "FLO.U32.SH R5, R5"),

    # FP32 rounding / conversion
    ("FRND",       "FRND.FLOOR R5, R5"),
    ("F2I",        "F2I.TRUNC.NTZ R5, R5"),
    ("I2F",        "I2F R5, R5"),

    # ── Round 3: SHFL, FP16 variants, FP32↔FP16 conversion ──────────────────

    # Warp shuffle — skip: InsKey operand order mismatch (P_R vs R_P) in cubit table
    # ("SHFL_IDX",   "SHFL.IDX R5, PT, R5, 0x0, 0x1f"),

    # FP16 packed (×2) ops — R5 = two packed FP16 values
    ("HADD2",      "HADD2 R5, R5, R6"),
    ("HMUL2",      "HMUL2 R5, R5, R6"),

    # FP32↔FP16 conversion
    # F2F.F32.F16 not in table as standalone — use F2F_R_R[F32,F64] which exists
    # ("F2F_F32_F16", "F2F.F32.F16 R5, R5"),   # InsKey form mismatch
    ("F2F_F16_F32", "F2F.F16.F32 R5, R5"),    # 8 cycles confirmed

    # MUFU.TANH
    ("MUFU_TANH",  "MUFU.TANH R5, R5"),        # 8 cycles confirmed

    # IMAD.SHL (shift-left multiply-add)
    ("IMAD_SHL",   "IMAD.SHL.U32 R5, R5, 0x2, RZ"),  # 4 cycles

    # BMSK.W variant
    ("BMSK_W",     "BMSK.W R5, R5, R6"),        # 4 cycles

    # VIMNMX — packed integer min/max with predicate comparison
    ("VIMNMX_S32", "VIMNMX.S32 R5, R5, 0x1, PT"),

    # ── Round 4: Tensor core ops ─────────────────────────────────────────────
    # HMMA.1688.F16: warp-level (32 threads), smaller shape
    # R0:R1 = Rd/Rc (accumulator pair), R4:R7 = Ra (A matrix), R8:R9 = Rb (B matrix)
    # NOTE: requires warpgroup (128 threads) on SM120 — fails with 32 threads
    # Skipped: ("HMMA",  "HMMA.16816.F32 R0, R8.ROW, R12.COL, R0"),
    # TODO: run with 128-thread harness for tensor core latency measurement
]


def build_kernel(asm: Assembler, sass_template: str, chain_len: int) -> list[Instruction]:
    """Build a latency benchmark kernel using cubit.Assembler.

    Layout:
      prologue (verbatim binary from nvcc stub)
      NOP wait-all-barriers (drain prologue loads)
      IMAD.WIDE R2, R0, 8, R2 (per-thread output offset)
      setup constants (R5=seed, R6=3, R7=1, R8=1, R4=R5)
      NOP stall=15 (drain)
      S2R R9, SR_CLOCKLO stall=31 (start clock, wait for result)
      chain × N stall=1 (serial RAW dependency)
      S2R R10, SR_CLOCKLO stall=31 (end clock, wait for result)
      IADD3 R9, R10, -R9 (diff)
      STG diff + chain result
      EXIT
    """
    instrs: list[Instruction] = []
    off = [0]  # mutable counter

    def add_binary(lo: int, hi: int):
        instrs.append(Instruction(off[0], lo, hi, "(binary)"))
        off[0] += 16

    def add(sass: str, **kwargs):
        inst = asm.assemble(sass, offset=off[0], **kwargs)
        instrs.append(inst)
        off[0] += 16

    # Prologue (binary — must match nvcc ABI exactly)
    for lo, hi in PROLOGUE_BINARY:
        add_binary(lo, hi)

    # Wait for all prologue barriers to complete
    add("NOP", stall=15, wait_mask=0x3F)

    # Per-thread output offset: R2:R3 += tid * 8
    add("IMAD.WIDE.U32 R2, R0, 0x8, R2", stall=15)

    # Setup constants
    add("IADD3 R5, PT, PT, R0, 0x1, RZ", stall=15)  # seed = tid+1 (avoid zero for MUFU)
    add("MOV R6, 0x3", stall=15)
    add("MOV R7, 0x1", stall=15)
    add("MOV R8, 0x1", stall=15)
    add("IADD3 R4, PT, PT, R5, RZ, RZ", stall=15)    # R4=R5 for fp64 pair

    # Drain
    add("NOP", stall=15)

    # Start clock with wbar=2, then wait on it
    add("S2R R9, SR_CLOCKLO", stall=15, write_bar=2)
    add("NOP", stall=15, wait_mask=0x04)  # wait barrier 2

    # Dependency chain (stall=1: hardware stalls on RAW reveal true latency)
    for _ in range(chain_len):
        add(sass_template, stall=1)

    # End clock with wbar=3, then wait on it
    add("S2R R10, SR_CLOCKLO", stall=1, write_bar=3)
    add("NOP", stall=15, wait_mask=0x08)  # wait barrier 3

    # diff = end - start
    add("IADD3 R9, PT, PT, R10, -R9, RZ", stall=15)

    # Store results
    add("STG.E desc[UR4][R2.64], R9", stall=15)
    add("STG.E desc[UR4][R2.64+0x4], R5", stall=15)

    # EXIT
    add("EXIT", stall=15)

    # Pad to 128-byte alignment
    instrs = asm.pad_to_alignment(instrs, 128)

    return instrs


def inject_and_run(op_name: str, sass_template: str, chain_len: int) -> dict:
    """Build kernel, inject into stub cubin, run on GPU."""
    with __import__('tempfile').TemporaryDirectory(prefix="latbench_") as tmpdir:
        cubin_path = os.path.join(tmpdir, "bench.cubin")

        # 1. Compile stub
        r = subprocess.run(
            [NVCC, "-cubin", "-arch=sm_120", "-o", cubin_path, STUB_CU],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return {"name": op_name, "error": f"nvcc: {r.stderr[:200]}"}

        # 2. Build kernel
        try:
            asm = Assembler()
            instrs = build_kernel(asm, sass_template, chain_len)
        except Exception as e:
            return {"name": op_name, "error": f"asm: {e}"}

        # 3. Inject into cubin
        try:
            cb = Cubin.from_file(cubin_path)
            ki = cb.get_kernel("encoder_test")

            needed = len(instrs) * 16
            if ki.text_size < needed:
                extra = (needed - ki.text_size + 15) // 16
                cb.grow_kernel_text(ki, extra)
                cb.save(cubin_path)
                cb = Cubin.from_file(cubin_path)
                ki = cb.get_kernel("encoder_test")

            for inst in instrs:
                cb.write_instruction(ki, inst.offset, inst.instr_word, inst.ctrl_word)

            # Fill remaining with NOPs
            nop = asm.nop()
            for i in range(len(instrs) * 16, ki.text_size, 16):
                cb.write_instruction(ki, i, nop.instr_word, nop.ctrl_word)

            cb.set_reg_count(ki, 14)
            cb.save(cubin_path)
        except Exception as e:
            import traceback
            return {"name": op_name, "error": f"inject: {e}",
                    "tb": traceback.format_exc()}

        # 4. Run and collect timing
        cycles_list = []
        last_stdout = ""
        for i in range(WARMUP + MEASURE):
            r = subprocess.run(
                [HARNESS, cubin_path, "encoder_test", "32", "0"],
                capture_output=True, text=True, timeout=10)
            last_stdout = r.stdout
            if r.returncode != 0:
                if i == 0:
                    return {"name": op_name,
                            "error": f"gpu: rc={r.returncode} {r.stderr[:200]}",
                            "stdout": r.stdout[:300]}
                continue
            m = re.search(r'\[0\]=0x([0-9a-fA-F]+)', r.stdout)
            if m and i >= WARMUP:
                val = int(m.group(1), 16)
                if 1 < val < 500000:
                    cycles_list.append(val)

        if not cycles_list:
            return {"name": op_name, "error": "no timing data",
                    "stdout": last_stdout[:500]}

        cycles_list.sort()
        median = cycles_list[len(cycles_list) // 2]
        lat = median / chain_len

        return {
            "name": op_name,
            "chain_length": chain_len,
            "median_cycles": median,
            "def_latency": round(lat, 2),
            "def_latency_int": round(lat),
            "min_cycles": min(cycles_list),
            "max_cycles": max(cycles_list),
            "samples": len(cycles_list),
        }


def main():
    parser = argparse.ArgumentParser(description="SM120 latency benchmark")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--op", type=str)
    parser.add_argument("--chain", type=int, default=CHAIN_LEN)
    args = parser.parse_args()

    ops = [(n, s) for n, s in OPCODES if not args.op or n == args.op]
    if args.op and not ops:
        print(f"Unknown: {args.op}. Available: {', '.join(n for n,s in OPCODES)}",
              file=sys.stderr)
        sys.exit(1)

    results = []
    for name, sass in ops:
        print(f"  {name:14s}", file=sys.stderr, end=" ", flush=True)
        r = inject_and_run(name, sass, args.chain)
        if "error" in r:
            print(f"ERROR: {r['error'][:60]}", file=sys.stderr)
        else:
            print(f"→ {r['def_latency']:5.1f} cycles", file=sys.stderr)
        results.append(r)

    if args.json:
        clean = [{k: v for k, v in r.items() if k not in ("stdout", "tb")}
                 for r in results]
        json.dump(clean, sys.stdout, indent=2)
        print()
    else:
        print(f"\n{'='*65}")
        print(f"SM120 Instruction Latencies (RTX 5090, chain={args.chain})")
        print(f"{'='*65}")
        print(f"{'Opcode':<14s} {'Latency':>8s} {'Median':>8s} {'Min':>7s} {'Max':>7s}")
        print(f"{'-'*65}")
        for r in results:
            if "error" in r:
                print(f"{r['name']:<14s} {'ERROR':>8s}   {r['error'][:40]}")
            else:
                print(f"{r['name']:<14s} {r['def_latency']:>7.1f}c "
                      f"{r['median_cycles']:>7d} {r['min_cycles']:>7d} {r['max_cycles']:>7d}")


if __name__ == "__main__":
    main()
