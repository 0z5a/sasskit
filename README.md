# sasskit

**Post-compilation SASS optimizer for NVIDIA SM120 (Blackwell / RTX 5090).**

Achieves **22% speedup** on real GPU kernels by optimizing SASS assembly after `ptxas` compilation — without modifying source code.

## What it does

sasskit takes a compiled CUDA binary (`.cubin`) and makes it faster through:

1. **Reforge** — empirical SASS-to-SASS optimization via mutation + GPU benchmarking
2. **Register recoloring** — reduce register count to improve occupancy
3. **Instruction scheduling** — stall count analysis and optimization

All optimizations are transparent: no source code changes, no recompilation.

## Results

### Reforge: 22% speedup on RTX 5090

```
$ sasskit reforge kernel.cubin -k KernelA -n 300 --temperature 0

Baseline: 115.8 ms/iter
Best:      94.7 ms/iter (1.22x) ✓

Key mutations:
  stall -1 on ISETP.NE.AND → 21.5% gain (ptxas over-stalled predicate cmp)
  stall -1 on LOP3.LUT     → 3.6% gain
  swap LOP3.LUT ↔ IADD3.X  → additional gain
```

ptxas is **not optimal** on SM120. Empirical mutation discovers improvements that analytical approaches miss.

### Register recoloring: 80 → 64 registers

```
$ sasskit optimize kernel.cubin -k KernelA -t 64

KernelA: 80 → 64 regs (8 spills, 61 trampolines) ✓
```

Reduces register pressure via graph coloring with spill-to-shared-memory trampolines. Verified on RTX 5090.

## How reforge works

```
┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
│ Decode SASS │────▶│  Mutate  │────▶│ Encode +  │────▶│  GPU     │
│  (cubit)    │     │ (random) │     │ Patch     │     │ Benchmark│
└─────────────┘     └──────────┘     └───────────┘     └──────────┘
                         ▲                                   │
                         └──── accept if faster ─────────────┘
```

**Mutations:** instruction swap, load hoisting, stall ±1, NOP removal

**Swap scope:** only SM120 `MOV RZ, imm32` pairs with identical, explicitly
recognized scheduling controls are admitted. These discard moves have no live
register or memory effects. Memory/load hoisting, live GPR destinations,
predicates, uniform registers, unknown forms and unsupported controls are
rejected in both proposal and application. The current whitelist does not
provide useful live-instruction scheduling optimizations: those require a model
of surrounding latency/barrier dependencies. Stall/NOP mutations remain
experimental and are not covered by this swap policy. Crash detection alone is
not a numerical correctness check.

**Speed:** ~1 second per iteration, 300 iterations = 5 minutes

Inspired by [CuAsmRL](https://arxiv.org/abs/2501.08071) (CGO '25, 9% avg on Ampere) but uses hill-climbing instead of RL and supports SM120 Blackwell.

## Installation

```bash
# Requires: cubit (SM120 assembler), RTX 5090 GPU, CUDA 12.8+
pip install -e .
```

## Usage

```bash
# SASS-to-SASS optimization (reforge)
sasskit reforge kernel.cubin -k MyKernel -n 300

# Register recoloring
sasskit optimize kernel.cubin -k MyKernel -t 64 -o optimized.cubin

# Analysis
sasskit analyze kernel.cubin -k MyKernel -t 64

# Disassembly
sasskit disassemble kernel.cubin -k MyKernel
```

Each reforge call uses a private temporary workspace. The input is read-only;
`--work-dir` selects the existing parent directory for temporary files. The best
result is published outside that workspace and remains available after cleanup:

```bash
sasskit reforge kernel.cubin -k MyKernel -n 300 --work-dir /tmp -o best.cubin
```

Existing output files (including input aliases) are never overwritten. Without
`-o`, each call creates `./reforge-results/<unique-run>/best.cubin`.
Python callers can use the keyword-only `output_path` and `work_dir` arguments;
`state.best_cubin_path` identifies the published result. A run with no accepted
mutation publishes the baseline if its benchmark is valid.

## Architecture

```
sasskit/
├── schedule/
│   ├── reforge.py      # SASS-to-SASS optimizer (mutation + GPU bench)
│   └── stall_opt.py    # Stall count analysis
├── recolor/
│   ├── coloring.py     # Graph coloring with spill management
│   ├── patcher.py      # Binary patching + trampoline generation
│   └── cli.py          # Command-line interface
├── analysis/
│   ├── cfg.py          # Control-flow graph
│   ├── liveness.py     # Register liveness analysis
│   └── interference.py # Interference graph
├── core/
│   ├── cubin.py        # CUDA ELF parser/writer
│   ├── decoder.py      # SM120 instruction decoder
│   └── isa_db.py       # ISA database interface
└── forge/
    └── engine.py       # LLM-driven SASS synthesis
```

## Key discoveries

- **SM120 stall field is 4 bits** `[44:41]`, bit 45 is yield (not 5-bit stall as previously assumed)
- **ptxas over-stalls ISETP** on SM120 by at least 1 cycle, costing 21% throughput
- **SM120 needs 5+ register headroom** above the highest used register (`EIATTR >= max_reg + 6`)
- **Shared memory spill slots must offset past kernel's static smem** (offset 0 conflicts with kernel data)

## Related projects

- [cubit](https://github.com/kacper-daftcode/cubit) — SM120 SASS assembler/disassembler (100% roundtrip accuracy)
- [blackwell-isa](https://github.com/kacper-daftcode/blackwell-isa) — SM120 ISA database (1,994 instruction forms)

## License

MIT
