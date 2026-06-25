# Design: SASS Instruction Scheduling Engine

## Status: PROPOSAL

## Authors: sasskit team

## Date: 2025-02-13

---

## 1. Problem Statement

NVIDIA's `ptxas` compiler produces SASS binaries with suboptimal instruction
schedules. This manifests as:

- **Unnecessary pipeline stalls.** Stall counts (`ctrl_word[3:0]`) set
  conservatively — the compiler waits longer than needed between dependent
  instructions.
- **Poor latency hiding.** Memory loads (`LDG`, `LDS`) not interleaved
  with independent compute — the pipeline stalls waiting for data that
  could have been requested earlier.
- **Missed dual-issue opportunities.** sm_120 can co-issue two instructions
  per cycle (integer+FP, integer+memory) but `ptxas` doesn't aggressively
  pair them.
- **Excessive barrier usage.** Write/read barriers (`ctrl_word[11:8]`,
  `[15:12]`) and wait masks (`ctrl_word[21:16]`) assigned conservatively,
  adding latency even when data is already available.

CuAsmRL (He & Yoneki, CGO 2025) demonstrated that RL-based schedule mutation
yields 9% geomean / 26% peak improvement on Ampere. Their approach proves the
opportunity exists. This design describes a **better** approach — faster,
more principled, and integrated with sasskit's register optimization.

---

## 2. Lessons from CuAsmRL

### What they got right

1. **SASS-level scheduling works.** Even well-optimized `-O3` code has ~9%
   room for improvement. This validates the entire endeavor.
2. **Restricting moves to within basic blocks.** Preserves control flow
   correctness without needing alias analysis or memory disambiguation.
3. **Latency inference from existing schedules.** Instead of micro-benchmarking
   every opcode, infer latencies from the stall counts `ptxas` placed between
   use-def pairs. If `ptxas` set stall=5 between an IMAD and its consumer,
   IMAD latency is ≤5. Aggregate across many kernels → latency table.
4. **Actual GPU execution for ground truth.** Cost models approximate; the GPU
   doesn't lie.

### What they got wrong

1. **RL is the wrong hammer.** Instruction scheduling within a basic block is
   a well-studied combinatorial optimization problem (list scheduling, modulo
   scheduling). Decades of compiler research solve it without training a neural
   network from scratch per kernel. PPO takes thousands of GPU evaluations to
   converge to what a good heuristic finds in milliseconds.

2. **The action space is crippled.** Only memory instructions can move, only by
   one position (swap with neighbor). This means:
   - Moving an instruction 10 positions takes 10 actions (10 GPU evaluations).
   - Compute instructions can never move.
   - No dual-issue pairing.
   - No stall count optimization independent of instruction order.

3. **Zero transfer learning.** The agent is trained from scratch for each
   kernel. A kernel with 3000 instructions needs thousands of episodes. Change
   one tile size → completely retrain. A cost model or LLM transfers instantly.

4. **Expensive.** Training takes O(hours) per kernel with GPU-in-the-loop for
   every single swap evaluation. Our target: O(minutes) including verification.

5. **No interaction with register allocation.** Instruction scheduling and
   register allocation are coupled — moving a load earlier extends its
   destination's live range, potentially increasing register pressure. CuAsmRL
   is oblivious to this. sasskit already has liveness analysis; a joint
   optimizer can respect pressure constraints.

---

## 3. Design Overview

### Architecture

```
Input: .cubin (SASS binary)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 1: ANALYSIS  (existing sasskit infrastructure)    │
│                                                         │
│  • Decode instructions (decoder.py)                     │
│  • Build CFG (cfg.py)                                   │
│  • Liveness analysis (liveness.py)                      │
│  • Interference graph (interference.py)                 │
│  ──── new ────                                          │
│  • Dependency DAG per basic block                       │
│  • Instruction classification & latency estimation      │
│  • Critical path analysis                               │
│  • Register pressure profile                            │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: CLASSICAL SCHEDULING  (new: schedule/)         │
│                                                         │
│  Multiple heuristics produce candidate schedules:       │
│   • Critical-path-first list scheduling                 │
│   • Memory-first (loads as early as possible)           │
│   • Dual-issue-aware pairing                            │
│   • Pressure-bounded scheduling (respect reg target)    │
│                                                         │
│  Cost model ranks candidates without GPU.               │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: LLM REFINEMENT  (Anthropic Claude)             │
│                                                         │
│  Present best schedule + annotations to Claude.         │
│  Claude identifies non-obvious improvements:            │
│   • Software pipelining patterns                        │
│   • Cross-iteration latency hiding in loops             │
│   • Instruction selection alternatives                  │
│                                                         │
│  Generates 2-3 refined candidates.                      │
│  Cost model filters.                                    │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: CONTROL WORD OPTIMIZATION  (new)               │
│                                                         │
│  For each surviving candidate:                          │
│   • Compute minimum legal stall counts                  │
│   • Assign barriers optimally (minimum barrier count)   │
│   • Set yield hints for warp scheduling                 │
│                                                         │
│  This is deterministic — no search needed.              │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 5: GPU VERIFICATION  (3-5 candidates only)        │
│                                                         │
│  Assemble → inject → execute → measure throughput.      │
│  Select winner.                                         │
│                                                         │
│  If winner beats original by >2% → accept.              │
│  If not → keep original schedule.                       │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
Output: Optimized .cubin
```

### Why this beats CuAsmRL

| Dimension | CuAsmRL | This design |
|-----------|---------|-------------|
| **GPU evaluations** | ~5000 per kernel | 3-5 per kernel |
| **Wall time** | Hours | Minutes |
| **Action space** | Swap memory inst ±1 | Full topological reordering + dual-issue + stall tuning |
| **Transfer** | None (retrain per kernel) | LLM knowledge transfers, cost model generalizes |
| **Register awareness** | None | Pressure-bounded scheduling |
| **Interpretability** | Black-box RL policy | Classical heuristics + LLM reasoning |
| **Correctness** | Empirical (GPU test) | Structural (dependency preservation) + GPU test |

---

## 4. Phase 1: Dependency Analysis

### 4.1 Dependency DAG

For each basic block, construct a directed acyclic graph of instruction
dependencies. An instruction B depends on instruction A if:

**True dependency (Read-After-Write, RAW):**
A writes register Rn, B reads Rn, no intervening write to Rn.

**Anti-dependency (Write-After-Read, WAR):**
A reads Rn, B writes Rn. After register renaming (sasskit recolor), most
WAR dependencies vanish. For scheduling, we can handle WAR by ensuring
we don't move a write before a prior read of the same register.

**Output dependency (Write-After-Write, WAW):**
A writes Rn, B writes Rn. Must preserve write order.

**Memory dependencies:**
- Load-after-store to same address (alias): conservatively assume ALL
  stores may alias any subsequent load unless provably independent.
- Store-after-store: must preserve order.
- Loads are freely reorderable with respect to each other.

**Barrier dependencies:**
- If instruction A sets write-barrier Bn, and instruction B waits on
  barrier Bn, then B depends on A.
- These encode variable-latency dependencies (memory ops).

### 4.2 Instruction classification

```python
@dataclass
class SchedInfo:
    """Scheduling-relevant properties of an instruction."""
    inst: Instruction
    # Classification
    is_memory: bool           # LDG, STG, LDS, STS, LDGSTS, ATOM, RED
    is_compute_int: bool      # IADD, IMAD, SHF, LOP3, SEL, ...
    is_compute_fp: bool       # FFMA, FADD, FMUL, ...
    is_control: bool          # BRA, EXIT, BSSY, BSYNC, BAR, ...
    is_tensor: bool           # HMMA, DMMA, ...

    # Latency
    latency_cycles: int       # Fixed latency (or 0 for variable)
    is_variable_latency: bool # True for memory ops

    # Dual-issue eligibility
    dual_issue_class: str     # "INT", "FP", "MEM", "CTRL", "TENSOR", "NONE"

    # Pressure impact
    regs_defined: set[int]    # Registers written (live range starts)
    regs_killed: set[int]     # Registers whose live ranges end HERE
    pressure_delta: int       # net change in live registers (+define, -kill)
```

### 4.3 Latency estimation

We use three sources, in order of confidence:

1. **Known latency table** (from micro-benchmarks and literature):
   ```
   IADD3:    4 cycles       LDG:   ~200-400 cycles (variable)
   IMAD:     5 cycles       LDS:   ~20-30 cycles (variable)
   IMAD.HI:  5 cycles       STS:   ~20-30 cycles (variable)
   FFMA:     4 cycles       LDGSTS: ~200-400 cycles (variable)
   SHF:      6 cycles       STG:   ~200-400 cycles (variable)
   LOP3:     4 cycles       ATOM:  ~200-400 cycles (variable)
   MOV:      2 cycles       BAR:   ~20-50 cycles (variable)
   SEL:      4 cycles
   ```

2. **Inferred from ptxas stall counts** (CuAsmRL's insight):
   For each use-def pair (A defines Rn, B uses Rn) in the same basic block,
   the accumulated stall count between A and B is an upper bound on A's
   latency. Take the minimum across all such pairs for each opcode.

3. **LLM knowledge** (fallback for exotic instructions):
   Ask Claude about unfamiliar opcodes. Claude has broad knowledge of GPU
   microarchitecture from training data.

### 4.4 Critical path

The critical path through the dependency DAG determines the theoretical
minimum execution time. For each instruction, compute:

- **ASAP** (As Soon As Possible): earliest cycle the instruction can execute.
- **ALAP** (As Late As Possible): latest cycle without delaying the block.
- **Slack** = ALAP - ASAP: how much freedom to schedule this instruction.

Instructions with slack=0 are on the critical path. These MUST be scheduled
optimally. Instructions with high slack can be moved freely for dual-issue
pairing or pressure control.

---

## 5. Phase 2: Classical Scheduling

### 5.1 List scheduling

The workhorse algorithm. Maintains a priority queue of ready instructions
(all predecessors in DAG already scheduled). Each cycle, picks the highest-
priority ready instruction and schedules it.

```python
def list_schedule(dag: DepDAG, priority_fn, pressure_limit: int = None):
    """Schedule instructions using list scheduling.

    Args:
        dag: Dependency DAG for one basic block
        priority_fn: (inst) -> priority value (higher = schedule first)
        pressure_limit: If set, delay definitions that would push
                        pressure above this limit (register-aware)
    """
    scheduled = []
    ready = [n for n in dag.nodes if dag.in_degree(n) == 0]
    cycle = 0

    while ready:
        # Sort by priority
        ready.sort(key=priority_fn, reverse=True)

        # Optionally check pressure constraint
        chosen = None
        for candidate in ready:
            if pressure_limit is not None:
                new_pressure = current_pressure + candidate.pressure_delta
                if new_pressure > pressure_limit and len(ready) > 1:
                    continue  # skip, try next
            chosen = candidate
            break

        if chosen is None:
            chosen = ready[0]  # forced to exceed pressure

        scheduled.append((cycle, chosen))
        ready.remove(chosen)

        # Release dependents
        for succ in dag.successors(chosen):
            if all(pred in scheduled_set for pred in dag.predecessors(succ)):
                ready.append(succ)

        cycle += chosen.latency
    return scheduled
```

### 5.2 Priority heuristics

We generate multiple schedules using different priority functions:

1. **Critical-path-first (CPF):**
   Priority = length of longest path from this instruction to any exit.
   Classic heuristic. Maximizes ILP by scheduling critical instructions first
   and filling gaps with non-critical work.

2. **Memory-first (MF):**
   Priority = is_memory * 1000 + critical_path_length.
   Schedules memory loads as early as possible to maximize latency hiding.
   Based on CuAsmRL's key insight that load placement is the biggest lever.

3. **Dual-issue-aware (DIA):**
   After primary scheduling with CPF, make a second pass that reorders
   adjacent instructions to maximize dual-issue pairs:
   - INT + FP: always eligible
   - INT + MEM: eligible if independent
   - FP + MEM: eligible if independent
   The cost model evaluates whether dual-issue reordering helps.

4. **Pressure-bounded (PB):**
   CPF with `pressure_limit = target_regs`. Delays loads if scheduling
   them would push register pressure above the target. This is the KEY
   interaction with recoloring: we schedule in a way that respects the
   register budget.

### 5.3 Why not ILP/constraint programming?

For basic blocks with 50-200 instructions, list scheduling with good
heuristics produces near-optimal results. ILP formulations exist (Winkel 2006,
Malik 2008) but:
- Solve time is exponential worst-case
- The objective function (minimize cycles with dual-issue and variable latency)
  is hard to express precisely in ILP
- We have an actual GPU to measure — ILP-optimal on a cost model might not be
  GPU-optimal due to model imprecision

The cost model + GPU verification loop is more practical than ILP optimality.

---

## 6. Phase 3: LLM Refinement (Anthropic Claude)

### 6.1 Why an LLM?

Classical heuristics are fast but local — they make greedy decisions per
instruction. An LLM can reason about **global patterns**:

- "This is a reduction loop — the accumulator chain is the bottleneck,
  not the loads. Interleaving the accumulator additions won't help, but
  unrolling the loop prologue would."
- "These three loads all hit L1 (addressed from the same base, stride 4).
  They'll be coalesced. The real bottleneck is the IMAD chain."
- "This block has a classic software pipelining opportunity: start iteration
  N+1's load during iteration N's compute."

These insights require understanding of GPU architecture, memory hierarchy,
and common optimization patterns. Claude has this knowledge.

### 6.2 When to invoke Claude

Not every basic block benefits from LLM analysis. Invoke Claude only for
**hot blocks** where:
- The block is in a loop (identified by back-edges in CFG)
- NCU profile data (if available) shows the block accounts for >10% of cycles
- The cost model estimates >5% improvement potential (gap between schedule and
  critical path lower bound)
- The classical heuristics disagree (different schedules have >3% cost model
  difference → uncertain optimization landscape)

### 6.3 Prompt design

```
ROLE: You are a GPU instruction scheduling expert for sm_120 (Blackwell).

BASIC BLOCK (from loop body, executes ~50000 times per kernel invocation):

[Instructions with annotations]
  Idx | Instruction                    | Lat | ASAP | ALAP | Slack | Deps
  ----+--------------------------------+-----+------+------+-------+--------
   0  | LDG.E R8, [R2.64]             | 200 |   0  |   0  |   0   | -
   1  | LDG.E R10, [R4.64]            | 200 |   0  |  10  |  10   | -
   2  | IMAD R12, R6, R7, R5          |   5 |   0  |  195 | 195   | -
   3  | IMAD.WIDE R14, R8, R9, RZ     |   5 | 200  | 200  |   0   | 0(R8)
   ...

CRITICAL PATH: 0 → 3 → 5 → 8 → 11 (estimated 225 cycles)

CURRENT BEST SCHEDULE (from CPF heuristic, cost model: 230 cycles):
  [schedule listing]

DUAL-ISSUE OPPORTUNITIES IDENTIFIED:
  - Inst 2 (INT) + Inst 6 (FP) — both ready at cycle 0, independent
  - Inst 7 (MEM) + Inst 9 (INT) — both ready at cycle 205

REGISTER PRESSURE: peak 58/64 at instruction 5

QUESTION: Suggest a better instruction ordering. Focus on:
1. Can any loads be moved earlier without exceeding register pressure 64?
2. Are there dual-issue pairs the heuristic missed?
3. Is there a software pipelining opportunity across loop iterations?

Reply with a reordered instruction list (by index) and brief reasoning.
```

### 6.4 Processing Claude's response

1. Parse the reordered instruction indices from Claude's response.
2. Validate: the reordering must be a valid topological sort of the DAG
   (if not, reject and log the invalid suggestion).
3. Compute control words for the new schedule.
4. Evaluate with cost model.
5. If cost model improvement >2% → add to GPU verification candidates.

### 6.5 Cost control

- **Token budget:** ~2000 input tokens + ~500 output tokens per block.
  At Sonnet pricing (~$3/M input, $15/M output): ~$0.01 per block.
- **Only hot blocks:** Typical kernel has 3-5 hot blocks. Total LLM cost
  per kernel: ~$0.05.
- **Caching:** If the same block appears in multiple kernels (common for
  library code), cache Claude's suggestions.

---

## 7. Phase 4: Control Word Optimization

### 7.1 Stall count minimization

Given a final instruction order, compute the **minimum legal stall count**
for each instruction. This is purely deterministic:

```python
def compute_min_stalls(schedule: list[SchedInst], dag: DepDAG):
    """Compute minimum stall counts that satisfy all dependencies.

    For each instruction i at position p:
      stall[i] = max over all predecessors j at position q:
        max(0, latency[j] - (sum of stalls between q and p))

    The idea: if IMAD at position 3 has latency 5, and its consumer is
    at position 6, we need at least 5 stall cycles distributed across
    positions 3, 4, and 5.
    """
    stalls = [1] * len(schedule)  # minimum 1 stall per instruction

    for i, inst in enumerate(schedule):
        for pred_idx, pred_latency in dag.predecessors_with_latency(inst):
            # Sum stalls between pred and this instruction
            gap = sum(stalls[k] for k in range(pred_idx + 1, i))
            needed = pred_latency - gap
            if needed > stalls[i]:
                stalls[i] = min(needed, 15)  # max stall count is 15

    return stalls
```

For variable-latency instructions (memory), use barriers instead of stalls:

### 7.2 Barrier assignment

sm_120 has 6 barriers (B0-B5). Barriers decouple the stall from the wait:
the producing instruction sets a write-barrier, and the consuming instruction
has the corresponding bit in its wait-mask.

```python
def assign_barriers(schedule, dag):
    """Assign barriers to variable-latency dependencies.

    Strategy: minimize barrier count using interval graph coloring.
    A barrier is "occupied" from the producing instruction to the
    consuming instruction. Two dependencies need different barriers
    only if their intervals overlap.
    """
    # Collect all variable-latency dependency intervals
    intervals = []
    for consumer_idx, inst in enumerate(schedule):
        for dep in dag.variable_latency_deps(inst):
            producer_idx = schedule.index(dep.producer)
            intervals.append((producer_idx, consumer_idx, dep))

    # Color intervals with minimum barriers (interval graph = perfect graph)
    # Sort by start, greedily assign smallest available barrier
    intervals.sort(key=lambda x: x[0])
    barrier_assignment = {}
    active = []  # (end_idx, barrier_id)

    for start, end, dep in intervals:
        # Release expired barriers
        active = [(e, b) for e, b in active if e > start]
        used = {b for _, b in active}
        # Assign smallest free barrier
        for b in range(6):
            if b not in used:
                barrier_assignment[dep] = b
                active.append((end, b))
                break
        else:
            # All 6 barriers in use — must add stall instead
            barrier_assignment[dep] = None  # fallback to stall

    return barrier_assignment
```

### 7.3 Yield hint optimization

The yield flag (`ctrl_word[4]`) hints the warp scheduler to switch to another
warp. Set yield=1 at points where:
- A long-latency operation just started (memory load issued, barrier set)
- The next instruction will stall waiting for a barrier
- We're at the end of a compute burst before a memory-dependent section

---

## 8. Phase 5: GPU Verification

### 8.1 Candidate assembly

For each surviving candidate schedule (typically 3-5):

1. **Reorder instructions** according to the schedule.
2. **Compute and apply control words** (stalls, barriers, yields).
3. **Assemble** via cubit (cuasm text → cubin binary).
4. **Inject** into the original cubin (replace the kernel's .text section).

### 8.2 Measurement protocol

Same as CuAsmRL's protocol (validated methodology):
- 100 warm-up iterations (flush caches, stabilize clocks)
- 100 measurement iterations
- Report median throughput (not mean — avoids outlier sensitivity)
- Compare against original schedule (baseline)

### 8.3 Acceptance criteria

- Improvement > 2%: accept (measurement noise is typically <1%)
- Improvement 0-2%: accept only if consistent across 3 repeated measurements
- Regression: reject, keep original

---

## 9. Joint Scheduling + Recoloring

This is sasskit's unique advantage over CuAsmRL. The two optimizations interact:

### 9.1 The phase-ordering problem

**Scheduling → Recoloring:** Changing instruction order changes live ranges.
Moving a load earlier extends its destination's live range (more registers
simultaneously live). A schedule that reduces stalls by 5% but increases
register pressure from 64 to 72 is a net loss.

**Recoloring → Scheduling:** Renaming registers may enable new dual-issue
opportunities (register bank conflicts) or remove false anti-dependencies
(WAR) that constrain scheduling.

### 9.2 Joint optimization strategy

```
1. Initial analysis
   - Decode, build CFG, liveness, interference graph
   - Identify target register count K

2. Pressure-bounded scheduling (Phase 2, PB heuristic)
   - Schedule with pressure_limit = K
   - This may sacrifice some latency hiding to stay within K registers

3. Recolor the scheduled code
   - Build new interference graph from new instruction order
   - Run DSatur/ILP graph coloring with target K
   - If success → pure rename, zero overhead

4. If step 3 fails (pressure > K even with bounded scheduling):
   - Relax pressure_limit slightly (K+4)
   - Re-schedule
   - Recolor with spills for the small overflow
   - Evaluate: does the scheduling gain outweigh the spill cost?

5. Compare candidates:
   Candidate A: original schedule + recolored (baseline sasskit)
   Candidate B: new schedule + recolored (joint optimization)
   Candidate C: new schedule + recolored + spills (if needed)
   → GPU verification picks the winner
```

### 9.3 The key insight

CuAsmRL improves throughput by ~9% via scheduling.
sasskit recoloring improves occupancy by 33-100% (3→4 blocks/SM).

**These multiply.** A kernel that gains 4 blocks/SM from recoloring AND 9%
from scheduling gets both benefits. And by scheduling with pressure awareness,
we avoid the scenario where scheduling improvement comes at the cost of
occupancy regression.

---

## 10. Cost Model

### 10.1 Cycle-accurate simulation

```python
class CostModel:
    """Estimate basic block execution time without GPU.

    Simulates execution cycle-by-cycle, tracking:
    - Pipeline state (which functional units are busy)
    - Barrier state (which barriers are pending)
    - Dual-issue opportunities
    """

    def estimate_cycles(self, schedule, stalls, barriers) -> int:
        cycle = 0
        barrier_ready = [0] * 6  # cycle when each barrier clears

        for i, inst in enumerate(schedule):
            # Wait for stall
            cycle += stalls[i]

            # Wait for barriers
            wait_mask = inst.wait_mask
            for b in range(6):
                if wait_mask & (1 << b):
                    cycle = max(cycle, barrier_ready[b])

            # Execute
            if inst.is_variable_latency:
                # Set barrier
                b = barriers.get(inst)
                if b is not None:
                    barrier_ready[b] = cycle + inst.estimated_latency

            # Dual-issue: if next instruction is eligible and independent
            if i + 1 < len(schedule):
                next_inst = schedule[i + 1]
                if self.can_dual_issue(inst, next_inst):
                    stalls[i + 1] = max(0, stalls[i + 1] - 1)

        return cycle
```

### 10.2 Calibration

The cost model is calibrated against actual GPU measurements:
1. Take 10-20 representative basic blocks from known kernels.
2. For each, generate 5-10 random valid schedules.
3. Measure actual throughput on GPU.
4. Fit latency parameters to minimize prediction error.

The model doesn't need to predict absolute cycle counts — it just needs to
**rank** candidate schedules correctly. Rank correlation (Spearman) > 0.8 is
sufficient for selecting the best among 5 candidates.

---

## 11. Implementation Plan

### Module structure

```
src/sasskit/
├── core/           # existing
├── analysis/       # existing + extensions
│   ├── cfg.py
│   ├── liveness.py
│   ├── interference.py
│   └── deps.py         ← NEW: dependency DAG construction
├── recolor/        # existing
├── schedule/       ← NEW MODULE
│   ├── __init__.py
│   ├── classify.py     # Instruction classification & latency tables
│   ├── dag.py          # Dependency DAG with ASAP/ALAP/slack
│   ├── list_sched.py   # List scheduling with priority heuristics
│   ├── dual_issue.py   # Dual-issue pairing pass
│   ├── ctrl_word.py    # Stall count & barrier optimization
│   ├── cost_model.py   # Cycle-accurate cost model
│   ├── llm_refine.py   # Claude integration for schedule refinement
│   └── cli.py          # CLI: sasskit schedule / sasskit optimize
└── forge/          # existing
```

### Phased implementation

**Phase A — Foundation (1-2 weeks):**
- `analysis/deps.py`: Dependency DAG construction from decoded instructions.
  Reuse existing `Instruction.dest_regs` / `src_regs` for RAW/WAR/WAW.
- `schedule/classify.py`: Instruction classification (memory/int/fp/control)
  and latency table for sm_120.
- `schedule/dag.py`: ASAP/ALAP/slack/critical-path on the dependency DAG.

**Phase B — Classical Scheduling (1-2 weeks):**
- `schedule/list_sched.py`: List scheduling with CPF, MF, PB heuristics.
- `schedule/ctrl_word.py`: Stall count minimization and barrier assignment.
- `schedule/cost_model.py`: Cycle-accurate simulation.
- CLI: `sasskit schedule kernel.cubin -k MyKernel`

**Phase C — LLM Integration (1 week):**
- `schedule/llm_refine.py`: Prompt construction from annotated basic blocks,
  response parsing, validation against DAG.
- Integration with existing `call_anthropic()` from forge module.

**Phase D — Joint Optimization (1 week):**
- `schedule/cli.py`: `sasskit optimize` command that runs
  recolor + schedule jointly.
- Pressure-bounded scheduling heuristic.
- End-to-end test: decode → schedule → recolor → patch → verify.

**Phase E — GPU Verification & Calibration (1 week):**
- Candidate assembly and injection pipeline.
- Measurement protocol (warm-up + median throughput).
- Cost model calibration against GPU measurements.

---

## 12. CLI Interface

```bash
# Analyze scheduling opportunities (no modification)
sasskit schedule analyze kernel.cubin -k MyKernel

# Output:
#   Basic blocks: 47
#   Hot blocks (in loops): 12
#   Estimated improvement: 8-15%
#   Critical path (block 7): 230 cycles, schedule: 280 cycles (18% gap)
#   Dual-issue opportunities: 45 pairs across hot blocks

# Apply scheduling optimization
sasskit schedule optimize kernel.cubin -k MyKernel -o optimized.cubin \
    --use-llm               # enable Claude refinement
    --pressure-limit 64     # respect register budget
    --verify                # GPU verification of candidates
    --benchmark 100         # 100 measurement iterations

# Joint recolor + schedule optimization (the full pipeline)
sasskit optimize kernel.cubin -k MyKernel -t 64 -o optimized.cubin \
    --schedule              # enable scheduling
    --use-llm               # enable Claude refinement
    --benchmark 100
```

---

## 13. Comparison to Prior Art

| | ptxas -O3 | CuAsmRL | maxas (manual) | **This design** |
|---|---|---|---|---|
| **Scheduling** | Static heuristics | RL search | Manual trial-and-error | Classical + LLM |
| **Register awareness** | Full (does both) | None | Full (manual) | Full (joint) |
| **GPU evaluations** | 0 (compile-time) | ~5000/kernel | ~50/kernel (manual) | 3-5/kernel |
| **Time per kernel** | Seconds | Hours | Days | Minutes |
| **Transfer learning** | N/A (hardcoded) | None | Human expertise | LLM + cost model |
| **Dual-issue** | Conservative | Not targeted | Aggressive (manual) | Systematic |
| **Architecture** | All NVIDIA | sm_80 (Ampere) | sm_50 (Maxwell) | sm_120 (Blackwell) |
| **Open-source** | No | No | Yes | Yes |

---

## 14. Risks and Mitigations

**Risk: Cost model is inaccurate.**
Mitigation: The cost model only needs to RANK candidates correctly, not predict
absolute times. GPU verification is the final arbiter. If cost model ranking
accuracy is <70%, fall back to evaluating more candidates on GPU (10 instead
of 5 — still 500x fewer than CuAsmRL).

**Risk: Claude suggests invalid schedules.**
Mitigation: Every suggestion is validated against the dependency DAG before
assembly. Invalid suggestions are rejected with zero cost. LLM is advisory,
not authoritative.

**Risk: Scheduling increases register pressure beyond target.**
Mitigation: Pressure-bounded scheduling (PB heuristic) explicitly constrains
pressure. Joint optimization evaluates the tradeoff and picks the best
overall candidate.

**Risk: cubit doesn't support sm_120 fully.**
Mitigation: sasskit's forge module already wraps cubit for sm_120 with
workarounds. Binary patching of control words (existing patcher.py) can handle
stall/barrier changes without full reassembly.

**Risk: Instruction latencies are wrong.**
Mitigation: Three-source latency estimation (table + inference + LLM). Cost
model calibration against GPU measurements corrects systematic errors.

---

## 15. Expected Impact

Based on CuAsmRL's results (9% geomean on Ampere) and sasskit's register
recoloring (33-100% occupancy improvement where applicable):

**Scheduling alone:** 5-15% throughput improvement on instruction-bound kernels.
Conservative estimate (CuAsmRL got 9% with a worse algorithm).

**Recoloring alone:** 0-30% throughput improvement on occupancy-bound kernels
(depends on whether occupancy is the actual bottleneck).

**Joint optimization:** Multiplicative. A kernel that's both occupancy-bound
and instruction-bound (common in crypto/HPC) could see 15-40% improvement.

**Cost per kernel:** ~$0.05 in LLM API calls + ~2 minutes GPU time.
Negligible compared to the GPU hours saved by faster kernels.
