"""NCU profiler integration for forge optimization loop.

SM120 (Blackwell) note: classic stall counters (smsp__warp_issue_stalled_*)
are not available. Instead we use:
  - sm__throughput: overall SM busy %
  - smsp__average_warp_latency_per_inst_executed.ratio: avg cycles per instruction
    (ideal = 1.0; high = latency stalls dominate)
  - sm__inst_executed.avg.per_cycle_active: IPC (ideal = 1.0 for 1 warp)
  - smsp__pipe_alu_cycles_active: INT pipeline utilization
  - smsp__pipe_fma_cycles_active: FMA pipeline utilization
"""
from __future__ import annotations
import csv, io, os, subprocess
from dataclasses import dataclass, field
from typing import Optional

NCU_BIN = os.environ.get("NCU_BIN", "/opt/nvidia/nsight-compute/2025.1.1/ncu")

METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__average_warp_latency_per_inst_executed.ratio",
    "smsp__pipe_alu_cycles_active.avg.per_second",
    "smsp__pipe_fma_cycles_active.avg.per_second",
]


@dataclass
class ProfileResult:
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def sm_throughput(self) -> float:
        return self.metrics.get("sm__throughput.avg.pct_of_peak_sustained_active", 0)

    @property
    def alu_pct(self) -> float:
        return self.metrics.get("smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", 0)

    @property
    def ipc(self) -> float:
        return self.metrics.get("sm__inst_executed.avg.per_cycle_active", 0)

    @property
    def avg_latency(self) -> float:
        """Average cycles-per-instruction: ideal=1.0, high=latency bound."""
        return self.metrics.get("smsp__average_warp_latency_per_inst_executed.ratio", 0)

    @property
    def alu_hz(self) -> float:
        return self.metrics.get("smsp__pipe_alu_cycles_active.avg.per_second", 0)

    @property
    def fma_hz(self) -> float:
        return self.metrics.get("smsp__pipe_fma_cycles_active.avg.per_second", 0)

    def format_for_prompt(self, label: str = "NCU PROFILE") -> str:
        if self.error:
            return f"{label}: (profiler unavailable: {self.error})"
        if not self.metrics:
            return f"{label}: (no metrics captured)"

        lines = [f"{label} (SM120 Blackwell, single warp):"]
        lines.append(f"  SM throughput:          {self.sm_throughput:5.1f}%  "
                     f"(% of peak SM active; 100% = fully utilized)")
        lines.append(f"  IPC:                    {self.ipc:5.2f}   "
                     f"(instructions/cycle; ideal=1.00 for 1 warp)")
        lines.append(f"  Avg cycles/instruction: {self.avg_latency:5.2f}   "
                     f"(ideal=1.00; >2.0 means heavy latency stalls)")
        if self.alu_pct > 0:
            lines.append(f"  INT pipe (ALU) pct:     {self.alu_pct:5.1f}%  "
                         f"(IADD3/IMAD utilization)")

        lines.append("")
        lines.append("  INTERPRETATION:")

        if self.avg_latency > 5.0:
            lines.append(
                f"  *** CRITICAL: avg {self.avg_latency:.1f} cycles/instruction — "
                "massive latency stalls! ***"
            )
            lines.append(
                "  IMAD.WIDE has 8-cycle latency. If consumer issues 1 instruction "
                "later, that's 7 wasted cycles."
            )
            lines.append(
                "  Fix: interleave 2+ independent multiply chains. "
                "Issue IMAD.WIDE for next partial product before consuming current result."
            )
        elif self.avg_latency > 2.5:
            lines.append(
                f"  ⚠ avg {self.avg_latency:.1f} cycles/instruction — significant latency stalls."
            )
            lines.append(
                "  Reduce IADD3 stall counts and interleave IMAD.WIDE chains more aggressively."
            )
        elif self.avg_latency < 1.5:
            lines.append(
                f"  ✓ avg {self.avg_latency:.1f} cycles/instruction — good latency hiding!"
            )
            lines.append("  Focus on reducing total instruction count to shrink loop body.")
        else:
            lines.append(
                f"  avg {self.avg_latency:.1f} cycles/instruction — moderate latency stalls."
            )

        if self.ipc < 0.4 and self.sm_throughput < 40:
            lines.append(
                f"  Only {self.ipc:.2f} IPC ({self.sm_throughput:.0f}% SM) — "
                "throughput is far below theoretical max."
            )
            lines.append(
                "  For a single-warp kernel on SM120, theoretical max is ~1 IPC when "
                "instructions are independent. Your encoded stall counts (control word) "
                "are adding wait cycles between instructions."
            )

        return "\n".join(lines)


def profile_cubin(cubin_path: str, kernel_name: str,
                  bench_bin: str = None, timeout: int = 60) -> ProfileResult:
    """Run NCU on a cubin and return parsed metrics."""
    if not os.path.exists(NCU_BIN):
        return ProfileResult(error=f"ncu not found at {NCU_BIN}")

    harness = bench_bin or os.environ.get('SASSKIT_BENCH_HARNESS', 'harnesses/bench_harness')
    metrics_str = ",".join(METRICS)

    cmd = [
        NCU_BIN, "--metrics", metrics_str, "--csv",
        harness, cubin_path, kernel_name, "1024"
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return _parse_ncu_csv(r.stdout)
    except subprocess.TimeoutExpired:
        return ProfileResult(error="NCU timeout")
    except Exception as e:
        return ProfileResult(error=str(e)[:100])


def _parse_ncu_csv(csv_text: str) -> ProfileResult:
    """Parse NCU CSV output, skipping any non-CSV lines (e.g. bench_harness stdout).

    NCU's --csv output is mixed with harness stdout. Find the CSV header
    (starts with '"ID"') and parse from there.
    """
    result = ProfileResult()
    metrics: dict[str, list[float]] = {}

    lines = csv_text.splitlines()
    csv_start = -1
    for idx, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            csv_start = idx
            break
    if csv_start < 0:
        result.error = "no CSV header found in NCU output"
        return result

    csv_block = "\n".join(lines[csv_start:])
    try:
        for row in csv.DictReader(io.StringIO(csv_block)):
            name = row.get("Metric Name", "").strip().strip('"')
            val_str = row.get("Metric Value", "").strip().strip('"')
            if name and val_str not in ("n/a", "", "-"):
                try:
                    val = float(val_str)
                    if name not in metrics:
                        metrics[name] = []
                    metrics[name].append(val)
                except ValueError:
                    pass
    except Exception as e:
        result.error = str(e)[:100]
        return result

    result.metrics = {k: sum(v) / len(v) for k, v in metrics.items() if v}
    return result
