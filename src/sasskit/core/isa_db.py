"""SM120 ISA database — scheduling, latency, and pipeline metadata.

Loads the unified ISA database from blackwell-isa/sm120.json and provides
query APIs for instruction scheduling properties.  Used by:

  - interference.py: per-opcode latency extension for R0/R1 hazard window
  - coloring.py: spill cost estimation (variable-latency ops cost more)
  - patcher.py: stall counts for trampoline instructions
  - forge/engine.py: latency table for LLM prompts

The database is loaded lazily on first access and cached as a module-level
singleton.  Path resolution: $BLACKWELL_ISA_DB env var, then known locations.

All scheduling data has a ``confidence`` field:
  - "measured":     from hardware microbenchmark
  - "ptxas_table":  from ptxas internal scheduling table (entry with data)
  - "inferred":     from pipeline class defaults or architecture knowledge
  - "synthetic":    manually estimated
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── data types ────────────────────────────────────────────────

@dataclass(frozen=True)
class SchedInfo:
    """Scheduling properties for one instruction opcode."""
    pipe_class: int
    pipe_name: str
    def_latency: int
    throughput: int
    is_variable_latency: bool
    confidence: str
    source: str


# ── default fallbacks ────────────────────────────────────────

_DEFAULT_INT = SchedInfo(
    pipe_class=37, pipe_name="INT_ARITH",
    def_latency=4, throughput=2,
    is_variable_latency=False,
    confidence="fallback", source="default INT_ARITH estimate",
)

_DEFAULT_FP = SchedInfo(
    pipe_class=36, pipe_name="FP_ARITH",
    def_latency=4, throughput=2,
    is_variable_latency=False,
    confidence="fallback", source="default FP_ARITH estimate",
)

_DEFAULT_MEM = SchedInfo(
    pipe_class=3, pipe_name="MEMORY",
    def_latency=200, throughput=4,
    is_variable_latency=True,
    confidence="fallback", source="default MEMORY estimate",
)

_DEFAULT_CTRL = SchedInfo(
    pipe_class=1, pipe_name="CTRL_FLOW",
    def_latency=0, throughput=1,
    is_variable_latency=False,
    confidence="fallback", source="default CTRL_FLOW estimate",
)

_DEFAULT_UNKNOWN = SchedInfo(
    pipe_class=0, pipe_name="UNKNOWN",
    def_latency=4, throughput=2,
    is_variable_latency=False,
    confidence="fallback", source="no data available",
)


# ── database class ───────────────────────────────────────────

class IsaDb:
    """SM120 ISA database loaded from sm120.json.

    Provides fast lookup by SASS base opcode (e.g. "IADD3", "LDG").
    """

    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        with open(self._path) as f:
            raw = json.load(f)

        self._meta = raw.get("_meta", {})
        self._pipe_classes: dict[str, str] = self._meta.get("pipe_classes", {})

        # Build base_op → SchedInfo index from instructions
        self._by_op: dict[str, SchedInfo] = {}
        for _key, inst in raw.get("instructions", {}).items():
            base_op = inst["base_op"]
            sched = inst.get("scheduling")
            if sched and base_op not in self._by_op:
                self._by_op[base_op] = SchedInfo(
                    pipe_class=sched.get("pipe_class", 0),
                    pipe_name=sched.get("pipe_name", "UNKNOWN"),
                    def_latency=sched.get("def_latency", 0),
                    throughput=sched.get("throughput", 1),
                    is_variable_latency=sched.get("is_variable_latency", False),
                    confidence=sched.get("confidence", "unknown"),
                    source=sched.get("source", ""),
                )

        # Also index sched_only entries (instructions without encoding)
        for _key, inst in raw.get("sched_only", {}).items():
            base_op = inst.get("base_op", "")
            sched = inst.get("scheduling")
            if sched and base_op and base_op not in self._by_op:
                self._by_op[base_op] = SchedInfo(
                    pipe_class=sched.get("pipe_class", 0),
                    pipe_name=sched.get("pipe_name", "UNKNOWN"),
                    def_latency=sched.get("def_latency", 0),
                    throughput=sched.get("throughput", 1),
                    is_variable_latency=sched.get("is_variable_latency", False),
                    confidence=sched.get("confidence", "unknown"),
                    source=sched.get("source", ""),
                )

        # Pipeline throughput from knob defaults (if available)
        pc = raw.get("pipeline_config", {})
        self._res_busy: dict[str, int] = pc.get("resource_busy_defaults", {})

    # ── queries ───────────────────────────────────────────────

    def get(self, base_op: str) -> SchedInfo:
        """Get scheduling info for a SASS base opcode.

        Returns a SchedInfo with confidence="fallback" for unknown opcodes.
        The fallback is chosen by opcode prefix heuristic.
        """
        info = self._by_op.get(base_op)
        if info is not None:
            return info
        return self._guess_fallback(base_op)

    def def_latency(self, base_op: str) -> int:
        """Def-to-use latency in cycles for the given opcode."""
        return self.get(base_op).def_latency

    def throughput(self, base_op: str) -> int:
        """Pipeline throughput (cycles per issue) for the given opcode."""
        return self.get(base_op).throughput

    def is_variable_latency(self, base_op: str) -> bool:
        """True if the instruction has variable latency (memory, special regs)."""
        return self.get(base_op).is_variable_latency

    def is_memory(self, base_op: str) -> bool:
        """True if the instruction is a memory operation."""
        return "MEMORY" in self.get(base_op).pipe_name

    def is_control(self, base_op: str) -> bool:
        """True if the instruction is control flow."""
        return self.get(base_op).pipe_name in ("CTRL_FLOW", "CTRL_FLOW_2")

    def pipe_name(self, base_op: str) -> str:
        """Pipeline name for the given opcode."""
        return self.get(base_op).pipe_name

    def classify(self, base_op: str) -> str:
        """Classify an instruction into a broad category.

        Returns one of: "int", "fp", "fp64", "memory", "control",
        "tensor", "special", "warp", "barrier", "unknown".
        """
        info = self.get(base_op)
        pn = info.pipe_name
        if pn == "INT_ARITH":
            return "int"
        if pn == "FP_ARITH":
            if info.def_latency >= 8:
                return "fp64"
            return "fp"
        if "MEMORY" in pn:
            return "memory"
        if pn in ("CTRL_FLOW", "CTRL_FLOW_2"):
            return "control"
        if pn in ("HMMA", "IMMA", "DMMA", "GMMA", "QMMA", "OMMA",
                   "MMA_SETUP", "MMA_SETUP_2", "MMA_EXEC", "MMA_EXEC_2"):
            return "tensor"
        if "SPECIAL" in pn or pn in ("CVT", "ADU"):
            return "special"
        if "WARP" in pn or pn in ("UDP", "UDP_2"):
            return "warp"
        if pn == "BARRIER":
            return "barrier"
        return "unknown"

    def all_ops(self) -> list[str]:
        """List all base opcodes with scheduling data."""
        return sorted(self._by_op.keys())

    def latency_table_for_prompt(self) -> str:
        """Format a latency table suitable for LLM prompts.

        Returns a compact multi-line string with per-opcode latency info
        for the most common instructions.
        """
        # Group by category, show representative ops
        categories = {
            "Integer ALU": [],
            "FP32 ALU": [],
            "FP64 ALU": [],
            "Memory": [],
            "Special": [],
            "Tensor": [],
            "Control": [],
        }
        for op, info in sorted(self._by_op.items()):
            cat = self.classify(op)
            lat = f"{info.def_latency} cycles"
            if info.is_variable_latency:
                lat += " (variable)"
            entry = f"{op}: {lat} [{info.pipe_name}]"

            if cat == "int":
                categories["Integer ALU"].append(entry)
            elif cat == "fp":
                categories["FP32 ALU"].append(entry)
            elif cat == "fp64":
                categories["FP64 ALU"].append(entry)
            elif cat == "memory":
                categories["Memory"].append(entry)
            elif cat == "special":
                categories["Special"].append(entry)
            elif cat == "tensor":
                categories["Tensor"].append(entry)
            elif cat == "control":
                categories["Control"].append(entry)

        lines = ["Instruction latencies (sm_120 Blackwell):"]
        for cat_name, entries in categories.items():
            if not entries:
                continue
            lines.append(f"\n  {cat_name}:")
            for entry in entries:
                lines.append(f"    {entry}")
        return "\n".join(lines)

    # ── internal ──────────────────────────────────────────────

    def _guess_fallback(self, base_op: str) -> SchedInfo:
        """Heuristic fallback for opcodes not in the database."""
        op = base_op.upper()
        # Memory
        if op.startswith(("LD", "ST", "ATOM", "RED")):
            return _DEFAULT_MEM
        # Control flow
        if op in ("BRA", "EXIT", "RET", "CALL", "BSSY", "BSYNC",
                   "BREAK", "BRX", "BRXU", "NOP", "YIELD"):
            return _DEFAULT_CTRL
        # FP
        if op.startswith(("F", "D", "H")) and not op.startswith("FLO"):
            return _DEFAULT_FP
        # Default to INT
        return _DEFAULT_INT

    def __repr__(self) -> str:
        return f"IsaDb({self._path.name}, {len(self._by_op)} ops)"


# ── module-level singleton ───────────────────────────────────

def _build_search_paths() -> list[str]:
    """Build list of candidate paths for sm120.json.

    All paths are relative or derived from environment — no hardcoded
    absolute paths.
    """
    paths = []
    # Relative to this source file: sasskit/src/sasskit/core/isa_db.py
    # → walk up to workspace and look for sibling blackwell-isa/
    cur = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = cur / "blackwell-isa" / "sm120.json"
        if candidate.is_file():
            paths.append(str(candidate))
            break
        if cur.parent == cur:
            break  # reached filesystem root
        cur = cur.parent
    return paths


_singleton: Optional[IsaDb] = None


def get_db() -> IsaDb:
    """Get the module-level ISA database singleton.

    Searches for sm120.json in order:
      1. ``$BLACKWELL_ISA_DB`` environment variable (explicit override)
      2. Sibling ``blackwell-isa/`` directory relative to the sasskit
         source tree (walks up parent directories)

    Raises FileNotFoundError if no database found.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    env_path = os.environ.get("BLACKWELL_ISA_DB")
    candidates = ([env_path] if env_path else []) + _build_search_paths()

    for path in candidates:
        if path and os.path.isfile(path):
            _singleton = IsaDb(path)
            return _singleton

    raise FileNotFoundError(
        "sm120.json not found. Set $BLACKWELL_ISA_DB or place "
        "blackwell-isa/sm120.json alongside the sasskit source tree. "
        f"Searched: {candidates}"
    )


def _reset_singleton() -> None:
    """Reset cached DB (for testing)."""
    global _singleton
    _singleton = None
