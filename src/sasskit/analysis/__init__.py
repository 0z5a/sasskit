"""Static analysis: CFG construction, liveness analysis, interference graphs."""
from .cfg import build_cfg, BasicBlock
from .liveness import compute_liveness, compute_live_at, find_max_pressure
from .interference import build_interference_graph
