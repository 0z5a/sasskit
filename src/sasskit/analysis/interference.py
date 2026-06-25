"""Interference graph construction from liveness data.

Two registers interfere (share an edge) if they are simultaneously live
at any point in the program. This means they cannot use the same
physical register.
"""

from __future__ import annotations

from .cfg import BasicBlock
from .liveness import compute_live_at


def build_interference_graph(
    blocks: list[BasicBlock],
    instructions: list | None = None,
) -> dict[int, set[int]]:
    """Build register interference graph from liveness data.
    
    Two registers interfere (share an edge) if they are simultaneously live
    at any point in the program. This means they cannot use the same
    physical register.
    
    Returns: adjacency list: reg -> set of interfering regs
    """
    live_at = compute_live_at(blocks, instructions=instructions)
    
    graph: dict[int, set[int]] = {}
    
    for _, live_set in live_at.items():
        for reg in live_set:
            if reg not in graph:
                graph[reg] = set()
        
        # Every pair of simultaneously-live registers interferes
        live_list = list(live_set)
        for i in range(len(live_list)):
            for j in range(i + 1, len(live_list)):
                r1, r2 = live_list[i], live_list[j]
                graph.setdefault(r1, set()).add(r2)
                graph.setdefault(r2, set()).add(r1)
    
    return graph
