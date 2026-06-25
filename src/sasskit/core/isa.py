"""SM120 SASS assembler — re-exports from cubit.assembler.

The assembler lives in cubit (the encoding backend).  This module
re-exports the public API for backward compatibility with sasskit code
that imports from ``sasskit.core.isa``.

Usage::

    from sasskit.core.isa import Assembler
    asm = Assembler()
    inst = asm.assemble("IADD3 R5, PT, PT, R5, R6, RZ", stall=1)
"""

from cubit.assembler import (
    Assembler as Sm120Assembler,  # backward compat alias
    Assembler,
    Instruction as AsmInstruction,  # backward compat alias
    Instruction,
    build_ctrl,
    merge_ctrl,
    decode_ctrl,
)


class AssembleError(Exception):
    """Assembly error (kept for backward compatibility)."""
    pass


# Legacy preset constants (kept for backward compatibility).
# Prefer using build_ctrl() or Assembler's stall= parameter directly.
CTRL_DEFAULT  = build_ctrl(stall=1,  yield_hint=True)
CTRL_DRAIN    = build_ctrl(stall=15, yield_hint=True)
CTRL_EXIT     = build_ctrl(stall=15, yield_hint=True)
CTRL_WAIT_ALL = build_ctrl(stall=15, yield_hint=True, wait_mask=0x3F)
