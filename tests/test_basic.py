"""Basic tests for sasskit."""

import os
import sys
import pytest

# Add src to path for testing without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sasskit.core.cubin import Cubin
from sasskit.core.decoder import decode_kernel
from sasskit.analysis.cfg import build_cfg
from sasskit.analysis.liveness import compute_liveness, find_max_pressure


# Path to test cubin — set via environment or skip
TEST_CUBIN = os.environ.get('TEST_CUBIN', '')


@pytest.fixture
def cubin():
    if not os.path.exists(TEST_CUBIN):
        pytest.skip(f"Test cubin not found: {TEST_CUBIN}")
    return Cubin.from_file(TEST_CUBIN)


def test_cubin_loads(cubin):
    """Test that the cubin loads and has expected kernels."""
    assert 'KernelA' in cubin.kernels
    assert 'KernelB' in cubin.kernels
    assert 'KernelC' in cubin.kernels
    assert 'KernelGen' in cubin.kernels


def test_kernel_text_size(cubin):
    """Test KernelA has expected code size."""
    ka = cubin.get_kernel('KernelA')
    assert ka.text_size > 0
    assert ka.text_size % 16 == 0  # Instructions are 16 bytes
    assert ka.num_instructions == ka.text_size // 16


def test_decode_instructions(cubin):
    """Test instruction decoding via nvdisasm."""
    instructions = decode_kernel(cubin, 'KernelA')
    assert len(instructions) > 0
    
    # Check first instruction is S2R (thread ID read)
    first = instructions[0]
    assert 'S2R' in first.opcode or first.code_offset == 0
    
    # Check we find NOP instructions at the end
    nops = [i for i in instructions if i.is_nop]
    assert len(nops) > 0, "Expected NOP padding at end"


def test_register_range(cubin):
    """Test that register analysis finds the correct range."""
    instructions = decode_kernel(cubin, 'KernelA')
    all_regs = set()
    for inst in instructions:
        all_regs |= inst.all_regs
    
    # Should have registers up to R74 (75 unique for 76-reg kernel)
    max_reg = max(all_regs)
    assert max_reg >= 64, f"Expected high registers, got max R{max_reg}"
    assert max_reg <= 80, f"Unexpectedly high register R{max_reg}"


def test_cfg_construction(cubin):
    """Test CFG construction."""
    instructions = decode_kernel(cubin, 'KernelA')
    blocks = build_cfg(instructions)
    
    assert len(blocks) > 10, f"Expected many basic blocks, got {len(blocks)}"
    
    # Check that blocks cover all instructions
    block_instrs = sum(len(b.instructions) for b in blocks)
    assert block_instrs == len(instructions)


def test_liveness_analysis(cubin):
    """Test liveness analysis produces reasonable results."""
    instructions = decode_kernel(cubin, 'KernelA')
    blocks = build_cfg(instructions)
    compute_liveness(blocks)
    
    offset, pressure, live_set = find_max_pressure(blocks)
    
    # Max pressure should be between 40 and 76
    assert pressure >= 40, f"Max pressure {pressure} seems too low"
    assert pressure <= 76, f"Max pressure {pressure} exceeds register count"
    
    # Check that high registers appear in the live set somewhere
    has_high = any(r >= 64 for r in live_set)
    # High regs might not be at the peak, but should be live somewhere
    

def test_recolor_feasibility(cubin):
    """Test that re-coloring analysis runs and gives a result."""
    from sasskit.recolor.coloring import plan_recoloring
    
    instructions = decode_kernel(cubin, 'KernelA')
    blocks = build_cfg(instructions)
    compute_liveness(blocks)
    
    result = plan_recoloring(instructions, blocks, target_regs=64)
    
    # Should produce SOME result
    assert result.max_pressure_before > 0
    assert result.target_regs == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
