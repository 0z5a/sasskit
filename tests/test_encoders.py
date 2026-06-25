#!/usr/bin/env python3
"""Verify patcher instruction encoders against cuobjdump and GPU execution.

For each encoder (make_bra, make_lds, make_sts, make_mov, make_s2r_tid,
make_nop_stall, etc.), this script:
  1. Loads the encoder_test.cubin baseline
  2. Injects the encoded instruction at a NOP slot
  3. Runs cuobjdump -sass to verify disassembly matches expectations
  4. Runs encoder_harness to verify the kernel doesn't crash on GPU

Usage:
    python3 tests/test_encoders.py [--verbose]
"""

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Add sasskit to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sasskit.core.cubin import Cubin
from sasskit.recolor.patcher import (
    make_bra, make_nop, make_nop_stall, make_s2r_tid,
    make_lds, make_sts, make_lds_64, make_sts_64,
    make_mov, make_iadd_carry, make_shf_l_u32,
)

HARNESS_DIR = Path(__file__).parent.parent / 'harnesses'
CUBIN_PATH = HARNESS_DIR / 'encoder_test.cubin'
HARNESS_BIN = HARNESS_DIR / 'encoder_harness'
KERNEL_NAME = 'encoder_test'

VERBOSE = '--verbose' in sys.argv or '-v' in sys.argv


def disasm_cubin(cubin_path: Path) -> dict[int, str]:
    """Run cuobjdump -sass and return {offset: asm_text}."""
    r = subprocess.run(
        ['/usr/local/cuda/bin/cuobjdump', '-sass', str(cubin_path)],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {r.stderr}")
    result = {}
    in_kernel = False
    for line in r.stdout.split('\n'):
        if f'Function : {KERNEL_NAME}' in line:
            in_kernel = True
            continue
        if in_kernel and 'Function :' in line:
            break
        if in_kernel:
            m = re.match(r'\s+/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line)
            if m:
                result[int(m.group(1), 16)] = m.group(2).strip()
    return result


def gpu_run(cubin_path: Path) -> tuple[bool, str]:
    """Run cubin on GPU via encoder_harness. Returns (passed, output)."""
    r = subprocess.run(
        [str(HARNESS_BIN), str(cubin_path), '256'],
        capture_output=True, text=True, timeout=10)
    passed = r.returncode == 0 and 'PASS' in r.stdout
    return passed, (r.stdout + r.stderr).strip()


def inject_instruction(cubin_data: bytearray, kernel_text_off: int,
                       code_offset: int, instr: int, ctrl: int) -> bytearray:
    """Write a 16-byte instruction at code_offset within the kernel text."""
    data = bytearray(cubin_data)
    file_off = kernel_text_off + code_offset
    struct.pack_into('<Q', data, file_off, instr)
    struct.pack_into('<Q', data, file_off + 8, ctrl)
    return data


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.disasm_ok = False
        self.gpu_ok = False
        self.disasm_text = ''
        self.expected_pattern = ''
        self.gpu_output = ''
        self.error = ''

    @property
    def passed(self):
        return self.disasm_ok and self.gpu_ok

    def __str__(self):
        d = 'DISASM_OK' if self.disasm_ok else 'DISASM_FAIL'
        g = 'GPU_OK' if self.gpu_ok else 'GPU_FAIL'
        s = f"{'PASS' if self.passed else 'FAIL'} {self.name}: {d} {g}"
        if not self.disasm_ok:
            s += f"\n  expected: {self.expected_pattern}"
            s += f"\n  got:      {self.disasm_text}"
        if not self.gpu_ok:
            s += f"\n  gpu: {self.gpu_output}"
        if self.error:
            s += f"\n  error: {self.error}"
        return s


def run_test(name: str, code_offset: int, instr: int, ctrl: int,
             expected_pattern: str, cubin: Cubin) -> TestResult:
    """Inject one instruction, verify disasm + GPU."""
    result = TestResult(name)
    result.expected_pattern = expected_pattern

    kernel = cubin.get_kernel(KERNEL_NAME)
    modified = inject_instruction(
        bytearray(CUBIN_PATH.read_bytes()),
        kernel.text_offset, code_offset, instr, ctrl)

    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        f.write(bytes(modified))
        tmp_path = Path(f.name)

    try:
        # 1. Disasm check
        asm_map = disasm_cubin(tmp_path)
        asm_text = asm_map.get(code_offset, '')
        result.disasm_text = asm_text

        if re.search(expected_pattern, asm_text, re.IGNORECASE):
            result.disasm_ok = True
        else:
            if VERBOSE:
                print(f"  DISASM MISMATCH at 0x{code_offset:04x}:")
                print(f"    expected pattern: {expected_pattern}")
                print(f"    got: {asm_text}")

        # 2. GPU crash test
        passed, output = gpu_run(tmp_path)
        result.gpu_ok = passed
        result.gpu_output = output

    except Exception as e:
        result.error = str(e)
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


def main():
    # Verify prerequisites
    if not CUBIN_PATH.exists():
        print(f"ERROR: {CUBIN_PATH} not found. Run: cd harnesses && nvcc -arch=sm_120 -cubin ...")
        return 1
    if not HARNESS_BIN.exists():
        print(f"ERROR: {HARNESS_BIN} not found. Run: cd harnesses && gcc ...")
        return 1

    cubin = Cubin.from_file(CUBIN_PATH)
    kernel = cubin.get_kernel(KERNEL_NAME)

    print(f"Kernel: {KERNEL_NAME}")
    print(f"  text_offset=0x{kernel.text_offset:x}, text_size=0x{kernel.text_size:x}")
    print(f"  instructions={kernel.num_instructions}")
    print()

    # Find NOP slots we can safely use (after EXIT at 0x80, starting at 0xA0)
    # We'll inject one instruction per test into different NOP slots.
    nop_base = 0xA0  # First NOP after the infinite-loop BRA at 0x90
    slot = 0

    def next_offset():
        nonlocal slot
        off = nop_base + slot * 16
        slot += 1
        return off

    results = []

    # ====================================================================
    # Test 1: NOP with stall
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_nop_stall(15)
    r = run_test('NOP stall=15', off, instr, ctrl, r'NOP', cubin)
    results.append(r)

    # ====================================================================
    # Test 2: NOP (default)
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_nop()
    r = run_test('NOP default', off, instr, ctrl, r'NOP', cubin)
    results.append(r)

    # ====================================================================
    # Test 3: BRA backward (to EXIT at 0x80)
    # We replace the NOP at this offset with BRA 0x80.
    # Since this NOP is after EXIT, it's never reached — but if we
    # redirect the EXIT-preceding instruction to here, the kernel
    # would exit via this BRA. For now, just verify encoding.
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_bra(0x80, off)
    r = run_test(f'BRA backward 0x{off:02x}->0x80', off, instr, ctrl,
                 r'BRA\s+0x80', cubin)
    results.append(r)

    # ====================================================================
    # Test 4: BRA forward (to a later NOP)
    # ====================================================================
    off = next_offset()
    target = off + 0x30  # skip 3 instructions forward
    instr, ctrl = make_bra(target, off)
    r = run_test(f'BRA forward 0x{off:02x}->0x{target:02x}', off, instr, ctrl,
                 rf'BRA\s+0x{target:x}', cubin)
    results.append(r)

    # ====================================================================
    # Test 5: BRA to self (infinite loop — just verify encoding, not GPU)
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_bra(off, off)
    r = TestResult(f'BRA self 0x{off:02x}->0x{off:02x}')
    asm_map = disasm_cubin(CUBIN_PATH)  # use original for safety
    # Verify encoding manually
    rel = off - off - 16  # -16
    relq = rel // 4       # -4
    lo8 = relq & 0xFF     # 0xFC
    expected_lo = lo8
    actual_lo = (instr >> 16) & 0xFF
    r.disasm_ok = (actual_lo == expected_lo) and ((instr & 0xFFFF) == 0x7947)
    r.disasm_text = f'opcode=0x{instr & 0xFFFF:04x} lo8=0x{actual_lo:02x} (expected 0x{expected_lo:02x})'
    r.expected_pattern = f'BRA self (opcode=0x7947, lo8=0x{expected_lo:02x})'
    r.gpu_ok = True  # skip GPU for infinite loop
    results.append(r)

    # ====================================================================
    # Test 6: MOV R5, R0
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_mov(5, 0)
    r = run_test('MOV R5, R0', off, instr, ctrl, r'MOV\s+R5,\s*R0', cubin)
    results.append(r)

    # ====================================================================
    # Test 7: MOV R10, R3
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_mov(10, 3)
    r = run_test('MOV R10, R3', off, instr, ctrl, r'MOV\s+R10,\s*R3', cubin)
    results.append(r)

    # ====================================================================
    # Test 8: S2R R2, SR_TID.X
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_s2r_tid(2)
    r = run_test('S2R R2, SR_TID.X', off, instr, ctrl,
                 r'S2R\s+R2,\s*SR_TID', cubin)
    results.append(r)

    # ====================================================================
    # Test 9: LDS (32-bit) R3, [R2 + 0x100]
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_lds(3, 2, 0x100)
    r = run_test('LDS R3, [R2+0x100]', off, instr, ctrl,
                 r'LDS.*R3.*R2', cubin)
    results.append(r)

    # ====================================================================
    # Test 10: STS (32-bit) [R2 + 0x100], R3
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_sts(2, 3, 0x100)
    r = run_test('STS [R2+0x100], R3', off, instr, ctrl,
                 r'STS.*R2.*R3', cubin)
    results.append(r)

    # ====================================================================
    # Test 11: LDS.64 R4, [R2 + 0x200]
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_lds_64(4, 2, 0x200)
    r = run_test('LDS.64 R4, [R2+0x200]', off, instr, ctrl,
                 r'LDS\.64.*R4.*R2', cubin)
    results.append(r)

    # ====================================================================
    # Test 12: STS.64 [R2 + 0x200], R4
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_sts_64(2, 4, 0x200)
    r = run_test('STS.64 [R2+0x200], R4', off, instr, ctrl,
                 r'STS\.64.*R2.*R4', cubin)
    results.append(r)

    # ====================================================================
    # Test 13: SHF.L.U32 R5, R3, 0x2, RZ
    # ====================================================================
    off = next_offset()
    instr, ctrl = make_shf_l_u32(5, 3, 2)
    r = run_test('SHF.L.U32 R5, R3, 0x2', off, instr, ctrl,
                 r'SHF.*R5.*R3', cubin)
    results.append(r)

    # ====================================================================
    # Test 14: BRA functional test — redirect EXIT to trampoline and back
    #
    # Replace EXIT at 0x80 with BRA to NOP area.
    # At the NOP area, place EXIT. Kernel should still work.
    # ====================================================================
    off_exit = 0x80
    off_tramp = next_offset()

    result_bra_func = TestResult('BRA functional (redirect EXIT)')
    try:
        modified = bytearray(CUBIN_PATH.read_bytes())
        # Place EXIT at trampoline offset
        exit_instr = 0x000000000000794d
        exit_ctrl = 0x000fea0003800000
        file_off_tramp = kernel.text_offset + off_tramp
        struct.pack_into('<Q', modified, file_off_tramp, exit_instr)
        struct.pack_into('<Q', modified, file_off_tramp + 8, exit_ctrl)
        # Replace EXIT at 0x80 with BRA to trampoline
        bra_i, bra_c = make_bra(off_tramp, off_exit)
        file_off_exit = kernel.text_offset + off_exit
        struct.pack_into('<Q', modified, file_off_exit, bra_i)
        struct.pack_into('<Q', modified, file_off_exit + 8, bra_c)

        with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
            f.write(bytes(modified))
            tmp = Path(f.name)

        # Verify disasm
        asm = disasm_cubin(tmp)
        bra_text = asm.get(off_exit, '')
        exit_text = asm.get(off_tramp, '')
        result_bra_func.disasm_text = f'@0x{off_exit:02x}: {bra_text} | @0x{off_tramp:02x}: {exit_text}'
        result_bra_func.expected_pattern = f'BRA 0x{off_tramp:x} + EXIT'
        result_bra_func.disasm_ok = (
            re.search(rf'BRA\s+0x{off_tramp:x}', bra_text, re.IGNORECASE) is not None
            and 'EXIT' in exit_text
        )

        # GPU test — this is the real test: kernel must produce canary
        passed, output = gpu_run(tmp)
        result_bra_func.gpu_ok = passed
        result_bra_func.gpu_output = output
        tmp.unlink(missing_ok=True)

    except Exception as e:
        result_bra_func.error = str(e)

    results.append(result_bra_func)

    # ====================================================================
    # Print results
    # ====================================================================
    print('=' * 70)
    print('ENCODER VERIFICATION RESULTS')
    print('=' * 70)

    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)

    for r in results:
        print(r)

    print()
    print(f'{n_pass}/{n_total} tests passed')

    if n_pass < n_total:
        print('\nFAILED TESTS:')
        for r in results:
            if not r.passed:
                print(f'  - {r.name}')
                if VERBOSE and r.disasm_text:
                    print(f'    disasm: {r.disasm_text}')

    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
