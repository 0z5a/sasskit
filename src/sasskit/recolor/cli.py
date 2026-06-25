"""Command-line interface for sasskit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sasskit.core.cubin import Cubin
from sasskit.core.decoder import decode_kernel
from sasskit.analysis import build_cfg, compute_liveness, compute_live_at, find_max_pressure
from sasskit.analysis.interference import build_interference_graph
from sasskit.recolor.coloring import plan_recoloring


def cmd_disassemble(args: argparse.Namespace) -> int:
    """Disassemble kernel(s) to clean SASS text (uses cubit if available)."""
    import subprocess
    import re
    import os as _os

    cubin = Cubin.from_file(args.input)
    kernels_to_dump = [args.kernel] if args.kernel else list(cubin.kernels.keys())

    # ── Try cubit disassemble first (no cuobjdump needed) ────────────────────
    from sasskit.core.decoder import _find_cubit
    cubit_bin = _find_cubit()
    TABLE = _os.environ.get('CUBIT_TABLE', 'tables/sm120.json')

    if cubit_bin and _os.path.isfile(TABLE) and not args.with_scheduling:
        lines_out: list[str] = []
        for kname in kernels_to_dump:
            cmd = [cubit_bin, 'disassemble', str(cubin.path), '-k', kname, '-t', TABLE]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and r.stdout.strip():
                if lines_out:
                    lines_out.append('')
                lines_out.extend(r.stdout.rstrip('\n').splitlines())

        if lines_out:
            output = '\n'.join(lines_out)
            if args.output:
                Path(args.output).write_text(output + '\n')
                print(f"Written {len(lines_out)} lines to {args.output}")
            else:
                print(output)
            return 0

    # ── Fall back to cuobjdump ────────────────────────────────────────────────
    result = subprocess.run(
        ['/usr/local/cuda/bin/cuobjdump', '-sass', str(cubin.path)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"Error: disassembly failed (no cubit or cuobjdump available)", file=sys.stderr)
        return 1

    _ann = re.compile(r'\s*[?&]\S+')
    lines_out: list[str] = []
    in_kernel = False
    current_kernel = ''
    pending: tuple | None = None

    for line in result.stdout.splitlines():
        m = re.search(r'Function\s*:\s*(\S+)', line)
        if m:
            current_kernel = m.group(1)
            in_kernel = current_kernel in kernels_to_dump
            if in_kernel:
                if lines_out:
                    lines_out.append('')
                lines_out.append(f'// {current_kernel}')
            continue
        if not in_kernel:
            continue
        m = re.match(
            r'\s+/\*([0-9a-f]+)\*/\s+(.*?)\s*/\*\s*0x([0-9a-fA-F]+)\s*\*/', line)
        if m:
            addr = int(m.group(1), 16)
            asm = _ann.sub('', m.group(2)).strip().rstrip(';').strip()
            pending = (addr, asm)
            continue
        m2 = re.match(r'\s+/\*\s*0x([0-9a-fA-F]+)\s*\*/', line)
        if m2 and pending is not None:
            addr, asm = pending
            ctrl = int(m2.group(1), 16)
            if args.with_scheduling:
                lines_out.append(f'  /*{addr:04x}*/  {asm} ;  /* ctrl: 0x{ctrl:016x} */')
            else:
                lines_out.append(f'  /*{addr:04x}*/  {asm} ;')
            pending = None

    output = '\n'.join(lines_out)
    if args.output:
        Path(args.output).write_text(output + '\n')
        print(f"Written {len(lines_out)} lines to {args.output}")
    else:
        print(output)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze register usage in a kernel."""
    cubin = Cubin.from_file(args.input)
    print(f"Loaded: {cubin}")
    print()
    
    kernel_name = args.kernel
    if not kernel_name:
        # Use the first kernel
        kernel_name = next(iter(cubin.kernels))
        print(f"No kernel specified, using: {kernel_name}")
    
    kernel = cubin.get_kernel(kernel_name)
    print(f"Kernel: {kernel.name}")
    print(f"  Registers: {kernel.reg_count}")
    print(f"  Stack: {kernel.max_stack} bytes")
    print(f"  Static shared memory: {kernel.shared_size} bytes")
    print(f"  Code size: {kernel.text_size} bytes ({kernel.num_instructions} instructions)")
    print()
    
    # Decode instructions
    print("Decoding instructions...")
    instructions = decode_kernel(cubin, kernel_name)
    print(f"  Decoded {len(instructions)} instructions")
    
    # Collect all referenced registers
    all_regs: set[int] = set()
    for inst in instructions:
        all_regs |= inst.all_regs
    
    max_reg = max(all_regs) if all_regs else 0
    print(f"  Register range: R0-R{max_reg} ({len(all_regs)} unique)")
    
    # Find high registers (>= target)
    target = args.target_regs or 64
    high_regs = {r for r in all_regs if r >= target}
    if high_regs:
        print(f"\n  Registers >= R{target}: {sorted(high_regs)}")
        for r in sorted(high_regs):
            uses = sum(1 for inst in instructions if r in inst.all_regs)
            print(f"    R{r}: {uses} references")
    
    # Build CFG and compute liveness
    print("\nBuilding control-flow graph...")
    blocks = build_cfg(instructions)
    print(f"  {len(blocks)} basic blocks")
    
    compute_liveness(blocks, instructions=instructions)
    
    offset, pressure, live_set = find_max_pressure(blocks)
    print(f"\n  Max register pressure: {pressure} registers")
    print(f"  At instruction offset: 0x{offset:04x}")
    
    if pressure <= target:
        print(f"\n  GOOD: Max pressure {pressure} <= target {target}")
        print(f"  Re-coloring to {target} registers should be feasible!")
    else:
        overshoot = pressure - target
        print(f"\n  Max pressure {pressure} > target {target} by {overshoot}")
        print(f"  Need to spill {overshoot} registers to shared memory")
    
    # Show registers live at peak
    if args.verbose:
        print(f"\n  Registers live at peak (0x{offset:04x}):")
        for r in sorted(live_set):
            print(f"    R{r}", end='')
        print()
    
    return 0


def cmd_recolor(args: argparse.Namespace) -> int:
    """Apply register re-coloring to a kernel."""
    cubin = Cubin.from_file(args.input)
    kernel_name = args.kernel
    if not kernel_name:
        kernel_name = next(iter(cubin.kernels))
    
    kernel = cubin.get_kernel(kernel_name)
    target = args.target_regs
    
    print(f"Re-coloring {kernel_name}: {kernel.reg_count} → {target} registers")
    
    # Decode
    instructions = decode_kernel(cubin, kernel_name)
    
    # Build CFG and liveness
    blocks = build_cfg(instructions)
    compute_liveness(blocks, instructions=instructions)
    
    # Plan re-coloring
    hot_loop = None
    if args.hot_start is not None and args.hot_end is not None:
        hot_loop = (args.hot_start, args.hot_end)
    
    result = plan_recoloring(instructions, blocks, target, hot_loop)
    
    if result.success:
        print(f"  Coloring successful!")
        print(f"  Pressure: {result.max_pressure_before} → {result.max_pressure_after}")
        if result.spilled:
            print(f"  Spills: {len(result.spilled)} registers")
            for s in result.spilled:
                print(f"    R{s.reg}: {s.reason}")
        
        renames = sum(1 for old, new in result.coloring.items() if old != new)
        print(f"  Register renames: {renames}")
        
        if not args.dry_run:
            from sasskit.recolor.patcher import apply_patch_plan
            plan = apply_patch_plan(cubin, kernel_name, result, instructions,
                                    target, blocks=blocks)
            print(f"\n{plan.summary}")
            
            output = args.output or args.input.replace('.cubin', '_recolored.cubin')
            cubin.save(output)
            print(f"\nSaved: {output}")
        else:
            print("\n  [dry-run] No changes written")
    else:
        print(f"  Coloring FAILED at target {target}")
        print(f"  Max pressure: {result.max_pressure_before}")
        print(f"  Spills needed: {len(result.spilled)}")
        return 1
    
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """All-in-one optimization: analyze → recolor → verify → save."""
    from sasskit.recolor.patcher import apply_patch_plan

    cubin = Cubin.from_file(args.input)
    target = args.target_regs
    kernels_to_opt = [args.kernel] if args.kernel else list(cubin.kernels.keys())

    results: list[tuple[str, str]] = []

    for kname in kernels_to_opt:
        kernel = cubin.get_kernel(kname)
        orig_regs = kernel.reg_count
        if orig_regs <= target:
            results.append((kname, f"already at {orig_regs} regs (≤ {target})"))
            continue

        instructions = decode_kernel(cubin, kname)
        blocks = build_cfg(instructions)
        compute_liveness(blocks, instructions=instructions)
        _, pressure, _ = find_max_pressure(blocks)

        result = plan_recoloring(instructions, blocks, target)
        if result.success:
            plan = apply_patch_plan(cubin, kname, result, instructions,
                                    target, blocks=blocks)
            n_renames = sum(1 for o, n in result.coloring.items() if o != n)
            if result.spilled:
                results.append((
                    kname,
                    f"{orig_regs} → {target} regs "
                    f"({n_renames} renames, {len(result.spilled)} spills, "
                    f"{len(plan.spill_trampolines)} trampolines)"
                ))
            else:
                results.append((
                    kname,
                    f"{orig_regs} → {target} regs "
                    f"(pure rename, {n_renames} renames)"
                ))
        else:
            results.append((kname, f"FAILED (pressure={pressure})"))

    output = args.output or args.input.replace('.cubin', '_optimized.cubin')
    cubin.save(output)

    print(f"\n{'='*60}")
    print(f"sasskit optimize: {args.input}")
    print(f"{'='*60}")
    for kname, status in results:
        print(f"  {kname}: {status}")
    print(f"\nSaved: {output}")
    return 0


def cmd_reforge(args: argparse.Namespace) -> int:
    """Run SASS-to-SASS optimizer."""
    from sasskit.schedule.reforge import reforge

    cubin = Cubin.from_file(args.input)
    kernel_name = args.kernel or next(iter(cubin.kernels))

    state = reforge(
        args.input, kernel_name,
        max_iters=args.iterations,
        bench_blocks=args.blocks,
        bench_threads=args.threads,
        bench_smem=args.smem,
        temperature=args.temperature,
        seed=args.seed,
    )

    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    """Optimize instruction stall counts without reordering."""
    from sasskit.schedule.stall_opt import apply_stall_optimization

    cubin = Cubin.from_file(args.input)
    kernels_to_opt = [args.kernel] if args.kernel else list(cubin.kernels.keys())

    for kname in kernels_to_opt:
        kernel = cubin.get_kernel(kname)
        instructions = decode_kernel(cubin, kname)
        blocks = build_cfg(instructions)

        stats = apply_stall_optimization(
            cubin, kernel, instructions, blocks, dry_run=args.dry_run)

        print(f"{kname}: {stats['modified']}/{stats['total_instructions']} "
              f"instructions optimized, "
              f"{stats['total_savings_cycles']} cycles saved"
              f"{' (dry-run)' if args.dry_run else ''}")

    if not args.dry_run:
        output = args.output or args.input.replace('.cubin', '_sched.cubin')
        cubin.save(output)
        print(f"Saved: {output}")

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a patched cubin."""
    cubin = Cubin.from_file(args.input)
    kernel_name = args.kernel or next(iter(cubin.kernels))
    kernel = cubin.get_kernel(kernel_name)
    
    print(f"Verifying {kernel_name}:")
    print(f"  Registers: {kernel.reg_count}")
    print(f"  Stack: {kernel.max_stack} bytes")
    
    # Decode and check for R >= reg_count
    instructions = decode_kernel(cubin, kernel_name)
    violations = []
    for inst in instructions:
        for reg in inst.all_regs:
            if reg >= kernel.reg_count and reg != 0xFF:
                violations.append((inst.code_offset, reg, inst.asm_text))
    
    if violations:
        print(f"\n  VIOLATIONS: {len(violations)} instructions use R >= R{kernel.reg_count}")
        for off, reg, asm in violations[:10]:
            print(f"    0x{off:04x}: R{reg} in '{asm}'")
        return 1
    else:
        print(f"  OK: All registers < R{kernel.reg_count}")
        return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Diagnose register encoding issues — find bit_position=-1 refs."""
    from sasskit.recolor.patcher import diagnose_encoding_mismatches, diagnose_unmatched_refs

    cubin = Cubin.from_file(args.input)
    kernel_name = args.kernel or next(iter(cubin.kernels))
    kernel = cubin.get_kernel(kernel_name)
    target = args.target_regs

    print(f"Diagnosing {kernel_name} (reg_count={kernel.reg_count}, target={target})")
    print()

    instructions = decode_kernel(cubin, kernel_name)

    # --- Part 1: All encoding mismatches (bit_position=-1) ---
    mismatches = diagnose_encoding_mismatches(instructions)
    if mismatches:
        print(f"=== Encoding mismatches (bit_position=-1): {len(mismatches)} ===")
        # Group by opcode for easier analysis
        by_opcode: dict[str, list] = {}
        for m in mismatches:
            op = m.asm_text.split()[0] if m.asm_text else '???'
            by_opcode.setdefault(op, []).append(m)

        for op in sorted(by_opcode):
            refs = by_opcode[op]
            print(f"\n  {op}: {len(refs)} unmatched refs")
            for m in refs[:5]:
                print(f"    0x{m.code_offset:04x}: R{m.reg_num} "
                      f"({m.field_name}) — {m.reason}")
            if len(refs) > 5:
                print(f"    ... and {len(refs) - 5} more")
    else:
        print("=== No encoding mismatches (all register fields resolved) ===")

    # --- Part 2: Registers >= target that would survive patching ---
    print()
    high_refs = [
        (inst, ref)
        for inst in instructions
        for ref in inst.reg_refs
        if ref.reg_num >= target and ref.reg_num != 0xFF
    ]
    if high_refs:
        print(f"=== Register refs >= R{target}: {len(high_refs)} ===")
        by_reg: dict[int, list] = {}
        for inst, ref in high_refs:
            by_reg.setdefault(ref.reg_num, []).append((inst, ref))

        for rn in sorted(by_reg):
            entries = by_reg[rn]
            bp_status = "OK" if all(
                ref.bit_position >= 0 for _, ref in entries
            ) else "HAS UNMATCHED"
            print(f"\n  R{rn}: {len(entries)} refs ({bp_status})")
            for inst, ref in entries[:5]:
                bp_str = (f"bit[{ref.bit_position}]"
                          if ref.bit_position >= 0
                          else "UNMATCHED")
                loc = "ctrl" if ref.in_ctrl_word else "instr"
                print(f"    0x{inst.code_offset:04x}: {ref.field_name} "
                      f"{bp_str} ({loc}) — {inst.asm_text[:60]}")
            if len(entries) > 5:
                print(f"    ... and {len(entries) - 5} more")
    else:
        print(f"=== No register refs >= R{target} ===")

    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    """Inject patched cubin sections back into an ELF object (.o) file.

    The .o file embeds the cubin inside a fatbin container.  This command
    finds the original cubin bytes in the .o and replaces them (entirely,
    or just the kernel text section when --text-only is given).
    """
    o_path = Path(args.input)
    o_data = bytearray(o_path.read_bytes())

    with open(args.cubin_orig, 'rb') as f:
        orig_data = f.read()
    with open(args.cubin_patched, 'rb') as f:
        patched_data = f.read()

    # --- locate the original cubin inside the .o file ---
    search_len = min(256, len(orig_data))
    fingerprint = orig_data[:search_len]
    cubin_pos = o_data.find(fingerprint)

    if cubin_pos == -1:
        print("ERROR: Could not find original cubin in .o file")
        print("  Make sure --cubin-orig matches the cubin embedded in the .o")
        return 1

    # Verify a larger stretch to be safe
    verify_len = min(4096, len(orig_data))
    if bytes(o_data[cubin_pos:cubin_pos + verify_len]) != orig_data[:verify_len]:
        print("ERROR: Fingerprint matched but extended verification failed")
        return 1

    print(f"Found cubin at offset 0x{cubin_pos:x} in {o_path.name}")

    if args.text_only:
        # --- replace only .text.KernelName ---
        orig_cubin = Cubin.from_file(args.cubin_orig)
        patched_cubin = Cubin.from_file(args.cubin_patched)
        kernel_name = args.kernel or next(iter(orig_cubin.kernels))

        orig_ka = orig_cubin.get_kernel(kernel_name)
        patched_ka = patched_cubin.get_kernel(kernel_name)

        patched_text = patched_data[
            patched_ka.text_offset:patched_ka.text_offset + patched_ka.text_size
        ]

        # If the target text section is larger, pad with NOPs
        if orig_ka.text_size > patched_ka.text_size:
            import struct
            NOP_INSTR = 0x0000000000007918
            NOP_CTRL  = 0x000fc00000000000
            pad_count = (orig_ka.text_size - patched_ka.text_size) // 16
            padding = b''
            for _ in range(pad_count):
                padding += struct.pack('<QQ', NOP_INSTR, NOP_CTRL)
            patched_text = patched_text + padding
            print(f"Padded .text.{kernel_name} with {pad_count} NOPs "
                  f"({patched_ka.text_size} → {orig_ka.text_size} bytes)")
        elif orig_ka.text_size < patched_ka.text_size:
            print(f"ERROR: .text.{kernel_name} in target is smaller than patched: "
                  f"{orig_ka.text_size} vs {patched_ka.text_size}")
            return 1
        dest_off = cubin_pos + orig_ka.text_offset
        orig_text = bytes(o_data[dest_off:dest_off + orig_ka.text_size])
        ndiff = sum(1 for a, b in zip(orig_text, patched_text) if a != b)

        o_data[dest_off:dest_off + orig_ka.text_size] = patched_text
        print(f"Replaced .text.{kernel_name}: {ndiff} changed bytes "
              f"({orig_ka.text_size} total)")

        # Also patch .nv.info metadata if changed
        if orig_ka.info_offset and orig_ka.info_size:
            p_info_off = patched_ka.info_offset or orig_ka.info_offset
            p_info_size = patched_ka.info_size or orig_ka.info_size
            if p_info_size == orig_ka.info_size:
                orig_info = orig_data[
                    orig_ka.info_offset:orig_ka.info_offset + orig_ka.info_size
                ]
                patched_info = patched_data[
                    p_info_off:p_info_off + p_info_size
                ]
                if orig_info != patched_info:
                    info_dest = cubin_pos + orig_ka.info_offset
                    o_data[info_dest:info_dest + orig_ka.info_size] = patched_info
                    info_d = sum(1 for a, b in zip(orig_info, patched_info) if a != b)
                    print(f"Patched .nv.info.{kernel_name}: {info_d} bytes")
    else:
        # --- replace the entire cubin (handles size changes) ---
        if bytes(o_data[cubin_pos:cubin_pos + len(orig_data)]) != orig_data:
            print("ERROR: Full cubin verification failed")
            return 1

        import struct as _st

        old_size = len(orig_data)
        new_size = len(patched_data)
        delta = new_size - old_size
        cubin_end_old = cubin_pos + old_size

        # --- Phase 1: collect all metadata patches BEFORE the splice ---
        # (After splice, offsets into the file change, so read first.)
        patches: list[tuple[int, bytes]] = []  # (file_offset, bytes_to_write)

        if delta != 0:
            # Fatbin wrapper (sits before the cubin, unaffected by splice)
            fatbin_magic = b'\x50\xed\x55\xba'
            fb_pos = o_data.rfind(fatbin_magic, 0, cubin_pos)
            if fb_pos >= 0:
                old_fat = _st.unpack_from('<Q', o_data, fb_pos + 8)[0]
                patches.append((fb_pos + 8, _st.pack('<Q', old_fat + delta)))
                print(f"  Fatbin fat_size: {old_fat} → {old_fat + delta}")

                fb_hdr_size = _st.unpack_from('<H', o_data, fb_pos + 6)[0]
                entry_off = fb_pos + fb_hdr_size
                old_body = _st.unpack_from('<Q', o_data, entry_off + 8)[0]
                patches.append((entry_off + 8, _st.pack('<Q', old_body + delta)))
                print(f"  Entry body_size: {old_body} → {old_body + delta}")

            # ELF section headers (located AFTER the cubin, will shift)
            e_shoff = _st.unpack_from('<Q', o_data, 0x28)[0]
            e_shnum = _st.unpack_from('<H', o_data, 0x3C)[0]
            e_shentsize = _st.unpack_from('<H', o_data, 0x3A)[0]
            FILE_SECTION_TYPES = {1, 2, 3, 4, 5, 6, 7, 9, 14, 17}

            # Read section info from CURRENT (pre-splice) positions
            sh_updates: list[tuple[int, int, int, int, int]] = []  # (i, sh_base, sh_off, sh_size, sh_type)
            for i in range(1, e_shnum):
                sh_base = e_shoff + i * e_shentsize
                if sh_base + 40 > len(o_data):
                    break
                sh_off = _st.unpack_from('<Q', o_data, sh_base + 24)[0]
                sh_size = _st.unpack_from('<Q', o_data, sh_base + 32)[0]
                sh_type = _st.unpack_from('<I', o_data, sh_base + 4)[0]
                sh_updates.append((i, sh_base, sh_off, sh_size, sh_type))

        # --- Phase 2: splice the cubin bytes ---
        o_data[cubin_pos:cubin_pos + old_size] = patched_data
        ndiff = sum(1 for a, b in zip(orig_data[:min(old_size, new_size)],
                                       patched_data[:min(old_size, new_size)])
                    if a != b)
        print(f"Replaced cubin: {ndiff} changed bytes "
              f"({old_size} → {new_size}, delta={delta:+d})")

        # --- Phase 3: apply fatbin metadata patches (before cubin, no shift) ---
        for off, data_bytes in patches:
            o_data[off:off + len(data_bytes)] = data_bytes

        if delta != 0:
            # --- Phase 4: update section headers (now at shifted positions) ---
            for i, sh_base_orig, sh_off, sh_size, sh_type in sh_updates:
                if sh_type not in FILE_SECTION_TYPES or sh_off == 0:
                    continue

                # Section headers themselves shifted if they're after cubin
                sh_base_new = sh_base_orig
                if sh_base_orig >= cubin_end_old:
                    sh_base_new = sh_base_orig + delta

                # Section CONTAINS the cubin: expand its size
                if sh_off <= cubin_pos < sh_off + sh_size:
                    _st.pack_into('<Q', o_data, sh_base_new + 32, sh_size + delta)
                    print(f"  Updated section [{i}] sh_size: "
                          f"{sh_size} → {sh_size + delta}")
                # Section starts AFTER the old cubin end: shift offset
                elif sh_off >= cubin_end_old:
                    _st.pack_into('<Q', o_data, sh_base_new + 24, sh_off + delta)

            # Update e_shoff if section headers are after the cubin
            if e_shoff >= cubin_end_old:
                _st.pack_into('<Q', o_data, 0x28, e_shoff + delta)
                print(f"  Updated e_shoff: 0x{e_shoff:x} → 0x{e_shoff + delta:x}")

    output = Path(args.output) if args.output else o_path
    output.write_bytes(bytes(o_data))
    print(f"Written: {output}")
    return 0


def cmd_fuzz(args: argparse.Namespace) -> int:
    """Run fuzz-testing with random register colorings."""
    from sasskit.recolor.fuzz import run_fuzz

    cubin = Cubin.from_file(args.input)
    kernel_name = args.kernel
    if not kernel_name:
        kernel_name = next(iter(cubin.kernels))

    result = run_fuzz(
        cubin_path=args.input,
        kernel_name=kernel_name,
        target_regs=args.target_regs,
        num_variants=args.num_variants,
        save_dir=args.save_dir,
        sass_test_bin=args.sass_test_bin,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print()
    print(result.summary())
    return 0 if result.passed > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='sasskit',
        description='SASS Toolkit — post-compilation CUDA binary modification',
    )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- disassemble ---
    p_dis = subparsers.add_parser('disassemble', help='Disassemble kernel(s) to clean SASS text')
    p_dis.add_argument('input', help='Input cubin or host ELF (executable / .o)')
    p_dis.add_argument('-k', '--kernel', help='Kernel name (default: all kernels)')
    p_dis.add_argument('-o', '--output', help='Output file (default: stdout)')
    p_dis.add_argument('--with-scheduling', action='store_true',
                       help='Include ctrl word as /* ctrl: 0x... */ comment')

    # --- recolor commands ---

    # analyze
    p_analyze = subparsers.add_parser('analyze', help='Analyze register usage')
    p_analyze.add_argument('input', help='Input cubin file')
    p_analyze.add_argument('-k', '--kernel', help='Kernel name (default: first)')
    p_analyze.add_argument('-t', '--target-regs', type=int, default=64,
                           help='Target register count (default: 64)')
    p_analyze.add_argument('-v', '--verbose', action='store_true')

    # recolor
    p_recolor = subparsers.add_parser('recolor', help='Apply register re-coloring')
    p_recolor.add_argument('input', help='Input cubin file')
    p_recolor.add_argument('-o', '--output', help='Output cubin file')
    p_recolor.add_argument('-k', '--kernel', help='Kernel name')
    p_recolor.add_argument('-t', '--target-regs', type=int, required=True,
                           help='Target register count')
    p_recolor.add_argument('--hot-start', type=lambda x: int(x, 0),
                           help='Hot loop start offset (hex)')
    p_recolor.add_argument('--hot-end', type=lambda x: int(x, 0),
                           help='Hot loop end offset (hex)')
    p_recolor.add_argument('--dry-run', action='store_true',
                           help="Plan but don't write")

    # verify
    p_verify = subparsers.add_parser('verify', help='Verify patched cubin')
    p_verify.add_argument('input', help='Input cubin file')
    p_verify.add_argument('-k', '--kernel', help='Kernel name')

    # diagnose
    p_diag = subparsers.add_parser(
        'diagnose',
        help='Diagnose register encoding issues (find bit_position=-1 refs)',
    )
    p_diag.add_argument('input', help='Input cubin file')
    p_diag.add_argument('-k', '--kernel', help='Kernel name')
    p_diag.add_argument('-t', '--target-regs', type=int, default=64,
                        help='Target register count (default: 64)')

    # inject
    p_inject = subparsers.add_parser(
        'inject',
        help='Inject patched cubin into an object (.o) file',
    )
    p_inject.add_argument('input', help='Input object (.o) file')
    p_inject.add_argument('--cubin-orig', required=True,
                          help='Original extracted cubin (must match the one in .o)')
    p_inject.add_argument('--cubin-patched', required=True,
                          help='Patched cubin to inject')
    p_inject.add_argument('-o', '--output',
                          help='Output .o file (default: overwrite input)')
    p_inject.add_argument('-k', '--kernel',
                          help='Kernel name (for --text-only mode)')
    p_inject.add_argument('--text-only', action='store_true',
                          help='Replace only .text.<kernel> section '
                               '(allows cross-cubin inject)')

    # fuzz
    p_fuzz = subparsers.add_parser(
        'fuzz',
        help='Fuzz-test random register colorings on the GPU',
    )
    p_fuzz.add_argument('input', help='Input cubin file')
    p_fuzz.add_argument('-k', '--kernel', help='Kernel name (default: first)')
    p_fuzz.add_argument('-t', '--target-regs', type=int, required=True,
                        help='Target register count')
    p_fuzz.add_argument('-n', '--num-variants', type=int, default=100,
                        help='Number of random colorings to try (default: 100)')
    p_fuzz.add_argument('--save-dir',
                        help='Directory to save passing cubin variants')
    p_fuzz.add_argument('--sass-test-bin',
                        help='Path to sass_test binary')
    p_fuzz.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    p_fuzz.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress per-variant output')

    # optimize (all-in-one)
    p_opt = subparsers.add_parser(
        'optimize',
        help='One-command optimization: analyze → recolor → verify → save',
    )
    p_opt.add_argument('input', help='Input cubin file')
    p_opt.add_argument('-o', '--output', help='Output cubin file')
    p_opt.add_argument('-k', '--kernel', help='Kernel name (default: all)')
    p_opt.add_argument('-t', '--target-regs', type=int, default=64,
                       help='Target register count (default: 64)')

    # schedule
    p_sched = subparsers.add_parser(
        'schedule',
        help='Optimize instruction stall counts (no reordering)',
    )
    p_sched.add_argument('input', help='Input cubin file')
    p_sched.add_argument('-o', '--output', help='Output cubin file')
    p_sched.add_argument('-k', '--kernel', help='Kernel name (default: all)')
    p_sched.add_argument('--dry-run', action='store_true',
                         help='Analyze but do not modify')

    # reforge
    p_reforge = subparsers.add_parser(
        'reforge',
        help='SASS-to-SASS optimizer: mutate → encode → GPU bench → accept/reject',
    )
    p_reforge.add_argument('input', help='Input cubin file')
    p_reforge.add_argument('-k', '--kernel', help='Kernel name (default: first)')
    p_reforge.add_argument('-n', '--iterations', type=int, default=200,
                           help='Max mutation iterations (default: 200)')
    p_reforge.add_argument('--blocks', type=int, default=1)
    p_reforge.add_argument('--threads', type=int, default=256)
    p_reforge.add_argument('--smem', type=int, default=28672)
    p_reforge.add_argument('--temperature', type=float, default=0.1)
    p_reforge.add_argument('--seed', type=int, default=42)

    # --- forge commands ---
    try:
        from sasskit.forge.cli import add_forge_subparsers, dispatch_forge
        add_forge_subparsers(subparsers)
        has_forge = True
    except ImportError:
        has_forge = False

    args = parser.parse_args()

    if args.command == 'disassemble':
        return cmd_disassemble(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    elif args.command == 'recolor':
        return cmd_recolor(args)
    elif args.command == 'optimize':
        return cmd_optimize(args)
    elif args.command == 'schedule':
        return cmd_schedule(args)
    elif args.command == 'reforge':
        return cmd_reforge(args)
    elif args.command == 'verify':
        return cmd_verify(args)
    elif args.command == 'diagnose':
        return cmd_diagnose(args)
    elif args.command == 'inject':
        return cmd_inject(args)
    elif args.command == 'fuzz':
        return cmd_fuzz(args)
    elif has_forge and args.command.startswith('forge-'):
        return dispatch_forge(args.command, args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
