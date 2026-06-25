#!/usr/bin/env python3
"""Parse cuobjdump -sass output and extract scheduling bits per opcode type.

Usage:
    export PATH=/usr/local/cuda-13.1/bin:$PATH
    cuobjdump -sass probe_sched.cubin | python3 parse_probe.py
    OR
    python3 parse_probe.py probe_sched.cubin [probe_mov_O0.cubin ...]
"""

import re
import sys
import subprocess
import os

# Bit field definitions for ctrl (hi64 of 128-bit instruction):
#   bits [63:58] = reuse flags (6 bits)
#   bits [57:41] = scheduling: stall + yield + barriers (17 bits)
#   bits [40:0]  = modifier bits (41 bits)
MODIFIER_MASK = (1 << 41) - 1
SCHED_MASK_17 = (1 << 17) - 1
REUSE_MASK_6  = (1 << 6) - 1

def parse_ctrl(ctrl_hi64):
    """Split ctrl word into reuse, scheduling, and modifier fields."""
    modifier = ctrl_hi64 & MODIFIER_MASK
    sched = (ctrl_hi64 >> 41) & SCHED_MASK_17
    reuse = (ctrl_hi64 >> 58) & REUSE_MASK_6
    return reuse, sched, modifier

def decode_sched(sched_17):
    """Decode the 17-bit scheduling field.
    
    SM 120 scheduling (approximate, from Ampere/Hopper docs extrapolation):
      bits [3:0]  = stall count (0-15 cycles)
      bit  [4]    = yield hint
      bits [9:5]  = barrier write mask? 
      bits [16:10] = barrier read mask?
    """
    stall = sched_17 & 0xF
    yield_bit = (sched_17 >> 4) & 1
    rest = sched_17 >> 5
    return stall, yield_bit, rest

def parse_cuobjdump_sass(text):
    """Parse cuobjdump -sass output into list of (offset, opcode, lo64, hi64)."""
    instructions = []
    
    # Pattern: /*offset*/ [predicate] OPCODE operands ; /* lo64 */
    #                                                    /* hi64 */
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        # Look for instruction line with offset
        m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\*\s*0x([0-9a-f]+)\s*\*/', lines[i])
        if m:
            offset = int(m.group(1), 16)
            asm_text = m.group(2).strip()
            lo64 = int(m.group(3), 16)
            
            # Next line should have ctrl word
            if i + 1 < len(lines):
                m2 = re.match(r'\s*/\*\s*0x([0-9a-f]+)\s*\*/', lines[i+1])
                if m2:
                    hi64 = int(m2.group(1), 16)
                    
                    # Extract opcode name (strip predicate guard)
                    opcode = asm_text
                    opcode = re.sub(r'^@!?P\d+\s+', '', opcode)  # strip predicate
                    opcode = opcode.split()[0] if opcode.split() else opcode
                    opcode = opcode.rstrip(';').strip()
                    
                    instructions.append((offset, opcode, lo64, hi64, asm_text))
                    i += 2
                    continue
        i += 1
    
    return instructions

def main():
    # Collect input
    if len(sys.argv) > 1 and not sys.stdin.isatty():
        text = sys.stdin.read()
    elif len(sys.argv) > 1:
        # Filenames provided
        text = ""
        for fname in sys.argv[1:]:
            if os.path.isfile(fname):
                result = subprocess.run(
                    ['cuobjdump', '-sass', fname],
                    capture_output=True, text=True
                )
                text += f"\n=== {fname} ===\n" + result.stdout
    else:
        text = sys.stdin.read()
    
    instructions = parse_cuobjdump_sass(text)
    
    # Group by opcode
    opcode_map = {}
    for offset, opcode, lo64, hi64, asm_text in instructions:
        if opcode not in opcode_map:
            opcode_map[opcode] = []
        reuse, sched, modifier = parse_ctrl(hi64)
        stall, yield_bit, sched_rest = decode_sched(sched)
        opcode_map[opcode].append({
            'offset': offset,
            'lo64': lo64,
            'hi64': hi64,
            'reuse': reuse,
            'sched': sched,
            'modifier': modifier,
            'stall': stall,
            'yield': yield_bit,
            'sched_rest': sched_rest,
            'asm': asm_text,
        })
    
    # Target opcodes for modmulp
    target_opcodes = {'MOV', 'IMAD.WIDE.U32', 'IADD', 'IADD.X', 'SEL'}
    
    print("=" * 100)
    print("SCHEDULING ANALYSIS — all opcodes found")
    print("=" * 100)
    
    for opcode in sorted(opcode_map.keys()):
        entries = opcode_map[opcode]
        marker = " ★★★ TARGET" if opcode in target_opcodes else ""
        print(f"\n{'─'*80}")
        print(f"  {opcode} ({len(entries)} instances){marker}")
        print(f"{'─'*80}")
        
        for e in entries:
            print(f"  off=0x{e['offset']:04x}  ctrl=0x{e['hi64']:016x}"
                  f"  reuse=0x{e['reuse']:02x} sched=0x{e['sched']:05x}"
                  f"  (stall={e['stall']:2d} yield={e['yield']} rest=0x{e['sched_rest']:03x})"
                  f"  mod=0x{e['modifier']:011x}")
            print(f"           {e['asm']}")
    
    # Summary table for target opcodes
    print(f"\n{'='*100}")
    print("SCHEDULING SUMMARY — target opcodes for modmulp_full.sass")
    print(f"{'='*100}")
    print(f"{'Opcode':<20} {'Count':>5}  {'Unique sched values (bits [57:41])'}")
    print(f"{'─'*80}")
    
    sched_table = {}
    for opcode in sorted(target_opcodes):
        if opcode in opcode_map:
            sched_vals = sorted(set(e['sched'] for e in opcode_map[opcode]))
            sched_table[opcode] = sched_vals
            sched_hex = ', '.join(f'0x{v:05x}' for v in sched_vals)
            print(f"{opcode:<20} {len(opcode_map[opcode]):>5}  {sched_hex}")
        else:
            print(f"{opcode:<20}     0  (not found in probe)")
    
    # Generate Python dict for isa.py
    print(f"\n{'='*100}")
    print("SUGGESTED SCHEDULING TABLE (for isa.py)")
    print("Use FIRST occurrence's full bits [63:41] as template.")
    print(f"{'='*100}")
    print()
    print("OPCODE_SCHED = {")
    for opcode in sorted(target_opcodes):
        if opcode in opcode_map:
            # Use first entry's full hi64 bits [63:41] as the scheduling template
            first = opcode_map[opcode][0]
            sched_template = first['hi64'] & ~MODIFIER_MASK
            print(f'    "{opcode}": 0x{sched_template:016x},  '
                  f'# reuse=0x{first["reuse"]:02x} sched=0x{first["sched"]:05x} '
                  f'stall={first["stall"]} yield={first["yield"]}')
        else:
            print(f'    "{opcode}": 0x000fe20000000000,  # NOT FOUND — using default')
    print("}")
    
    # Also output ALL opcodes' scheduling for reference
    print()
    print("# ALL opcodes found:")
    print("ALL_OPCODE_SCHED = {")
    for opcode in sorted(opcode_map.keys()):
        first = opcode_map[opcode][0]
        sched_template = first['hi64'] & ~MODIFIER_MASK
        print(f'    "{opcode}": 0x{sched_template:016x},  '
              f'# stall={first["stall"]} yield={first["yield"]}')
    print("}")

if __name__ == '__main__':
    main()
