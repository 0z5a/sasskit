"""sm_120 SASS instruction decoder with control-word register field support.

On sm_120, register fields appear in TWO places:
  Instruction word: bits[23:16]=Rd, [31:24]=Ra, [39:32]=Rb, [47:40]=Rc
  Control word: bits[7:0]=Rc2 (accumulator in IMAD with immediate, etc.)
"""
from __future__ import annotations
import re, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from .cubin import Cubin, KernelInfo

@dataclass
class RegRef:
    reg_num: int
    is_dest: bool
    is_pair: bool
    bit_position: int
    field_name: str
    in_ctrl_word: bool = False
    is_quad: bool = False

@dataclass  
class Instruction:
    code_offset: int
    instr_word: int
    ctrl_word: int
    asm_text: str
    opcode: str
    reg_refs: list[RegRef] = field(default_factory=list)
    is_predicated: bool = False
    predicate: Optional[str] = None
    
    @property
    def opcode_bits(self): return self.instr_word & 0xFFFF
    @property
    def stall(self): return self.ctrl_word & 0xF
    @property
    def is_nop(self): return self.opcode_bits == 0x7918
    @property
    def is_branch(self): return self.opcode.startswith('BRA') or self.opcode == 'EXIT'
    @property
    def dest_regs(self):
        r = []
        for ref in self.reg_refs:
            if ref.is_dest:
                r.append(ref.reg_num)
                if getattr(ref, 'is_quad', False):
                    r.extend([ref.reg_num + 1, ref.reg_num + 2, ref.reg_num + 3])
                elif ref.is_pair:
                    r.append(ref.reg_num + 1)
        return r
    @property
    def src_regs(self):
        r = []
        for ref in self.reg_refs:
            if not ref.is_dest:
                r.append(ref.reg_num)
                if getattr(ref, 'is_quad', False):
                    r.extend([ref.reg_num + 1, ref.reg_num + 2, ref.reg_num + 3])
                elif ref.is_pair:
                    r.append(ref.reg_num + 1)
        return r
    @property
    def all_regs(self): return set(self.dest_regs + self.src_regs)

def _find_cubit() -> str | None:
    """Find cubit binary via CUBIT_BIN env var, then PATH."""
    import shutil, os
    explicit = os.environ.get('CUBIT_BIN')
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit
    return shutil.which('cubit')


def disassemble_kernel(cubin_path, kernel_name):
    """Disassemble a kernel using cubit (no cuobjdump required) with fallback."""
    cubit = _find_cubit()
    if cubit:
        return _disassemble_with_cubit(cubit, cubin_path, kernel_name)
    return _disassemble_with_cuobjdump(cubin_path, kernel_name)


def _disassemble_with_cubit(cubit_bin: str, cubin_path, kernel_name: str) -> dict:
    """Disassemble using cubit disassemble (our own decoder)."""
    import os
    TABLE = os.environ.get('CUBIT_TABLE', 'tables/sm120.json')
    if not os.path.isfile(TABLE):
        return _disassemble_with_cuobjdump(cubin_path, kernel_name)

    result = subprocess.run(
        [cubit_bin, 'disassemble', str(cubin_path),
         '-k', kernel_name, '-t', TABLE],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # Fall back to cuobjdump if cubit fails
        return _disassemble_with_cuobjdump(cubin_path, kernel_name)

    instructions = {}
    for line in result.stdout.splitlines():
        # cubit format: /*addr*/  ASM ;
        m = re.match(r'^\s*/\*([0-9a-f]+)\*/\s+(.+?)\s*;?\s*$', line)
        if m:
            addr = int(m.group(1), 16)
            asm = m.group(2).strip().rstrip(';').strip()
            instructions[addr] = asm
    return instructions


def _disassemble_with_cuobjdump(cubin_path, kernel_name: str) -> dict:
    """Disassemble using cuobjdump -sass (original implementation)."""
    result = subprocess.run(
        ['/usr/local/cuda/bin/cuobjdump', '-sass', str(cubin_path)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {result.stderr}")
    instructions = {}
    in_kernel = False
    for line in result.stdout.split('\n'):
        if f'Function : {kernel_name}' in line:
            in_kernel = True; continue
        if in_kernel and 'Function :' in line: break
        if in_kernel:
            m = re.match(r'\s+/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*\s*0x[0-9a-f]+\s*\*/', line)
            if m:
                instructions[int(m.group(1), 16)] = m.group(2).strip()
    return instructions

def _parse_reg_refs(asm_text, instr_word, ctrl_word):
    """Extract register refs from asm text, checking BOTH instruction and control words.

    Cross-references register names from ``cuobjdump -sass`` text with the
    binary encoding.  A register name ``Rn`` in the asm text is matched to
    the first bit-field whose byte value equals ``n``.  This naturally
    avoids touching immediate fields: an immediate ``0x10`` in the asm text
    does NOT match ``\\bR(\\d+)\\b`` (no ``R`` prefix), so the Rb field
    won't be treated as a register.

    For format 0x8/0xC, the Rc operand (if a register) lives in the
    control word bits[7:0] rather than the instruction word bits[47:40].
    """
    refs = []
    text = re.sub(r'^@!?P\d+\s+', '', asm_text) if asm_text.startswith('@') else asm_text
    reg_matches = list(re.finditer(r'\bR(\d+)\b', text))

    # Determine if Rc lives in the control word or instruction word.
    #
    # On sm_120, bit 11 of the opcode selects the Rb source format:
    #   bit 11 = 0  →  register Rb (bits[39:32])
    #   bit 11 = 1  →  immediate Rb / UR Rb
    #
    # ctrl[7:0] holds the Rc accumulator for IMAD/SHF/LEA instructions,
    # but holds the UR descriptor index for memory instructions (LDG, STG,
    # LD, ST, ATOM, RED).  We must NEVER treat the UR descriptor as Rc.
    #
    # Strategy:
    #   Memory instructions: never use Rc_ctrl (ctrl[7:0] = UR descriptor)
    #   Compute instructions with bit11=1: Rc_ctrl checked BEFORE Rb
    #   Compute instructions with bit11=0: Rc_ctrl as fallback only if
    #       instr bits[47:40] is empty (IMAD.WIDE/HI in register mode)
    bare_op = re.sub(r'^@!?P\d+\s+', '', asm_text).split()[0] if asm_text else ''
    is_mem = bare_op.startswith(('LD', 'ST', 'ATOM', 'RED'))
    rc_in_ctrl = bool((instr_word >> 11) & 1)
    rc_instr_byte = (instr_word >> 40) & 0xFF

    # Fields: (bit_pos, name, in_ctrl_word)
    if is_mem:
        # Memory instructions: ctrl[7:0] is UR descriptor, NOT Rc.
        # Never include Rc_ctrl.
        fields = [
            (16, 'Rd', False), (24, 'Ra', False),
            (32, 'Rb', False), (40, 'Rc', False),
        ]
    elif rc_in_ctrl:
        # Compute with immediate/UR Rb: Rc at ctrl[7:0].
        # Check Rc_ctrl BEFORE Rb to prevent false match on immediate.
        fields = [
            (16, 'Rd', False), (24, 'Ra', False),
            (0, 'Rc_ctrl', True),
            (32, 'Rb', False), (40, 'Rc', False),
        ]
    elif rc_instr_byte in (0x00, 0xFF):
        # Register Rb, but Rc field empty → Rc overflows to ctrl[7:0]
        # (IMAD.WIDE, IMAD.HI, SHF in register mode).
        fields = [
            (16, 'Rd', False), (24, 'Ra', False), (32, 'Rb', False),
            (40, 'Rc', False), (0, 'Rc_ctrl', True),
        ]
    else:
        # Register Rb with real Rc at instr bits[47:40].
        fields = [
            (16, 'Rd', False), (24, 'Ra', False), (32, 'Rb', False),
            (40, 'Rc', False),
        ]

    used = set()

    for m in reg_matches:
        rn = int(m.group(1))
        if rn > 127: continue
        matched = False
        for bp, fn, ic in fields:
            key = (bp, ic)
            if key in used: continue
            word = ctrl_word if ic else instr_word
            if ((word >> bp) & 0xFF) == rn:
                is_d = (fn == 'Rd')
                # Pair detection rules for SM 70+:
                #
                # .WIDE (IMAD.WIDE): Rd:Rd+1 = 64-bit result, Ra is 32-bit scalar → only Rd is pair
                # .64 arithmetic (IADD.64): Rd:Rd+1 and Ra:Ra+1 both 64-bit → Rd and Ra are pairs
                # .64 memory (STS.64, LDS.64, LDC.64): Rd:Rd+1 = 64-bit data, Ra = 32-bit address → only Rd
                # LDG/STG with Ra.64: Ra:Ra+1 = 64-bit address → Ra is pair
                # .128 memory (LDS.128, STG.128): Rd is quad
                #
                bare_op = re.sub(r'^@!?P\d+\s+', '', asm_text).split()[0] if asm_text else ''
                is_mem = bare_op.startswith(('LD', 'ST', 'ATOM', 'RED'))
                
                # Check if .64 is in the OPCODE (e.g. "IADD.64", "LDC.64", "STS.64")
                # NOT in an operand (e.g. "R12.64" which is address syntax)
                opcode_has_64 = '.64' in bare_op
                
                is_p = False
                if '.WIDE' in bare_op:
                    # .WIDE: Rd is 64-bit result, Rc/Rc_ctrl is 64-bit accumulator
                    # Ra is 32-bit scalar, Rb is 32-bit scalar/immediate
                    is_p = (fn in ('Rd', 'Rc', 'Rc_ctrl'))
                elif opcode_has_64:
                    if is_mem:
                        # memory .64: data register is pair (Rd for loads, Rb for stores)
                        is_p = (fn in ('Rd', 'Rb'))
                    else:
                        # arithmetic .64: all register operands are 64-bit pairs
                        # e.g. IADD.64 Rd, Ra, Rb → Rd:Rd+1 = Ra:Ra+1 + Rb:Rb+1
                        is_p = (fn in ('Rd', 'Ra', 'Rb', 'Rc', 'Rc_ctrl'))
                # LDG/STG with Ra.64 syntax → Ra is 64-bit address pair
                if fn == 'Ra' and re.search(r'R' + str(rn) + r'\.64', asm_text):
                    is_p = True
                is_q = '.128' in asm_text and (is_d or (is_mem and fn == 'Rb'))
                if is_q: is_p = False  # quad supersedes pair
                refs.append(RegRef(rn, is_d, is_p, bp, fn, ic, is_q))
                used.add(key)
                matched = True
                break
        if not matched:
            refs.append(RegRef(rn, False, False, -1, 'unknown'))
    return refs

def decode_kernel(cubin, kernel_name):
    kernel = cubin.get_kernel(kernel_name)
    asm_map = disassemble_kernel(cubin.path, kernel_name)
    instructions = []
    for i in range(kernel.num_instructions):
        off = i * 16
        iw, cw = cubin.read_instruction(kernel, off)
        asm = asm_map.get(off, '')
        op, pred, is_pred = '', None, False
        if asm:
            t = asm
            pm = re.match(r'^(@!?P\d+)\s+(.*)', t)
            if pm: is_pred, pred, t = True, pm.group(1), pm.group(2)
            op = t.split()[0] if t.split() else ''
        rr = _parse_reg_refs(asm, iw, cw) if asm else []
        instructions.append(Instruction(off, iw, cw, asm, op, rr, is_pred, pred))
    return instructions

def rename_register(cubin, kernel, code_offset, bit_position, old_reg, new_reg, in_ctrl_word=False):
    """Rename a register field. Handles both instruction-word and control-word fields."""
    instr, ctrl = cubin.read_instruction(kernel, code_offset)
    word = ctrl if in_ctrl_word else instr
    if ((word >> bit_position) & 0xFF) != old_reg:
        return False
    word = (word & ~(0xFF << bit_position)) | ((new_reg & 0xFF) << bit_position)
    if in_ctrl_word:
        cubin.write_instruction(kernel, code_offset, instr, word)
    else:
        cubin.write_instruction(kernel, code_offset, word, ctrl)
    return True


# Branch-like opcodes that encode offsets in register bit fields (NOT actual registers).
# Format: lower 16 bits of instruction word (lo & 0xFFFF).
_BRANCH_OPCODES = frozenset({
    0x1947,  # BRA / @Px BRA (conditional branch)
    0x0947,  # @P0 BRA
    0x7941,  # BSYNC
    0x7945,  # BSSY
    0x7942,  # BRX
    0x7943,  # JMP
    0x7946,  # CALL
    0x7947,  # RET
    0x7940,  # EXIT / BRA.U
})

# Register byte positions inside a 128-bit SASS instruction (lo word = bytes 0..7).
# (bit_position_in_lo, field_name, is_ctrl_word)
_REG_FIELDS = [
    (16, 'Rd', False),   # bits [23:16] – destination
    (24, 'Ra', False),   # bits [31:24] – source A
    (32, 'Rb', False),   # bits [39:32] – source B (only when not immediate)
    (40, 'Rc', False),   # bits [47:40] – source C (only for 3-op instr)
    (0,  'Rc_ctrl', True),  # ctrl-word bits [7:0] – Rc accumulator for IMAD etc.
]


def rename_register_all_sections(cubin, kernel_name, rename_map):
    """Rename registers in ALL text sections of a kernel (including .nv.capmerc.text.*).

    ``rename_map`` is a dict ``{old_reg: new_reg}``.

    Avoids patching:
    - Branch instructions (their "register" fields encode jump offsets).
    - Byte positions that are part of the opcode (bytes 0 and 1 of the lo word).
    - Bytes in the hi word above the ctrl-word Rc field (those are barrier/scheduling bits).

    Returns the total number of byte patches applied.
    """
    import struct

    data = cubin.data
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shentsize = struct.unpack_from('<H', data, 58)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]
    strtab_off = struct.unpack_from('<Q', data, e_shoff + e_shstrndx * e_shentsize + 24)[0]
    strtab_size = struct.unpack_from('<Q', data, e_shoff + e_shstrndx * e_shentsize + 32)[0]
    strtab = bytes(data[strtab_off:strtab_off + strtab_size])

    def _sec_name(n):
        end = strtab.index(b'\x00', n)
        return strtab[n:end].decode()

    total = 0
    old_set = set(rename_map.keys())

    for i in range(e_shnum):
        sh_off = e_shoff + i * e_shentsize
        sh_name_off = struct.unpack_from('<I', data, sh_off)[0]
        sh_file_off = struct.unpack_from('<Q', data, sh_off + 24)[0]
        sh_size = struct.unpack_from('<Q', data, sh_off + 32)[0]
        sec_name = _sec_name(sh_name_off)

        # Only process text sections belonging to this kernel
        if not (sec_name == f'.text.{kernel_name}' or
                sec_name == f'.nv.capmerc.text.{kernel_name}'):
            continue

        for j in range(0, sh_size - 16, 16):
            base = sh_file_off + j
            lo = struct.unpack_from('<Q', data, base)[0]
            ctrl = struct.unpack_from('<Q', data, base + 8)[0]

            opcode = lo & 0xFFFF
            if opcode in _BRANCH_OPCODES:
                continue  # branch offsets live in register fields – do not touch

            has_imm = bool((lo >> 11) & 1)  # bit 11: Rb is immediate/UR

            for bit_pos, fname, is_ctrl in _REG_FIELDS:
                # Skip Rb register field when instruction uses immediate Rb
                if has_imm and fname == 'Rb':
                    continue
                word = ctrl if is_ctrl else lo
                byte_val = (word >> bit_pos) & 0xFF
                if byte_val in old_set:
                    new_r = rename_map[byte_val]
                    new_word = (word & ~(0xFF << bit_pos)) | ((new_r & 0xFF) << bit_pos)
                    file_off = base + (8 if is_ctrl else 0)
                    struct.pack_into('<Q', data, file_off, new_word)
                    total += 1

    return total
