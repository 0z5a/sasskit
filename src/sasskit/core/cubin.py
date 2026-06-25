"""CUDA cubin (ELF) parser.

Handles reading and writing cubin files, extracting kernel text sections,
parsing .nv.info attributes (register count, shared memory, etc.), and
modifying ELF metadata for re-coloring.

Cubin files are standard ELF64 with NVIDIA-specific sections:
  .text.<KernelName>     — SASS machine code (16 bytes per instruction)
  .nv.info.<KernelName>  — Kernel attributes (reg count, smem size, ...)
  .nv.shared.<KernelName>— Static shared memory (NOBITS section, size only)
  .nv.constant0.<KernelName> — Constant bank 0 data

Reference: CuAssembler's CubinFile and CuNVInfo classes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from elftools.elf.elffile import ELFFile


# .nv.info attribute codes (from CuAssembler + reverse engineering)
# Format: each attribute is a (tag, size, data) triple.
NVINFO_ATTR_REGCOUNT = 0x1b04  # EIATTR_REGCOUNT: u32 register count
NVINFO_ATTR_MAX_STACK = 0x2304  # EIATTR_MAX_STACK_SIZE: u32 bytes
NVINFO_ATTR_MIN_STACK = 0x1204  # EIATTR_MIN_STACK_SIZE: u32 bytes
NVINFO_ATTR_SMEM_SIZE = 0x1e04  # EIATTR_SHARED: u32 static smem bytes


@dataclass
class KernelInfo:
    """Parsed information about a single kernel in a cubin."""
    name: str
    text_offset: int       # File offset of .text.<name> section
    text_size: int         # Size in bytes
    text_index: int        # ELF section index
    info_offset: Optional[int] = None  # File offset of .nv.info.<name>
    info_size: Optional[int] = None
    shared_size: int = 0   # Static shared memory (bytes)
    reg_count: int = 0     # Number of registers
    max_stack: int = 0     # Stack frame size (bytes)
    reserved_smem: int = 0 # Reserved shared memory for spills (bytes)
    
    @property
    def num_instructions(self) -> int:
        return self.text_size // 16


@dataclass
class Cubin:
    """Parsed cubin file with read/write capability.
    
    Usage:
        cubin = Cubin.from_file("kernel.cubin")
        kernel = cubin.get_kernel("KernelA")
        print(f"Registers: {kernel.reg_count}")
        
        # Read an instruction (16 bytes: 8-byte instr + 8-byte ctrl)
        instr, ctrl = cubin.read_instruction(kernel, offset=0x3D30)
        
        # Modify and write back
        cubin.write_instruction(kernel, offset=0x3D30, instr=new_instr, ctrl=new_ctrl)
        cubin.set_reg_count(kernel, 64)
        cubin.save("kernel_patched.cubin")
    """
    path: Path
    data: bytearray
    kernels: dict[str, KernelInfo] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: str | Path) -> Cubin:
        """Load and parse a cubin or host ELF (executable / .o file).

        If *path* is a host binary (x86-64 executable or object) with an
        embedded CUDA fatbin, the CUDA cubin is extracted automatically via
        ``cuobjdump --extract-elf`` and its bytes are used for parsing.
        The original *path* is kept so ``cuobjdump -sass`` calls still work.
        """
        path = Path(path)
        data = bytearray(path.read_bytes())

        # Detect host ELF by reading e_machine from bytes 18-19 (u16 LE).
        # EM_CUDA = 0xBE = 190; anything else is a host binary.
        ELF_MAGIC = b'\x7fELF'
        EM_CUDA = 190
        if (len(data) >= 20
                and data[:4] == ELF_MAGIC
                and int.from_bytes(data[18:20], 'little') != EM_CUDA):
            data = cls._extract_cubin_bytes(path)

        cubin = cls(path=path, data=data)
        cubin._parse()
        return cubin

    @classmethod
    def _extract_cubin_bytes(cls, path: Path) -> bytearray:
        """Extract CUDA cubin from a host ELF via cuobjdump --extract-elf."""
        import subprocess
        import tempfile
        import os

        CUOBJDUMP = '/usr/local/cuda/bin/cuobjdump'
        abs_path = path.resolve()

        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [CUOBJDUMP, '--extract-elf', 'all', str(abs_path)],
                cwd=tmpdir, capture_output=True,
            )
            cubins = sorted(f for f in os.listdir(tmpdir) if f.endswith('.cubin'))
            if not cubins:
                raise RuntimeError(
                    f"No CUDA cubins found in '{path}'. "
                    "Make sure the binary was compiled with CUDA and that "
                    "cuobjdump is on PATH."
                )
            cubin_path = os.path.join(tmpdir, cubins[-1])
            return bytearray(open(cubin_path, 'rb').read())
    
    def _parse(self) -> None:
        """Parse ELF sections to find kernels and their attributes."""
        from io import BytesIO
        elf = ELFFile(BytesIO(bytes(self.data)))
        
        # First pass: find all .text.<name> sections (kernel code)
        text_sections: dict[str, tuple[int, int, int]] = {}
        for i, sec in enumerate(elf.iter_sections()):
            name = sec.name
            if name.startswith('.text.') and sec['sh_type'] == 'SHT_PROGBITS':
                kernel_name = name[6:]  # strip '.text.'
                text_sections[kernel_name] = (sec['sh_offset'], sec['sh_size'], i)
        
        # Second pass: find .nv.info, .nv.shared, etc.
        info_sections: dict[str, tuple[int, int]] = {}
        shared_sections: dict[str, int] = {}
        for sec in elf.iter_sections():
            name = sec.name
            if name.startswith('.nv.info.'):
                kname = name[9:]
                info_sections[kname] = (sec['sh_offset'], sec['sh_size'])
            elif name.startswith('.nv.shared.'):
                kname = name[11:]
                shared_sections[kname] = sec['sh_size']
        
        # Build KernelInfo for each kernel
        for kname, (off, size, idx) in text_sections.items():
            ki = KernelInfo(
                name=kname,
                text_offset=off,
                text_size=size,
                text_index=idx,
            )
            if kname in info_sections:
                ki.info_offset, ki.info_size = info_sections[kname]
                self._parse_nv_info(ki)
            if kname in shared_sections:
                ki.shared_size = shared_sections[kname]
            self.kernels[kname] = ki
    
    def _parse_nv_info(self, ki: KernelInfo) -> None:
        """Parse .nv.info section to extract kernel attributes.
        
        NV info format (sm_75+): sequence of (u16 tag, u16 size, data[size]) entries.
        Some tags are 3-byte: (u8 format, u8 attr, u8 size) — depends on format byte.
        
        The format is not fully documented. We look for known attribute patterns.
        """
        if ki.info_offset is None or ki.info_size is None:
            return
        
        data = self.data[ki.info_offset:ki.info_offset + ki.info_size]
        pos = 0
        while pos + 4 <= len(data):
            # Try to parse as (format, attr, size_or_value) triples
            fmt = data[pos]
            attr = data[pos + 1]
            
            if fmt == 0x04:
                # Format 4: (0x04, attr, u16 size, data[size])
                if pos + 4 > len(data):
                    break
                size = struct.unpack_from('<H', data, pos + 2)[0]
                if pos + 4 + size > len(data):
                    break
                payload = data[pos + 4:pos + 4 + size]
                self._handle_nv_attr(ki, attr, payload)
                pos += 4 + size
            elif fmt == 0x03:
                # Format 3: (0x03, attr, u8 value) — 3 bytes total
                if pos + 3 > len(data):
                    break
                value = data[pos + 2]
                self._handle_nv_attr_byte(ki, attr, value)
                pos += 3
            elif fmt == 0x02:
                # Format 2: (0x02, attr, u8 value) — 3 bytes total
                if pos + 3 > len(data):
                    break
                value = data[pos + 2]
                self._handle_nv_attr_byte(ki, attr, value, fmt=0x02)
                pos += 3
            else:
                pos += 1  # Skip unknown format bytes
    
    def _handle_nv_attr(self, ki: KernelInfo, attr: int, payload: bytes) -> None:
        """Handle a parsed .nv.info attribute with payload."""
        if attr == 0x1b and len(payload) >= 4:
            # EIATTR_REGCOUNT (observed: format=0x04, attr=0x1b)
            # But actually seen as byte at attr 0x4c position
            pass
        if attr == 0x1c and len(payload) >= 4:
            # EIATTR_MAX_STACK_SIZE
            ki.max_stack = struct.unpack_from('<I', payload, 0)[0]
        if attr == 0x1e and len(payload) >= 4:
            # EIATTR_SHARED — but usually in .nv.shared section size
            pass
    
    def _handle_nv_attr_byte(self, ki: KernelInfo, attr: int, value: int, fmt: int = 3) -> None:
        """Handle a 3-byte .nv.info attribute."""
        if attr == 0x1b and fmt == 0x03:
            # EIATTR_REGCOUNT format 3: (03, 1b, count)
            ki.reg_count = value
        if fmt == 0x02 and attr >= 0x20:
            # Format 2: attr byte IS often the register count (attr=0x4c=76)
            # Heuristic: if we haven't found reg count yet and attr looks like
            # a valid register count (32-255), use it
            if ki.reg_count == 0 or ki.reg_count > 200:
                if 16 <= attr <= 200:
                    ki.reg_count = attr
    
    def get_kernel(self, name: str) -> KernelInfo:
        """Get kernel info by name. Raises KeyError if not found."""
        if name not in self.kernels:
            available = ', '.join(self.kernels.keys())
            raise KeyError(f"Kernel '{name}' not found. Available: {available}")
        return self.kernels[name]
    
    def read_instruction(self, kernel: KernelInfo, code_offset: int) -> tuple[int, int]:
        """Read a 16-byte instruction at the given code offset within a kernel.
        
        Returns (instruction_word, control_word) as u64 pair.
        """
        file_off = kernel.text_offset + code_offset
        instr = struct.unpack_from('<Q', self.data, file_off)[0]
        ctrl = struct.unpack_from('<Q', self.data, file_off + 8)[0]
        return instr, ctrl
    
    def write_instruction(self, kernel: KernelInfo, code_offset: int,
                          instr: int, ctrl: int) -> None:
        """Write a 16-byte instruction at the given code offset."""
        file_off = kernel.text_offset + code_offset
        struct.pack_into('<Q', self.data, file_off, instr)
        struct.pack_into('<Q', self.data, file_off + 8, ctrl)
    
    def find_reg_count_offset(self, kernel: KernelInfo) -> Optional[int]:
        """Find the file offset of the register count byte in .nv.info.
        
        Returns the file offset where the register count value is stored,
        or None if not found.
        """
        if kernel.info_offset is None or kernel.info_size is None:
            return None
        
        data = self.data[kernel.info_offset:kernel.info_offset + kernel.info_size]
        pos = 0
        while pos + 3 <= len(data):
            fmt = data[pos]
            attr = data[pos + 1]
            if fmt == 0x03 and attr == 0x1b:
                # Register count is at pos+2 (1 byte)
                return kernel.info_offset + pos + 2
            elif fmt == 0x02 and attr == 0x4c:
                # Alternative encoding seen in some cubins
                return kernel.info_offset + pos + 2
            elif fmt == 0x04:
                if pos + 4 > len(data):
                    break
                size = struct.unpack_from('<H', data, pos + 2)[0]
                pos += 4 + size
            elif fmt in (0x02, 0x03):
                pos += 3
            else:
                pos += 1
        return None
    
    def set_reg_count(self, kernel: KernelInfo, new_count: int) -> bool:
        """Patch the register count in .nv.info. Returns True on success."""
        offset = self.find_reg_count_offset(kernel)
        if offset is None:
            # Brute-force: search for the current reg count value
            if kernel.info_offset and kernel.info_size:
                data = self.data[kernel.info_offset:kernel.info_offset + kernel.info_size]
                for i in range(len(data)):
                    if data[i] == kernel.reg_count:
                        # Verify context: should be preceded by format+attr bytes
                        if i >= 2 and data[i-1] == 0x1b:
                            offset = kernel.info_offset + i
                            break
            if offset is None:
                return False
        
        self.data[offset] = new_count & 0xFF
        kernel.reg_count = new_count
        return True
    
    def set_max_stack(self, kernel: KernelInfo, new_stack: int) -> bool:
        """Patch the max stack size in .nv.info. Returns True on success."""
        if kernel.info_offset is None or kernel.info_size is None:
            return False
        
        data = self.data[kernel.info_offset:kernel.info_offset + kernel.info_size]
        pos = 0
        while pos + 4 <= len(data):
            fmt = data[pos]
            attr = data[pos + 1]
            if fmt == 0x04 and attr == 0x1c:
                size = struct.unpack_from('<H', data, pos + 2)[0]
                if size >= 4:
                    file_off = kernel.info_offset + pos + 4
                    struct.pack_into('<I', self.data, file_off, new_stack)
                    kernel.max_stack = new_stack
                    return True
                break
            elif fmt == 0x04:
                size = struct.unpack_from('<H', data, pos + 2)[0]
                pos += 4 + size
            elif fmt in (0x02, 0x03):
                pos += 3
            else:
                pos += 1
        return False
    
    def grow_kernel_text(self, kernel: KernelInfo,
                         extra_nop_count: int) -> list[int]:
        """Extend a kernel's .text section with extra NOP instructions.

        Inserts *extra_nop_count* NOP instructions (16 bytes each) at the
        end of the kernel's text section.  Updates ELF section headers,
        program headers, and internal bookkeeping so the cubin remains
        structurally valid.

        Returns: list of code offsets (relative to kernel text) for the
                 new NOP slots.
        """
        NOP_INSTR = 0x0000000000007918
        NOP_CTRL  = 0x000fc00000000000
        extra_bytes = extra_nop_count * 16
        insert_pos = kernel.text_offset + kernel.text_size

        # Build NOP block
        nop_block = bytearray()
        for _ in range(extra_nop_count):
            nop_block += struct.pack('<QQ', NOP_INSTR, NOP_CTRL)

        # Insert into the data buffer
        self.data[insert_pos:insert_pos] = nop_block

        # --- Update ELF headers ---
        # Section header table offset
        e_shoff = struct.unpack_from('<Q', self.data, 0x28)[0]
        if e_shoff >= insert_pos:
            struct.pack_into('<Q', self.data, 0x28, e_shoff + extra_bytes)

        # Program header table offset
        e_phoff = struct.unpack_from('<Q', self.data, 0x20)[0]
        if e_phoff >= insert_pos:
            struct.pack_into('<Q', self.data, 0x20, e_phoff + extra_bytes)

        # Re-read (possibly shifted) e_shoff
        e_shoff = struct.unpack_from('<Q', self.data, 0x28)[0]
        e_shnum = struct.unpack_from('<H', self.data, 0x3C)[0]
        e_shentsize = struct.unpack_from('<H', self.data, 0x3A)[0]

        for i in range(e_shnum):
            sh_base = e_shoff + i * e_shentsize
            sh_off = struct.unpack_from('<Q', self.data, sh_base + 24)[0]
            sh_size = struct.unpack_from('<Q', self.data, sh_base + 32)[0]

            if sh_off == kernel.text_offset and sh_size == kernel.text_size:
                # This is our text section — grow it
                struct.pack_into('<Q', self.data, sh_base + 32,
                                 kernel.text_size + extra_bytes)
            elif sh_off >= insert_pos:
                # Section starts at or after the insert point — shift
                struct.pack_into('<Q', self.data, sh_base + 24,
                                 sh_off + extra_bytes)

        # Program headers (re-read after potential shift)
        e_phoff = struct.unpack_from('<Q', self.data, 0x20)[0]
        e_phnum = struct.unpack_from('<H', self.data, 0x38)[0]
        e_phentsize = struct.unpack_from('<H', self.data, 0x36)[0]

        for i in range(e_phnum):
            ph_base = e_phoff + i * e_phentsize
            p_offset = struct.unpack_from('<Q', self.data, ph_base + 8)[0]
            p_filesz = struct.unpack_from('<Q', self.data, ph_base + 32)[0]
            p_memsz = struct.unpack_from('<Q', self.data, ph_base + 40)[0]

            if p_offset <= kernel.text_offset < p_offset + p_filesz:
                # Segment contains our text section — grow
                struct.pack_into('<Q', self.data, ph_base + 32,
                                 p_filesz + extra_bytes)
                struct.pack_into('<Q', self.data, ph_base + 40,
                                 p_memsz + extra_bytes)
            elif p_offset >= insert_pos:
                # Segment starts after insert — shift
                struct.pack_into('<Q', self.data, ph_base + 8,
                                 p_offset + extra_bytes)

        # --- Update internal bookkeeping ---
        old_text_size = kernel.text_size
        kernel.text_size += extra_bytes

        # Shift other kernels' offsets if needed
        for name, k in self.kernels.items():
            if k is kernel:
                continue
            if k.text_offset >= insert_pos:
                k.text_offset += extra_bytes
            if k.info_offset is not None and k.info_offset >= insert_pos:
                k.info_offset += extra_bytes

        # Return code offsets for the new NOP slots
        return [old_text_size + j * 16 for j in range(extra_nop_count)]

    def save(self, path: str | Path) -> None:
        """Write the (possibly modified) cubin to a file."""
        Path(path).write_bytes(bytes(self.data))
    
    def __repr__(self) -> str:
        kernels_str = ', '.join(
            f"{k.name}(R{k.reg_count},S{k.max_stack})" 
            for k in self.kernels.values()
        )
        return f"Cubin({self.path.name}, kernels=[{kernels_str}])"
