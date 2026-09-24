"""Swap policy tests never execute rejected instruction sequences on a GPU."""
from dataclasses import replace

import pytest

from sasskit.analysis import build_cfg
from sasskit.core.cubin import Cubin, KernelInfo
from sasskit.core.decoder import Instruction, RegRef
from sasskit.core.isa import build_ctrl
from sasskit.schedule import reforge as rf


def move(offset=0, immediate=1):
    return Instruction(offset, (immediate << 32) | (255 << 16) | 0x7802,
                       build_ctrl(stall=4) | 0xF00, f'MOV RZ, 0x{immediate:x}', 'MOV')


def test_audited_discard_moves():
    assert rf._can_swap(move(), move(16, 2))
    assert move().dest_regs == []


@pytest.mark.parametrize('text', [
    'STG.E [R2], R3', 'LDG.E R4, [R5]', 'STS [R2], R3', 'LDS R4, [R5]',
    'STL [R2], R3', 'LDL R4, [R5]', 'LDC R4, c[0][0]', 'TEX R4, R5',
    'ATOM.E.ADD R4, [R5], R6', 'RED.E.ADD [R5], R6', 'MEMBAR.SC.GPU',
    'BAR.SYNC 0', 'WARPSYNC 0xffffffff', 'LDGSTS [R2], [R4]',
    'DEPBAR.LE SB0, 0', 'UTMALDG.1D [UR2], [UR4]', 'BRA 0x30', 'CALL 0x30',
    'EXIT', 'RET', 'BSSY B0, 0x30', 'BSYNC B0', 'MUFU.RCP R2, R3',
    'HMMA.16816.F32 R2, R4, R6, R8', 'ISETP.EQ.AND P0, PT, R2, R3, PT',
    '@P0 MOV RZ, 0x1', '@!P0 MOV RZ, 0x1', '@UP0 MOV RZ, 0x1',
    'MOV RZ, UR0', 'MOV RZ, R0', 'MOV R1, 0x1', 'MOV.64 RZ, 0x1',
    'IADD3.X R2, P0, P1, R3, R4, R5, P0, P1', 'UNKNOWN R2',
    'MOV.UNKNOWN RZ, 0x1', 'MOV RZ, R0.reuse', '',
])
def test_unmodeled_forms_rejected_both_directions(text):
    inst = replace(move(), asm_text=text, opcode=text.split()[0] if text else '')
    assert not rf._can_swap(inst, move(16, 2))
    assert not rf._can_swap(move(16, 2), inst)


@pytest.mark.parametrize('kwargs', [
    {'wait_mask': 1}, {'read_bar': 0}, {'write_bar': 1}, {'stall': 3},
])
def test_scheduling_constraints(kwargs):
    control = {'stall': 4, **kwargs}
    inst = replace(move(), ctrl_word=build_ctrl(**control) | 0xF00)
    assert not rf._can_swap(inst, move(16, 2))


@pytest.mark.parametrize('bit', [0, 40, 58, 59, 60, 61, 62, 63])
def test_unknown_control_and_reuse_bits(bit):
    assert not rf._can_swap(replace(move(), ctrl_word=move().ctrl_word ^ (1 << bit)), move(16, 2))


@pytest.mark.parametrize('refs', [
    [RegRef(1, True, False, 16, 'Rd')],
    [RegRef(1, False, True, 24, 'Ra')],
    [RegRef(1, False, False, 24, 'Ra', is_quad=True)],
    [RegRef(0, False, False, -1, 'unknown')],
])
def test_live_or_unresolved_registers_rejected(refs):
    assert not rf._can_swap(replace(move(), reg_refs=refs), move(16, 2))


def test_decoded_text_must_match_encoding():
    assert not rf._can_swap(replace(move(), instr_word=move().instr_word ^ (1 << 32)), move(16, 2))
    assert not rf._can_swap(replace(move(), is_predicated=True), move(16, 2))
    assert not rf._can_swap(replace(move(), predicate='@PT'), move(16, 2))


def test_memory_alias_oracle():
    memory = {'address': 1}
    memory['address'] = 7
    original = memory['address']
    memory['address'] = 1
    reordered = memory['address']
    memory['address'] = 7
    assert original != reordered
    store = Instruction(0, 0, 0, 'STG [R2], R3', 'STG', [RegRef(3, False, False, 32, 'Rb')])
    load = Instruction(16, 0, 0, 'LDG R4, [R5]', 'LDG', [RegRef(4, True, False, 16, 'Rd')])
    assert not rf._can_swap(store, load)


def program(tmp_path):
    instructions = [move(i * 16, i + 1) for i in range(4)]
    blocks = build_cfg(instructions)
    kernel = KernelInfo('test', 0, 64, 1)
    binary = Cubin(tmp_path / 'test', bytearray(64), {'test': kernel})
    for inst in instructions:
        binary.write_instruction(kernel, inst.code_offset, inst.instr_word, inst.ctrl_word)
    return binary, kernel, instructions, blocks


def test_generators_and_application(tmp_path, monkeypatch):
    binary, kernel, instructions, blocks = program(tmp_path)
    monkeypatch.setattr(rf.random, 'randrange', lambda *args: 0 if len(args) == 1 else 1)
    proposal = rf.gen_swap_adjacent(instructions, blocks)
    assert proposal is not None
    assert rf.apply_mutation(binary, kernel, instructions, blocks, proposal)
    assert not rf.apply_mutation(binary, kernel, instructions, blocks, proposal)  # stale
    assert rf.gen_swap_load_up(instructions, blocks) is None


def test_unsafe_manual_proposal_and_load_hoist(tmp_path, monkeypatch):
    binary, kernel, instructions, blocks = program(tmp_path)
    instructions[2].opcode = 'LDG.E'
    instructions[2].asm_text = 'LDG.E R4, [R5]'
    monkeypatch.setattr(rf.random, 'randrange', lambda *args: 0 if len(args) == 1 else 1)
    assert rf.gen_swap_adjacent(instructions, blocks) is None
    assert rf.gen_swap_load_up(instructions, blocks) is None
    original = bytes(binary.data)
    for kind, index in [(rf.MutationType.SWAP_ADJACENT, 1), (rf.MutationType.SWAP_LOAD_UP, 2)]:
        assert not rf.apply_mutation(binary, kernel, instructions, blocks, rf.Mutation(kind, 0, index))
    assert bytes(binary.data) == original


def test_block_boundary_is_exclusive(tmp_path):
    binary, kernel, instructions, blocks = program(tmp_path)
    instructions[1] = Instruction(16, 0, 0, 'EXIT', 'EXIT')
    blocks = build_cfg(instructions)
    assert rf._get_block_instructions(instructions, blocks[0]) == instructions[:2]
    assert rf._get_block_instructions(instructions, blocks[0]) is blocks[0].instructions
    assert not rf.apply_mutation(binary, kernel, instructions, blocks,
                                 rf.Mutation(rf.MutationType.SWAP_ADJACENT, 0, 1))


@pytest.mark.parametrize('block,index', [(-1, 0), (1, 0), (0, -1), (0, 4), (0, 3)])
def test_invalid_positions(tmp_path, block, index):
    binary, kernel, instructions, blocks = program(tmp_path)
    assert not rf.apply_mutation(binary, kernel, instructions, blocks,
                                 rf.Mutation(rf.MutationType.SWAP_ADJACENT, block, index))
