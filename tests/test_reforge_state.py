"""Deterministic search transactions with a tiny test-only binary format."""
import struct
from pathlib import Path

import pytest

from sasskit.core.cubin import Cubin, KernelInfo
from sasskit.core.decoder import Instruction, RegRef
from sasskit.schedule import reforge as rf


class ToyCubin(Cubin):
    @classmethod
    def from_file(cls, path):
        return cls(Path(path), bytearray(Path(path).read_bytes()), {
            'test': KernelInfo('test', 16, 64, 1),
            'other': KernelInfo('other', 80, 16, 2),
        })


def decode_toy(program, name):
    # Catches decoding mutated memory through the original file path.
    assert bytes(program.data) == program.path.read_bytes()
    result = []
    for offset in range(0, 64, 16):
        lo, hi = program.read_instruction(program.get_kernel(name), offset)
        dest, src = {1: (1, 2), 2: (3, 4), 3: (5, 1), 4: (0, 0)}[lo]
        refs = [RegRef(dest, True, False, 16, 'Rd'), RegRef(src, False, False, 24, 'Ra')]
        op = 'EXIT' if lo == 4 else 'IADD3'
        result.append(Instruction(offset, lo, hi, f'{op} R{dest}, R{src}', op, refs))
    return result


@pytest.fixture
def search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'source'
    source.write_bytes(b'header unchanged' + b''.join(struct.pack('<QQ', i, 0) for i in (1, 2, 3, 4, 99)))
    baseline = source.read_bytes()
    monkeypatch.setattr(rf, 'Cubin', ToyCubin)
    monkeypatch.setattr(rf, 'decode_kernel', decode_toy)
    monkeypatch.setattr(rf, 'gpu_test', lambda *args: True)
    monkeypatch.setattr(rf.random, 'random', lambda: 0.0)
    # The toy IR exercises snapshot freshness, not hardware scheduling legality.
    monkeypatch.setattr(rf, '_can_swap', lambda a, b: not (
        rf._has_raw_dep(a, b) or rf._has_war_dep(a, b) or rf._has_waw_dep(a, b)))
    seen, binaries, states = [], [], []
    real_state = rf.ReforgeState

    def state_factory(**kwargs):
        state = real_state(**kwargs)
        states.append(state)
        return state
    monkeypatch.setattr(rf, 'ReforgeState', state_factory)

    def run(times, proposals, **kwargs):
        timing = iter(times)
        proposal = iter(proposals)

        def generate(instructions, blocks):
            assert all(inst is instructions[inst.code_offset // 16]
                       for block in blocks for inst in block.instructions)
            seen.append([i.instr_word for i in instructions])
            index = next(proposal)
            if index is None or not rf._can_swap(instructions[index], instructions[index + 1]):
                return None
            return rf.Mutation(rf.MutationType.SWAP_ADJACENT, 0, index)

        def bench(path, *args):
            binary = Path(path).read_bytes()
            assert binary[:16] == baseline[:16] and binary[80:] == baseline[80:]
            binaries.append(binary)
            return next(timing)

        monkeypatch.setattr(rf, 'MUTATION_GENERATORS', [(generate, 1.0)])
        monkeypatch.setattr(rf, 'gpu_bench', bench)
        state = rf.reforge(str(source), 'test', max_iters=len(proposals),
                           verbose=False, **kwargs)
        assert source.read_bytes() == baseline
        return state
    return run, seen, binaries, states, source


def words(binary):
    return [struct.unpack_from('<Q', binary, i)[0] for i in range(16, 80, 16)]


def test_second_proposal_uses_accepted_instructions(search):
    run, seen, binaries, states, source = search
    state = run([10, 9], [0, 1], temperature=0)
    assert seen == [[1, 2, 3, 4], [2, 1, 3, 4]]
    assert state.accepted == 1 and state.generation == 1
    assert words(Path(state.best_cubin_path).read_bytes()) == [2, 1, 3, 4]
    assert [i.instr_word for i in state.instructions] == [2, 1, 3, 4]
    assert state.instructions[1].dest_regs == [1]
    assert state.instructions[2].src_regs == [1]


def test_accepted_non_best_refreshes_current_only(search):
    run, seen, binaries, states, source = search
    state = run([10, 9, 10], [0, 0], temperature=1)
    assert state.accepted == 2 and state.generation == 2
    assert state.current_time_ms == 10 and state.best_time_ms == 9
    assert [i.instr_word for i in state.instructions] == [1, 2, 3, 4]
    assert words(Path(state.best_cubin_path).read_bytes()) == [2, 1, 3, 4]


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1, 0, 11])
def test_rejected_candidate_does_not_pollute_next_proposal(search, invalid):
    run, seen, binaries, states, source = search
    state = run([10, 9, invalid, 8], [0, 0, 0], temperature=0)
    assert seen == [[1, 2, 3, 4], [2, 1, 3, 4], [2, 1, 3, 4]]
    assert state.accepted == 2 and state.rejected == 1
    assert state.current_time_ms == state.best_time_ms == 8
    assert words(Path(state.best_cubin_path).read_bytes()) == [1, 2, 3, 4]


def test_validation_rejection_leaves_current(search, monkeypatch):
    run, seen, binaries, states, source = search
    monkeypatch.setattr(rf, 'gpu_test', lambda *args: False)
    state = run([10], [0])
    assert state.accepted == 0 and state.rejected == 1
    assert [i.instr_word for i in state.instructions] == [1, 2, 3, 4]
    assert Path(state.best_cubin_path).read_bytes() == source.read_bytes()


def test_apply_false_leaves_current(search, monkeypatch):
    run, seen, binaries, states, source = search
    monkeypatch.setattr(rf, 'apply_mutation', lambda *args: False)
    state = run([10], [0])
    assert state.generation == state.accepted == state.rejected == 0
    assert Path(state.best_cubin_path).read_bytes() == source.read_bytes()


@pytest.mark.parametrize('failure', ['decode', 'benchmark'])
def test_exception_keeps_known_good_state(search, monkeypatch, failure):
    run, seen, binaries, states, source = search
    if failure == 'decode':
        def decoder(program, name):
            if program.path.name == 'candidate.cubin':
                current = program.path.with_name('current.cubin')
                assert current.read_bytes() == source.read_bytes()
                raise RuntimeError('candidate decode failed')
            return decode_toy(program, name)
        monkeypatch.setattr(rf, 'decode_kernel', decoder)
    error = RuntimeError if failure == 'decode' else StopIteration
    with pytest.raises(error):
        run([10, 9] if failure == 'decode' else [10], [0])
    assert states[0].accepted == 0
    assert states[0].current_time_ms == 10
    assert [i.instr_word for i in states[0].instructions] == [1, 2, 3, 4]
    assert states[0].best_cubin_path is None


def test_control_word_only_refreshes_analysis(search, monkeypatch):
    run, seen, binaries, states, source = search
    def apply(program, kernel, instructions, blocks, mutation):
        lo, hi = program.read_instruction(kernel, 0)
        program.write_instruction(kernel, 0, lo, 6 << 41)
        return True
    monkeypatch.setattr(rf, 'apply_mutation', apply)
    state = run([10, 9], [0])
    assert state.instructions[0].ctrl_word == 6 << 41
    assert state.blocks[0].instructions[0] is state.instructions[0]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0, -1])
def test_invalid_baseline_stops_before_mutations(search, value):
    run, seen, binaries, states, source = search
    with pytest.raises(ValueError, match='Baseline'):
        run([value], [0])
    assert seen == []


def test_rejected_candidates_do_not_repeat_disassembly(search, monkeypatch):
    run, seen, binaries, states, source = search
    decoded = []
    def decoder(program, name):
        decoded.append(program.path)
        return decode_toy(program, name)
    monkeypatch.setattr(rf, 'decode_kernel', decoder)
    state = run([10, 11, 12], [0, 0], temperature=0)
    assert state.rejected == 2
    assert decoded == [source]
