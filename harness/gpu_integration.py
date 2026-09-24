"""Numerical, snapshot, and multi-worker integration on an explicit PTX ABI."""
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from sasskit.analysis import build_cfg
from sasskit.core.cubin import Cubin
from sasskit.core import decoder
from sasskit.core.isa import build_ctrl
from sasskit.schedule import reforge as rf

ROOT = Path(__file__).resolve().parent.parent
RUN = subprocess.run


def no_timeout(*args, **kwargs):
    kwargs.pop('timeout', None)
    return RUN(*args, **kwargs)


def validate(path, seed):
    result = RUN([str(ROOT / 'harness/validate'), str(path), str(seed)],
                 capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def worker(uuid, count, index):
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    source = ROOT / 'audited.cubin'
    token = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []

    def bench(path, *args):
        actual = validate(path, index + 1000)
        assert actual['uuid_hex'] == uuid.removeprefix('GPU-').replace('-', '')
        calls.append(actual)
        return actual['event_us'] / 1000

    with patch.object(rf, 'gpu_bench', bench), patch.object(subprocess, 'run', no_timeout):
        state = rf.reforge(str(source), 'affine', max_iters=0, verbose=False,
                           output_path=ROOT / f'evidence/gpu-{count}-{index}.cubin',
                           work_dir=ROOT / 'tmp')
    assert hashlib.sha256(Path(state.best_cubin_path).read_bytes()).hexdigest() == token
    (ROOT / f'evidence/gpu-{count}-{index}.json').write_text(json.dumps(calls))


def prepare():
    program = Cubin.from_file(ROOT / 'fixture.cubin')
    kernel = program.get_kernel('affine')
    before = bytes(program.data)
    # Two compiler-preserved nanosleep(0) slots reserve a reachable location
    # for the audited discard moves without changing instruction offsets.
    with patch.object(subprocess, 'run', no_timeout):
        instructions = decoder.decode_kernel(program, 'affine')
    pair = next(i for i in range(len(instructions)-1)
                if instructions[i].opcode == 'NANOSLEEP' and instructions[i+1].opcode == 'NANOSLEEP')
    offsets = [instructions[pair].code_offset, instructions[pair+1].code_offset]
    assert offsets[1] < next(i.code_offset for i in instructions if i.opcode == 'EXIT')
    for offset, immediate in zip(offsets, [1, 2]):
        program.write_instruction(kernel, offset, (immediate << 32) | (255 << 16) | 0x7802,
                                  build_ctrl(stall=4) | 0xF00)
    program.save(ROOT / 'audited.cubin')
    changes = [i for i,(a,b) in enumerate(zip(before, program.data)) if a != b]
    assert all(any(kernel.text_offset+off <= i < kernel.text_offset+off+16 for off in offsets) for i in changes)
    records = {}
    for backend in ['cuobjdump', 'cubit']:
        if backend == 'cubit':
            os.environ['CUBIT_BIN'] = str(ROOT / 'target-cli/release/cubit')
        else:
            os.environ.pop('CUBIT_BIN', None)
        fresh = Cubin.from_file(ROOT / 'audited.cubin')
        start = time.perf_counter()
        with patch.object(subprocess, 'run', no_timeout):
            inst = decoder.decode_kernel(fresh, 'affine')
        blocks = build_cfg(inst)
        decode_ms = (time.perf_counter() - start) * 1000
        a,b = [i for i in inst if i.code_offset in offsets]
        assert rf._can_swap(a,b), (backend,a,b)
        block_index = next(i for i,b in enumerate(blocks) if a in b.instructions)
        proposal = rf.Mutation(rf.MutationType.SWAP_ADJACENT, block_index, blocks[block_index].instructions.index(a))
        assert rf.apply_mutation(fresh, fresh.get_kernel('affine'), inst, blocks, proposal)
        output = ROOT / f'swap-{backend}.cubin'
        fresh.save(output)
        reloaded = Cubin.from_file(output)
        with patch.object(subprocess, 'run', no_timeout):
            decoded = decoder.decode_kernel(reloaded, 'affine')
        assert [i.instr_word for i in decoded if i.code_offset in offsets] == [b.instr_word,a.instr_word]
        assert reloaded.data[:kernel.text_offset] == program.data[:kernel.text_offset]
        assert reloaded.data[kernel.text_offset+kernel.text_size:] == program.data[kernel.text_offset+kernel.text_size:]
        records[backend] = {'decode_cfg_ms': decode_ms, 'offsets': offsets,
                            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                            'numeric': [validate(output, seed) for seed in (0, 1, 20260924)]}
        decisions = iter([10.0, 9.0, 11.0, 8.0])
        proposals, numeric = [], []
        def generate(current, cfg):
            chosen = [i for i in current if i.code_offset in offsets]
            proposals.append([i.instr_word >> 32 for i in chosen])
            index = next(i for i,b in enumerate(cfg) if chosen[0] in b.instructions)
            return rf.Mutation(rf.MutationType.SWAP_ADJACENT, index,
                               cfg[index].instructions.index(chosen[0]))
        def bench(path, *args):
            numeric.append(validate(path, 92024))
            return next(decisions)  # Scripted acceptance decisions, not GPU timings.
        with patch.object(subprocess, 'run', no_timeout), \
             patch.object(rf, 'MUTATION_GENERATORS', [(generate, 1.0)]), \
             patch.object(rf, 'gpu_test', lambda path,*args: validate(path, 72024)['numerically_correct']), \
             patch.object(rf, 'gpu_bench', bench):
            state = rf.reforge(str(ROOT / 'audited.cubin'), 'affine', max_iters=3,
                               temperature=0, verbose=False, work_dir=ROOT / 'tmp',
                               output_path=ROOT / f'evidence/snapshot-final-{backend}.cubin')
        assert proposals == [[1,2], [2,1], [2,1]]
        assert state.accepted == 2 and state.rejected == 1 and state.generation == 2
        assert [i.instr_word >> 32 for i in state.instructions if i.code_offset in offsets] == [1,2]
        records[backend]['transaction'] = {'proposals': proposals, 'accepted': state.accepted,
                                          'rejected': state.rejected, 'oracle_runs': numeric,
                                          'decision_times_are_scripted': True}
    (ROOT / 'evidence/decoder-gpu.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records), flush=True)


def main():
    if sys.argv[1] == 'prepare':
        prepare()
        return
    count = int(sys.argv[1])
    uuids = RUN(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                capture_output=True, text=True, check=True).stdout.splitlines()
    ctx = multiprocessing.get_context('spawn')
    processes = [ctx.Process(target=worker, args=(uuids[i], count, i)) for i in range(count)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
    print(f'{count} GPU workers passed', flush=True)


if __name__ == '__main__':
    main()
