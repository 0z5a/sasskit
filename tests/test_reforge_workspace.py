"""File ownership tests; no CUDA runtime is required."""
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest

from sasskit.schedule import reforge as rf


class TokenCubin:
    def __init__(self, path):
        self.path = Path(path)
        self.token = self.path.read_bytes()

    @classmethod
    def from_file(cls, path):
        return cls(path)

    def get_kernel(self, name):
        return name

    def save(self, path):
        Path(path).write_bytes(self.token)


def run_token(source, barrier=None, **kwargs):
    token = Path(source).read_bytes()
    saved = []

    def save(obj, path):
        saved.append(Path(path))
        Path(path).write_bytes(obj.token)

    def bench(path, *args):
        if barrier is not None:
            barrier.wait()
        assert Path(path).read_bytes() == token, 'another run overwrote this cubin'
        return 1.0

    with patch.object(rf, 'Cubin', TokenCubin), \
         patch.object(TokenCubin, 'save', save), \
         patch.object(rf, 'decode_kernel', return_value=[]), \
         patch.object(rf, 'gpu_bench', bench):
        state = rf.reforge(str(source), 'test', max_iters=0, verbose=False, **kwargs)
    return state, saved


def worker(source, cwd, barrier):
    import os
    os.chdir(cwd)
    state, saved = run_token(source, barrier)
    assert Path(state.best_cubin_path).read_bytes() == Path(source).read_bytes()
    assert not saved[0].parent.exists()


@pytest.mark.parametrize('count', [2, 8])
def test_concurrent_calls_share_cwd_and_temp_root(tmp_path, count):
    ctx = multiprocessing.get_context('spawn')
    barrier = ctx.Barrier(count)
    processes = []
    for i in range(count):
        source = tmp_path / f'input-{i}.cubin'
        source.write_bytes(str(i).encode())
        process = ctx.Process(target=worker, args=(source, tmp_path, barrier))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    assert [p.exitcode for p in processes] == [0] * count
    assert len(list((tmp_path / 'reforge-results').glob('*/best.cubin'))) == count


def test_repeated_calls_and_spaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'input with spaces.cubin'
    source.write_bytes(b'original')
    root = tmp_path / 'temporary root'
    root.mkdir()
    first, first_saved = run_token(source, work_dir=root)
    second, second_saved = run_token(source, work_dir=root)
    assert first_saved[0].parent != second_saved[0].parent
    assert all(p.parent == first_saved[0].parent for p in first_saved)
    assert list(root.iterdir()) == []
    for state in (first, second):
        assert Path(state.best_cubin_path).read_bytes() == b'original'
        assert state.cubin_path == str(source)
    assert first.best_cubin_path != second.best_cubin_path
    assert source.read_bytes() == b'original'


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
def test_input_alias_rejected(tmp_path, alias):
    source = tmp_path / 'input'
    source.write_bytes(b'original')
    output = tmp_path / 'output'
    if alias == 'same':
        output = source
    elif alias == 'symlink':
        output.symlink_to(source)
    else:
        output.hardlink_to(source)
    with pytest.raises(ValueError, match='alias'):
        run_token(source, output_path=output)
    assert source.read_bytes() == b'original'


@pytest.mark.parametrize('error', [RuntimeError('benchmark failed'), KeyboardInterrupt()])
def test_failure_cleans_only_owned_workspace(tmp_path, error):
    source = tmp_path / 'input'
    source.write_bytes(b'original')
    other = tmp_path / 'reforge-other'
    other.mkdir()
    (other / 'current.cubin').write_bytes(b'other')
    with patch.object(rf, 'Cubin', TokenCubin), \
         patch.object(rf, 'decode_kernel', return_value=[]), \
         patch.object(rf, 'gpu_bench', side_effect=error):
        with pytest.raises(type(error)):
            rf.reforge(str(source), 'test', work_dir=tmp_path, verbose=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['input', 'reforge-other']
    assert (other / 'current.cubin').read_bytes() == b'other'


def test_publish_failure_and_collision(tmp_path):
    source = tmp_path / 'input'
    source.write_bytes(b'original')
    output = tmp_path / 'output'
    with patch.object(rf.os, 'link', side_effect=OSError('publication failed')):
        with pytest.raises(OSError, match='publication failed'):
            run_token(source, output_path=output, work_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['input']
    output.write_bytes(b'preexisting')
    with pytest.raises(FileExistsError):
        run_token(source, output_path=output)
    assert output.read_bytes() == b'preexisting'
    assert source.read_bytes() == b'original'


def publish_worker(source, output, barrier, results):
    barrier.wait()
    try:
        rf._publish_best(Path(source), Path(output))
        results.put('published')
    except FileExistsError:
        results.put('conflict')


def test_atomic_publish_collision(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    barrier, results = ctx.Barrier(2), ctx.Queue()
    output = tmp_path / 'shared-output'
    processes = []
    for i in range(2):
        source = tmp_path / f'input-{i}'
        source.write_bytes(str(i).encode() * 65536)
        process = ctx.Process(target=publish_worker, args=(source, output, barrier, results))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    assert [p.exitcode for p in processes] == [0, 0]
    assert sorted(results.get() for _ in processes) == ['conflict', 'published']
    assert output.read_bytes() in (b'0' * 65536, b'1' * 65536)
    assert not list(tmp_path.glob('.reforge-*'))


def test_cli_defaults_and_options(tmp_path, monkeypatch, capsys):
    from sasskit.recolor import cli
    source = tmp_path / 'input'
    source.write_bytes(b'original')
    with patch.object(cli.Cubin, 'from_file') as load, patch.object(rf, 'reforge') as run:
        load.return_value.kernels = {'test': object()}
        monkeypatch.setattr('sys.argv', ['sasskit', 'reforge', str(source)])
        assert cli.main() == 0
        assert run.call_args.kwargs['output_path'] is None
        assert run.call_args.kwargs['work_dir'] is None
    monkeypatch.setattr('sys.argv', ['sasskit', 'reforge', '--help'])
    with pytest.raises(SystemExit, match='0'):
        cli.main()
    assert 'reforge-results' in ''.join(capsys.readouterr().out.split())
