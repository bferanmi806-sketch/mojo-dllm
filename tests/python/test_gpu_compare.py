"""A faster result must not hide a failed or shorter-work benchmark run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def harness(tmp_path, monkeypatch, mode):
    # main() sets its device environment; restore the caller after each test.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'test-device')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MOJO_DLLM_GPU_PROFILE', '1')
    source = Path(__file__).resolve().parents[2] / 'bench/compare_gpu.py'
    spec = importlib.util.spec_from_file_location('gpu_compare', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / 'new'
    (root / 'bench').mkdir(parents=True)
    (root / 'bench/config-gpu.json').write_text(json.dumps({'prompts': ['hello']}))
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'fixture')
    output = tmp_path / 'out.json'
    monkeypatch.setattr('sys.argv', ['compare_gpu', '--base', str(tmp_path / 'base'),
                        '--new', str(root), '--model', str(model), '--gpu', 'assigned',
                        '--out', str(output), '--reps', '1', '--prompts', '1'])
    monkeypatch.setattr(mod, 'command', lambda cmd: '' if 'status' in cmd else 'fixture')
    bench = SimpleNamespace(ROOT=None)
    calls = []

    def run(c, p, threads, extra):
        variant = bench.ROOT.name
        calls.append(variant)
        assert extra == ['--device', 'gpu']
        if mode == 'fail' and variant == 'new':
            return dict(ok=False, rc=1, error='fixture failure')
        ms = 20 if variant == 'base' else 10
        return dict(ok=True, generate_s=ms * 32 / 1000, ms_per_step=ms,
                    forward_passes=16 if mode == 'both_short' or mode == 'short' and variant == 'new' else 32,
                    seq_len=150, text='same')

    bench.run_mojo = run
    monkeypatch.setattr(mod, 'load_bench', lambda root: bench)
    return mod, output, calls


def test_gpu_compare_interleaves_and_excludes_warmup(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    mod.main()
    result = json.loads(output.read_text())
    assert calls == ['new', 'base', 'base', 'new']
    assert not result['partial']
    assert result['summary'][0]['pairs'] == 1
    assert result['summary'][0]['median_paired_speedup'] == 2


@pytest.mark.parametrize('mode', ['fail', 'short', 'both_short'])
def test_gpu_compare_rejects_invalid_speedups(tmp_path, monkeypatch, mode):
    mod, output, _ = harness(tmp_path, monkeypatch, mode)
    with pytest.raises(SystemExit, match='failed|mismatch'):
        mod.main()
    assert json.loads(output.read_text())['partial']
