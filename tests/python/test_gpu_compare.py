"""A faster result must not hide a failed or shorter-work benchmark run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeArm:
    """What isolation.begin_arm returns, without a preflight of this machine: each run is called and reported clean."""

    def __init__(self):
        self.began = []
        self.windows = 0
        self.window_kwargs = []
        self.finished = False

    def begin(self, name, **kw):
        self.began.append((name, kw))
        return self

    def run(self, fn, **kw):
        self.windows += 1
        self.window_kwargs.append(kw)
        return fn(), {'isolation': {'clean': True, 'reasons': []}}

    def finish(self):
        self.finished = True


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
    for checkout in ('base', 'new'):
        (tmp_path / checkout / 'build').mkdir(parents=True, exist_ok=True)
        (tmp_path / checkout / 'build/mojo-dllm').write_bytes(f'binary of {checkout}'.encode())
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'fixture')
    output = tmp_path / 'out.json'
    monkeypatch.setattr('sys.argv', ['compare_gpu', '--base', str(tmp_path / 'base'),
                        '--new', str(root), '--model', str(model), '--gpu', '1',
                        '--out', str(output), '--reps', '1', '--prompts', '1'])
    def command(cmd):
        if 'status' in cmd:
            return ''
        if 'rev-parse' in cmd:
            return 'head-of-' + Path(cmd[2]).name
        if cmd[0] == 'nvidia-smi':
            assert cmd[1:3] == ['-i', '1']
            return 'Fixture GPU, GPU-1111-aaaa, 00000000:02:00.0, 6144 MiB, 595.1, 150.00 W'
        return 'fixture'

    monkeypatch.setattr(mod, 'command', command)
    arm = FakeArm()
    monkeypatch.setattr(mod.isolation, 'begin_arm', lambda *a, **k: arm.begin(*a, **k))
    mod.fake_arm = arm
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


def argv(tmp_path, monkeypatch, **replace):
    """The harness command line with some flags replaced, e.g. new=path."""
    args = ['compare_gpu', '--base', str(tmp_path / 'base'), '--new', str(tmp_path / 'new'),
            '--model', str(tmp_path / 'model.gguf'), '--gpu', '1', '--out', str(tmp_path / 'out.json'),
            '--reps', '1', '--prompts', '1']
    for flag, value in replace.items():
        args[args.index('--' + flag) + 1] = value
    monkeypatch.setattr('sys.argv', args)


def refused(mod, output, calls, match):
    with pytest.raises(SystemExit, match=match):
        mod.main()
    assert calls == [] and not output.exists() and mod.fake_arm.began == [], 'something ran before the refusal'


def test_the_same_checkout_twice_is_refused(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    argv(tmp_path, monkeypatch, base=str(tmp_path / 'new'))
    refused(mod, output, calls, 'same checkout')


def test_a_symlink_to_the_other_checkout_is_the_same_checkout(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    (tmp_path / 'alias').symlink_to(tmp_path / 'new')
    argv(tmp_path, monkeypatch, base=str(tmp_path / 'alias'))
    refused(mod, output, calls, 'same checkout')


def test_two_checkouts_at_one_clean_commit_are_refused(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    command = mod.command
    monkeypatch.setattr(mod, 'command', lambda cmd: 'one-commit' if 'rev-parse' in cmd else command(cmd))
    refused(mod, output, calls, 'same commit')


def test_byte_identical_binaries_are_refused(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    (tmp_path / 'base/build/mojo-dllm').write_bytes(b'binary of new')
    refused(mod, output, calls, 'byte-identical')


def test_one_commit_with_an_edited_src_is_compared_by_the_binary_hash(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    command = mod.command

    def dirty_new(cmd):
        if 'rev-parse' in cmd:
            return 'one-commit'
        if 'status' in cmd:
            return ' M src/main.mojo' if Path(cmd[2]).name == 'new' else ''
        return command(cmd)

    monkeypatch.setattr(mod, 'command', dirty_new)
    mod.main()
    assert json.loads(output.read_text())['versions'] == {'base': 'one-commit', 'new': 'one-commit-dirty'}


def test_two_edited_checkouts_at_one_commit_are_compared_by_the_binary_hash(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    command = mod.command
    monkeypatch.setattr(mod, 'command', lambda cmd: ' M src/a.mojo' if 'status' in cmd else
                        'one-commit' if 'rev-parse' in cmd else command(cmd))
    mod.main()
    assert json.loads(output.read_text())['versions'] == {'base': 'one-commit-dirty', 'new': 'one-commit-dirty'}


def test_a_missing_binary_is_refused(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    (tmp_path / 'base/build/mojo-dllm').unlink()
    refused(mod, output, calls, 'does not exist')


@pytest.mark.parametrize('prompts', [[], [''], 'hello', None])
def test_an_empty_prompt_list_is_refused(tmp_path, monkeypatch, prompts):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    (tmp_path / 'new/bench/config-gpu.json').write_text(json.dumps({'prompts': prompts}))
    refused(mod, output, calls, 'no usable prompts')


def test_the_evidence_records_each_binary_and_the_card(tmp_path, monkeypatch):
    import hashlib
    import os
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    seen = {}
    run = mod.load_bench(None).run_mojo

    def spy(c, p, threads, extra):
        seen['order'], seen['visible'] = os.environ.get('CUDA_DEVICE_ORDER'), os.environ.get('CUDA_VISIBLE_DEVICES')
        return run(c, p, threads, extra)

    mod.load_bench(None).run_mojo = spy
    mod.main()
    doc = json.loads(output.read_text())
    assert doc['binary_sha256'] == {k: hashlib.sha256(f'binary of {k}'.encode()).hexdigest() for k in ('base', 'new')}
    assert seen == {'order': 'PCI_BUS_ID', 'visible': '1'}
    assert doc['gpu_index'] == '1' and doc['gpu_uuid'] == 'GPU-1111-aaaa' and doc['gpu_pci_bus_id'] == '00000000:02:00.0'
    assert doc['threads'] == 8 and doc['steps'] == 32 and doc['block_length'] == 32
    assert {r['rep'] for r in doc['runs']} == {-1, 0}


@pytest.mark.parametrize('selector', ['assigned', '', '0,1', '-1', 'GPU-'])
def test_a_gpu_that_is_neither_an_index_nor_a_uuid_is_refused(tmp_path, monkeypatch, selector):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    argv(tmp_path, monkeypatch, gpu=selector)
    with pytest.raises(SystemExit, match='--gpu'):
        mod.main()
    assert calls == []


@pytest.mark.parametrize('answer', ['1', '', 'Fixture GPU, 1, 00000000:02:00.0, 6144 MiB'])
def test_the_card_nvidia_smi_names_must_have_a_uuid(tmp_path, monkeypatch, answer):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    command = mod.command
    monkeypatch.setattr(mod, 'command', lambda cmd: answer if cmd[0] == 'nvidia-smi' else command(cmd))
    with pytest.raises(SystemExit, match='did not name a card'):
        mod.main()
    assert calls == []


def test_every_run_is_inside_an_isolation_window_after_one_preflight(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    mod.main()
    arm = mod.fake_arm
    assert len(arm.began) == 1 and arm.began[0][1]['device'] == 'gpu' and arm.began[0][1]['allow_noisy'] is False
    assert arm.windows == len(calls) == 4 and arm.finished
    assert all(kw == {'max_quiet_wait_s': 300, 'quiet_sample_s': 2.0} for kw in arm.window_kwargs)
    assert all(r['isolation']['clean'] is True for r in json.loads(output.read_text())['runs'])


def test_a_failed_preflight_stops_before_any_run(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')

    def failed(*a, **k):
        raise mod.isolation.IsolationError('the machine is not fit for a timed run')

    monkeypatch.setattr(mod.isolation, 'begin_arm', failed)
    with pytest.raises(SystemExit, match='not fit'):
        mod.main()
    assert calls == []


def test_allow_noisy_reaches_the_preflight(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'ok')
    monkeypatch.setattr('sys.argv', __import__('sys').argv + ['--allow-noisy'])
    mod.main()
    assert mod.fake_arm.began[0][1]['allow_noisy'] is True


def test_the_window_is_closed_when_a_run_fails(tmp_path, monkeypatch):
    mod, output, calls = harness(tmp_path, monkeypatch, 'fail')
    with pytest.raises(SystemExit):
        mod.main()
    assert mod.fake_arm.finished
