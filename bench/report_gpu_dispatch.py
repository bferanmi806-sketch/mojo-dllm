#!/usr/bin/env python3
"""Regenerate bench/results/6000d-dispatch/README.md from the committed JSON.

Every figure and every sentence about the numbers is read or computed from the files: the cards, their memory and
driver from the `machine.gpu` line of each timing file, the prompt and round counts, the forward passes and the block
length from the runs, which cases are unchanged from the paired ratios, and the check counts from the exactness files.
What the measurement files do not hold (the SM count, the Mojo and max-core versions, the compile target and the
author's notes about the host and the model sources) is read from the `environment` object of a timing file or, for
the files written before compare_gpu recorded it, from `environment.json` beside them. A value that is in neither is
an error: nothing here is typed in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHS = ('llada', 'dream')
LABELS = {'llada': 'LLaDA', 'dream': 'Dream'}
UNCHANGED_WITHIN = 0.005        # a paired speedup this close to 1 is reported as unchanged
WIDE_RANGE = 0.05               # a pair range wider than this is explained from the prompt lengths
METHOD_PHRASES = ('no early stop', 'no threshold or dual cache', 'no profiler synchronization',
                  'first pair per length is warmup')
ENVIRONMENT_KEYS = ('sm_count', 'mojo', 'max_core', 'compiled_for', 'host_note', 'source_note')
NUMBER_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')


def words(n: int, capital: bool = False) -> str:
    text = NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)
    return text.capitalize() if capital else text


def fail(message: str) -> None:
    raise SystemExit(f'report_gpu_dispatch: {message}')


def card(data: dict, arch: str) -> dict:
    """The card of a timing file, read from the nvidia-smi line it recorded: name, uuid, bus id, memory, driver."""
    fields = [f.strip() for f in str(data.get('machine', {}).get('gpu', '')).split(',')]
    if len(fields) < 5 or not fields[1].startswith('GPU-') or not fields[3].endswith(' MiB'):
        fail(f'{arch}.json holds no nvidia-smi line with a name, uuid, memory and driver: {fields}')
    return dict(name=fields[0].removeprefix('NVIDIA '), uuid=fields[1], memory_mib=int(fields[3].removesuffix(' MiB')),
                driver=fields[4])


def environment(datas: dict, dest: Path) -> dict:
    env = {}
    for data in datas.values():
        env.update(data.get('environment') or {})
    side = dest / 'environment.json'
    if side.is_file():
        for key, value in json.loads(side.read_text()).items():
            env.setdefault(key, value)
    missing = [k for k in ENVIRONMENT_KEYS if k not in env or env[k] in (None, '')]
    if missing:
        fail(f'the environment is not recorded: {missing} are in neither a timing file nor {side.name}')
    return env


def check_runs(arch: str, data: dict) -> None:
    if data['partial']:
        fail(f'{arch}.json is a partial run')
    if not data['runs'] or not data['summary']:
        fail(f'{arch}.json holds no runs or no summary')
    bad = [r for r in data['runs'] if not r['ok'] or r['forward_passes'] != data['steps']]
    if bad:
        fail(f'{arch}.json holds {len(bad)} runs that failed or did not do {data["steps"]} forward passes')
    for phrase in METHOD_PHRASES:
        if phrase not in data['method']:
            fail(f'{arch}.json method no longer says "{phrase}", so the prose that says it cannot be generated')


def same(datas: dict, key: str, label: str):
    values = {json.dumps(key_of(d), sort_keys=True) for d in datas.values() for key_of in [lambda x: x[key]]}
    if len(values) != 1:
        fail(f'the timing files disagree on {label}: {sorted(values)}')
    return next(iter(datas.values()))[key]


def unchanged_sentence(datas: dict) -> str:
    parts = []
    for arch, data in datas.items():
        flat = [row['gen_length'] for row in data['summary'] if abs(row['median_paired_speedup'] - 1) <= UNCHANGED_WITHIN]
        if flat:
            parts.append(f"{LABELS[arch]} at {' and '.join(map(str, flat))} generated tokens")
    if not parts:
        return f'No case has a median paired speedup within {UNCHANGED_WITHIN:.1%} of 1.'
    return (f"{'; '.join(parts)}: effectively unchanged, the median paired speedup being within "
            f'{UNCHANGED_WITHIN:.1%} of 1.')


def range_sentence(datas: dict) -> str:
    widest = max(((max(r['paired_speedups']) - min(r['paired_speedups']), arch, r) for arch, d in datas.items() for r in d['summary']),
                 key=lambda t: t[0])
    spread, arch, row = widest
    if spread <= WIDE_RANGE:
        return 'The benefit depends on shape.'
    lengths = sorted({r['seq_len'] for r in datas[arch]['runs'] if r['gen_length'] == row['gen_length'] and r['rep'] >= 0})
    why = (f' The prompts give canvases of {", ".join(map(str, lengths))} tokens at that length.'
           if len(lengths) > 1 else ' The prompts give one canvas length there, so the spread is not from prompt length.')
    return (f'The benefit depends on shape. The widest pair range is {LABELS[arch]} at {row["gen_length"]} generated tokens, '
            f'{min(row["paired_speedups"]):.3f} to {max(row["paired_speedups"]):.3f}x.{why}')


def render(root: Path = ROOT) -> str:
    dest = root / 'bench/results/6000d-dispatch'
    datas = {arch: json.loads((dest / f'{arch}.json').read_text()) for arch in ARCHS}
    for arch, data in datas.items():
        check_runs(arch, data)
    cards = {arch: card(data, arch) for arch, data in datas.items()}
    uuids = {c['uuid'] for c in cards.values()}
    shape = {(c['name'], c['memory_mib'], c['driver']) for c in cards.values()}
    if len(shape) != 1:
        fail(f'the two timing files name different cards or drivers: {sorted(shape)}')
    name, memory, driver = next(iter(shape))
    env = environment(datas, dest)
    steps, block = same(datas, 'steps', 'forward passes'), same(datas, 'block_length', 'block length')
    prompts = same(datas, 'prompts', 'prompts')
    rounds = {len({r['rep'] for r in d['runs'] if r['rep'] >= 0}) for d in datas.values()}
    if len(rounds) != 1:
        fail(f'the timing files hold different numbers of measured rounds: {sorted(rounds)}')
    lines = [f'# {name} Q4_K dispatch measurement', '',
             'Generated by `python3 bench/report_gpu_dispatch.py`.', '',
             f'{words(len(uuids), True)} {name} GPUs, {env["sm_count"]} SMs each, {memory:,} MiB, driver {driver}; '
             f'Mojo {env["mojo"]}, max-core {env["max_core"]}, compiled for {env["compiled_for"]}. '
             'Each model ran on its own GPU, and each A/B pair used the same GPU. '
             f'{env["host_note"]}', '',
             f'{env["source_note"]}', '',
             f'{words(len(prompts), True)} prompts, {words(next(iter(rounds)))} measured rounds per length; '
             'a warmup pair per length is excluded. '
             f'All runs use {steps} forward passes, block length {block}, no early stop, threshold or dual cache. '
             'These timings cover generation, excluding model startup, and use no synchronizing profiler. '
             'Speedup is the median of paired generation-time ratios, not the ratio of the two pooled medians.', '',
             '| Model | Generated tokens | Base ms/step | New ms/step | Paired speedup | Pair range |',
             '|---|---:|---:|---:|---:|---:|']
    for arch, data in datas.items():
        for row in data['summary']:
            a, b = row['median_ms_per_step']['base'], row['median_ms_per_step']['new']
            ratios = row['paired_speedups']
            lines.append(f"| {arch} | {row['gen_length']} | {a:.3f} | {b:.3f} | {row['median_paired_speedup']:.3f}x | "
                         f"{min(ratios):.3f} to {max(ratios):.3f}x |")
    lines += ['', unchanged_sentence(datas) + ' ' + range_sentence(datas), '', '## Preservation checks', '']
    for arch in ARCHS:
        exact = json.loads((root / f'bench/checks/2026-10-30-6000d-dispatch-{arch}-exact.json').read_text())
        if not exact['logits'] or not exact['generation']:
            fail(f'the exactness file of {arch} holds no logit canvases or no generated token lists')
        if not all(r['bytes_equal'] for r in exact['logits']) or not all(r['tokens_equal'] for r in exact['generation']):
            fail(f'the exactness file of {arch} records a difference between the base and the new implementation')
        count = sum(r['values'] for r in exact['logits'])
        lines.append(f"- {arch}: {len(exact['logits'])} logit canvases, {count:,} finite float values checked byte for byte; "
                     f"{len(exact['generation'])} complete generated token lists equal. Mask ID {exact['mask_id']}.")
    lines += ['', 'These checks compare the base GPU implementation with the new GPU implementation. '
              'They do not establish independent model parity or improved model accuracy.', '',
              '## Reproduce', '',
              'Build the base and candidate commits recorded in `llada.json`/`dream.json` with '
              '`MOJO_DLLM_GPU_ARCH=sm_120 scripts/build.sh`. Run the commands in '
              '`docs/development.md` under “Comparing GPU dispatch changes”, adding '
              '`--lengths 128,256,512,1024` for the full matrix.', '',
              'The `tiles-*.jsonl` files are the initial synthetic-shape exploration on GPU 5, '
              'with fixed-width launches from base 70a0880. Build '
              '`bench/bench_gpu_tiles.mojo` with `scripts/build.sh`, then run it with token '
              'counts 64, 148, 276 and 532. These batched microbenchmarks include enqueue and '
              'synchronization time; they are not a substitute for the paired generation runs. '
              'The dispatch wave count is a heuristic, not an occupancy-counter measurement.', '']
    return '\n'.join(lines)


def main(root: Path = ROOT) -> None:
    text = render(root)
    (root / 'bench/results/6000d-dispatch/README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
