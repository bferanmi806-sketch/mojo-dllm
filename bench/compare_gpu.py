"""Interleave two GPU builds using the existing run_bench.py timing/parser.

usage: python3 bench/compare_gpu.py --base CHECKOUT --new CHECKOUT --model M.gguf --gpu INDEX_OR_UUID --out FILE.json
                                    [--lengths 128] [--reps 3] [--prompts 3] [--allow-noisy]

Refuses a base and a new that are the same checkout, the same clean commit or the same binary, and records the
SHA-256 of each build/mojo-dllm. --gpu is a PCI-order index (CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA and nvidia-smi
agree) or a GPU-... uuid; the recorded card is read back from nvidia-smi for that selector. The isolation preflight
and a window around each run are the ones run_bench uses (bench/isolation.py), read-only: no watchdog, no power
profile change, no process management.
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_evidence  # noqa: E402  (bench/ab_evidence.py)
import isolation  # noqa: E402  (bench/isolation.py)
import spec  # noqa: E402  (bench/spec.py)

def command(args):
    return subprocess.check_output(args, text=True).strip()


def load_bench(root):
    spec = importlib.util.spec_from_file_location('run_bench', root / 'bench/run_bench.py')
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lengths', default='128')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--prompts', type=int, default=3)
    ap.add_argument('--allow-noisy', action='store_true', help='continue after a failed isolation preflight')
    args = ap.parse_args()
    if args.reps < 1 or not 1 <= args.prompts <= 3:
        ap.error('reps must be positive and prompts must be between 1 and 3')
    ab_evidence.require_gpu_selector(args.gpu)
    roots = {k: Path(getattr(args, k)).resolve() for k in ('base', 'new')}
    bench = load_bench(roots['new'])
    # CUDA numbers cards fastest first by default and nvidia-smi by PCI bus: one order for both.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.pop('MOJO_DLLM_GPU_PROFILE', None)
    os.environ['OMP_NUM_THREADS'] = str(spec.COMPARE_THREADS)
    prompts = json.loads((roots['new'] / 'bench/config-gpu.json').read_text())['prompts']
    ab_evidence.require_prompts(prompts, 'bench/config-gpu.json')
    prompts = prompts[:args.prompts]
    versions = {k: ab_evidence.head_label(p, command) for k, p in roots.items()}
    hashes = {k: ab_evidence.binary_sha256(p) for k, p in roots.items()}
    ab_evidence.require_distinct_builds(roots, versions, hashes)
    query = ['nvidia-smi', '-i', args.gpu, '--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,power.limit', '--format=csv,noheader']
    meta = spec.result_meta(versions, spec.COMPARE_THREADS)
    card = ab_evidence.parse_gpu_query(command(query), args.gpu)
    doc = dict(date=dt.datetime.now(dt.timezone.utc).isoformat(), partial=True,
               method='run_bench.run_mojo; interleaved AB/BA, no early stop, no threshold or dual cache, no profiler synchronization; first pair per length is warmup',
               machine=dict(gpu=card['query'], cpu=command(['lscpu'])),
               versions=meta['versions'], binary_sha256=hashes, model=Path(args.model).name,
               model_bytes=Path(args.model).stat().st_size, gpu_index=args.gpu, gpu_uuid=card['uuid'],
               gpu_pci_bus_id=card['pci_bus_id'],
               threads=meta['threads'], steps=spec.COMPARE_STEPS, block_length=spec.COMPARE_BLOCK_LENGTH,
               prompts=prompts, runs=[])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(doc, indent=2) + '\n')
    try:
        arm = isolation.begin_arm(output.stem, device='gpu', need_ram_gb=isolation.need_ram_gb(args.model),
                                  runs_dir=output.parent / 'env', allow_noisy=args.allow_noisy,
                                  versions={'mojo-dllm-binary-sha256': hashes['new']})
    except isolation.IsolationError as e:
        raise SystemExit(str(e)) from e
    try:
        for length in map(int, args.lengths.split(',')):
            config = dict(models=dict(gguf=args.model), gen_length=length, steps=spec.COMPARE_STEPS,
                          block_length=spec.COMPARE_BLOCK_LENGTH)
            for rep in range(spec.WARMUP_REP, args.reps):
                for pi, prompt in enumerate(prompts if rep != spec.WARMUP_REP else prompts[:1]):
                    for variant in (('base', 'new') if rep % 2 == 0 else ('new', 'base')):
                        bench.ROOT = roots[variant]
                        result, pinfo = arm.run(
                            lambda: bench.run_mojo(config, prompt, spec.COMPARE_THREADS, ['--device', 'gpu']),
                            max_quiet_wait_s=300, quiet_sample_s=2.0)
                        result.update(variant=variant, gen_length=length, rep=rep, prompt_index=pi,
                                      isolation=pinfo['isolation'])
                        doc['runs'].append(result)
                        save()
                        print(length, rep, pi, variant, result.get('ms_per_step', result), flush=True)
                        if not result['ok']:
                            raise SystemExit('benchmark run failed; partial evidence retained')
    finally:
        arm.finish()
    summaries = []
    for length in map(int, args.lengths.split(',')):
        pairs = []
        same_text = True
        for rep in range(args.reps):
            for pi in range(len(prompts)):
                pair = {r['variant']: r for r in doc['runs'] if r['gen_length'] == length and r['rep'] == rep and r['prompt_index'] == pi}
                a, b = pair['base'], pair['new']
                if not a['forward_passes'] == b['forward_passes'] == doc['steps'] or a['seq_len'] != b['seq_len']:
                    raise SystemExit('workload mismatch')
                pairs.append(a['generate_s'] / b['generate_s'])
                same_text &= a['text'] == b['text']
        per_variant = {v: statistics.median(r['ms_per_step'] for r in doc['runs'] if r['variant'] == v and r['gen_length'] == length and r['rep'] != spec.WARMUP_REP) for v in roots}
        summaries.append(dict(gen_length=length, median_ms_per_step=per_variant,
                              median_paired_speedup=statistics.median(pairs), paired_speedups=pairs,
                              all_text_equal=same_text, pairs=len(pairs)))
    doc.update(partial=False, summary=summaries)
    save()
    print(json.dumps(summaries, indent=2))

if __name__ == '__main__':
    main()
