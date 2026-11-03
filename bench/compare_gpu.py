"""Interleave two GPU builds using the existing run_bench.py timing/parser.

No watchdog, system power changes, or unrelated process management.
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess

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
    args = ap.parse_args()
    if args.reps < 1 or not 1 <= args.prompts <= 3:
        ap.error('reps must be positive and prompts must be between 1 and 3')
    roots = {k: Path(getattr(args, k)).resolve() for k in ('base', 'new')}
    bench = load_bench(roots['new'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.pop('MOJO_DLLM_GPU_PROFILE', None)
    os.environ['OMP_NUM_THREADS'] = '8'
    prompts = json.loads((roots['new'] / 'bench/config-gpu.json').read_text())['prompts'][:args.prompts]
    versions = {k: command(['git', '-C', str(p), 'rev-parse', 'HEAD']) + ('-dirty' if command(['git', '-C', str(p), 'status', '--porcelain', '--', 'src']) else '') for k, p in roots.items()}
    query = ['nvidia-smi', '-i', args.gpu, '--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,power.limit', '--format=csv,noheader']
    doc = dict(date=dt.datetime.now(dt.timezone.utc).isoformat(), partial=True,
               method='run_bench.run_mojo; interleaved AB/BA, no early stop, no threshold or dual cache, no profiler synchronization; first pair per length is warmup',
               machine=dict(gpu=command(query), cpu=command(['lscpu'])),
               versions=versions, model=Path(args.model).name,
               model_bytes=Path(args.model).stat().st_size, gpu_uuid=args.gpu,
               threads=8, steps=32, block_length=32, prompts=prompts, runs=[])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(doc, indent=2) + '\n')
    for length in map(int, args.lengths.split(',')):
        config = dict(models=dict(gguf=args.model), gen_length=length, steps=32, block_length=32)
        for rep in range(-1, args.reps):
            for pi, prompt in enumerate(prompts if rep >= 0 else prompts[:1]):
                for variant in (('base', 'new') if rep % 2 == 0 else ('new', 'base')):
                    bench.ROOT = roots[variant]
                    result = bench.run_mojo(config, prompt, 8, ['--device', 'gpu'])
                    result.update(variant=variant, gen_length=length, rep=rep, prompt_index=pi)
                    doc['runs'].append(result)
                    save()
                    print(length, rep, pi, variant, result.get('ms_per_step', result), flush=True)
                    if not result['ok']:
                        raise SystemExit('benchmark run failed; partial evidence retained')
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
        per_variant = {v: statistics.median(r['ms_per_step'] for r in doc['runs'] if r['variant'] == v and r['gen_length'] == length and r['rep'] >= 0) for v in roots}
        summaries.append(dict(gen_length=length, median_ms_per_step=per_variant,
                              median_paired_speedup=statistics.median(pairs), paired_speedups=pairs,
                              all_text_equal=same_text, pairs=len(pairs)))
    doc.update(partial=False, summary=summaries)
    save()
    print(json.dumps(summaries, indent=2))

if __name__ == '__main__':
    main()
