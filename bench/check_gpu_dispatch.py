#!/usr/bin/env python3
"""Check exact GPU logits and generated tokens between two builds.

Usage: python3 bench/check_gpu_dispatch.py --base BASE --new NEW --model M.gguf
       --arch llada|dream --out evidence.json
CUDA_VISIBLE_DEVICES selects the assigned GPU (set CUDA_DEVICE_ORDER=PCI_BUS_ID so the index matches
nvidia-smi; both variables are recorded in the evidence). No CPU or external-runtime parity
is implied: this checks that dispatch preserves the base GPU computation. A base and a new that are the same
checkout, the same clean commit or byte-identical binaries are refused, and the SHA-256 of each binary is recorded.
"""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_evidence  # noqa: E402  (bench/ab_evidence.py)
import spec  # noqa: E402  (bench/spec.py)

ARCH_IDS = spec.ARCH_IDS


def run(cmd):
    return subprocess.check_output(cmd, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--arch', choices=['llada', 'dream'], required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    roots = {k: Path(getattr(args, k)).resolve() for k in ('base', 'new')}
    bins = {k: str(p / 'build/mojo-dllm') for k, p in roots.items()}
    mask, vocab = ARCH_IDS[args.arch]['mask'], ARCH_IDS[args.arch]['vocab']
    model = str(Path(args.model).resolve())
    prompts = json.loads((roots['new'] / 'bench/config-gpu.json').read_text())['prompts']
    ab_evidence.require_prompts(prompts, 'bench/config-gpu.json')
    versions = {k: ab_evidence.head_label(p, run) for k, p in roots.items()}
    hashes = {k: ab_evidence.binary_sha256(p) for k, p in roots.items()}
    ab_evidence.require_distinct_builds(roots, versions, hashes)
    doc = dict(arch=args.arch, model=Path(model).name, mask_id=mask,
               method='Exact base/new GPU logits and token lists; not independent model parity',
               versions=versions, binary_sha256=hashes,
               cuda=dict(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         device_order=os.environ.get('CUDA_DEVICE_ORDER')),
               logits=[], generation=[])
    with tempfile.TemporaryDirectory(prefix='mojo-dispatch-') as tmp:
        for masked in [64, 128, 256, 512]:
            for pi, prompt in enumerate(prompts[:2]):
                toks = json.loads(run([bins['base'], 'tokenize', '--model', model, '--prompt', prompt]).splitlines()[-1])
                ids = toks + [mask] * masked
                files = {}
                for variant, binary in bins.items():
                    f = Path(tmp) / (variant + '.f32')
                    run([binary, 'logits', '--model', model, '--tokens', ','.join(map(str, ids)),
                         '--rows', ','.join(map(str, range(len(ids)))), '--out', str(f), '--device', 'gpu', '--threads', '8'])
                    files[variant] = f.read_bytes()
                    if len(files[variant]) != len(ids) * vocab * 4:
                        raise RuntimeError('unexpected logit file size')
                    values = array('f')
                    values.frombytes(files[variant])
                    if not all(map(math.isfinite, values)):
                        raise RuntimeError('non-finite logits')
                same = files['base'] == files['new']
                doc['logits'].append(dict(masked=masked, prompt_index=pi, rows=len(ids),
                                          values=len(ids) * vocab, bytes_equal=same))
                print('logits', masked, pi, same, flush=True)
                if not same:
                    raise RuntimeError('GPU logits differ')
        for pi, prompt in enumerate(prompts):
            results = {}
            for variant, binary in bins.items():
                output = run([binary, 'run', '--model', model, '--prompt', prompt, '--device', 'gpu',
                              '--threads', '8', '--max-tokens', '128', '--steps', '32',
                              # Dream has no blocks and ignores --block-length; the flag is for LLaDA only.
                              *([] if args.arch == 'dream' else ['--block-length', '32']),
                              '--no-early-stop', '--json'])
                results[variant] = json.loads(next(l for l in output.splitlines() if l.startswith('{')))
            a, b = results['base'], results['new']
            same = a['tokens'] == b['tokens'] and a['forward_passes'] == b['forward_passes'] == 32
            doc['generation'].append(dict(prompt_index=pi, tokens_equal=same, tokens=a['tokens'],
                                          forward_passes=a['forward_passes']))
            print('generation', pi, same, flush=True)
            if not same:
                raise RuntimeError('generated tokens or workload differ')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2) + '\n')


if __name__ == '__main__':
    main()
