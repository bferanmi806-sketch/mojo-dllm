#!/usr/bin/env python3
"""Show that the DiffusionGemma bound of bench/parity.py rejects the bugs it exists to catch.

usage: uv run --only-group fixtures python bench/check_dg_bound.py check CASES_DIR
           [--dropped DIR --dropped-layer N] [--out bench/checks/<date>-dg-bound.json]
       uv run --only-group fixtures python bench/check_dg_bound.py make-mutant MODEL.gguf --layer N --out MUTANT.gguf

CASES_DIR is the `--phase ref` output of bench/parity.py (build/dg-parity/<model>). The check takes the
reference's unified canvas logits of every case and compares each of these against them, with the same
`compare` and the same bound the real comparison uses:

  control        the logits against themselves: the bound must accept perfect agreement.
  soft-cap       the logits with the final soft-cap removed, cap * atanh(logit / cap): a runtime that
                 forgot the tanh at the end of the output head.
  dropped scale  the logits of a run where one layer's decoder output scale was set to 1.0, which is
                 what a runtime that forgot to multiply that layer's canvas rows by it computes. The run
                 is `bench/parity.py --phase ref` on a model made by `make-mutant`, passed with --dropped.

The exit status is 1 when the control is rejected or any mutant is accepted: a bound that lets a known bug
through proves nothing about a runtime it passes. make-mutant copies the GGUF and overwrites the four bytes
of `blk.N.layer_output_scale.weight`; the original file is never written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity  # noqa: E402

ROOT = HERE.parent


def load_cases(folder: pathlib.Path) -> dict[str, tuple[dict, pathlib.Path]]:
    cases = {}
    for case_json in parity.dg_case_files(folder):
        ref = case_json.parent / "unified.ref.f32"
        if not ref.is_file():
            raise SystemExit(f"refusing: {ref} is missing; run bench/parity.py --phase ref first")
        cases[case_json.parent.name] = (json.loads(case_json.read_text()), ref)
    if not cases:
        raise SystemExit(f"refusing: no cases under {folder}")
    return cases


def check(args: argparse.Namespace) -> int:
    import numpy as np

    bound = parity.llada_bound(ROOT)
    cases = load_cases(pathlib.Path(args.cases))
    dropped = load_cases(pathlib.Path(args.dropped)) if args.dropped else {}
    if args.dropped and set(dropped) != set(cases):
        raise SystemExit("refusing: --dropped holds different cases from CASES_DIR")
    results, failures = [], []
    for name, (meta, ref_path) in cases.items():
        vocab = meta["n_vocab"]
        ref = parity._load_f32(ref_path, vocab)
        if ref.size == 0:
            raise SystemExit(f"refusing: {ref_path} is empty")
        cap = float(meta["final_logit_softcapping"])
        # The reference holds cap * tanh(z / cap); undo it. The clip keeps a logit that rounded to exactly
        # +-cap finite, and is part of the mutant, not of the comparison.
        uncapped = cap * np.arctanh(np.clip(ref / cap, -1 + 1e-9, 1 - 1e-9))
        mutants = {"control": (ref, False), "soft-cap removed": (uncapped, True)}
        if dropped:
            mutants[f"layer {args.dropped_layer} output scale dropped"] = (
                parity._load_f32(dropped[name][1], vocab),
                True,
            )
        for label, (candidate, must_fail) in mutants.items():
            m = parity.dg_metrics(candidate, ref, 0)
            violations = parity.dg_violations(m, bound)
            rejected = bool(violations)
            ok = rejected == must_fail
            if not ok:
                failures.append(f"{name} {label}: {'rejected' if rejected else 'accepted'}, expected "
                                f"{'rejected' if must_fail else 'accepted'}")
            results.append(dict(case=name, prompt=meta["prompt"], mutant=label, rejected=rejected,
                                expected_rejected=must_fail, violations=violations, **m))
            print(f"{name} {label:40} rejected={rejected!s:5} min_cos={m['min_cosine']:.5f} "
                  f"argmax={m['argmax_agree']}/{m['rows']} top5={m['mean_top5_overlap']:.2f} "
                  f"max_rel_rms={m['max_rel_rms']:.4f} top_logit_err={m['top1_max_rel_err']:.4f}  {violations}")
    if args.out:
        first = next(iter(cases.values()))[0]
        doc = dict(date=dt.datetime.now().isoformat(timespec="seconds"), bound=bound, model=first["model"],
                   model_sha256=first["model_sha256"], llama_cpp_commit=first["llama_cpp_commit"],
                   dropped_layer=args.dropped_layer if args.dropped else None,
                   dropped_model=(
                       {k: next(iter(dropped.values()))[0][k] for k in ("model", "model_sha256")} if dropped else None
                   ),
                   results=results,
                   all_as_expected=not failures)
        pathlib.Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {args.out}")
    if failures:
        print("the bound does not separate good from bad here:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


def make_mutant(args: argparse.Namespace) -> int:
    from gguf import GGUFReader

    src, out = pathlib.Path(args.model), pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists")
    name = f"blk.{args.layer}.layer_output_scale.weight"
    reader = GGUFReader(str(src))
    found = [t for t in reader.tensors if t.name == name]
    if len(found) != 1:
        raise SystemExit(f"refusing: {src} has {len(found)} tensors named {name}")
    tensor = found[0]
    if tensor.n_elements != 1 or tensor.tensor_type.name != "F32":
        raise SystemExit(f"refusing: {name} is not a single f32 ({tensor.n_elements} x {tensor.tensor_type.name})")
    before = float(tensor.data[0])
    if before == 1.0:
        raise SystemExit(f"refusing: {name} is already 1.0, so dropping it changes nothing")
    shutil.copyfile(src, out)
    with out.open("r+b") as f:
        f.seek(int(tensor.data_offset))
        if struct.unpack("<f", f.read(4))[0] != before:
            out.unlink()
            raise SystemExit(f"refusing: the bytes at the offset of {name} are not the value gguf read")
        f.seek(int(tensor.data_offset))
        f.write(struct.pack("<f", 1.0))
    after = float([t for t in GGUFReader(str(out)).tensors if t.name == name][0].data[0])
    if after != 1.0:
        out.unlink()
        raise SystemExit(f"refusing: {name} reads {after} after the patch, not 1.0")
    print(f"{name}: {before} -> {after} in {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("cases")
    c.add_argument("--dropped", default=None)
    c.add_argument("--dropped-layer", type=int, default=None)
    c.add_argument("--out", default=None)
    mk = sub.add_parser("make-mutant")
    mk.add_argument("model")
    mk.add_argument("--layer", type=int, required=True)
    mk.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "check":
        if (args.dropped is None) != (args.dropped_layer is None):
            ap.error("--dropped and --dropped-layer go together")
        return check(args)
    return make_mutant(args)


if __name__ == "__main__":
    raise SystemExit(main())
