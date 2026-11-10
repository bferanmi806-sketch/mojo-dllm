#!/usr/bin/env python3
"""Render the benchmark table from bench/results/*.json into README and docs.

usage: bench_table.py --write     rewrite the tables between the markers
       bench_table.py --check     exit 1 if a table differs from the evidence
       --allow-noisy              with either: publish a timed GSM8K round that has noisy or unchecked rows

No benchmark number in this repository is typed by hand. The tables live
between `<!-- bench:start -->` and `<!-- bench:end -->` and are produced from
the result files that bench/run_bench.py writes. A partial result file (a run
that was stopped) is refused, and so is a runtime that appears in two files.
"""

from __future__ import annotations

import json
import math
import statistics
import pathlib
import sys
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "README.md", ROOT / "docs" / "benchmarks.md", ROOT / "docs" / "correctness.md"]
START, END = "<!-- bench:start -->", "<!-- bench:end -->"

ROWS = [
    ("mojo-dllm-gpu", "**mojo-dllm** `--device gpu`", "Mojo"),
    ("llama.cpp-cuda", "llama.cpp `llama-diffusion-cli`, CUDA, all layers offloaded", "C/C++, CUDA"),
    ("mojo-dllm", "**mojo-dllm**", "Mojo"),
    ("mojo-dllm-full-logits", "mojo-dllm, logits for every position (ablation)", "Mojo"),
    ("llama.cpp", "llama.cpp `llama-diffusion-cli`", "C/C++"),
    ("diffuse-cpp", "diffuse-cpp, cache off", "C++"),
    ("diffuse-cpp-cache-entropy-exit", "diffuse-cpp, cache + `entropy_exit` †", "C++"),
]


def model_label(doc: dict) -> str:
    name = pathlib.Path(doc["config"]["models"]["gguf"]).name
    for suffix in (".gguf", ".Q4_K_M", "-Q4_K_M"):
        name = name.replace(suffix, "")
    return name


GPU_RUNTIMES = {"mojo-dllm-gpu", "llama.cpp-cuda", "llama-bench-cuda"}


def is_gpu(doc: dict) -> bool:
    """A result file is a GPU benchmark if it ran GPU runtimes; mixing the two is refused."""
    kinds = {rt in GPU_RUNTIMES for rt in doc["summary"]}
    if len(kinds) != 1:
        raise SystemExit("bench_table: a result file mixes CPU and GPU runtimes")
    return kinds.pop()


def check_summary(name: str, doc: dict) -> None:
    """The table rows come from `summary`, the ratio line from `runs`: refuse a file in which they disagree."""
    for rt, sm in doc["summary"].items():
        ok = [r for r in doc["runs"] if r["runtime"] == rt and r.get("ok")]
        if sm["runs_ok"] != len(ok):
            raise SystemExit(f"bench_table: {name}: the summary of {rt} says runs_ok {sm['runs_ok']}, its runs hold {len(ok)}")
        for state in ("clean", "noisy", "unchecked"):
            got = sum(1 for r in ok if window_state(r) == state)
            if f"{state}_runs" in sm and sm[f"{state}_runs"] != got:
                raise SystemExit(f"bench_table: {name}: the summary of {rt} says {state}_runs {sm[f'{state}_runs']}, "
                                 f"its runs hold {got}")
        ms = [r["ms_per_step"] for r in ok if r.get("ms_per_step") is not None]
        if ms and not math.isclose(sm["ms_per_step_median"], statistics.median(ms), rel_tol=1e-9):
            raise SystemExit(f"bench_table: {name}: the summary of {rt} says ms_per_step_median "
                             f"{sm['ms_per_step_median']}, its runs give {statistics.median(ms)}")


def load() -> dict[str, dict]:
    """Result files grouped by model: {label: {merged, meta, sources}}."""
    files = sorted((ROOT / "bench" / "results").glob("*.json"))
    if not files:
        raise SystemExit("bench_table: no result files under bench/results")
    groups: dict[str, dict] = {}
    for f in files:
        doc = json.loads(f.read_text())
        if doc.get("partial", True):
            raise SystemExit(f"bench_table: {f.name} is a partial run; finish or delete it")
        if doc["config"].get("kind") == "fairness":
            continue
        gpu = is_gpu(doc)
        label = model_label(doc) + (" (GPU)" if gpu else "")
        g = groups.setdefault(label, dict(merged={}, meta={}, sources=[], gpu=gpu))
        check_summary(f.name, doc)
        g["sources"].append(f.name)
        g.setdefault("runs", []).extend(doc["runs"])
        g.setdefault("files", []).append((f.name, doc["runs"]))
        g.setdefault("busy_threshold", doc["config"].get("busy_threshold", 0.1))
        for rt, s in doc["summary"].items():
            if rt in g["merged"]:
                raise SystemExit(f"bench_table: {label} / {rt} appears in more than one result file")
            if s["runs_ok"] == 0:
                continue
            g["merged"][rt] = s
        meta = g["meta"]
        meta.setdefault("machine", doc["machine"])
        meta.setdefault("config", doc["config"])
        meta.setdefault("versions", {}).update(doc["versions"])
        meta.setdefault("reps", doc["reps"])
    return groups


def fairness() -> str:
    """Thread sweep and forward-pass-only tables from the fairness result files."""
    files = sorted((ROOT / "bench" / "results").glob("*.json"))
    docs = [(f, json.loads(f.read_text())) for f in files]
    docs = [(f, d) for f, d in docs if d["config"].get("kind") == "fairness"]
    out: list[str] = []
    for f, doc in sorted(docs, key=lambda fd: is_gpu(fd[1])):
        # A runtime whose runs all failed has no medians; leave it out here
        # and let the run's own exit status (run_bench exits 1) report it.
        s = {k: v for k, v in doc["summary"].items() if v["runs_ok"] > 0}
        if not s:
            raise SystemExit(f"bench_table: {f.name} has no successful run")
        c = doc["config"]
        m = doc["machine"]
        label = model_label(doc)
        seq = sorted({r["seq_len"] for r in doc["runs"] if r.get("ok")})
        canvas = f"{seq[0]} tokens" if len(seq) == 1 else "varies"
        runs = f"median of {doc['reps']} runs, one prompt, canvas {canvas}"
        if out:
            out.append("")
        if is_gpu(doc):
            gpu_desc = (m.get("gpu") or "unknown GPU").split(",")[0].strip()
            out += [
                f"**{label} on the GPU, forward pass alone**: {gpu_desc}; {runs}.",
                "",
                "| Runtime | ms / step | forward pass alone |",
                "|---|---:|---:|",
            ]
            g = s.get("mojo-dllm-gpu")
            lb = s.get("llama-bench-cuda")
            if g:
                out.append(f"| mojo-dllm `--device gpu` | {g['ms_per_step_median']:,.0f} | {g['forward_ms_median']:,.0f} ms |")
            if lb:
                out.append(f"| llama.cpp CUDA, `llama-bench -p {seq[0]}` | | {lb['forward_ms_median']:,.0f} ms |")
        else:
            out += [
                f"**{label}, llama.cpp thread sweep and forward pass alone**: {m['cpu']}; {runs}.",
                "",
                "| Runtime | threads | ms / step | forward pass alone |",
                "|---|---:|---:|---:|",
            ]
            mj = s.get("mojo-dllm")
            if mj:
                out.append(
                    f"| mojo-dllm | {c['threads']} | {mj['ms_per_step_median']:,.0f} | {mj['forward_ms_median']:,.0f} ms |"
                )
            ts = sorted({int(k.split(":t")[1]) for k in s if k.startswith(("llama.cpp:t", "llama-bench:t"))})
            for t in ts:
                d = s.get(f"llama.cpp:t{t}")
                lb = s.get(f"llama-bench:t{t}")
                step = f"{d['ms_per_step_median']:,.0f}" if d and d["ms_per_step_median"] else ""
                fwd = f"{lb['forward_ms_median']:,.0f} ms" if lb and lb["forward_ms_median"] else ""
                out.append(f"| llama.cpp | {t} | {step} | {fwd} |")
        busy = [r["cpu_busy_before"] for r in doc["runs"]]
        steps = "" if is_gpu(doc) else "The ms / step column runs `llama-diffusion-cli`. "
        cpu_note = f" CPU busy before a run: at most {max(busy):.1%}."
        out += [
            "",
            f"{steps}The forward pass alone is mojo-dllm's measured forward pass against llama.cpp's "
            f"`llama-bench` at the same token count. `llama-bench` applies a causal mask and computes "
            f"logits for one position; mojo-dllm's figure includes logits for up to "
            f"{c['block_length']} positions and copying the canvas in.{cpu_note} "
            f"Evidence: `bench/results/{f.name}`.",
        ]
    return "\n".join(out)


class PairedRatio(NamedTuple):
    median: float
    low: float
    high: float
    pairs: int
    below: int


def paired_ratio(runs: list[dict], num: str, den: str) -> PairedRatio | None:
    """ms / step of runtime `num` over `den`, for each (rep, prompt) in which both finished."""
    steps: dict[str, dict[tuple[int, int], float]] = {num: {}, den: {}}
    for r in runs:
        if r.get("ok") and r["runtime"] in steps and r.get("ms_per_step") is not None:
            key = (r["rep"], r["prompt_index"])
            if key in steps[r["runtime"]]:
                raise SystemExit(f"bench_table: two runs of {r['runtime']} share rep {key[0]} and prompt {key[1]}")
            steps[r["runtime"]][key] = r["ms_per_step"]
    ratios = [steps[num][k] / steps[den][k] for k in steps[num] if k in steps[den]]
    if not ratios:
        return None
    return PairedRatio(statistics.median(ratios), min(ratios), max(ratios), len(ratios), sum(1 for x in ratios if x < 1))


def gpu_ratio(label: str, g: dict) -> PairedRatio | None:
    """The ratio of one result file that measured both runtimes; refused when two files each hold one."""
    names = ("mojo-dllm-gpu", "llama.cpp-cuda")
    if not all(n in g["merged"] for n in names):
        return None
    both = [runs for _, runs in g["files"] if all(any(r["runtime"] == n and r.get("ok") for r in runs) for n in names)]
    if not both:
        raise SystemExit(f"bench_table: {label}: mojo-dllm-gpu and llama.cpp-cuda come from different result files; "
                         "a ratio needs one run that measured both")
    pr = paired_ratio(both[0], *names)
    if pr is None:
        raise SystemExit(f"bench_table: {label}: no run finished in both runtimes")
    return pr


def busy_before(r: dict) -> float:
    """The CPU load a run started under; a run without it is refused, never read as a quiet machine."""
    if "cpu_busy_before" not in r:
        raise SystemExit(f"bench_table: run {r.get('runtime')} rep {r.get('rep')} has no cpu_busy_before")
    return r["cpu_busy_before"]


def window_state(r: dict) -> str:
    """`clean`, `noisy` or `unchecked` for a run's isolation window (the same rule as bench/run_bench.py)."""
    iso = r.get("isolation")
    if not isinstance(iso, dict):
        return "unchecked"
    return "clean" if iso.get("clean") is True and not iso.get("noisy") else "noisy"


def busy_note(runs: list[dict], threshold: float, rows: list) -> list[str]:
    """Disclose runs that started on a busy CPU or ended in a noisy window, with medians over the others.

    The harness waits for a quiet CPU but gives up after max_quiet_wait_s and
    runs anyway, recording how busy the CPU was; a run whose isolation window
    saw other CPU, swap, thermal or GPU activity is kept as `noisy`. Those runs
    stay in the table; this note says how many there were and what the quiet
    runs alone give. In a file where some runs have a window and some do not,
    the runs without one are counted as unchecked.
    """
    ok = [r for r in runs if r.get("ok")]
    busy = [r for r in ok if busy_before(r) > threshold]
    states = {id(r): window_state(r) for r in ok}
    noisy = [r for r in ok if states[id(r)] == "noisy"]
    has_windows = any(v != "unchecked" for v in states.values())
    unchecked = [r for r in ok if states[id(r)] == "unchecked"] if has_windows else []
    disturbed = [r for r in ok if busy_before(r) > threshold or states[id(r)] == "noisy"]
    disturbed_ids = {id(r) for r in disturbed}
    if not disturbed and not unchecked:
        return []
    parts = []
    for key, label, _ in rows:
        quiet = [r for r in ok if r["runtime"] == key and id(r) not in disturbed_ids and r.get("ms_per_step") is not None]
        if not quiet or not any(r["runtime"] == key for r in disturbed):
            continue
        step = statistics.median(r["ms_per_step"] for r in quiet)
        start = statistics.median(r["startup_s"] for r in quiet)
        name = label.replace("**", "")
        parts.append(f"{name} {step:,.0f} ms / step and {start:.1f} s startup ({len(quiet)} runs)")
    sentences = []
    if busy:
        sentences.append(f"{len(busy)} of {len(ok)} runs started with the CPU more than {threshold:.0%} busy "
                         "(the harness waits for a quiet CPU, then runs anyway and records the load).")
    if noisy:
        sentences.append(f"{len(noisy)} of {len(ok)} runs ended in a noisy isolation window (other CPU, swap, "
                         "thermal or GPU activity) and stay in the medians.")
    if unchecked:
        sentences.append(f"{len(unchecked)} of {len(ok)} runs carry no isolation window.")
    note = " ".join(sentences)
    if parts:
        note += " The quiet runs alone give " + "; ".join(parts) + "."
    return ["", note]


def render() -> str:
    groups = load()
    order = sorted(groups, key=lambda k: (not k.startswith("LLaDA"), groups[k]["gpu"], k))
    out: list[str] = []
    for label in order:
        g = groups[label]
        merged, meta, sources = g["merged"], g["meta"], g["sources"]
        c = meta["config"]
        m = meta["machine"]
        if out:
            out.append("")
        if g["gpu"]:
            gpu_desc = m.get("gpu") or "unknown GPU"
            parts = [p.strip() for p in gpu_desc.split(",")]
            where = f"{parts[0]} ({parts[2]}, driver {parts[1]}), host {m['cpu']}" if len(parts) == 3 else gpu_desc
            out += [
                f"**{label.removesuffix(' (GPU)')}** on the GPU, Q4_K_M. {where}. Each prompt: "
                f"{c['gen_length']} generated tokens, {c['steps']} denoising steps; median of "
                f"{meta['reps']} runs x {len(c['prompts'])} prompts. Peak RSS is host memory.",
                "",
                "| Runtime | ms / step | tokens / s | startup | peak RSS | language |",
                "|---|---:|---:|---:|---:|---|",
            ]
        else:
            out += [
                f"**{label}**, Q4_K_M. {m['cpu']}, {c['threads']} threads, {m['ram_gb']} GB RAM, "
                f"`{m['power_profile']}` power profile. Each prompt: {c['gen_length']} generated tokens, "
                f"{c['steps']} denoising steps; median of {meta['reps']} runs x {len(c['prompts'])} prompts.",
                "",
                "| Runtime | ms / step | tokens / s | startup | peak RSS | language |",
                "|---|---:|---:|---:|---:|---|",
            ]
        for key, row_label, lang in ROWS:
            s = merged.get(key)
            if s is None:
                continue
            out.append(
                f"| {row_label} | {s['ms_per_step_median']:,.0f} | {s['tokens_per_s_median']:.2f} | "
                f"{s['startup_s_median']:.1f} s | {s['peak_rss_gb_max']:.1f} GB | {lang} |"
            )
        if g["gpu"]:
            pr = gpu_ratio(label, g)
            if pr:
                out += [
                    "",
                    f"Step time of mojo-dllm over llama.cpp: median {pr.median:.3f}, range {pr.low:.3f} to "
                    f"{pr.high:.3f} across {pr.pairs} paired runs, with mojo-dllm's step shorter in {pr.below} of "
                    "them. A ratio below 1 means mojo-dllm's step is shorter.",
                ]
        v = meta["versions"]
        out += [
            "",
            f"Versions: mojo-dllm `{v.get('mojo-dllm')}`, {v.get('mojo')}, llama.cpp `{v.get('llama.cpp')}`"
            + (f", diffuse-cpp `{v['diffuse-cpp']}`" if any(k.startswith("diffuse") for k in merged) else "")
            + f". Evidence: {', '.join(f'`bench/results/{s}`' for s in sources)}.",
        ]
        out += busy_note(g["runs"], g["busy_threshold"], ROWS)
        if "diffuse-cpp-cache-entropy-exit" in merged:
            out += [
                "",
                "† Not the same work: the inter-step cache reuses stale K/V and `entropy_exit` can stop early. "
                "Shown because it is diffuse-cpp's recommended mode.",
            ]
    fair = fairness()
    if fair:
        out += ["", fair]
    return "\n".join(out)


def parity() -> tuple[str, str]:
    """Keep the newest run of each (architecture, device, model)."""
    files = sorted((ROOT / "bench" / "parity").glob("*.json"))
    if not files:
        raise SystemExit("bench_table: no parity evidence under bench/parity")
    newest: dict[tuple[str, str, str], tuple[pathlib.Path, dict]] = {}
    for f in files:
        doc = json.loads(f.read_text())
        arch = doc.get("arch", "llada")
        key = (arch, doc.get("device", "cpu"), doc.get("model", arch))
        previous = newest.get(key)
        # A legacy name and a new model-qualified name can coexist on one day.
        # Their lexical order does not tell us which run happened last.
        if previous is None or (doc["date"], f.name) > (previous[1]["date"], previous[0].name):
            newest[key] = (f, doc)
    lg: list[str] = []
    gn: list[str] = []
    for key in sorted(newest, key=lambda k: (k[0] != "llada", k[0], k[1] != "cpu", k[1], k[2])):
        arch, device, model = key
        f, doc = newest[key]
        name = model + (", GPU" if device == "gpu" else "")
        src = f"`bench/parity/{f.name}` (mojo-dllm `{doc['versions']['mojo-dllm']}`)"
        if lg:
            lg.append("")
            gn.append("")
        lg += [
            f"**{name}**",
            "",
            "| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS "
            "| same argmax | top-5 overlap |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in doc["logits"]:
            lg.append(
                f"| {c['seq_len']} tokens | {c['rows']} | {c['median_cosine']:.5f} | "
                f"{c['masked_median_cosine']:.5f} | {c['min_cosine']:.4f} (row {c['worst_rows'][0]}) | "
                f"{c['median_rel_rms']:.3f} | {c['argmax_agree']} / {c['rows']} | {c['mean_top5_overlap']:.2f} / 5 |"
            )
        if "reference2" in doc:
            r2 = doc["reference2"]
            lg += [
                "",
                f"The same canvases against a second reference, {r2}, and the two references "
                "against each other:",
                "",
                "| Canvas | compared | median cosine | worst cosine | median rel. RMS | same argmax |",
                "|---:|---|---:|---:|---:|---:|",
            ]
            for c in doc["logits"]:
                for label, m in (
                    ("mojo-dllm vs llama.cpp CPU", c),
                    (f"mojo-dllm vs {r2}", c["vs_ref2"]),
                    (f"llama.cpp CPU vs {r2}", c["ref_vs_ref2"]),
                ):
                    lg.append(
                        f"| {c['seq_len']} tokens | {label} | {m['median_cosine']:.5f} | "
                        f"{m['min_cosine']:.4f} | {m['median_rel_rms']:.3f} | {m['argmax_agree']} / {m['rows']} |"
                    )
        lg += ["", f"Evidence: {src}."]
        gn += [
            f"**{name}**",
            "",
            "| Prompt | generated | steps | same tokens | first difference |",
            "|---|---:|---:|---:|---:|",
        ]
        for g in doc["generation"]:
            first = "none" if g["first_divergence"] is None else f"position {g['first_divergence']}"
            gn.append(
                f"| {g['prompt']} | {g['gen_length']} | {g['steps']} | "
                f"{g['matching_tokens']} / {g['gen_length']} | {first} |"
            )
        gn += ["", f"Evidence: {src}."]
    return "\n".join(lg), "\n".join(gn)


def quality() -> str:
    files = sorted((ROOT / "bench" / "quality").glob("*.json"))
    if not files:
        return ""
    out: list[str] = []
    for f in files:
        doc = json.loads(f.read_text())
        if out:
            out.append("")
        out += [
            f"**{doc['model']}**, {doc['gen_length']} generated tokens.",
            "",
            "| Prompt | steps | tokens / step | time | repeated words | output (first 120 characters) |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in doc["runs"]:
            snippet = r["text"][:120].replace("\n", " ").replace("|", "\\|")
            out.append(
                f"| {r['prompt']} | {r['steps']} | {r['tokens_per_step']:g} | {r['generate_s']:.0f} s | "
                f"{r['repeated_words']} | {snippet} |"
            )
        out += ["", f"Evidence: `bench/quality/{f.name}` (mojo-dllm `{doc['versions']['mojo-dllm']}`)."]
    return "\n".join(out)


def count(v: object, spec: str = ".1f") -> str:
    """A count or a mean for a table cell; `n/a` for a null (an engine that does not count forward passes)."""
    return "n/a" if v is None else format(v, spec)


def isolation_sentence(docs: list[dict]) -> str:
    """What the timed rounds say about the machine they ran on, from their own `noisy_rows` and `unchecked_rows`.

    A document written before isolation windows existed has neither key: nothing records that the machine
    was otherwise idle, and the sentence says so instead of claiming it.
    """
    recorded = [d for d in docs if "noisy_rows" in d and "unchecked_rows" in d]
    if len(recorded) < len(docs):
        return ("These rounds carry no isolation record, so nothing here shows that the machine was otherwise "
                "idle.")
    noisy = sum(sum(d["noisy_rows"].values()) for d in docs)
    unchecked = sum(sum(d["unchecked_rows"].values()) for d in docs)
    if noisy == 0 and unchecked == 0:
        return "Every timed problem ran in an isolation window that came back clean."
    return (f"Isolation: {noisy} noisy and {unchecked} unchecked problem rows across the timed arms; "
            "times average over the clean rows only where the table says so.")


def refuse_unclean_timing(f: pathlib.Path, d: dict) -> None:
    """A timed round whose clean rows are fewer than its problems is not published unless the caller says so."""
    if "clean_rows" in d and d["clean_rows"] < d["problems"]:
        raise SystemExit(
            f"bench_table: {f.name} has {d['clean_rows']} clean rows for {d['problems']} problems "
            f"(noisy {d.get('noisy_rows')}, unchecked {d.get('unchecked_rows')}); run the round again, "
            "or pass --allow-noisy to publish it with those counts"
        )


def eval_table(allow_noisy: bool = False) -> str:
    """GSM8K paired comparisons from bench/eval/*-vs-*.json. Accuracy comes from the
    full comparison; time per problem from the `--limit` one, the shorter round
    (generate_s in the other rounds carries whatever else ran)."""
    full: dict[tuple, tuple] = {}
    timed: dict[tuple, tuple] = {}
    for f in sorted((ROOT / "bench" / "eval").glob("*-vs-*.json")):
        doc = json.loads(f.read_text())
        if doc.get("limit") and not allow_noisy:
            refuse_unclean_timing(f, doc)
        (timed if doc.get("limit") else full)[tuple(doc["arms"])] = (f, doc)
    if not full:
        return ""
    bases = {a for a, _ in full}
    if len(bases) != 1:
        raise SystemExit(f"bench_table: eval comparisons use several baselines: {sorted(bases)}")
    base = bases.pop()
    docs = [d for _, d in full.values()]
    if len({d["accuracy"][base] for d in docs}) != 1 or len({d["problems"] for d in docs}) != 1:
        raise SystemExit("bench_table: the eval comparisons disagree on the baseline arm")
    first = docs[0]
    cfg = first["config"][base]

    def secs(pair: tuple, arm: str) -> str:
        if pair not in timed:
            return "not measured"
        d = timed[pair][1]
        clean = (d.get("generate_s_mean_clean") or {}).get(arm)
        mean = d["generate_s_mean"][arm] if clean is None and "generate_s_mean_clean" not in d else clean
        return "no clean row" if mean is None else f"{mean:.1f} s"

    any_pair = next(iter(full))
    out = [
        f"**{pathlib.Path(cfg['model']).name}**, GSM8K, {first['problems']} problems, "
        f"flags `{' '.join(cfg['flags'])}`, mojo-dllm `{cfg['version']}`.",
        "",
        "| Arm | accuracy | vs baseline | 95% CI | discordant | McNemar p | passes / problem | time / problem |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {base} | {first['accuracy'][base]:.1%} | | | | | "
        f"{count(first['forward_passes_mean'][base])} | {secs(any_pair, base)} |",
    ]
    other_weights: list[str] = []
    for pair, (f, d) in full.items():
        arm = pair[1]
        lo, hi = d["ci95"]
        if d.get("weights_differ"):
            other_weights.append(f"{arm} ran {d['models'][arm]}, the baseline {d['models'][base]}")
            arm_cell = f"{arm} \u2021"
        else:
            arm_cell = arm
        out.append(
            f"| {arm_cell} | {d['accuracy'][arm]:.1%} | {d['difference'] * 100:+.1f} pt | "
            f"{lo * 100:+.1f} to {hi * 100:+.1f} pt | {d['discordant']['a_only']} / {d['discordant']['b_only']} | "
            f"{d['mcnemar_p']:.2f} | {count(d['forward_passes_mean'][arm])} | {secs(pair, arm)} |"
        )
    limits = sorted({d["limit"] for _, d in timed.values()})
    files = [f for f, _ in full.values()] + [f for f, _ in timed.values()]
    out += [
        "",
        f"Discordant counts problems only the baseline solved / only the arm solved. Time per problem is "
        f"the mean over the first {', '.join(map(str, limits))} problems of each arm. "
        + isolation_sentence([d for _, d in timed.values()]),
        "",
        "Evidence: " + ", ".join(f"`bench/eval/{f.name}`" for f in files) + ".",
    ]
    if other_weights:
        out[-3:-3] = ["", "\u2021 " + "; ".join(other_weights) + ": the accuracy difference includes the weights."]
    return "\n".join(out)


def splice(text: str, table: str, start: str = START, end: str = END) -> str:
    if start not in text or end not in text:
        return text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{table}\n{end}{tail}"


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--allow-noisy"]
    allow_noisy = len(args) != len(sys.argv) - 1
    mode = args[0] if args else "--check"
    table = render()
    logit_table, gen_table = parity()
    quality_table = quality()
    gsm8k_table = eval_table(allow_noisy)
    bad = []
    for t in TARGETS:
        if not t.exists():
            continue
        cur = t.read_text()
        new = splice(cur, table)
        new = splice(new, logit_table, "<!-- parity:start -->", "<!-- parity:end -->")
        new = splice(new, gen_table, "<!-- genparity:start -->", "<!-- genparity:end -->")
        if quality_table:
            new = splice(new, quality_table, "<!-- quality:start -->", "<!-- quality:end -->")
        if gsm8k_table:
            new = splice(new, gsm8k_table, "<!-- eval:start -->", "<!-- eval:end -->")
        if mode == "--write":
            t.write_text(new)
            print(f"bench_table: wrote {t.relative_to(ROOT)}")
        elif new != cur:
            bad.append(str(t.relative_to(ROOT)))
    if bad:
        print("bench_table: out of date with bench/results: " + ", ".join(bad), file=sys.stderr)
        print("            run: python3 scripts/bench_table.py --write", file=sys.stderr)
        return 1
    if mode == "--check":
        print("bench_table: tables match the evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
