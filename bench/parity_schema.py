"""The shape of a bench/parity/*.json document and the invariants its numbers must keep.

`bench/parity.py` writes these files, `scripts/bench_table.py` reads them, and a test runs `validate_parity` over every
committed one, so the reader accepts what the writer can produce and nothing else. The fields the tool cannot know
itself (a GGUF metadata audit, the model's source revision, the reference build flags, a smoke run) are optional and
carried by hand; a rerun of the tool leaves them out, and the validator accepts that.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, TypedDict


class Metrics(TypedDict):
    """One `parity.compare` result: logits of mojo-dllm against a reference, row by row."""

    rows: int
    min_cosine: float
    median_cosine: float
    masked_median_cosine: float
    max_rel_rms: float
    median_rel_rms: float
    argmax_agree: int
    mean_top5_overlap: float
    worst_rows: list[int]


class DgMetrics(Metrics):
    """`Metrics` plus the worst relative error of a row's largest logit (DiffusionGemma)."""

    top1_max_rel_err: float


class Bound(TypedDict):
    """The bound a DiffusionGemma comparison must stay inside, taken from the LLaDA evidence."""

    min_cosine: float
    argmax_fraction: float
    mean_top5_overlap: float
    max_rel_rms: float
    max_top1_rel_err: float
    source: str


class Generation(TypedDict):
    prompt: str
    gen_length: int
    block_length: int
    steps: int
    matching_tokens: int
    first_divergence: int | None
    mojo_text: str


class ParityDoc(TypedDict):
    """The keys every parity file has. Optional ones are listed in OPTIONAL_KEYS and checked when present."""

    date: str
    model: str
    versions: dict[str, str]
    reference: str
    logits: list[dict[str, Any]]
    generation: list[Generation]


METRIC_KEYS = tuple(Metrics.__annotations__)
VIOLATION_KEYS = ("min_cosine", "argmax_agree", "rows", "mean_top5_overlap", "max_rel_rms", "top1_max_rel_err")
BOUND_KEYS = tuple(k for k in Bound.__annotations__ if k != "source")   # the numbers a comparison reads
REQUIRED_KEYS = tuple(ParityDoc.__annotations__)
OPTIONAL_KEYS = (
    "arch", "device", "reference2", "threads", "machine", "measurement_scope", "metadata_audit", "model_source",
    "reference_build", "smoke", "model_sha256", "bound",
)
ARCHES = ("llada", "dream", "diffusion-gemma")
EPS = 1e-6      # a cosine of 1.0000001 is rounding, not a violation


class ParityError(ValueError):
    """A parity document, bound or metrics dict that breaks the schema; the message lists every problem."""


def require_keys(d: object, keys: tuple[str, ...], what: str) -> None:
    """Raise ParityError naming the missing keys, so a typo fails at once and not after a long run."""
    if not isinstance(d, dict):
        raise ParityError(f"{what} is not a dict")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ParityError(f"{what} lacks {missing}; it has {sorted(d)}")


def _int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def metric_problems(m: object, where: str, full: bool = True) -> list[str]:
    """Problems with one metrics dict; `full` asks for every `Metrics` key, otherwise only those a bound reads."""
    try:
        require_keys(m, METRIC_KEYS if full else VIOLATION_KEYS[:-1] if False else METRIC_KEYS, where)
    except ParityError as e:
        return [str(e)]
    assert isinstance(m, dict)
    out: list[str] = []
    rows = m["rows"]
    if not _int(rows) or rows < 1:
        return [f"{where}: rows is {rows!r}, not an integer of at least 1"]
    if not _int(m["argmax_agree"]) or not 0 <= m["argmax_agree"] <= rows:
        out.append(f"{where}: argmax_agree {m['argmax_agree']!r} is outside 0..rows ({rows})")
    for k in ("min_cosine", "median_cosine", "masked_median_cosine", "max_rel_rms", "median_rel_rms",
              "mean_top5_overlap"):
        if not _num(m[k]):
            out.append(f"{where}: {k} is {m[k]!r}, not a finite number")
    if out:
        return out
    for k in ("min_cosine", "median_cosine", "masked_median_cosine"):
        if not -1 - EPS <= m[k] <= 1 + EPS:
            out.append(f"{where}: {k} {m[k]} is not a cosine")
    if m["min_cosine"] > m["median_cosine"] + EPS or m["min_cosine"] > m["masked_median_cosine"] + EPS:
        out.append(f"{where}: min_cosine {m['min_cosine']} is above a median ({m['median_cosine']}, "
                   f"{m['masked_median_cosine']})")
    if not 0 <= m["mean_top5_overlap"] <= 5:
        out.append(f"{where}: mean_top5_overlap {m['mean_top5_overlap']} is outside 0..5")
    if m["median_rel_rms"] < 0 or m["max_rel_rms"] < m["median_rel_rms"] - EPS:
        out.append(f"{where}: max_rel_rms {m['max_rel_rms']} is below median_rel_rms {m['median_rel_rms']} or negative")
    worst = m["worst_rows"]
    if (not isinstance(worst, list) or not 1 <= len(worst) <= 3
            or not all(_int(i) and 0 <= i < rows for i in worst)):
        out.append(f"{where}: worst_rows {worst!r} is not 1 to 3 row indexes below {rows}")
    return out


def generation_problems(g: object, where: str) -> list[str]:
    try:
        require_keys(g, ("prompt", "gen_length", "block_length", "steps", "matching_tokens", "first_divergence",
                         "mojo_text"), where)
    except ParityError as e:
        return [str(e)]
    assert isinstance(g, dict)
    n, same, first = g["gen_length"], g["matching_tokens"], g["first_divergence"]
    if not _int(n) or n < 1:
        return [f"{where}: gen_length is {n!r}, not an integer of at least 1"]
    out: list[str] = []
    if not _int(same) or not 0 <= same <= n:
        return [f"{where}: matching_tokens {same!r} is outside 0..gen_length ({n})"]
    if same == n and first is not None:
        out.append(f"{where}: first_divergence is {first!r} although all {n} tokens match")
    if same < n and (not _int(first) or not 0 <= first < n or first > same):
        out.append(f"{where}: first_divergence {first!r} does not fit {same} matching of {n} tokens "
                   "(it must be an index of at most matching_tokens)")
    mine, ref = g.get("mojo_generated_tokens"), g.get("reference_generated_tokens")
    if (mine is None) != (ref is None):
        out.append(f"{where}: only one of mojo_generated_tokens and reference_generated_tokens is present")
    elif mine is not None:
        if not (isinstance(mine, list) and isinstance(ref, list) and len(mine) == len(ref) == n):
            out.append(f"{where}: the generated token lists are not {n} long each")
        else:
            equal = sum(1 for a, b in zip(mine, ref) if a == b)
            where_first = next((i for i, (a, b) in enumerate(zip(mine, ref)) if a != b), None)
            if equal != same:
                out.append(f"{where}: matching_tokens is {same} but the token lists agree in {equal} places")
            if where_first != first:
                out.append(f"{where}: first_divergence is {first!r} but the token lists first differ at {where_first!r}")
    return out


def parity_problems(doc: object) -> list[str]:
    """Every way `doc` breaks the schema or an invariant of its numbers; empty when it is a valid parity file."""
    try:
        require_keys(doc, REQUIRED_KEYS, "the document")
    except ParityError as e:
        return [str(e)]
    assert isinstance(doc, dict)
    out: list[str] = []
    try:
        dt.datetime.fromisoformat(doc["date"])
    except (TypeError, ValueError):
        out.append(f"date {doc['date']!r} is not an ISO timestamp")
    if not isinstance(doc["model"], str) or not doc["model"]:
        out.append("model is not a non-empty string")
    versions = doc["versions"]
    if not isinstance(versions, dict) or not isinstance(versions.get("mojo-dllm"), str) or not versions["mojo-dllm"]:
        out.append("versions has no mojo-dllm build")
    if not isinstance(doc["reference"], str) or not doc["reference"]:
        out.append("reference is not a non-empty string")
    if doc.get("arch", "llada") not in ARCHES:
        out.append(f"arch {doc.get('arch')!r} is not one of {ARCHES}")
    if doc.get("device", "cpu") not in ("cpu", "gpu"):
        out.append(f"device {doc.get('device')!r} is not cpu or gpu")
    extra = sorted(set(doc) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if extra:
        out.append(f"unknown keys {extra}: add them to parity_schema.OPTIONAL_KEYS if the writer produces them")
    if not isinstance(doc["logits"], list) or not isinstance(doc["generation"], list):
        return out + ["logits and generation must be lists"]
    if not doc["logits"]:
        out.append("logits is empty")
    for i, entry in enumerate(doc["logits"]):
        out += metric_problems(entry, f"logits[{i}]")
        if isinstance(entry, dict):
            if not isinstance(entry.get("prompt"), str) or not _int(entry.get("seq_len")):
                out.append(f"logits[{i}] has no prompt string or no integer seq_len")
            if "reference2" in doc:
                for k in ("vs_ref2", "ref_vs_ref2"):
                    out += metric_problems(entry.get(k), f"logits[{i}].{k}")
    for i, g in enumerate(doc["generation"]):
        out += generation_problems(g, f"generation[{i}]")
    return out


def validate_parity(doc: object) -> None:
    """Raise ParityError listing every problem when `doc` is not a valid parity file."""
    problems = parity_problems(doc)
    if problems:
        raise ParityError("; ".join(problems))
