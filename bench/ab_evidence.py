"""Checks that an A/B comparison of two builds compares two different builds, shared by the GPU A/B tools.

A speedup or an "exact" claim means nothing when both sides are the same binary, when a stale
`build/mojo-dllm` carries the label of the new commit, or when the recorded card is another card. The tools
(compare_gpu.py, check_gpu_dispatch.py, scripts/compare_logits.py) call these before they measure anything.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from typing import Callable, Sequence

BINARY = "build/mojo-dllm"
GPU_SELECTOR = re.compile(r"[0-9]+|GPU-[0-9A-Fa-f-]+")


def binary_sha256(root: pathlib.Path) -> str:
    """The SHA-256 of root/build/mojo-dllm: the identity of the build that produced the evidence."""
    path = root / BINARY
    if not path.is_file():
        raise SystemExit(f"{path} does not exist: build that checkout first (scripts/build.sh)")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_sha256_or_none(root: pathlib.Path) -> str | None:
    """The hash, or None when the checkout has no binary (a tool that records but does not compare builds)."""
    return binary_sha256(root) if (root / BINARY).is_file() else None


def head_label(root: pathlib.Path, run: Callable[[list[str]], str], *, short: bool = False) -> str:
    """`HEAD`, with `-dirty` when src/ has uncommitted changes."""
    rev = ["git", "-C", str(root), "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    dirty = run(["git", "-C", str(root), "status", "--porcelain", "--", "src"]).strip()
    return run(rev).strip() + ("-dirty" if dirty else "")


def require_distinct_builds(roots: dict[str, pathlib.Path], heads: dict[str, str], hashes: dict[str, str]) -> None:
    """Refuse a base and a new that are the same checkout, the same clean commit or the same binary bytes.

    Two checkouts at one commit are allowed when one has uncommitted changes in src/ (its label ends in
    `-dirty`); the binary hash then decides.
    """
    if roots["base"] == roots["new"]:
        raise SystemExit(f"--base and --new are the same checkout ({roots['base']}): nothing to compare")
    if heads["base"] == heads["new"] and not heads["base"].endswith("-dirty"):
        raise SystemExit(f"--base and --new are at the same commit ({heads['base']}): nothing to compare")
    if hashes["base"] == hashes["new"]:
        raise SystemExit(f"the two binaries are byte-identical (sha256 {hashes['base'][:16]}): "
                         "the comparison would measure one build against itself, or one of them is stale")


def require_prompts(prompts: Sequence[object], source: str) -> None:
    """An empty prompt list would run zero iterations and write an evidence file with nothing in it."""
    if not isinstance(prompts, (list, tuple)) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        raise SystemExit(f"{source} lists no usable prompts (a non-empty list of non-empty strings is required)")


def require_gpu_selector(selector: str) -> str:
    """A GPU index or a `GPU-...` uuid: the form CUDA_VISIBLE_DEVICES and `nvidia-smi -i` both take."""
    if not GPU_SELECTOR.fullmatch(selector):
        raise SystemExit(f"--gpu must be an index (0, 1, ...) or a GPU-... uuid, got {selector!r}")
    return selector


def parse_gpu_query(text: str, selector: str) -> dict:
    """The card `nvidia-smi -i SELECTOR --query-gpu=name,uuid,pci.bus_id,...` described, as separate fields."""
    fields = [f.strip() for f in text.strip().splitlines()[0].split(",")] if text.strip() else []
    if len(fields) < 3 or not fields[1].startswith("GPU-"):
        raise SystemExit(f"nvidia-smi -i {selector} did not name a card (name, GPU-uuid, bus id): {text.strip()[:200]!r}")
    return {"name": fields[0], "uuid": fields[1], "pci_bus_id": fields[2], "query": text.strip()}

