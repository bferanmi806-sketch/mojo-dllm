"""Model references, choosing a file inside a repo, and getting it onto disk."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from launcher.errors import LauncherError
from launcher.ui import human_size
from launcher.verified import VerifiedStore

DEFAULT_MODEL = "llada-8b"
ALIASES = {"llada-8b": ("mradermacher/LLaDA-8B-Instruct-GGUF", "Q4_K_M")}
# The quants the binary's matrix kernels load (Q4_K and Q6_K tensors), in the order tried
# when no quant is requested. A Q8_0 or F16 file downloads fine and then fails to load.
DEFAULT_QUANTS = ("Q4_K_M", "Q4_K_S", "Q6_K")

# Owner and name never start with "." or "-", so "..", "./x" and "-x/y" are not repos.
_REPO = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*/[A-Za-z0-9_][A-Za-z0-9._-]*(:[A-Za-z0-9._-]+)?$")
_SHARD = re.compile(r"-\d{5}-of-\d{5}")


class Hub(Protocol):
    cache_dir: Path
    offline: bool

    def list_files(self, repo: str) -> dict[str, int]: ...
    def cached_path(self, repo: str, filename: str) -> Path | None: ...
    def download(self, repo: str, filename: str) -> Path: ...
    def expected_sha256(self, repo: str, filename: str) -> str | None: ...


@dataclass(frozen=True)
class ModelRef:
    path: Path | None = None
    repo: str | None = None
    quant: str | None = None


@dataclass(frozen=True)
class ModelPlan:
    repo: str
    filename: str
    size: int | None
    cached: Path | None

    @property
    def label(self) -> str:
        return f"{self.repo}/{self.filename}"


def parse_model_ref(text: str) -> ModelRef:
    path = Path(text)
    if text and path.is_file():
        return ModelRef(path=path)
    if text.endswith(".gguf"):
        raise LauncherError(f"no such file: {text}\nCheck the path, or use llada-8b to download the default model.")
    if text in ALIASES:
        repo, quant = ALIASES[text]
        return ModelRef(repo=repo, quant=quant)
    if _REPO.match(text):
        repo, _, quant = text.partition(":")
        return ModelRef(repo=repo, quant=quant or None)
    raise LauncherError(
        f"unknown model {text!r}. Accepted: a path to a .gguf file, the alias llada-8b, "
        "or owner/repo[:QUANT]."
    )


def _has_quant(name: str, quant: str) -> bool:
    # "_" may precede a quant (m_Q5_K_M) but not follow it: Q4_K is not Q4_K_S.
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(quant)}(?![A-Za-z0-9_])", name, re.I) is not None


def pick_gguf(files: Sequence[str], quant: str | None) -> str:
    ggufs = [
        f for f in files
        if f.endswith(".gguf") and not _SHARD.search(f) and not f.rsplit("/", 1)[-1].lower().startswith("mmproj")
    ]
    if not ggufs:
        raise LauncherError("the repo has no .gguf file (split shards are not supported)")
    wanted = (quant,) if quant else DEFAULT_QUANTS
    for q in wanted:
        for name in ggufs:
            if _has_quant(name, q):
                return name
    listing = "\n".join(f"  {f}" for f in ggufs)
    if quant:
        raise LauncherError(f"no GGUF file matches quant {quant}\nAvailable GGUF files:\n{listing}")
    raise LauncherError(
        "the repo has no Q4_K_M, Q4_K_S or Q6_K file, the quants mojo-dllm can load\n"
        f"Available GGUF files:\n{listing}\n"
        "Pick one with owner/repo:QUANT, for example owner/repo:Q4_K_M."
    )


def plan_model(ref: ModelRef, hub: Hub) -> ModelPlan:
    assert ref.repo is not None
    files = hub.list_files(ref.repo)
    if hub.offline and not files:
        raise LauncherError(
            f"{ref.repo} ({ref.quant or 'default quant'}) has no GGUF file in the local cache "
            f"({hub.cache_dir}), and offline mode is on.\n"
            f"On a connected machine run ./mojo-dllm pull {ref.repo}"
            + (f":{ref.quant}" if ref.quant else "")
            + ", then copy the cache over or run again without --offline."
        )
    filename = pick_gguf(list(files), ref.quant)
    return ModelPlan(ref.repo, filename, files.get(filename), hub.cached_path(ref.repo, filename))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(path: Path, plan: ModelPlan, hub: Hub, store: VerifiedStore) -> None:
    """Compare the file with the hub's sha256; record it on a match, delete it otherwise."""
    expected = hub.expected_sha256(plan.repo, plan.filename)
    if expected is None:
        _delete(path)
        raise LauncherError(f"Hugging Face reported no sha256 for {plan.label}, so the download cannot be verified.")
    actual = _sha256(path)
    if actual != expected:
        _delete(path)
        raise LauncherError(
            f"sha256 of {plan.label} does not match the hub's. The file was deleted.\n"
            "Run the command again to download it fresh."
        )
    store.record(path, actual)


def fetch_model(plan: ModelPlan, hub: Hub, out: TextIO, store: VerifiedStore) -> Path:
    size = f" ({human_size(plan.size)})" if plan.size is not None else ""
    print(f"Downloading {plan.label}{size}", file=out, flush=True)
    path = hub.download(plan.repo, plan.filename)
    _check_sha256(path, plan, hub, store)
    print("sha256 verified", file=out, flush=True)
    return path


def use_cached(plan: ModelPlan, hub: Hub, out: TextIO, store: VerifiedStore) -> tuple[Path, bool]:
    """The cached file and whether its sha256 has been checked.

    An interrupted download, or another tool, can leave an unchecked file in
    the cache. Online it is checked once. Offline there is nothing to check
    against, so it is used with a warning and reported as not verified.
    """
    path = plan.cached
    assert path is not None
    if store.is_verified(path):
        return path, True
    if hub.offline:
        print(
            f"warning: {plan.label} in the cache was not verified against its sha256, "
            "and offline mode cannot check it now",
            file=out,
        )
        return path, False
    print(f"Verifying {plan.label} against the hub's sha256 (once)", file=out, flush=True)
    _check_sha256(path, plan, hub, store)
    print("sha256 verified", file=out, flush=True)
    return path, True


def _delete(path: Path) -> None:
    path.resolve().unlink(missing_ok=True)
    path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ModelResult:
    path: Path
    verified: bool  # False only for a cached file that could not be checked offline


def already_downloaded_line(verified: bool) -> str:
    return "already downloaded" if verified else "already downloaded (not verified)"


def ensure_model_status(
    text: str, hub: Hub | None, out: TextIO, *, announce_cached: bool, store: VerifiedStore
) -> ModelResult:
    """Resolve a model reference to a local file, downloading it when needed."""
    ref = parse_model_ref(text)
    if ref.path is not None:
        return ModelResult(ref.path, True)
    assert hub is not None
    plan = plan_model(ref, hub)
    if plan.cached is not None:
        path, verified = use_cached(plan, hub, out, store)
        if announce_cached:
            print(f"{already_downloaded_line(verified)}\n{path}", file=out)
        return ModelResult(path, verified)
    path = fetch_model(plan, hub, out, store)
    if announce_cached:
        print(f"{path} ({human_size(path.stat().st_size)})", file=out)
    return ModelResult(path, True)


def ensure_model(text: str, hub: Hub | None, out: TextIO, *, announce_cached: bool, store: VerifiedStore) -> Path:
    return ensure_model_status(text, hub, out, announce_cached=announce_cached, store=store).path
