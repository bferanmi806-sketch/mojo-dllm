"""Hugging Face access: the cache layout, file listing, download and checksum.

Cache lookups read the cache directory directly, so doctor and --offline never
import huggingface_hub or open a socket. The hub is imported only inside the
methods that talk to the network.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from launcher.errors import LauncherError


def hf_cache_dir(env: Mapping[str, str], home: Path) -> Path:
    """The directory Hugging Face tools and llama.cpp's -hf share."""
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    base = Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else home / ".cache"
    return base / "huggingface" / "hub"


def offline_requested(flag: bool, env: Mapping[str, str]) -> bool:
    return flag or env.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}


def snapshot_revision(path: Path) -> str | None:
    """The commit a cached file belongs to: the directory under snapshots/ that holds it."""
    parts = path.parts
    for i in range(len(parts) - 2):
        if parts[i] == "snapshots":
            return parts[i + 1]
    return None


class Hub:
    def __init__(self, cache_dir: Path | None = None, offline: bool = False) -> None:
        self.cache_dir = cache_dir or hf_cache_dir(os.environ, Path.home())
        self.offline = offline

    def _snapshots(self, repo: str) -> list[Path]:
        owner, _, name = repo.partition("/")
        root = self.cache_dir / f"models--{owner}--{name}" / "snapshots"
        # Newest revision first.
        return sorted(
            (p for p in root.glob("*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def cached_path(self, repo: str, filename: str) -> Path | None:
        for snap in self._snapshots(repo):
            path = snap / filename
            if path.is_file():
                return path
        return None

    def list_files(self, repo: str) -> dict[str, int]:
        if self.offline:
            files: dict[str, int] = {}
            for snap in reversed(self._snapshots(repo)):
                for p in snap.rglob("*.gguf"):
                    if p.is_file():
                        files[p.relative_to(snap).as_posix()] = p.stat().st_size
            return files
        from huggingface_hub import HfApi

        with _hub_errors(repo, self.cache_dir):
            return {
                f.path: f.size
                for f in HfApi().list_repo_tree(repo, recursive=True)
                if hasattr(f, "size")
            }

    def download(self, repo: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        with _hub_errors(repo, self.cache_dir):
            return Path(hf_hub_download(repo, filename, cache_dir=self.cache_dir))

    def expected_sha256(self, repo: str, filename: str, revision: str | None = None) -> str | None:
        """The hub's sha256 of the file at `revision` (the default branch when None)."""
        from huggingface_hub import HfApi

        with _hub_errors(repo, self.cache_dir):
            (info,) = HfApi().get_paths_info(repo, [filename], revision=revision)
        return info.lfs.sha256 if getattr(info, "lfs", None) else None


class _hub_errors:
    """Turn the hub's exceptions into launcher errors that name the next command."""

    def __init__(self, repo: str, cache_dir: Path | None = None) -> None:
        self.repo = repo
        self.cache_dir = cache_dir

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        import httpx2
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

        if isinstance(exc, GatedRepoError):
            raise LauncherError(
                f"{self.repo} is gated. Accept its terms on huggingface.co, then set HF_TOKEN "
                "to a token that can read it."
            ) from exc
        if isinstance(exc, RepositoryNotFoundError):
            raise LauncherError(
                f"{self.repo} was not found on Hugging Face, or it is private. "
                "Check the name, or set HF_TOKEN for a private repo."
            ) from exc
        # httpx2 is the hub's HTTP client. HfHubHTTPError is one of its HTTPError subclasses and
        # also an OSError, so the status errors come first; a bare OSError is a local failure.
        if isinstance(exc, HfHubHTTPError) or isinstance(exc, httpx2.HTTPStatusError):
            status = exc.response.status_code
            raise LauncherError(
                f"Hugging Face answered HTTP {status} for {self.repo}: {exc}\n"
                "Check the repo name and HF_TOKEN, or try again later."
            ) from exc
        if isinstance(exc, httpx2.HTTPError):
            raise LauncherError(
                f"could not reach Hugging Face for {self.repo}: {exc}\n"
                "Check the network, or add --offline to use files already in the cache."
            ) from exc
        if isinstance(exc, OSError):
            where = self.cache_dir if self.cache_dir is not None else "the Hugging Face cache"
            raise LauncherError(
                f"cannot write to {where} for {self.repo}: {exc.strerror or exc}\n"
                "Free space or fix the permissions there, or set HF_HOME to a writable directory."
            ) from exc
        return False
