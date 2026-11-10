"""Which cached model files have passed their sha256 check.

The Hugging Face cache cannot say whether a file was ever verified: an
interrupted run, or another tool, can leave one there. After a check passes,
the file's size and mtime are recorded here, and a file whose record is
missing or stale is hashed again before it is used.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from launcher.errors import LauncherError


def default_state_path(env: Mapping[str, str], home: Path) -> Path:
    base = Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else home / ".cache"
    return base / "mojo-dllm" / "verified.json"


def default_store() -> VerifiedStore:
    return VerifiedStore(default_state_path(os.environ, Path.home()))


class VerifiedStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, dict[str, object]]:
        # An unreadable file is the same as no records: every file is hashed again.
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _key(blob: Path) -> str:
        return str(blob.resolve())

    def is_verified(self, blob: Path) -> bool:
        try:
            stat = blob.stat()
        except OSError:
            return False
        record = self._load().get(self._key(blob))
        return (
            isinstance(record, dict)
            and record.get("size") == stat.st_size
            and record.get("mtime_ns") == stat.st_mtime_ns
        )

    def record(self, blob: Path, sha256: str, stat: os.stat_result | None = None) -> None:
        """Remember that `blob` matched its sha256.

        Pass the stat taken before the file was hashed: a file replaced while it was being
        hashed then fails is_verified instead of being recorded with its new size and mtime.
        """
        try:
            self._write(blob, sha256, stat if stat is not None else blob.stat())
        except OSError as e:
            raise LauncherError(
                f"cannot record the check of {blob.name} in {self.path}: {e.strerror or e}. "
                "The file matched its sha256, and it will be checked again on the next run. "
                f"Make {self.path.parent} writable, or set XDG_CACHE_HOME to a directory that is."
            ) from e

    def _write(self, blob: Path, sha256: str, stat: os.stat_result) -> None:
        records = self._load()
        records[self._key(blob)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".verified-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
