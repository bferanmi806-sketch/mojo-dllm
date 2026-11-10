"""One way to write a result file: refuse to replace a dated file by accident, and never leave a partial one.

Result files are named by date, so a second run on the same day used to overwrite the first with its first
partial write, and a kill during `write_text` left truncated JSON. `refuse_existing` is called before the long
work starts (a refusal after an hour of measuring discards the hour); `write_atomic` writes a temporary file in
the same directory and renames it over the target, so a reader sees the old file or the whole new one.
"""

from __future__ import annotations

import os
import pathlib
import tempfile


def refuse_existing(path: pathlib.Path, force: bool = False) -> None:
    """Exit when `path` exists, naming it, unless `force` (the caller's --force) says to replace it."""
    if path.exists() and not force:
        raise SystemExit(f"{path} exists: it may be committed evidence. Move it aside, or pass --force to replace it.")


def write_atomic(path: pathlib.Path, text: str) -> None:
    """Replace `path` with `text` in one step; a failure leaves the old file (or none) and no temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def write_new(path: pathlib.Path, text: str, force: bool = False) -> None:
    """`refuse_existing`, then `write_atomic`."""
    refuse_existing(path, force)
    write_atomic(path, text)
