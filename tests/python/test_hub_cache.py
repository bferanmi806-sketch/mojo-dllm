from __future__ import annotations

import os
from pathlib import Path

from launcher.hub import Hub, hf_cache_dir

HOME = Path("/home/u")


def test_cache_dir_precedence() -> None:
    assert hf_cache_dir({"HF_HUB_CACHE": "/a", "HF_HOME": "/b"}, HOME) == Path("/a")
    assert hf_cache_dir({"HF_HOME": "/b"}, HOME) == Path("/b/hub")
    assert hf_cache_dir({"XDG_CACHE_HOME": "/x"}, HOME) == Path("/x/huggingface/hub")
    assert hf_cache_dir({}, HOME) == HOME / ".cache/huggingface/hub"


def _snapshot(cache: Path, repo: str, rev: str, name: str, mtime: float, content: bytes = b"12345") -> Path:
    owner, _, model = repo.partition("/")
    snap = cache / f"models--{owner}--{model}" / "snapshots" / rev
    blob = cache / f"models--{owner}--{model}" / "blobs" / f"blob-{rev}-{name}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    snap.mkdir(parents=True, exist_ok=True)
    link = snap / name
    link.symlink_to(blob)
    # _snapshots sorts on the mtime of the snapshot directory, so that is the one to set, after the link
    # exists (creating the link moves it). The blob's own mtime plays no part.
    os.utime(snap, (mtime, mtime))
    return link


def test_cache_lookup_reads_the_huggingface_layout(tmp_path: Path) -> None:
    link = _snapshot(tmp_path, "o/r", "rev1", "m.Q4_K_M.gguf", 10)
    hub = Hub(tmp_path)
    assert hub.cached_path("o/r", "m.Q4_K_M.gguf") == link
    assert hub.cached_path("o/r", "other.gguf") is None
    assert hub.cached_path("o/none", "m.Q4_K_M.gguf") is None


def test_offline_listing_is_what_the_cache_holds(tmp_path: Path) -> None:
    _snapshot(tmp_path, "o/r", "rev1", "m.Q4_K_M.gguf", 10)
    _snapshot(tmp_path, "o/r", "rev2", "m.Q8_0.gguf", 20)
    assert Hub(tmp_path, offline=True).list_files("o/r") == {
        "m.Q4_K_M.gguf": 5,
        "m.Q8_0.gguf": 5,
    }


# Spread-out values: two directories made microseconds apart never decide the order by chance.
OLD_MTIME, NEW_MTIME = 1_000_000_000, 1_700_000_000


def test_the_newest_revision_wins(tmp_path: Path) -> None:
    _snapshot(tmp_path, "o/r", "old", "m.gguf", OLD_MTIME)
    new = _snapshot(tmp_path, "o/r", "new", "m.gguf", NEW_MTIME)
    assert Hub(tmp_path).cached_path("o/r", "m.gguf") == new


def test_the_newest_revision_wins_whatever_order_the_directories_were_made_in(tmp_path: Path) -> None:
    # The newer snapshot is created first, so neither creation order nor name order can give the answer.
    new = _snapshot(tmp_path, "o/r", "a-new", "m.gguf", NEW_MTIME)
    _snapshot(tmp_path, "o/r", "z-old", "m.gguf", OLD_MTIME)
    assert Hub(tmp_path).cached_path("o/r", "m.gguf") == new


def test_offline_the_newest_revisions_size_wins_for_a_name_both_hold(tmp_path: Path) -> None:
    _snapshot(tmp_path, "o/r", "a-new", "m.gguf", NEW_MTIME, b"1234567890")
    _snapshot(tmp_path, "o/r", "z-old", "m.gguf", OLD_MTIME, b"123")
    assert Hub(tmp_path, offline=True).list_files("o/r") == {"m.gguf": 10}


def test_a_dangling_link_is_not_a_cached_file(tmp_path: Path) -> None:
    link = _snapshot(tmp_path, "o/r", "rev1", "m.gguf", OLD_MTIME)
    link.resolve().unlink()
    hub = Hub(tmp_path, offline=True)
    assert link.is_symlink() and not link.exists()
    assert hub.cached_path("o/r", "m.gguf") is None and hub.list_files("o/r") == {}


def test_a_repo_with_no_cache_directory_is_empty(tmp_path: Path) -> None:
    hub = Hub(tmp_path / "absent", offline=True)
    assert hub.list_files("o/r") == {} and hub.cached_path("o/r", "m.gguf") is None


def test_every_file_the_offline_listing_names_is_in_the_cache(tmp_path: Path) -> None:
    # plan_model picks from this listing, so an offline plan always has a cached file.
    _snapshot(tmp_path, "o/r", "rev1", "m.Q4_K_M.gguf", 10)
    _snapshot(tmp_path, "o/r", "rev2", "m.Q8_0.gguf", 20)
    _snapshot(tmp_path, "o/r", "rev2", "m.Q2_K.gguf", 20)
    hub = Hub(tmp_path, offline=True)
    listed = hub.list_files("o/r")
    assert listed and all(hub.cached_path("o/r", name) is not None for name in listed)
