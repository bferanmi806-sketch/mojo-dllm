"""A cached file is checked against the revision it was cached at, and a mismatch deletes only what the cache owns."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from launcher.errors import LauncherError
from launcher.hub import Hub, snapshot_revision
from launcher.models import ensure_model
from launcher.verified import VerifiedStore

REPO = "o/r"
NAME = "m.Q4_K_M.gguf"
OLD, NEW = b"the file as cached at A", b"the file after upstream re-uploaded it"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHfApi:
    """Answers by revision: the hash of OLD at commit A, of NEW at main. Records every revision it was asked for."""

    revisions: list[str | None] = []

    def list_repo_tree(self, repo: str, recursive: bool = False) -> list[SimpleNamespace]:
        return [SimpleNamespace(path=NAME, size=len(NEW))]

    def get_paths_info(self, repo: str, paths: list[str], revision: str | None = None) -> list[SimpleNamespace]:
        FakeHfApi.revisions.append(revision)
        data = OLD if revision == "A" else NEW
        return [SimpleNamespace(lfs=SimpleNamespace(sha256=sha(data)))]


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeHfApi]:
    FakeHfApi.revisions = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return FakeHfApi


def layout(cache: Path, rev: str, content: bytes, *, blob: Path | None = None) -> tuple[Path, Path]:
    """The real Hugging Face layout: snapshots/<rev>/<name> is a symlink to blobs/<hash>."""
    root = cache / "models--o--r"
    target = blob or root / "blobs" / sha(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    link = root / "snapshots" / rev / NAME
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return link, target


def test_the_revision_is_the_snapshot_directory_name(tmp_path: Path) -> None:
    link, _ = layout(tmp_path, "A", OLD)
    assert snapshot_revision(link) == "A"
    nested = tmp_path / "models--o--r" / "snapshots" / "B" / "sub" / "m.gguf"
    assert snapshot_revision(nested) == "B"
    assert snapshot_revision(tmp_path / "elsewhere" / "m.gguf") is None


def test_expected_sha256_asks_for_the_given_revision(fake_api: type[FakeHfApi]) -> None:
    hub = Hub(Path("/unused"))
    assert hub.expected_sha256(REPO, NAME, revision="A") == sha(OLD)
    assert hub.expected_sha256(REPO, NAME) == sha(NEW)
    assert fake_api.revisions == ["A", None]


def test_a_file_cached_at_an_older_revision_is_kept_when_main_moved(
    fake_api: type[FakeHfApi], tmp_path: Path
) -> None:
    cache = tmp_path / "hub"
    link, blob = layout(cache, "A", OLD)
    store = VerifiedStore(tmp_path / "verified.json")
    got = ensure_model(f"{REPO}:Q4_K_M", Hub(cache), io.StringIO(), announce_cached=False, store=store)
    assert got == link and link.is_symlink() and blob.read_bytes() == OLD
    assert store.is_verified(link) and fake_api.revisions == ["A"]


def test_a_mismatch_against_the_files_own_revision_removes_the_link_and_the_blob(
    fake_api: type[FakeHfApi], tmp_path: Path
) -> None:
    cache = tmp_path / "hub"
    link, blob = layout(cache, "A", b"truncated")
    with pytest.raises(LauncherError, match="does not match"):
        ensure_model(
            f"{REPO}:Q4_K_M", Hub(cache), io.StringIO(), announce_cached=False,
            store=VerifiedStore(tmp_path / "verified.json"),
        )
    assert not link.is_symlink() and not blob.exists()


def test_a_link_to_a_target_outside_the_cache_is_removed_and_the_target_is_kept(
    fake_api: type[FakeHfApi], tmp_path: Path
) -> None:
    cache = tmp_path / "hub"
    outside = tmp_path / "llama.cpp" / "model.gguf"
    link, _ = layout(cache, "A", b"truncated", blob=outside)
    with pytest.raises(LauncherError, match="does not match"):
        ensure_model(
            f"{REPO}:Q4_K_M", Hub(cache), io.StringIO(), announce_cached=False,
            store=VerifiedStore(tmp_path / "verified.json"),
        )
    assert not link.is_symlink() and outside.read_bytes() == b"truncated"


def test_a_hub_that_lists_no_such_path_is_a_launcher_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Empty(FakeHfApi):
        def get_paths_info(self, repo, paths, revision=None):  # type: ignore[no-untyped-def]
            return []

    monkeypatch.setattr(huggingface_hub, "HfApi", Empty)
    with pytest.raises(LauncherError, match="no file m.Q4_K_M.gguf"):
        Hub(Path("/unused")).expected_sha256(REPO, NAME)


def test_a_cache_root_that_contains_snapshots_does_not_hide_the_revision(tmp_path: Path) -> None:
    """HF_HOME=/data/snapshots puts a `snapshots` component before the repo folder; the revision is the one after
    the repo folder's own `snapshots`."""
    cache = tmp_path / "snapshots" / "hub"
    link, _ = layout(cache, "REV", OLD)
    assert "snapshots" in link.parts[: link.parts.index("models--o--r")]
    assert snapshot_revision(link) == "REV"
    assert snapshot_revision(cache / "snapshots" / "hub" / "m.gguf") is None, "no repo folder, no revision"



def test_a_file_in_the_blobs_folder_of_the_repo_has_no_revision(tmp_path: Path) -> None:
    _, blob = layout(tmp_path, "A", OLD)
    assert blob.parent.name == "blobs" and snapshot_revision(blob) is None
