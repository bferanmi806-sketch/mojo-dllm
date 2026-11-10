from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from launcher.errors import LauncherError
from launcher.verified import VerifiedStore, default_state_path

HOME = Path("/home/u")
SHA = "ab" * 32


def _blob(tmp_path: Path, content: bytes = b"12345") -> Path:
    blob = tmp_path / "blob"
    blob.write_bytes(content)
    return blob


def test_state_path_follows_xdg_cache_home() -> None:
    assert default_state_path({"XDG_CACHE_HOME": "/x"}, HOME) == Path("/x/mojo-dllm/verified.json")
    assert default_state_path({}, HOME) == HOME / ".cache/mojo-dllm/verified.json"


def test_a_file_with_no_record_is_unverified(tmp_path: Path) -> None:
    assert not VerifiedStore(tmp_path / "state" / "verified.json").is_verified(_blob(tmp_path))


def test_a_recorded_file_is_verified_across_store_instances(tmp_path: Path) -> None:
    state, blob = tmp_path / "state" / "verified.json", _blob(tmp_path)
    VerifiedStore(state).record(blob, SHA)
    assert VerifiedStore(state).is_verified(blob)


def test_the_record_holds_size_mtime_and_sha256_under_the_resolved_path(tmp_path: Path) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    link = tmp_path / "snap.gguf"
    link.symlink_to(blob)
    VerifiedStore(state).record(link, SHA)
    data = json.loads(state.read_text())
    assert data == {str(blob.resolve()): {"size": 5, "mtime_ns": blob.stat().st_mtime_ns, "sha256": SHA}}
    assert VerifiedStore(state).is_verified(blob)


def test_a_changed_size_or_mtime_unverifies_the_file(tmp_path: Path) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    store = VerifiedStore(state)
    store.record(blob, SHA)
    os.utime(blob, ns=(1, blob.stat().st_mtime_ns + 1_000_000))
    assert not store.is_verified(blob)
    store.record(blob, SHA)
    blob.write_bytes(b"1234567")
    assert not store.is_verified(blob)


def test_records_for_other_files_survive_a_new_record(tmp_path: Path) -> None:
    state = tmp_path / "verified.json"
    a, b = _blob(tmp_path), tmp_path / "other"
    b.write_bytes(b"zz")
    VerifiedStore(state).record(a, SHA)
    VerifiedStore(state).record(b, SHA)
    assert VerifiedStore(state).is_verified(a) and VerifiedStore(state).is_verified(b)


def test_an_unreadable_state_file_means_unverified_and_is_replaced(tmp_path: Path) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    state.write_text("{not json")
    assert not VerifiedStore(state).is_verified(blob)
    VerifiedStore(state).record(blob, SHA)
    assert VerifiedStore(state).is_verified(blob)


def test_the_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path, monkeypatch) -> None:
    state, blob = tmp_path / "s" / "verified.json", _blob(tmp_path)
    replaced: list[tuple[str, str]] = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append((str(src), str(dst))), real(src, dst))[1])
    VerifiedStore(state).record(blob, SHA)
    assert len(replaced) == 1 and replaced[0][1] == str(state) and replaced[0][0] != str(state)
    assert [p.name for p in state.parent.iterdir()] == ["verified.json"]


def test_a_vanished_file_is_unverified_not_an_error(tmp_path: Path) -> None:
    assert not VerifiedStore(tmp_path / "verified.json").is_verified(tmp_path / "gone")


def test_a_state_path_under_a_file_is_a_launcher_error_naming_the_path(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    state = blocker / "verified.json"
    with pytest.raises(LauncherError) as e:
        VerifiedStore(state).record(_blob(tmp_path), SHA)
    assert str(state) in str(e.value) and "checked again" in str(e.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions, so the write would succeed")
def test_an_unwritable_state_directory_is_a_launcher_error_naming_the_path(tmp_path: Path) -> None:
    directory = tmp_path / "locked"
    directory.mkdir()
    directory.chmod(0o500)
    try:
        with pytest.raises(LauncherError) as e:
            VerifiedStore(directory / "verified.json").record(_blob(tmp_path), SHA)
    finally:
        directory.chmod(0o700)
    assert str(directory) in str(e.value)


def test_a_failing_write_keeps_the_old_state_and_leaves_no_temp_file(tmp_path: Path, monkeypatch) -> None:
    state, blob = tmp_path / "s" / "verified.json", _blob(tmp_path)
    VerifiedStore(state).record(blob, SHA)
    before = state.read_text()
    other = tmp_path / "other"
    other.write_bytes(b"zz")

    def failing_dump(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("launcher.verified.json.dump", failing_dump)
    with pytest.raises(LauncherError, match="No space left on device"):
        VerifiedStore(state).record(other, SHA)
    assert state.read_text() == before
    assert [p.name for p in state.parent.iterdir()] == ["verified.json"]


def test_an_interrupt_during_the_write_removes_the_temp_file_and_propagates(tmp_path: Path, monkeypatch) -> None:
    state, blob = tmp_path / "s" / "verified.json", _blob(tmp_path)

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("launcher.verified.json.dump", interrupted)
    with pytest.raises(KeyboardInterrupt):
        VerifiedStore(state).record(blob, SHA)
    assert list(state.parent.iterdir()) == []


def test_a_record_uses_the_stat_taken_before_the_file_was_hashed(tmp_path: Path) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    store = VerifiedStore(state)
    before = blob.stat()
    blob.write_bytes(b"a different file, replaced after the hash was computed")
    store.record(blob, SHA, before)
    assert not store.is_verified(blob)


@pytest.mark.parametrize("content", ["[]", '"x"', "3", "null"])
def test_valid_json_that_is_not_an_object_is_treated_as_empty(tmp_path: Path, content: str) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    state.write_text(content)
    store = VerifiedStore(state)
    assert not store.is_verified(blob)
    store.record(blob, SHA)
    assert store.is_verified(blob)


@pytest.mark.parametrize("record", ["x", 3, None, ["size", 5]])
def test_a_record_that_is_not_an_object_is_unverified(tmp_path: Path, record: object) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    state.write_text(json.dumps({str(blob.resolve()): record}))
    assert not VerifiedStore(state).is_verified(blob)
