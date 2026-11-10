from __future__ import annotations

from pathlib import Path

import pytest

from launcher.errors import LauncherError
from launcher import probes
from launcher.probes import read_mojo_pin, read_project_version
from launcher.verified import VerifiedStore


def test_the_pin_is_read_from_pyproject() -> None:
    assert read_mojo_pin('dependencies = [\n    "mojo==1.1.0",\n    "max-core==26.6.0",\n]') == "1.1.0"


def test_a_missing_pin_is_a_launcher_error_naming_the_file() -> None:
    with pytest.raises(LauncherError, match=r"pyproject\.toml has no mojo==X pin"):
        read_mojo_pin('dependencies = ["max-core==26.6.0"]')


def test_the_version_is_read_from_pyproject() -> None:
    assert read_project_version('[project]\nname = "x"\nversion = "0.1.0"\n') == "0.1.0"


def test_another_table_cannot_override_the_version() -> None:
    text = '[tool.x]\nversion = "9"\n[project]\nname = "x"\nversion = "0.1.0"\n'
    assert read_project_version(text) == "0.1.0"


@pytest.mark.parametrize("text", ['[project]\nname = "x"\n', '', 'not = [toml'])
def test_a_missing_version_is_a_launcher_error_naming_the_file(text: str) -> None:
    with pytest.raises(LauncherError, match=r"pyproject\.toml has no \[project\] version"):
        read_project_version(text)


def _machine_without_tools(monkeypatch: pytest.MonkeyPatch, cached: Path | None, store: VerifiedStore) -> None:
    monkeypatch.setattr(probes.shutil, "which", lambda name: None)
    monkeypatch.setattr(probes, "cached_default_model", lambda hub: cached)
    monkeypatch.setattr(probes, "default_store", lambda: store)


def test_real_probes_says_whether_the_cached_model_was_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    store = VerifiedStore(tmp_path / "verified.json")
    _machine_without_tools(monkeypatch, model, store)
    unchecked = probes.real_probes()
    assert unchecked.cached_model == model and unchecked.verified is False
    store.record(model, "0" * 64)
    assert probes.real_probes().verified is True


def test_real_probes_with_no_cached_model_is_not_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _machine_without_tools(monkeypatch, None, VerifiedStore(tmp_path / "verified.json"))
    got = probes.real_probes()
    assert got.cached_model is None and got.verified is False
