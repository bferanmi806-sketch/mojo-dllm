from __future__ import annotations

import pytest

from launcher.errors import LauncherError
from launcher.probes import read_mojo_pin, read_project_version


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
