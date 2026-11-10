"""The ./mojo-dllm script: its dependency pin, its lock, and that it starts."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "mojo-dllm"
LOCK = ROOT / "mojo-dllm.lock"


def header_pin() -> str:
    match = re.search(r'^# dependencies = \["huggingface-hub==([^"]+)"\]$', SCRIPT.read_text(), re.M)
    assert match is not None, "mojo-dllm has no `# dependencies = [\"huggingface-hub==X\"]` line"
    return match.group(1)


def locked_packages() -> dict[str, dict]:
    assert LOCK.is_file(), "mojo-dllm.lock is missing: run `uv lock --script mojo-dllm`"
    return {p["name"]: p for p in tomllib.loads(LOCK.read_text())["package"]}


def test_the_script_pin_is_the_version_the_tests_run_with() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert f"huggingface-hub=={header_pin()}" in project["dependency-groups"]["test"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    tested = next(p["version"] for p in lock["package"] if p["name"] == "huggingface-hub")
    assert tested == header_pin(), "uv.lock tests another huggingface-hub than the script ships"


def test_the_script_lock_pins_the_same_hub_version() -> None:
    assert locked_packages()["huggingface-hub"]["version"] == header_pin()
    manifest = tomllib.loads(LOCK.read_text())["manifest"]["requirements"]
    assert manifest == [{"name": "huggingface-hub", "specifier": f"=={header_pin()}"}]


def test_every_locked_package_carries_hashes() -> None:
    # The point of the lock: a transitive release that changes is refused, not installed.
    for name, package in locked_packages().items():
        artifacts = [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]
        assert artifacts, f"{name} has no artifact in mojo-dllm.lock"
        assert all(a["hash"].startswith("sha256:") for a in artifacts), name


def test_the_script_starts_and_prints_the_project_version() -> None:
    # `python3 -I` drops the script's directory from sys.path, so `from launcher.cli import main` could
    # not work under it; -E -s still ignores PYTHON* variables and the user site. It runs from another
    # directory so nothing depends on the working directory.
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    done = subprocess.run(
        [sys.executable, "-E", "-s", str(SCRIPT), "--version"],
        capture_output=True, text=True, timeout=60, cwd=ROOT.parent,
    )
    assert (done.returncode, done.stdout, done.stderr) == (0, f"mojo-dllm {version}\n", "")
