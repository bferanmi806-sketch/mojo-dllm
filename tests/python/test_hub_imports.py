"""doctor and --offline never import huggingface_hub or open a socket (the claim in launcher/hub.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CODE = """
import socket, sys
sys.path.insert(0, {root!r})

def no_socket(*args, **kwargs):
    raise AssertionError("opened a socket")

socket.socket = no_socket
socket.create_connection = no_socket

from pathlib import Path
from launcher import doctor, probes
from launcher.hub import Hub

cache = Path({cache!r})
snap = cache / "models--o--r" / "snapshots" / "rev1"
snap.mkdir(parents=True)
(snap / "m.Q4_K_M.gguf").write_bytes(b"12345")
hub = Hub(cache, offline=True)
assert hub.list_files("o/r") == {{"m.Q4_K_M.gguf": 5}}
assert hub.cached_path("o/r", "m.Q4_K_M.gguf") == snap / "m.Q4_K_M.gguf"
assert hub.list_files("o/none") == {{}}
path, why = probes.cached_default_model(hub)
assert path is None and "local cache" in why, (path, why)
leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("huggingface_hub", "httpx2", "httpx"))
print("leaked:", leaked)
"""


def run(code: str, cache: Path) -> subprocess.CompletedProcess[str]:
    # -I: no PYTHON* variables and no user site, so the child sees only what the launcher imports.
    return subprocess.run(
        [sys.executable, "-I", "-c", code.format(root=str(ROOT), cache=str(cache))],
        capture_output=True, text=True, timeout=60, cwd=cache.parent,
    )


def test_the_offline_cache_and_doctor_never_import_the_hub_or_open_a_socket(tmp_path: Path) -> None:
    done = run(CODE, tmp_path / "hub")
    assert (done.returncode, done.stderr) == (0, ""), done.stderr
    assert done.stdout == "leaked: []\n"


def test_the_probe_reports_a_hub_import_when_one_is_forced(tmp_path: Path) -> None:
    # The check above must be able to fail: importing the hub by hand shows up in the same listing.
    done = run(CODE.replace("import socket, sys", "import huggingface_hub, socket, sys", 1), tmp_path / "hub")
    assert done.returncode == 0 and "huggingface_hub" in done.stdout
