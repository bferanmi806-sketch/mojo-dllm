from __future__ import annotations

import errno
import re
import tomllib
from pathlib import Path

import httpx2
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
import pytest

from launcher.errors import LauncherError
from launcher.hub import _hub_errors

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("connection refused"),
        httpx2.ReadTimeout("timed out"),
    ],
    ids=["connect", "timeout"],
)
def test_network_failures_name_the_offline_flag(error: Exception) -> None:
    with pytest.raises(LauncherError) as e:
        with _hub_errors("a/b"):
            raise error
    assert "could not reach Hugging Face for a/b" in str(e.value)
    assert "--offline" in str(e.value)


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("GET", "https://huggingface.co/api/x"))


def _raised(error: Exception) -> str:
    with pytest.raises(LauncherError) as e:
        with _hub_errors("a/b", cache_dir=Path("/cache/hub")):
            raise error
    return str(e.value)


def test_a_gated_repo_names_the_token_and_not_the_network() -> None:
    msg = _raised(GatedRepoError("gated", response=_response(403)))
    assert "a/b is gated" in msg and "HF_TOKEN" in msg and "--offline" not in msg


def test_a_missing_repo_says_it_was_not_found() -> None:
    msg = _raised(RepositoryNotFoundError("gone", response=_response(404)))
    assert "was not found" in msg and "--offline" not in msg


@pytest.mark.parametrize("status", [401, 404, 500, 503])
def test_an_http_status_error_reports_its_status_code(status: int) -> None:
    msg = _raised(HfHubHTTPError("boom", response=_response(status)))
    assert f"HTTP {status}" in msg and "a/b" in msg and "--offline" not in msg


def test_a_full_disk_names_the_cache_and_not_the_network() -> None:
    msg = _raised(OSError(errno.ENOSPC, "No space left on device"))
    assert "cannot write to /cache/hub" in msg and "No space left on device" in msg
    assert "network" not in msg and "--offline" not in msg


def test_a_read_only_cache_names_the_cache() -> None:
    msg = _raised(PermissionError(errno.EACCES, "Permission denied"))
    assert "cannot write to /cache/hub" in msg and "Permission denied" in msg


def test_other_exceptions_pass_through() -> None:
    with pytest.raises(KeyError):
        with _hub_errors("a/b"):
            raise KeyError("x")


def test_the_test_group_pins_the_hub_version_the_script_uses() -> None:
    script = (ROOT / "mojo-dllm").read_text()
    in_script = re.search(r'"(huggingface-hub==[^"]+)"', script)
    assert in_script is not None, "mojo-dllm has no huggingface-hub pin"
    group = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]["test"]
    assert in_script.group(1) in group
