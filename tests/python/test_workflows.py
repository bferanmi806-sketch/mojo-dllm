"""The workflows under .github/workflows and scripts/ci-local.sh, which must run the same gates.

PyYAML is not in the `test` dependency group, so the files are read line by line: a job is a key at two spaces of
indent, a step is a list item at six, and a step's keys sit at eight.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CI_LOCAL = ROOT / "scripts" / "ci-local.sh"
GPU_BUILD = "bash scripts/build.sh tests/gpu/test_gpu.mojo build/tests/test_gpu"


def read_jobs(path: Path) -> dict[str, dict]:
    """{job: {"runs-on": str, "steps": [{"name", "if", "uses", "run"}]}} for one workflow file."""
    jobs: dict[str, dict] = {}
    job: dict | None = None
    step: dict | None = None
    block_key: str | None = None
    in_jobs = False
    for raw in path.read_text().splitlines():
        if raw.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs or not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        text = raw.strip()
        if block_key is not None and step is not None and indent > 8:
            step[block_key] += text + "\n"
            continue
        block_key = None
        if indent == 2 and text.endswith(":"):
            job = {"steps": []}
            jobs[text[:-1]] = job
        elif job is not None and indent == 4 and text.startswith("runs-on:"):
            job["runs-on"] = text.split(":", 1)[1].strip()
        elif job is not None and indent == 6 and text.startswith("- "):
            step = {"run": "", "name": "", "if": "", "uses": ""}
            job["steps"].append(step)
            text = text[2:]
            indent = 8
        if step is not None and indent == 8:
            key, _, value = text.partition(":")
            if key in step:
                value = value.strip()
                if value in ("|", ">"):
                    block_key = key
                else:
                    step[key] = value
    return jobs


def md_lint_arguments(text: str) -> list[str]:
    match = re.search(r"markdownlint-cli2((?: \"[^\"]+\")+)", text)
    assert match is not None, "no markdownlint-cli2 command with quoted globs found"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_the_ci_jobs_are_read() -> None:
    jobs = read_jobs(WORKFLOWS / "ci.yml")
    assert set(jobs) >= {"mojo", "docs"} and len(jobs) == 3
    assert all(job["steps"] for job in jobs.values())


def test_the_job_that_runs_the_checks_is_named_for_what_it_does() -> None:
    jobs = read_jobs(WORKFLOWS / "ci.yml")
    assert "checks" in jobs and "hygiene" not in jobs


def test_no_step_after_the_first_command_is_skipped_when_an_earlier_one_fails() -> None:
    """Every `run` step of the checks job after the first carries `!cancelled()`, the install of markdownlint
    included: without it a failed launcher test skips the install and markdownlint then reports "command not found"."""
    steps = [s for s in read_jobs(WORKFLOWS / "ci.yml")["checks"]["steps"] if s["run"]]
    assert len(steps) >= 10
    bare = [s["name"] for s in steps[1:] if s["if"] != "${{ !cancelled() }}"]
    assert bare == []


def test_markdownlint_runs_over_the_same_files_in_both_gates() -> None:
    ci = md_lint_arguments(
        "\n".join(s["run"] for s in read_jobs(WORKFLOWS / "ci.yml")["checks"]["steps"] if "markdownlint-cli2 " in s["run"])
    )
    local = md_lint_arguments(CI_LOCAL.read_text())
    assert ci == local
    assert {"assets/*.md", ".github/*.md", "#node_modules"} <= set(ci)


def test_the_gpu_test_suite_is_built_in_both_gates() -> None:
    runs = [s["run"].strip() for s in read_jobs(WORKFLOWS / "ci.yml")["mojo"]["steps"]]
    assert GPU_BUILD in runs
    assert GPU_BUILD in CI_LOCAL.read_text()


def test_the_secret_scan_runs_on_the_image_the_other_workflows_use() -> None:
    scan = read_jobs(WORKFLOWS / "secret-scan.yml")["trufflehog"]["runs-on"]
    others = {job["runs-on"] for name in ("ci.yml", "docs.yml") for job in read_jobs(WORKFLOWS / name).values()}
    assert others == {scan} == {"ubuntu-24.04"}


def test_the_secret_scan_header_names_every_trigger() -> None:
    text = (WORKFLOWS / "secret-scan.yml").read_text()
    header = text.split("\non:", 1)[0]
    assert "pull requests" in header and "push" in header and "weekly" in header
    assert "check_hygiene.sh --tracked" in header, "the header must name the backstop for unverified secrets"


def run_hygiene(
    tmp_path: Path, content: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """check_hygiene.sh --tracked in a repository whose one tracked file holds `content`."""
    tmp_path = tmp_path / "repo"  # tmp_path itself holds the fake machine tree some fixtures put there
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "check_hygiene.sh").write_bytes((ROOT / "scripts" / "check_hygiene.sh").read_bytes())
    (tmp_path / "f.txt").write_text(content)
    for cmd in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True)
    return subprocess.run(
        ["bash", "scripts/check_hygiene.sh", "--tracked"], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )


def test_hygiene_accepts_plain_text(tmp_path: Path) -> None:
    result = run_hygiene(tmp_path, "nothing to see\npostgresql://localhost/db\n")
    assert result.returncode == 0, result.stderr


def test_hygiene_refuses_a_private_key_block(tmp_path: Path) -> None:
    result = run_hygiene(tmp_path, "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nabc\n")
    assert result.returncode == 1 and "private key" in result.stderr.lower()


def test_hygiene_refuses_a_database_url_with_a_password(tmp_path: Path) -> None:
    result = run_hygiene(tmp_path, "DATABASE_URL=postgresql://app:" + "s3cretvalue@db.example.com:5432/app\n")
    assert result.returncode == 1 and "database url" in result.stderr.lower()


def test_hygiene_refuses_when_grep_itself_fails(tmp_path: Path) -> None:
    """grep exits 2 on a bad pattern. That must stop the scan, never read as "no match"."""
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "grep").write_text("#!/bin/sh\nexit 2\n")
    (stub / "grep").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}"}
    result = run_hygiene(tmp_path, "nothing to see\n", env=env)
    assert result.returncode == 1 and "nothing was checked" in result.stderr
