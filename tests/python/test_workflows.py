"""The workflows under .github/workflows and scripts/ci-local.sh, which must run the same gates.

PyYAML is not in the `test` dependency group, so the files are read line by line: a job is a key at two spaces of
indent, a step is a list item at six, and a step's keys sit at eight.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

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


def test_the_job_keeps_the_name_branch_protection_requires() -> None:
    """`main` requires a status check named hygiene. A renamed job never reports it and blocks every merge."""
    text = (WORKFLOWS / "ci.yml").read_text()
    jobs = read_jobs(WORKFLOWS / "ci.yml")
    assert set(jobs) == {"hygiene", "mojo", "docs"}
    before = text.split("\n  hygiene:", 1)[0].splitlines()[-1]
    assert before.lstrip().startswith("#") and "required status check" in before


def test_no_step_after_the_first_command_is_skipped_when_an_earlier_one_fails() -> None:
    """Every `run` step of the hygiene job after the first carries `!cancelled()`, the install of markdownlint
    included: without it a failed launcher test skips the install and markdownlint then reports "command not found"."""
    steps = [s for s in read_jobs(WORKFLOWS / "ci.yml")["hygiene"]["steps"] if s["run"]]
    assert len(steps) >= 10
    bare = [s["name"] for s in steps[1:] if s["if"] != "${{ !cancelled() }}"]
    assert bare == []


def test_markdownlint_runs_over_the_same_files_in_both_gates() -> None:
    ci = md_lint_arguments(
        "\n".join(s["run"] for s in read_jobs(WORKFLOWS / "ci.yml")["hygiene"]["steps"] if "markdownlint-cli2 " in s["run"])
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
    tmp_path: Path, content: str, env: dict[str, str] | None = None, name: str = "f.txt", mode: str = "--tracked"
) -> subprocess.CompletedProcess[str]:
    """check_hygiene.sh --tracked in a repository whose one tracked file holds `content`."""
    tmp_path = tmp_path / "repo"  # tmp_path itself holds the fake machine tree some fixtures put there
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "check_hygiene.sh").write_bytes((ROOT / "scripts" / "check_hygiene.sh").read_bytes())
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_text(content)
    for cmd in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True)
    return subprocess.run(
        ["bash", "scripts/check_hygiene.sh", mode], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
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


# ---- ci.yml against ci-local.sh -------------------------------------------------------------------------------

# Commands CI runs that ci-local.sh does not: they set up the runner, or finish a step ci-local.sh does another way.
CI_ONLY = (
    "npm install --global markdownlint-cli2@",
    "uv sync --frozen",
    "git diff --exit-code",
    "set -euo pipefail",
    "curl ",
    "echo ",
    "tar ",
)
# Commands ci-local.sh runs that ci.yml does not (the runner has no GPU, so CI only builds the suite).
LOCAL_ONLY = ("scripts/run-gpu-tests.sh",)
# What names a gate: a script of the repository or the tool a step runs.
GATE = re.compile(
    r"(?:scripts|bench)/[\w.-]+\.(?:sh|py)|shellcheck|yamllint|markdownlint-cli2|pytest|mdbook build|uv lock --script|mojo format"
)


def ci_commands() -> list[str]:
    """Every command of every `run:` step in ci.yml, one per `&&` part and per line, with `uv run --frozen mojo`
    written as ci-local.sh's $MOJO."""
    commands = []
    for job in read_jobs(WORKFLOWS / "ci.yml").values():
        for step in job["steps"]:
            for line in step["run"].replace("\\\n", " ").splitlines():
                for part in line.split("&&"):
                    part = part.strip().replace("uv run --frozen mojo", "$MOJO")
                    if part and not part.startswith("#"):
                        commands.append(part)
    return commands


def gates(text: str) -> set[str]:
    """The gates a text names. ci-local.sh's own path (its usage line) is not one."""
    return set(GATE.findall(text.replace("$MOJO", "mojo"))) - {"scripts/ci-local.sh"}


def test_every_command_ci_runs_is_in_ci_local() -> None:
    local = " ".join(CI_LOCAL.read_text().split())
    checked = [c for c in ci_commands() if not c.startswith(CI_ONLY)]
    assert len(checked) >= 15, checked
    # ci-local.sh lays a long command out over lines; both sides compare with the whitespace collapsed.
    missing = [c for c in checked if " ".join(c.split()) not in local]
    assert missing == []


def test_ci_local_runs_no_gate_that_ci_lacks() -> None:
    ci = gates(" ".join(ci_commands()))
    local = gates(re.sub(r'step "[^"]*"', "step", CI_LOCAL.read_text()))      # a step label names no gate
    extra = {g for g in local - ci if not g.startswith(LOCAL_ONLY)}
    assert extra == set()
    assert gates(" ".join(ci_commands())) - local == set()


# ---- scripts/ci-local.sh itself, with every tool replaced by a stub ---------------------------------------------

STUB = """#!/bin/bash
# Records the call, and fails when the call contains $STUB_FAIL.
echo "$(basename "$0") $*" >> "$STUB_LOG"
if [ -n "${STUB_FAIL:-}" ] && [[ "$(basename "$0") $*" == *"$STUB_FAIL"* ]]; then exit 1; fi
if [ "$(basename "$0")" = nvidia-smi ]; then printf '%s' "${STUB_GPU:-}"; fi
exit 0
"""
STUB_TOOLS = ("uv", "fakemojo", "shellcheck", "markdownlint-cli2", "mdbook", "gh", "nvidia-smi", "python3")
STUB_SCRIPTS = (
    "check_hygiene.sh",
    "check_format.sh",
    "run_extra_group_tests.sh",
    "build.sh",
    "run-tests.sh",
    "check_no_gpu.sh",
    "check_visual.sh",
    "run-gpu-tests.sh",
    "check_gpu_test_count.sh",
    "verify-action-pins.sh",
)
REAL_TOOLS = ("bash", "basename", "dirname", "git", "sha256sum", "grep", "timeout", "cat", "sed")


def run_ci_local(
    tmp_path: Path, *args: str, fail: str = "", gpu: str = "", without: tuple[str, ...] = ()
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """scripts/ci-local.sh copied into a scratch repository whose scripts and tools are stubs. Returns the result
    and the stub log, one line per call."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "ci-local.sh").write_bytes(CI_LOCAL.read_bytes())
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name in STUB_SCRIPTS:
        (repo / "scripts" / name).write_text(STUB)
    for name in STUB_TOOLS:
        if name not in without:
            (stubs / name).write_text(STUB)
            (stubs / name).chmod(0o755)
    for name in REAL_TOOLS:
        (stubs / name).symlink_to(subprocess.run(["which", name], capture_output=True, text=True, check=True).stdout.strip())
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    log = tmp_path / "calls.log"
    log.touch()
    env = {
        "PATH": str(stubs),
        "HOME": str(tmp_path / "home"),
        "MOJO": "fakemojo",
        "STUB_LOG": str(log),
        "STUB_FAIL": fail,
        "STUB_GPU": gpu,
    }
    result = subprocess.run(
        ["bash", "scripts/ci-local.sh", *args], cwd=repo, env=env, capture_output=True, text=True, check=False, timeout=60
    )
    return result, log.read_text().splitlines()


def summary(result: subprocess.CompletedProcess[str]) -> list[str]:
    return result.stdout.split("==> summary", 1)[1].splitlines()[1:]


def test_ci_local_passes_when_every_step_passes(tmp_path: Path) -> None:
    result, calls = run_ci_local(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = summary(result)
    assert lines[-1] == "ci-local: 18 steps passed, 1 skipped"
    assert "  PASS  launcher tests" in lines and not [l for l in lines if "FAIL" in l]
    assert len(calls) >= 17


def test_without_a_gpu_ci_local_builds_the_gpu_suite_and_records_a_skip(tmp_path: Path) -> None:
    result, calls = run_ci_local(tmp_path)
    assert "build.sh tests/gpu/test_gpu.mojo build/tests/test_gpu" in calls
    assert not any(c.startswith("run-gpu-tests.sh") for c in calls)
    lines = summary(result)
    assert "  PASS  GPU test suite builds" in lines
    assert any(l.startswith("  SKIP  GPU tests") for l in lines)
    assert lines[-1] == "ci-local: 18 steps passed, 1 skipped"


def test_the_gpu_test_count_is_compared_with_its_pin_on_every_machine(tmp_path: Path) -> None:
    for gpu in ("", "GPU 0: NVIDIA GeForce RTX 3050 (UUID: GPU-x)\n"):
        result, calls = run_ci_local(tmp_path / f"gpu{bool(gpu)}", gpu=gpu)
        assert "check_gpu_test_count.sh " in [c.split(" ")[0] + " " for c in calls], calls
        assert "  PASS  GPU test count matches its pin" in summary(result)
    failing, _ = run_ci_local(tmp_path / "fail", fail="check_gpu_test_count.sh")
    assert failing.returncode == 1 and "  FAIL  GPU test count matches its pin" in summary(failing)


def test_with_a_gpu_ci_local_runs_the_gpu_suite_and_does_not_skip(tmp_path: Path) -> None:
    result, calls = run_ci_local(tmp_path, gpu="GPU 0: NVIDIA GeForce RTX 3050 (UUID: GPU-x)\n")
    assert any(c.startswith("run-gpu-tests.sh") for c in calls)
    lines = summary(result)
    assert "  PASS  GPU tests (count pinned)" in lines and lines[-1] == "ci-local: 18 steps passed, 0 skipped"


def test_the_fixture_gate_is_a_step_and_its_failure_is_reported(tmp_path: Path) -> None:
    result, calls = run_ci_local(tmp_path)
    assert "  PASS  fixtures match their generators" in summary(result)
    assert any(c.startswith("uv ") and "scripts/check_fixtures.py" in c for c in calls)
    failing, _ = run_ci_local(tmp_path / "fail", fail="scripts/check_fixtures.py")
    assert failing.returncode == 1 and "  FAIL  fixtures match their generators" in summary(failing)


def test_a_failing_step_does_not_stop_the_later_ones_and_the_exit_is_1(tmp_path: Path) -> None:
    result, calls = run_ci_local(tmp_path, fail="pytest tests/python")
    assert result.returncode == 1
    lines = summary(result)
    assert "  FAIL  launcher tests" in lines
    assert "  PASS  docs build + links" in lines, "steps after the failure must still run"
    assert lines[-1] == "ci-local: 1 step(s) failed"


def test_a_missing_tool_is_a_failure_and_not_a_skip(tmp_path: Path) -> None:
    result, _ = run_ci_local(tmp_path, without=("shellcheck",))
    assert result.returncode == 1
    lines = summary(result)
    assert "  FAIL  shellcheck" in lines
    assert not any(l.startswith("  SKIP") and "shellcheck" in l for l in lines)
    assert "missing tool: shellcheck" in result.stderr


def test_fast_mode_runs_only_the_pre_commit_set(tmp_path: Path) -> None:
    result, calls = run_ci_local(tmp_path, "--fast")
    assert result.returncode == 0, result.stdout
    assert [l.strip() for l in summary(result)[:3]] == [
        "PASS  hygiene (secrets, employer name)",
        "PASS  mojo format",
        "PASS  build -Werror",
    ]
    assert not any(c.startswith(("run-tests.sh", "shellcheck", "uv ")) for c in calls)


def test_an_unknown_flag_exits_2_before_any_step(tmp_path: Path) -> None:
    result, calls = run_ci_local(tmp_path, "--nope")
    assert result.returncode == 2 and "unknown flag" in result.stderr and calls == []


def test_the_launcher_tests_check_that_the_lock_matches_pyproject_in_both_gates() -> None:
    """`uv run --frozen` trusts uv.lock without comparing it with pyproject.toml; `--locked` fails on a mismatch."""
    command = "uv run --locked --only-group test pytest tests/python -q"
    assert command in ci_commands()
    assert command in " ".join(CI_LOCAL.read_text().split())


EM, EN = "\u2014", "\u2013"


def test_hygiene_refuses_an_em_dash_in_markdown_and_names_the_line(tmp_path: Path) -> None:
    result = run_hygiene(tmp_path, f"first line\nsecond {EM} line\n", name="docs/a.md")
    assert result.returncode == 1
    assert "DASH: docs/a.md:2" in result.stderr


def test_hygiene_refuses_an_en_dash_in_markdown(tmp_path: Path) -> None:
    result = run_hygiene(tmp_path, f"a range 1{EN}9\n", name="README.md")
    assert result.returncode == 1 and "DASH: README.md:1" in result.stderr


def test_hygiene_checks_staged_markdown_too(tmp_path: Path) -> None:
    result = run_hygiene(tmp_path, f"x {EM} y\n", name="b.md", mode="--staged")
    assert result.returncode == 1 and "DASH: b.md:1" in result.stderr


def test_hygiene_leaves_dashes_in_other_files_and_in_committed_evidence_alone(tmp_path: Path) -> None:
    assert run_hygiene(tmp_path / "a", f"x {EM} y\n", name="notes.txt").returncode == 0
    assert run_hygiene(tmp_path / "b", f"1.1{EN}1.2x\n", name="bench/results/run-1/README.md").returncode == 0


def test_hygiene_accepts_a_hyphen_and_a_minus_in_markdown(tmp_path: Path) -> None:
    result = run_hygiene(tmp_path, "a-b, c\u2212d, 1 to 9\n", name="ok.md")
    assert result.returncode == 0, result.stderr


def test_no_tracked_markdown_file_outside_the_evidence_has_a_dash() -> None:
    """Independent of the script: read every tracked Markdown file here. The gate runs the script over the tree."""
    names = subprocess.run(["git", "ls-files", "-z", "--", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True)
    offenders = []
    for name in names.stdout.split("\0"):
        if name and not name.startswith("bench/results/"):
            for number, line in enumerate((ROOT / name).read_text().splitlines(), start=1):
                if EM in line or EN in line:
                    offenders.append(f"{name}:{number}")
    assert offenders == []


def test_the_spec_is_linted_like_every_other_document() -> None:
    assert "#docs/spec.md" not in " ".join(md_lint_arguments(CI_LOCAL.read_text()))


@pytest.mark.parametrize(
    "scheme", ["postgres", "postgresql", "mysql", "mariadb", "mongodb", "mongodb+srv", "redis", "amqp"]
)
def test_hygiene_refuses_a_database_url_with_a_password_for_each_scheme(tmp_path: Path, scheme: str) -> None:
    """Each alternative of the pattern is a separate way to miss a secret."""
    result = run_hygiene(tmp_path, f"URL={scheme}://app:" + "s3cretvalue@db.example.com/app\n")
    assert result.returncode == 1 and "database url" in result.stderr.lower(), scheme


def test_a_raw_mojo_format_step_counts_as_a_gate() -> None:
    assert gates("uv run --frozen mojo format -q src tests bench") == {"mojo format"}
    assert gates("bash scripts/check_format.sh") == {"scripts/check_format.sh"}
