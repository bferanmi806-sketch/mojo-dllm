"""scripts/action-pin-comments.py and scripts/verify-action-pins.sh can turn red (#251).

The gate resolves each pinned SHA through the GitHub API, so the tests put a `gh` stub and a `uv` stub on PATH: `gh`
answers from a table, `uv` runs the Python script with the interpreter of the test run. Nothing touches the network.
PyYAML comes from the `lint` group, so the module is skipped, with that reason shown, when it is missing; CI runs
it in the step that installs the extra groups (see .github/workflows/ci.yml).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("yaml", reason="PyYAML is in the `lint` group, not the `test` group; CI runs this file with both")

ROOT = Path(__file__).resolve().parents[2]
COMMENTS = ROOT / "scripts" / "action-pin-comments.py"
VERIFY = ROOT / "scripts" / "verify-action-pins.sh"
SHA = "a" * 40
SHA2 = "b" * 40
SHA3 = "c" * 40

GH_STUB = r"""#!/usr/bin/env python3
import json, os, sys
table = json.load(open(os.environ["GH_TABLE"]))
with open(os.environ["GH_LOG"], "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
if table.get("fail"):
    sys.exit(1)
path = sys.argv[2]
if "/git/ref/tags/" in path:
    repo, tag = path[len("repos/"):].split("/git/ref/tags/")
    kind, sha = table["refs"][f"{repo}@{tag}"]
    print(table.get("refname", {}).get(f"{repo}@{tag}", f"refs/tags/{tag}"), kind, sha)
elif "/git/tags/" in path:
    repo, sha = path[len("repos/"):].split("/git/tags/")
    kind, sha2 = table["tags"][f"{repo}@{sha}"]
    print(kind, sha2)
else:
    sys.exit(2)
"""
UV_STUB = r"""#!/usr/bin/env bash
# `uv run --frozen --only-group lint python SCRIPT ROOT`: run SCRIPT with the test interpreter.
if [ -n "${UV_EMPTY:-}" ]; then exit 0; fi
if [ -n "${UV_FAIL:-}" ]; then printf 'actions/checkout\t%s\tv4\tx.yml:1\n' "$UV_FAIL_SHA"; echo "uv: boom" >&2; exit 3; fi
while [ "$1" != python ]; do shift; done
shift
exec "$PYTHON_FOR_STUB" "$@"
"""


class Rig:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.root = tmp / "repo"
        (self.root / ".github/workflows").mkdir(parents=True)
        self.bin = tmp / "bin"
        self.bin.mkdir()
        for name, body in (("gh", GH_STUB), ("uv", UV_STUB)):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        for tool in ("bash", "dirname", "mktemp", "rm", "cat", "env", "python3", "sed", "head"):
            found = shutil.which(tool)
            if found:
                (self.bin / tool).symlink_to(found)
        self.table: dict = {"refs": {}, "tags": {}}
        self.log = tmp / "gh.log"

    def workflow(self, text: str, name: str = "ci.yml") -> None:
        (self.root / ".github/workflows" / name).write_text(textwrap.dedent(text))

    def uses(self, *lines: str) -> None:
        steps = "\n".join(f"      - uses: {line}" for line in lines)
        self.workflow(f"name: t\non: push\njobs:\n  j:\n    runs-on: ubuntu-24.04\n    steps:\n{steps}\n")

    def run(self, **env: str) -> subprocess.CompletedProcess[str]:
        table = self.tmp / "table.json"
        table.write_text(json.dumps(self.table))
        full = {"PATH": str(self.bin), "GH_TABLE": str(table), "GH_LOG": str(self.log), "PYTHON_FOR_STUB": sys.executable,
                "HOME": str(self.tmp), **env}
        return subprocess.run(["bash", str(VERIFY), str(self.root)], env=full, capture_output=True, text=True, timeout=120)

    def gh_calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


# verify-action-pins.sh --------------------------------------------------------------------------------------

def test_a_pin_that_matches_its_tag_passes(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4.1.1")
    rig.table["refs"]["actions/checkout@v4.1.1"] = ["commit", SHA]
    p = rig.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "OK actions/checkout" in p.stdout and "checked 1 pins" in p.stdout


def test_a_pin_that_is_not_the_commit_of_its_tag_fails(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4.1.1")
    rig.table["refs"]["actions/checkout@v4.1.1"] = ["commit", SHA2]
    p = rig.run()
    assert p.returncode == 1 and "FAIL actions/checkout" in p.stdout and "bbbbbbbb" in p.stdout


def test_an_annotated_tag_is_peeled_to_its_commit(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4")
    rig.table["refs"]["actions/checkout@v4"] = ["tag", SHA2]
    rig.table["tags"][f"actions/checkout@{SHA2}"] = ["commit", SHA]
    p = rig.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert rig.gh_calls() == ["api repos/actions/checkout/git/ref/tags/v4 --jq .ref + \" \" + .object.type + \" \" + .object.sha",
                              f"api repos/actions/checkout/git/tags/{SHA2} --jq .object.type + \" \" + .object.sha"]


def chain(rig: Rig, depth: int) -> None:
    """A tag that points to `depth` nested tag objects and then the commit SHA."""
    shas = [f"{i:040x}" for i in range(1, depth + 1)]
    rig.table["refs"]["actions/checkout@v4"] = ["tag", shas[0]]
    for i, s in enumerate(shas):
        rig.table["tags"][f"actions/checkout@{s}"] = ["tag", shas[i + 1]] if i + 1 < depth else ["commit", SHA]


def test_a_chain_of_sixteen_tag_objects_is_followed_and_seventeen_is_refused(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4")
    chain(rig, 16)
    assert rig.run().returncode == 0
    chain(rig, 17)
    p = rig.run()
    assert p.returncode == 1 and "ERROR actions/checkout" in p.stdout


def test_a_tag_that_cannot_be_resolved_is_an_error_not_a_pass(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4")
    rig.table["fail"] = True                                  # gh exits 1: an expired token, no network
    p = rig.run()
    assert p.returncode == 1 and "ERROR" in p.stdout and "cannot resolve tag v4" in p.stdout


def test_a_ref_whose_name_is_another_tag_is_refused(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4")
    rig.table["refs"]["actions/checkout@v4"] = ["commit", SHA]
    rig.table["refname"] = {"actions/checkout@v4": "refs/tags/v4.1"}     # the API matched a longer name
    assert rig.run().returncode == 1


@pytest.mark.parametrize("kind,sha", [("blob", SHA), ("commit", "xyz"), ("commit", "A" * 40), ("commit", SHA[:39])])
def test_an_object_that_is_no_commit_sha_is_an_error(rig: Rig, kind: str, sha: str) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4")
    rig.table["refs"]["actions/checkout@v4"] = [kind, sha]
    p = rig.run()
    assert p.returncode == 1 and "ERROR" in p.stdout


def test_one_resolution_is_cached_per_repository_and_claim(rig: Rig) -> None:
    rig.uses(f"github/codeql-action/init@{SHA}  # v3", f"github/codeql-action/analyze@{SHA}  # v3")
    rig.table["refs"]["github/codeql-action@v3"] = ["commit", SHA]
    p = rig.run()
    assert p.returncode == 0 and "checked 2 pins" in p.stdout
    assert len(rig.gh_calls()) == 1                           # a subpath shares its repository's tags


def test_two_claims_of_one_repository_are_each_resolved(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v3", f"actions/checkout@{SHA2}  # v4")
    rig.table["refs"]["actions/checkout@v3"] = ["commit", SHA]
    rig.table["refs"]["actions/checkout@v4"] = ["commit", SHA3]       # the second pin is wrong: a cache keyed on the repo alone hides it
    p = rig.run()
    assert p.returncode == 1 and "FAIL actions/checkout" in p.stdout and len(rig.gh_calls()) == 2


def test_no_pinned_actions_is_a_failure(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4")
    p = rig.run(UV_EMPTY="1")                                  # the extractor printed nothing and succeeded
    assert p.returncode == 1 and "no pinned actions" in p.stderr


def test_a_failing_extractor_stops_the_gate_even_after_it_printed_a_row(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4")
    rig.table["refs"]["actions/checkout@v4"] = ["commit", SHA]       # the half-printed row would verify
    p = rig.run(UV_FAIL="1", UV_FAIL_SHA=SHA)
    assert p.returncode == 1 and "checked" not in p.stdout and rig.gh_calls() == []


def test_a_workflow_with_a_branch_ref_fails_in_the_gate(rig: Rig) -> None:
    rig.uses("actions/checkout@main")
    p = rig.run()
    assert p.returncode == 1 and "non-SHA" in p.stderr


# action-pin-comments.py -------------------------------------------------------------------------------------

def extract(rig: Rig) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(COMMENTS), str(rig.root)], capture_output=True, text=True, timeout=60)


def test_the_extractor_prints_action_sha_claim_and_location(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA.upper()}  # v4.1.1")
    p = extract(rig)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == f"actions/checkout\t{SHA}\tv4.1.1\t.github/workflows/ci.yml:7"


def test_a_pin_without_a_version_comment_is_refused(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}")
    p = extract(rig)
    assert p.returncode != 0 and "missing version comment" in p.stderr


@pytest.mark.parametrize("comment", ["# v4 and more words", "#", "# "])
def test_an_ambiguous_or_empty_comment_is_refused(rig: Rig, comment: str) -> None:
    rig.uses(f"actions/checkout@{SHA}  {comment}")
    assert extract(rig).returncode != 0


def test_a_reference_that_is_not_a_sha_is_refused(rig: Rig) -> None:
    for ref in ("actions/checkout@v4", "actions/checkout@" + SHA[:39], "actions/checkout"):
        rig.uses(ref)
        p = extract(rig)
        assert p.returncode != 0 and "non-SHA" in p.stderr, ref


def test_local_and_docker_references_are_skipped_but_a_pin_is_still_required(rig: Rig) -> None:
    rig.uses("./.github/actions/setup", "docker://alpine:3.20", f"actions/checkout@{SHA}  # v4")
    (rig.root / ".github/actions/setup").mkdir(parents=True)
    (rig.root / ".github/actions/setup/action.yml").write_text(
        f"name: s\ndescription: d\nruns:\n  using: composite\n  steps:\n    - uses: actions/cache@{SHA2}  # v4.2.0\n")
    p = extract(rig)
    assert p.returncode == 0, p.stderr
    assert [r.split("\t")[0] for r in p.stdout.strip().splitlines()] == ["actions/checkout", "actions/cache"]


def test_a_local_reference_outside_the_scanned_directories_is_refused(rig: Rig) -> None:
    rig.uses(f"actions/checkout@{SHA}  # v4", "./scripts/thing")
    p = extract(rig)
    assert p.returncode != 0 and "never scanned" in p.stderr


def test_a_hash_inside_a_quoted_value_is_not_the_comment(rig: Rig) -> None:
    rig.workflow(f"""\
        name: t
        on: push
        jobs:
          j:
            runs-on: ubuntu-24.04
            steps:
              - uses: "actions/checkout@{SHA}"  # v4
                with: {{ref: "a # b"}}
              - run: echo "x # y"
        """)
    p = extract(rig)
    assert p.returncode == 0 and p.stdout.split("\t")[2] == "v4"


def test_a_flow_mapping_step_keeps_its_comment(rig: Rig) -> None:
    rig.workflow(f"name: t\non: push\njobs:\n  j:\n    runs-on: x\n    steps:\n      - {{uses: actions/checkout@{SHA}}}  # v4\n")
    p = extract(rig)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split("\t")[2] == "v4"


def test_a_reusable_workflow_job_is_checked(rig: Rig) -> None:
    rig.workflow(f"name: t\non: push\njobs:\n  j:\n    uses: org/repo/.github/workflows/w.yml@{SHA}  # v1\n")
    p = extract(rig)
    assert p.returncode == 0 and p.stdout.split("\t")[0] == "org/repo/.github/workflows/w.yml"


def test_no_workflow_or_no_pin_is_refused(rig: Rig) -> None:
    rig.workflow("name: t\non: push\njobs:\n  j:\n    runs-on: x\n    steps:\n      - run: true\n")
    p = extract(rig)
    assert p.returncode != 0 and "no pinned actions" in p.stderr
    for f in (rig.root / ".github/workflows").iterdir():
        f.unlink()
    assert extract(rig).returncode != 0


def test_an_unparseable_workflow_is_refused(rig: Rig) -> None:
    rig.workflow("name: [unclosed\n")
    assert extract(rig).returncode != 0


def test_the_real_workflows_of_this_repository_extract(tmp_path: Path) -> None:
    p = subprocess.run([sys.executable, str(COMMENTS), str(ROOT)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    rows = [r.split("\t") for r in p.stdout.strip().splitlines()]
    assert rows and all(len(r) == 4 and len(r[1]) == 40 and r[2].startswith("v") for r in rows)
    assert os.path.exists(ROOT / ".github/workflows/ci.yml")


def test_a_hash_inside_a_quoted_value_on_the_same_line_is_not_the_comment(rig: Rig) -> None:
    rig.workflow(f"name: t\non: push\njobs:\n  j:\n    runs-on: x\n    steps:\n"
                 f"      - {{uses: actions/checkout@{SHA}, with: {{ref: \"a # b\"}}}}  # v4\n")
    p = extract(rig)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split("\t")[2] == "v4"
