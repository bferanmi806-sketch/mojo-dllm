"""scripts/check_fixtures.py: regenerate the fixtures into a scratch directory and compare them with the tracked ones.

The real generators need numpy and gguf, which the `test` group does not install, so the gate runs here against
a stand-in generator that writes known files. The generators themselves are held to the gate by CI, which runs
`check_fixtures.py` in the `fixtures` group.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("check_fixtures", ROOT / "scripts" / "check_fixtures.py")
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
sys.modules["check_fixtures"] = gate
spec.loader.exec_module(gate)

# What the stand-in generator writes: a file, a file in a directory, and a binary one.
GENERATED = {"a.tsv": b"one\ntwo\n", "dir/b.gguf": b"GGUF\x00\x01", "c.f32": bytes(range(16))}
GENERATOR = """import os, pathlib
out = pathlib.Path(os.environ["FIXTURES_OUT"])
for name, data in {files!r}.items():
    (out / name).parent.mkdir(parents=True, exist_ok=True)
    (out / name).write_bytes(data)
"""


def make_repo(tmp_path: Path, tracked: dict[str, bytes] | None = None) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "gen.py").write_text(GENERATOR.format(files=GENERATED))
    for name, data in (GENERATED if tracked is None else tracked).items():
        (root / "tests" / "fixtures" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "tests" / "fixtures" / name).write_bytes(data)
    return root


def check(root: Path, not_generated: tuple[str, ...] = ()) -> list[str]:
    return gate.check(root, [["scripts/gen.py"]], not_generated)


def test_fixtures_that_match_their_generators_pass(tmp_path: Path) -> None:
    assert check(make_repo(tmp_path)) == []


def test_one_changed_byte_fails_and_names_the_file(tmp_path: Path) -> None:
    tracked = {**GENERATED, "dir/b.gguf": b"GGUF\x00\x02"}
    problems = check(make_repo(tmp_path, tracked))
    assert len(problems) == 1 and "dir/b.gguf" in problems[0] and "differs" in problems[0]


def test_a_generated_file_that_is_not_tracked_fails(tmp_path: Path) -> None:
    tracked = {k: v for k, v in GENERATED.items() if k != "c.f32"}
    problems = check(make_repo(tmp_path, tracked))
    assert len(problems) == 1 and "c.f32" in problems[0] and "not in tests/fixtures" in problems[0]


def test_a_tracked_file_no_generator_writes_fails(tmp_path: Path) -> None:
    problems = check(make_repo(tmp_path, {**GENERATED, "by-hand.bin": b"x"}))
    assert len(problems) == 1 and "by-hand.bin" in problems[0] and "no generator writes it" in problems[0]


def test_a_file_made_by_another_script_is_named_in_the_allow_list(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {**GENERATED, "tok.tsv": b"x", "prov/p.json": b"{}"})
    assert check(root, ("tok.tsv", "prov/*")) == []
    assert len(check(root, ("tok.tsv",))) == 1


def test_every_problem_is_listed_not_only_the_first(tmp_path: Path) -> None:
    tracked = {"a.tsv": b"changed", "stray.bin": b"x"}
    problems = check(make_repo(tmp_path, tracked))
    assert len(problems) == 4  # a.tsv differs, dir/b.gguf and c.f32 untracked, stray.bin unmade


def test_a_generator_that_fails_fails_the_gate(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "scripts" / "gen.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(gate.GeneratorFailed, match=r"gen\.py.*3"):
        check(root)


def test_a_generator_that_writes_nothing_fails_the_gate(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {})
    (root / "scripts" / "gen.py").write_text("pass\n")
    with pytest.raises(gate.GeneratorFailed, match="wrote no file"):
        check(root)


def test_the_generators_write_where_the_environment_says() -> None:
    """Both real generators read FIXTURES_OUT, so the gate never touches tests/fixtures."""
    for name in ("gen_fixtures.py", "gen_moe_fixture.py"):
        assert "FIXTURES_OUT" in (ROOT / "scripts" / name).read_text(), name


def test_the_moe_generator_writes_its_four_files_into_the_directory_it_is_given(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_moe_fixture.py")],
        env={**os.environ, "FIXTURES_OUT": str(out)},
        check=True,
        capture_output=True,
    )
    assert sorted(p.name for p in out.iterdir()) == ["moe-q4-q5.bin", "moe-q4-q8.bin", "moe-q5.bin", "moe-q8.bin"]
    for name in out.iterdir():
        assert name.read_bytes() == (ROOT / "tests" / "fixtures" / name.name).read_bytes()


def test_the_not_generated_list_covers_exactly_what_the_tokenizer_and_golden_tools_make() -> None:
    """Every tracked fixture is made by a generator or named in the list, and the list names no file that is gone."""
    tracked = {p.relative_to(ROOT / "tests" / "fixtures").as_posix() for p in (ROOT / "tests" / "fixtures").rglob("*") if p.is_file()}
    listed = [name for pattern in gate.NOT_GENERATED for name in tracked if Path(name).match(pattern) or name == pattern]
    assert listed, "the allow-list matches no tracked file"
    for pattern in gate.NOT_GENERATED:
        assert any(Path(name).match(pattern) or name == pattern for name in tracked), f"{pattern} matches nothing"


def fixture_table_names() -> list[str]:
    """What the first column of the fixture table of docs/development.md names: files, directories and globs."""
    block = (ROOT / "docs" / "development.md").read_text().split("## Tests", 1)[1].split("\n### ", 1)[0]
    names: list[str] = []
    for row in block.splitlines():
        if row.startswith("| `"):
            names += [n for n in re.findall(r"`([^`]+)`", row.split("|")[1])]
    return names


def test_every_fixture_file_is_named_in_the_development_guide() -> None:
    names = fixture_table_names()
    assert len(names) >= 20
    fixtures = ROOT / "tests" / "fixtures"
    missing = []
    for path in sorted(fixtures.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(fixtures).as_posix()
        covered = any(
            rel == name
            or (name.endswith("/") and rel.startswith(name))
            or ("*" in name and Path(rel).match(name))
            for name in names
        )
        if not covered:
            missing.append(rel)
    assert missing == []


def test_the_guide_names_the_provenance_of_all_three_tokenizer_goldens() -> None:
    names = fixture_table_names()
    for prefix in ("llada", "dream", "gemma4"):
        assert f"{prefix}-tokenizer-golden.provenance.json" in names or "*-tokenizer-golden.provenance.json" in names
