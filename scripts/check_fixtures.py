#!/usr/bin/env python3
"""Regenerate the test fixtures into a scratch directory and compare them with tests/fixtures byte for byte.

Run with:  uv run --frozen --only-group fixtures python scripts/check_fixtures.py

Every fixture "comes from a script, never from hand", and nothing checked that. A fixture edited by hand, a
generator changed without regenerating, or a file left behind by a deleted generator passed every gate. The gate
fails on each of those: a regenerated file that differs from the tracked one, a regenerated file that is not
tracked, and a tracked file that no generator writes and that NOT_GENERATED does not name.

The generators write where FIXTURES_OUT says (default tests/fixtures), so this never touches the tracked files.
The tokenizer fixtures and goldens need a real model file and llama.cpp, and golden-generation.tsv needs a Mojo
build, so they are listed in NOT_GENERATED and not compared here.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATORS = [["scripts/gen_fixtures.py"], ["scripts/gen_moe_fixture.py"]]
NOT_GENERATED = (
    "*-tokenizer.gguf",  # scripts/gen_tokenizer_fixture.py, from a real model file
    "*-tokenizer-golden.tsv",  # the same script, from llama-tokenize
    "*-tokenizer-golden.provenance.json",  # the same script, next to the golden it describes
    "golden-generation.tsv",  # UPDATE_GOLDEN=1 build/tests/test_golden
)


class GeneratorFailed(Exception):
    pass


def _files(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    return {p.relative_to(directory).as_posix(): p for p in sorted(directory.rglob("*")) if p.is_file()}


def _named(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pathlib.PurePosixPath(name).match(pattern) for pattern in patterns)


def check(root: pathlib.Path, generators: list[list[str]], not_generated: tuple[str, ...]) -> list[str]:
    """The problems found, one line each; an empty list means every fixture regenerates byte for byte."""
    tracked = _files(root / "tests" / "fixtures")
    scratch_parent = root / "build"
    scratch_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fixtures-", dir=scratch_parent) as scratch:
        env = {**os.environ, "FIXTURES_OUT": scratch}
        for command in generators:
            result = subprocess.run([sys.executable, *command], cwd=root, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise GeneratorFailed(f"{' '.join(command)} exited {result.returncode}: {result.stderr.strip()[-400:]}")
        made = _files(pathlib.Path(scratch))
        if not made:
            raise GeneratorFailed(f"{' '.join(g[0] for g in generators)} wrote no file")
        problems = []
        for name, path in made.items():
            if name not in tracked:
                problems.append(f"{name}: a generator writes it and it is not in tests/fixtures")
            elif path.read_bytes() != tracked[name].read_bytes():
                problems.append(f"{name}: differs from what its generator writes")
        for name in tracked:
            if name not in made and not _named(name, not_generated):
                problems.append(f"{name}: tracked in tests/fixtures and no generator writes it")
        return problems


def main() -> int:
    try:
        problems = check(ROOT, GENERATORS, NOT_GENERATED)
    except GeneratorFailed as error:
        print(f"check_fixtures: a generator failed: {error}", file=sys.stderr)
        return 1
    if problems:
        print("check_fixtures: the fixtures do not match their generators:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print("Regenerate with scripts/gen_fixtures.py and scripts/gen_moe_fixture.py and commit the result.", file=sys.stderr)
        return 1
    print("check_fixtures: every generated fixture matches byte for byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
