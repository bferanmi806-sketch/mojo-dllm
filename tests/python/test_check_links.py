"""scripts/check_links.py fails on a broken relative link and refuses to pass when it found no link at all (#251).

The script takes its root from its own location, so it is copied into a throwaway tree with `docs/` and `book/`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "site"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    shutil.copy(ROOT / "scripts/check_links.py", root / "scripts/check_links.py")
    return root


def run(tree: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(tree / "scripts/check_links.py")], capture_output=True, text=True, timeout=60)


def test_resolving_links_pass_and_are_counted(tree: Path) -> None:
    (tree / "README.md").write_text("[a](docs/a.md) and [b](docs/a.md#section)\n")
    (tree / "docs/a.md").write_text("[back](../README.md)\n")
    p = run(tree)
    assert p.returncode == 0 and "3 relative links resolve" in p.stdout


def test_a_broken_link_fails_and_names_the_file_and_the_target(tree: Path) -> None:
    (tree / "README.md").write_text("[ok](docs/a.md) [gone](docs/missing.md)\n")
    (tree / "docs/a.md").write_text("x\n")
    p = run(tree)
    assert p.returncode == 1 and "README.md -> docs/missing.md" in p.stderr and "ok" not in p.stderr.replace("broken", "")


def test_a_broken_link_in_a_nested_doc_is_found_relative_to_that_doc(tree: Path) -> None:
    (tree / "docs/sub").mkdir()
    (tree / "docs/sub/n.md").write_text("[up](../a.md) [bad](a.md)\n")
    (tree / "docs/a.md").write_text("x\n")
    (tree / "README.md").write_text("[d](docs/a.md)\n")
    p = run(tree)
    assert p.returncode == 1 and "docs/sub/n.md -> a.md" in p.stderr and "../a.md" not in p.stderr


def test_no_relative_link_at_all_is_a_refusal(tree: Path) -> None:
    (tree / "README.md").write_text("no links here, only [text](https://example.org) and [a](#anchor)\n")
    p = run(tree)
    assert p.returncode == 1 and "found no relative links at all" in p.stderr


def test_external_anchor_and_site_root_links_are_not_checked(tree: Path) -> None:
    (tree / "README.md").write_text("[a](https://x.org/y) [b](#top) [c](/abs/path) [d](mailto:a@b.c) [e](docs/a.md)\n")
    (tree / "docs/a.md").write_text("x\n")
    p = run(tree)
    assert p.returncode == 0 and "1 relative links resolve" in p.stdout


def test_a_titled_link_is_checked_by_its_target(tree: Path) -> None:
    (tree / "README.md").write_text('[ok](docs/a.md "the title") [bad](docs/zz.md "t")\n')
    (tree / "docs/a.md").write_text("x\n")
    p = run(tree)
    assert p.returncode == 1 and "docs/zz.md" in p.stderr and "docs/a.md" not in p.stderr.replace("docs/zz.md", "")


def test_a_broken_link_in_the_built_book_fails(tree: Path) -> None:
    (tree / "README.md").write_text("[a](docs/a.md)\n")
    (tree / "docs/a.md").write_text("x\n")
    (tree / "book").mkdir()
    (tree / "book/index.html").write_text('<a href="a.html">ok</a><a href="gone.html?x=1#y">bad</a><img src="i.png">')
    (tree / "book/a.html").write_text("x")
    p = run(tree)
    assert p.returncode == 1 and "book/index.html -> gone.html?x=1#y" in p.stderr and "i.png" in p.stderr


def test_only_links_in_the_book_count_when_they_are_the_only_ones(tree: Path) -> None:
    (tree / "README.md").write_text("no links\n")
    (tree / "book").mkdir()
    (tree / "book/a.html").write_text('<a href="b.html">b</a>')
    (tree / "book/b.html").write_text("x")
    p = run(tree)
    assert p.returncode == 0 and "1 relative links resolve" in p.stdout


def test_the_repository_itself_has_no_broken_link() -> None:
    p = subprocess.run([sys.executable, str(ROOT / "scripts/check_links.py")], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr


def test_a_reference_definition_is_checked(tree: Path) -> None:
    (tree / "README.md").write_text("[ok][a] and [bad][b]\n\n[a]: docs/a.md\n[b]: docs/gone.md \"title\"\n[c]: https://x.org\n")
    (tree / "docs/a.md").write_text("x\n")
    p = run(tree)
    assert p.returncode == 1 and "README.md -> docs/gone.md" in p.stderr and "https" not in p.stderr


def test_a_link_inside_a_fenced_code_block_is_an_example_and_is_not_checked(tree: Path) -> None:
    (tree / "README.md").write_text("[real](docs/a.md)\n\n```markdown\n[example](docs/never-exists.md)\n```\n\n"
                                    "~~~\n[tilde](docs/also-missing.md)\n~~~\n\n[after](docs/a.md)\n")
    (tree / "docs/a.md").write_text("x\n")
    p = run(tree)
    assert p.returncode == 0 and "2 relative links resolve" in p.stdout


def test_text_after_an_unclosed_fence_is_not_swallowed_silently_for_the_links_before_it(tree: Path) -> None:
    (tree / "README.md").write_text("[bad](docs/missing.md)\n\n```\n[x](docs/y.md)\n")
    (tree / "docs/a.md").write_text("x\n")
    p = run(tree)
    assert p.returncode == 1 and "docs/missing.md" in p.stderr


def test_an_angle_bracket_target_is_unwrapped(tree: Path) -> None:
    (tree / "README.md").write_text("[a](<docs/a.md>) [b](<docs/b.md> \"t\")\n")
    (tree / "docs/a.md").write_text("x\n")
    p = run(tree)
    assert p.returncode == 1 and "docs/b.md" in p.stderr and "docs/a.md" not in p.stderr.replace("docs/b.md", "")


def test_a_longer_fence_is_not_closed_by_a_shorter_one_inside_it(tree: Path) -> None:
    """A four-backtick block that holds a three-backtick block ends at the four-backtick line, not the inner one."""
    (tree / "README.md").write_text(
        "````md\n```\n[ex](gone.md)\n```\n````\n\n[real-broken](missing1.md)\n\ntext\n\n```\ncode\n```\n[also-broken](missing2.md)\n")
    p = run(tree)
    assert p.returncode == 1
    assert "missing1.md" in p.stderr and "missing2.md" in p.stderr and "gone.md" not in p.stderr


def test_a_fence_closes_only_on_the_same_character_and_at_least_the_same_length(tree: Path) -> None:
    (tree / "docs/a.md").write_text("x\n")
    (tree / "README.md").write_text("~~~~\n```\n[ex](gone1.md)\n~~~\n[ex2](gone2.md)\n~~~~~\n[ok](docs/a.md)\n")
    p = run(tree)
    assert p.returncode == 0 and "1 relative links resolve" in p.stdout


def test_a_closing_fence_may_be_longer_and_may_not_carry_text(tree: Path) -> None:
    (tree / "docs/a.md").write_text("x\n")
    (tree / "README.md").write_text("```\n[ex](gone.md)\n```info\n[still](gone2.md)\n`````\n[ok](docs/a.md)\n")
    p = run(tree)
    assert p.returncode == 0 and "1 relative links resolve" in p.stdout


def test_a_tilde_line_does_not_close_a_backtick_fence(tree: Path) -> None:
    (tree / "docs/a.md").write_text("x\n")
    (tree / "README.md").write_text("```\n[ex](gone.md)\n~~~\n[ex2](gone2.md)\n```\n[ok](docs/a.md)\n")
    p = run(tree)
    assert p.returncode == 0 and "1 relative links resolve" in p.stdout
