#!/usr/bin/env python3
"""Check relative links in the Markdown sources and in the built mdBook.

usage: check_links.py

Fails on a relative link whose target file does not exist. External links are
not fetched (that would make the gate depend on the network); anchors are not
checked. Refuses to pass when it found nothing to check. Inline links, with or
without a title (`[a](b.md "title")`), and reference definitions (`[a]: b.md`)
are read; a link inside a fenced code block is an example, not a link, and is
not read.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
MD_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*$", re.M)
FENCE = re.compile(r"^ {0,3}(```|~~~).*?^ {0,3}\1[ \t]*$", re.M | re.S)
HTML_LINK = re.compile(r'(?:href|src)="([^"]+)"')


def is_relative(target: str) -> bool:
    """Relative paths only: no scheme, no fragment-only link, no site-root path."""
    return not re.match(r"^[a-z]+:|^#|^/", target)


def markdown_targets(text: str) -> list[str]:
    """Every link target of a Markdown text outside fenced code blocks."""
    text = FENCE.sub("", text)
    return MD_LINK.findall(text) + MD_DEFINITION.findall(text)


def main() -> int:
    checked = 0
    broken = []
    md_files = [p for p in ROOT.glob("*.md")] + list((ROOT / "docs").rglob("*.md"))
    for f in md_files:
        for target in markdown_targets(f.read_text()):
            if not is_relative(target):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            checked += 1
            if not (f.parent / path).resolve().exists():
                broken.append(f"{f.relative_to(ROOT)} -> {target}")
    book = ROOT / "book"
    for f in book.rglob("*.html") if book.exists() else []:
        for target in HTML_LINK.findall(f.read_text(errors="replace")):
            if not is_relative(target):
                continue
            path = target.split("#", 1)[0].split("?", 1)[0]
            if not path:
                continue
            checked += 1
            if not (f.parent / path).resolve().exists():
                broken.append(f"{f.relative_to(ROOT)} -> {target}")
    if checked == 0:
        print("check_links: found no relative links at all; refusing to pass", file=sys.stderr)
        return 1
    if broken:
        print("check_links: broken relative links:", file=sys.stderr)
        for b in broken:
            print("  " + b, file=sys.stderr)
        return 1
    print(f"check_links: {checked} relative links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
