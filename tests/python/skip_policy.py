"""A pytest plugin: a skipped test that tests/python/allowed-skips.txt does not name fails the run.

`pytest -q` prints a skip as one letter among the dots, so a test that silently stops running looks like a pass.
This plugin lists every skip in the summary (`-rs`, set in pyproject.toml) and turns the exit status to 1 for a skip
whose test id and reason no allow-list entry matches. The tests it guards run `-p skip_policy`.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

DEFAULT_ALLOWLIST = Path(__file__).with_name("allowed-skips.txt")
Entry = tuple[str, str]


def load_allowlist(path: Path) -> list[Entry]:
    """The (test id glob, reason substring) pairs of an allow-list file; a malformed line is an error."""
    entries: list[Entry] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        glob, sep, reason = (part.strip() for part in line.partition("|"))
        if not sep or not glob or not reason:
            raise ValueError(f"{path}:{number}: expected '<test id glob> | <reason text>', got {raw!r}")
        entries.append((glob, reason))
    return entries


def is_allowed(nodeid: str, reason: str, entries: list[Entry]) -> bool:
    return any(fnmatch.fnmatchcase(nodeid, glob) and text in reason for glob, text in entries)


class SkipPolicy:
    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        self.unexpected: list[tuple[str, str]] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if not report.skipped or hasattr(report, "wasxfail"):
            return
        longrepr = report.longrepr
        reason = str(longrepr[2] if isinstance(longrepr, tuple) else longrepr)
        reason = reason.removeprefix("Skipped: ")
        if not is_allowed(report.nodeid, reason, self.entries):
            self.unexpected.append((report.nodeid, reason))

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if not self.unexpected:
            return
        terminalreporter.section("unexpected skips", red=True)
        for nodeid, reason in self.unexpected:
            terminalreporter.write_line(f"unexpected skip: {nodeid}: {reason}")
        terminalreporter.write_line("Fix the test, or name the skip in tests/python/allowed-skips.txt.")

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.unexpected and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--skip-allowlist", default=str(DEFAULT_ALLOWLIST), help="file of skips that may happen")


def pytest_configure(config: pytest.Config) -> None:
    path = Path(config.getoption("--skip-allowlist"))
    if not path.is_file():
        raise pytest.UsageError(f"skip allow-list {path} does not exist; no skip can be judged")
    try:
        entries = load_allowlist(path)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    config.pluginmanager.register(SkipPolicy(entries), "skip-policy")
