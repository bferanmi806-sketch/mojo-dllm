"""docs/benchmarks.md: no command in a runbook carries a secret in its argument list.

The argument list of a running process is readable by every user of the machine (`/proc/PID/cmdline` is mode 0444),
so a key passed as an argument is exposed for as long as the command runs. The Vast.ai runbook reads the API key into
the environment variable the CLI reads (`VAST_API_KEY`), where only the owner and root can read it.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs" / "benchmarks.md"
OLD_LINE = 'read -rs -p "key: " K; echo; vastai set api-key "$K"; unset K'      # what the runbook had before
SECRET_NAME = r"[A-Za-z_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Za-z_]*"


def fenced_blocks(text: str) -> list[list[str]]:
    """The lines of every fenced code block, with the list indent of a block inside a numbered step removed."""
    blocks: list[list[str]] = []
    current: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if inside:
                blocks.append(current)
                current = []
            inside = not inside
        elif inside:
            current.append(line.strip())
    assert not inside, "an unclosed code fence"
    return blocks


def argv_secrets(line: str) -> list[str]:
    """What on one shell line puts a secret in an argument list: an `api-key` argument or option, or the expansion
    of a variable named like a secret. Reading into the variable (`read -rs VAR`) and `export VAR` do not expand it."""
    found = re.findall(r"(?i)(?:--api-key|\bapi-key)\b", line)
    found += re.findall(r"\$\{?" + SECRET_NAME + r"\}?", line)
    return found


def test_the_runbooks_have_code_blocks_to_scan() -> None:
    blocks = fenced_blocks(DOC.read_text())
    joined = "\n".join("\n".join(b) for b in blocks)
    assert len(blocks) >= 10
    assert "vastai create instance" in joined and "sglang serve" in joined and "vllm serve" in joined


def test_the_detector_finds_the_line_the_runbook_used_to_have() -> None:
    assert argv_secrets(OLD_LINE)
    assert argv_secrets("vastai --api-key abc search offers")
    assert argv_secrets('curl -H "Authorization: Bearer $HF_TOKEN" x')


def test_the_detector_accepts_reading_and_exporting_the_variable() -> None:
    assert argv_secrets('read -rs -p "key: " VAST_API_KEY; echo; export VAST_API_KEY') == []
    assert argv_secrets('vastai create ssh-key "$(cat ~/.ssh/vast_dgemma.pub)"') == []


def test_no_command_in_the_docs_puts_a_secret_in_its_argument_list() -> None:
    bad = [(line, argv_secrets(line)) for block in fenced_blocks(DOC.read_text()) for line in block if argv_secrets(line)]
    assert bad == []


def test_the_vast_api_key_is_read_without_echo_into_the_variable_the_cli_reads() -> None:
    lines = [line for block in fenced_blocks(DOC.read_text()) for line in block]
    reads = [l for l in lines if l.startswith("read -rs") and "VAST_API_KEY" in l and "export VAST_API_KEY" in l]
    assert len(reads) == 1, reads
    assert not any("set api-key" in l for l in lines)
