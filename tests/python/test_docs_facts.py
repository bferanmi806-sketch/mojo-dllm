"""Statements in the documents that the code or the evidence can check: the source map, the tensor types, the flag
limits and the commit named under a benchmark table."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = load_module("bench_table_docs_facts", ROOT / "scripts" / "bench_table.py")


def text(path: str) -> str:
    return (ROOT / path).read_text()


# ---- the source map in docs/architecture.md ----------------------------------------------------------------------


def source_map_paths() -> list[str]:
    """The paths in the first column of the Source map table, with `a`, `b` pairs split."""
    block = text("docs/architecture.md").split("## Source map", 1)[1].split("\n## ", 1)[0]
    paths: list[str] = []
    for row in block.splitlines():
        if row.startswith("| `"):
            paths += re.findall(r"`([^`]+)`", row.split("|")[1])
    return paths


def test_every_source_file_is_in_the_source_map() -> None:
    listed = source_map_paths()
    assert len(listed) >= 15
    missing = []
    for path in sorted((ROOT / "src").rglob("*.mojo")):
        rel = path.relative_to(ROOT).as_posix()
        if path.name == "__init__.mojo":
            continue
        if not any(rel == entry or (entry.endswith("/") and rel.startswith(entry)) for entry in listed):
            missing.append(rel)
    assert missing == []


def test_every_path_in_the_source_map_exists() -> None:
    assert [p for p in source_map_paths() if not (ROOT / p).exists()] == []


def test_the_sampler_row_names_all_three_loops() -> None:
    row = next(r for r in text("docs/architecture.md").splitlines() if "`src/mojo_dllm/diffusion/sampler.mojo`" in r)
    assert "LLaDA" in row and "Dream" in row and "DiffusionGemma" in row


# ---- tensor types -------------------------------------------------------------------------------------------------

DECODERS = ["F32", "F16", "BF16", "Q5_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"]


def decoder_types_in_code() -> set[str]:
    """The types `has_decoder` accepts, read from src/mojo_dllm/quant/kquants.mojo, as the names ggml prints."""
    body = text("src/mojo_dllm/quant/kquants.mojo").split("def has_decoder(", 1)[1].split("\ndef ", 1)[0]
    constants = set(re.findall(r"GGML_([A-Z0-9_]+)", body))
    assert constants, "has_decoder names no GGML_ constant; update this test"
    return constants


def test_the_decoder_list_in_this_test_is_the_one_the_code_accepts() -> None:
    assert decoder_types_in_code() == set(DECODERS)


@pytest.mark.parametrize("path", ["README.md", "docs/introduction.md", "docs/architecture.md", "docs/model-porting.md"])
def test_the_documents_list_the_same_decoder_types(path: str) -> None:
    """Every sentence that lists the decodable tensor types lists all eight, in the order of the code."""
    flat = " ".join(text(path).split())
    sentences = [m.group(0) for m in re.finditer(r"F32[ ,/(]+F16[^.;|]*?Q6_K", flat)]
    assert sentences, f"{path} no longer lists the decoder types; update this test"
    for sentence in sentences:
        assert re.findall(r"F32|F16|BF16|Q5_0|Q8_0|Q4_K|Q5_K|Q6_K", sentence) == DECODERS, (path, sentence)


# ---- flag limits --------------------------------------------------------------------------------------------------


def error_row() -> str:
    return next(r for r in text("docs/cli.md").splitlines() if r.startswith("| `ERROR: flag --steps must be at least 1"))


def test_the_range_error_row_lists_every_limit_the_code_enforces() -> None:
    row = error_row()
    assert re.search(r"`--eb-steps`[^.]*at least 1|at least 1[^.]*`--eb-steps`", row)
    assert re.search(r"`--eb-stability`[^.]*at least 0", row)
    assert re.search(r"`--eb-t-max`[^.]*above 0|above 0[^.]*`--eb-t-max`", row)
    for flag in ("--eb-t-min", "--eb-bound", "--eb-confidence"):
        assert flag in row


def test_the_flag_table_gives_the_limit_of_each_eb_flag() -> None:
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| `--(eb-[a-z-]+)[^|]*\|.*$", text("docs/cli.md"), re.M)}
    assert "at least 1" in rows["eb-steps"]
    assert "above 0" in rows["eb-t-max"]
    assert "at least 0" in rows["eb-bound"] and "at least 0" in rows["eb-confidence"]


def test_the_limits_in_the_documents_match_the_rule_functions() -> None:
    rules = text("src/mojo_dllm/diffusion/sampler.mojo")
    assert 'flag == "eb-t-max" or flag == "prev-temp-inv"' in rules and '"must be above 0" if x <= 0' in rules
    assert 'if flag == "eb-stability"' in rules and '"must be at least 0" if n < 0' in rules
    assert '"must be at least 1" if n < 1' in rules


# ---- the versions under a benchmark table -----------------------------------------------------------------------


def test_the_version_line_names_each_commit_with_the_file_it_comes_from() -> None:
    expected = "mojo-dllm `bdd268b` (cpu-llada) and `dd0a647` (cpu-diffuse, diffuse-cpp rows only), Mojo 1.1.0"
    for path in ("README.md", "docs/benchmarks.md"):
        assert expected in text(path), path


def test_a_single_commit_is_named_once() -> None:
    sources = [("2026-10-03-cpu-dream.json", {"mojo-dllm": "abc1234"}, ["mojo-dllm", "llama.cpp"])]
    assert tables.mojo_dllm_versions(sources) == "`abc1234`"


def test_two_commits_name_their_files_and_say_which_rows_one_of_them_holds() -> None:
    sources = [
        ("2026-10-03-cpu-diffuse.json", {"mojo-dllm": "dd0a647"}, ["diffuse-cpp", "diffuse-cpp-cache-entropy-exit"]),
        ("2026-10-03-cpu-llada.json", {"mojo-dllm": "bdd268b"}, ["mojo-dllm", "llama.cpp"]),
    ]
    assert tables.mojo_dllm_versions(sources) == "`bdd268b` (cpu-llada) and `dd0a647` (cpu-diffuse, diffuse-cpp rows only)"


def test_two_files_with_one_commit_do_not_repeat_it() -> None:
    sources = [
        ("2026-10-03-cpu-a.json", {"mojo-dllm": "abc1234"}, ["mojo-dllm"]),
        ("2026-10-04-cpu-b.json", {"mojo-dllm": "abc1234"}, ["llama.cpp"]),
    ]
    assert tables.mojo_dllm_versions(sources) == "`abc1234`"
