"""scripts/gen_fixtures.py reproduces every committed byte, in any order, and removes what it no longer makes (#253).

Needs numpy and gguf (the `fixtures` group): the module is skipped, with that reason shown, when either is missing, and
CI runs it in the step that installs the extra groups.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="numpy is in the `fixtures` group, not the `test` group; CI runs this file there")
pytest.importorskip("gguf", reason="gguf is in the `fixtures` group, not the `test` group; CI runs this file there")

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"
sys.path.insert(0, str(ROOT / "scripts"))


def load():
    spec = importlib.util.spec_from_file_location("gen_fixtures_t", ROOT / "scripts/gen_fixtures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_fixtures_t"] = module
    spec.loader.exec_module(module)
    return module


gf = load()


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "fixtures"
    target.mkdir()
    monkeypatch.setattr(gf, "OUT", target)
    return target


def run_main() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        gf.main()
    return buf.getvalue()


def test_the_three_functions_that_shared_a_stream_give_the_committed_bytes_in_any_order(out: Path) -> None:
    for order in ((gf.kquants, gf.tiny_llada, gf.tiny_dream), (gf.tiny_dream, gf.tiny_llada, gf.kquants),
                  (gf.tiny_llada, gf.tiny_dream, gf.kquants)):
        for f in out.iterdir():
            f.unlink()
        with redirect_stdout(io.StringIO()):
            for fn in order:
                fn()
        for name in ("kquants.gguf", "tiny-llada.gguf", "tiny-dream.gguf"):
            assert (out / name).read_bytes() == (FIXTURES / name).read_bytes(), (name, [f.__name__ for f in order])


def test_a_function_run_twice_gives_the_same_bytes(out: Path) -> None:
    with redirect_stdout(io.StringIO()):
        gf.tiny_llada()
        first = (out / "tiny-llada.gguf").read_bytes()
        gf.tiny_llada()
    assert (out / "tiny-llada.gguf").read_bytes() == first == (FIXTURES / "tiny-llada.gguf").read_bytes()


def test_the_stream_positions_are_where_the_shared_stream_stood() -> None:
    import numpy as np
    shared = np.random.default_rng(gf.SEED)
    assert gf.stream(0).bit_generator.state == shared.bit_generator.state
    assert gf.stream(5).uniform() == pytest.approx(np.random.default_rng(gf.SEED).uniform(size=6)[5])
    assert (gf.KQUANTS_SKIP, gf.TINY_LLADA_SKIP, gf.TINY_DREAM_SKIP) == (0, 548, 149284)


def test_a_full_regeneration_is_byte_identical_to_every_committed_fixture(out: Path) -> None:
    run_main()
    made = [line for line in (out / gf.MANIFEST).read_text().splitlines()]
    assert len(made) > 200 and made == sorted(made)
    different = [rel for rel in made if (out / rel).read_bytes() != (FIXTURES / rel).read_bytes()]
    assert different == []
    assert (FIXTURES / gf.MANIFEST).read_text() == (out / gf.MANIFEST).read_text()
    for rel in made:
        assert (FIXTURES / rel).is_file()


def test_a_file_an_earlier_run_made_and_this_one_does_not_is_removed(out: Path) -> None:
    run_main()
    (out / "gone.gguf").write_bytes(b"x")
    (out / "old-dir").mkdir()
    (out / "old-dir" / "x.gguf").write_bytes(b"y")
    (out / gf.MANIFEST).write_text((out / gf.MANIFEST).read_text() + "gone.gguf\nold-dir/x.gguf\n")
    text = run_main()
    assert not (out / "gone.gguf").exists() and not (out / "old-dir").exists()
    assert "removed stale" in text and "gone.gguf" in text
    assert "gone.gguf" not in (out / gf.MANIFEST).read_text()


def test_a_file_another_generator_wrote_is_left_alone(out: Path) -> None:
    run_main()
    (out / "moe-q4-q5.bin").write_bytes(b"from gen_moe_fixture")
    (out / "dream-tokenizer.gguf").write_bytes(b"from gen_tokenizer_fixture")
    run_main()
    assert (out / "moe-q4-q5.bin").read_bytes() == b"from gen_moe_fixture"
    assert (out / "dream-tokenizer.gguf").read_bytes() == b"from gen_tokenizer_fixture"


def test_an_unchanged_file_is_not_rewritten(out: Path) -> None:
    run_main()
    target = out / "kquants.gguf"
    before = target.stat().st_mtime_ns
    run_main()
    assert target.stat().st_mtime_ns == before


def test_a_failed_generation_leaves_the_folder_as_it_was(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_main()
    snapshot = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}

    def boom() -> None:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(gf, "entropy_rows", boom)
    with pytest.raises(RuntimeError):
        run_main()
    assert gf.OUT == out
    assert {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()} == snapshot
    assert not [p for p in out.parent.iterdir() if p.name.startswith("fixtures-")]
