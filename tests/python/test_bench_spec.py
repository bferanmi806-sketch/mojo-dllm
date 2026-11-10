"""One module holds what several benchmark tools repeat (#244): ids, prompts, llama.cpp settings, seeds, timeouts."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "scripts"))
import spec  # noqa: E402  (bench/spec.py)


def load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    assert s is not None and s.loader is not None
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


rb = load("run_bench_spec", ROOT / "bench/run_bench.py")
parity = load("parity_spec", ROOT / "bench/parity.py")
dispatch = load("check_gpu_dispatch_spec", ROOT / "bench/check_gpu_dispatch.py")
compare_logits = load("compare_logits_spec", ROOT / "scripts/compare_logits.py")
quality = load("quality_spec", ROOT / "bench/quality.py")
import engines  # noqa: E402  (bench/engines)
from engines import EngineError, GenConfig  # noqa: E402
from engines import hf, llamacpp  # noqa: E402,F401
import eval_gsm8k  # noqa: E402  (bench/eval_gsm8k.py)


def test_every_tool_reads_the_same_ids() -> None:
    assert parity.SPECS is spec.ARCH_IDS
    assert dispatch.ARCH_IDS is spec.ARCH_IDS
    assert compare_logits.MASK == spec.ARCH_IDS["llada"]["mask"]
    assert hf.LLADA_MASK_ID == spec.ARCH_IDS["llada"]["mask"]


def test_the_ids_are_the_published_ones() -> None:
    assert spec.ARCH_IDS == {"llada": dict(vocab=126464, mask=126336), "dream": dict(vocab=152064, mask=151666)}


def test_every_committed_config_lists_a_prefix_of_the_prompts() -> None:
    configs = sorted((ROOT / "bench").glob("config*.json"))
    assert len(configs) >= 8
    for path in configs:
        prompts = json.loads(path.read_text())["prompts"]
        assert prompts == list(spec.PROMPTS[: len(prompts)]), path.name


def test_the_other_tools_take_their_prompts_from_the_spec() -> None:
    assert quality.PROMPTS == list(spec.PROMPTS[:2])
    assert parity.LOGIT_CASES[0].prompt == spec.PROMPTS[0]
    assert parity.LOGIT_CASES[1].prompt == spec.PROMPTS[2]
    assert [c.prompt for c in parity.GEN_CASES] == list(spec.PROMPTS[:2])


def test_the_default_prompt_of_the_scripts_is_the_first() -> None:
    for name in ("compare_logits.py", "gpu_profile.py"):
        text = (ROOT / "scripts" / name).read_text()
        assert "spec.PROMPTS[0]" in text, name
        assert '"Explain speculative decoding in two sentences."' not in text, name


@pytest.mark.parametrize("cuda", [False, True])
def test_run_bench_and_the_engine_build_the_same_llama_command(monkeypatch: pytest.MonkeyPatch, cuda: bool) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(rb, "timed", lambda cmd: (seen.append(cmd) or (1, "", "stop", 0.0, 0)))
    cfg = {"llama_cpp": "/llama", "models": {"gguf": "m.gguf"}, "steps": 32}
    rb.run_llama(cfg, "hello", 6, 72, cuda=cuda)
    engine = llamacpp.LlamaCpp.__new__(llamacpp.LlamaCpp)
    engine.llama_cpp, engine.threads = Path("/llama"), 6
    mine = engine.command("hello", GenConfig(model="m.gguf", device="gpu" if cuda else "cpu", steps=32), 72)
    assert seen[0][1:] == mine[1:]
    assert mine[mine.index("--diffusion-eps") + 1] == spec.LLAMA_EPS


def test_the_llama_flags_are_these() -> None:
    assert spec.llama_diffusion_args(72, 32, 6, True) == [
        "-c", "72", "-b", "72", "-ub", "72", "--diffusion-steps", "32", "--diffusion-eps", "0.001", "--temp", "0",
        "-t", "6", "-tb", "6", "--seed", "42", "-ngl", "99", "-v"]
    assert spec.llama_diffusion_args(72, 32, 6, False, "block", 16)[8:10] == ["--diffusion-block-length", "16"]
    assert "-ngl" not in spec.llama_diffusion_args(72, 32, 6, False)


def test_one_timeout_default_and_one_seed_set() -> None:
    assert GenConfig(model="m").timeout_s == spec.GENERATION_TIMEOUT_S
    assert eval_gsm8k.DEFAULT_TIMEOUT_S == spec.GENERATION_TIMEOUT_S
    assert parity.DG_CANVAS_SEED == spec.DG_CANVAS_SEED and parity.DG_NOISE_SEED == spec.DG_NOISE_SEED
    assert parity.DG_PREV_TEMP_INV == 1.0 / spec.DG_FIRST_STEP_TEMP
    assert "1.0 / 0.8" not in (ROOT / "bench/parity.py").read_text()


def test_gen_fixtures_uses_the_shared_temperature() -> None:
    tree = ast.parse((ROOT / "scripts/gen_fixtures.py").read_text())
    assert "1.0 / 0.8" not in ast.unparse(tree)


def test_a_device_outside_cpu_and_gpu_is_refused_by_the_config() -> None:
    with pytest.raises(EngineError, match="device"):
        GenConfig(model="m", device="tpu")
    assert GenConfig(model="m", device="cpu").device == "cpu"
    with pytest.raises(EngineError, match="schedule"):
        GenConfig(model="m", schedule="cosine")


def test_an_unknown_runtime_name_is_refused_by_name() -> None:
    with pytest.raises(SystemExit) as e:
        rb.select_runtimes(["mojo-dllm", "mojo-dlm-gpu"], "", 1)
    assert "mojo-dlm-gpu" in str(e.value)


def test_the_thread_count_in_a_runtime_name_is_a_positive_integer() -> None:
    rb.check_runtime_names(["llama.cpp:t12", "llama-bench:t4"])
    for bad in ("llama.cpp:t0", "llama.cpp:t", "llama.cpp:tx", "llama-bench:t-1", "llama.cpp:t4 "):
        with pytest.raises(SystemExit):
            rb.check_runtime_names([bad])


def test_the_committed_configs_name_only_known_runtimes() -> None:
    for path in sorted((ROOT / "bench").glob("config*.json")):
        names = json.loads(path.read_text())["runtimes"]
        assert rb.select_runtimes(names, "", 1) == names, path.name


def test_an_unknown_runtime_fails_before_the_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def must_not_start(*a: object, **k: object) -> None:
        raise AssertionError("the preflight started")

    monkeypatch.setattr(rb.isolation, "begin_arm", must_not_start)
    monkeypatch.setattr(rb.isolation, "tuned", must_not_start)
    monkeypatch.setattr(rb.isolation, "get_machine", must_not_start)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    config = tmp_path / "c.json"
    config.write_text(json.dumps({
        "models": {"gguf": str(model)}, "llama_cpp": "/x", "diffuse_cpp": "/y", "runtimes": ["mojo-dllm", "mojo-dlm"],
        "prompts": ["p"], "gen_length": 8, "steps": 8, "block_length": 8, "threads": 1,
        "busy_threshold": 0.1, "max_quiet_wait_s": 1}))
    monkeypatch.setattr(sys, "argv", ["run_bench", "--config", str(config), "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit) as e:
        rb.main()
    assert "mojo-dlm" in str(e.value)


def test_result_meta_has_an_integer_thread_count_and_the_versions() -> None:
    m = spec.result_meta({"mojo-dllm": "abc1234"}, "12")
    assert m["threads"] == 12 and isinstance(m["threads"], int)
    assert m["versions"] == {"mojo-dllm": "abc1234"} and m["date"][:2] == "20"
    for bad in ("x", 0, -3, None, True):
        with pytest.raises(ValueError, match="threads"):
            spec.result_meta({}, bad)


def test_quality_takes_the_meta_from_the_spec() -> None:
    assert "spec.result_meta" in (ROOT / "bench/quality.py").read_text()
