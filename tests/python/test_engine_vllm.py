"""bench/engines/vllm.py: a client for a RUNNING vLLM server's OpenAI-compatible endpoint, as a GSM8K engine.

The server is the fake in fake_vllm_server.py (standard library, 127.0.0.1, ephemeral port). No vLLM, no GPU, no
weights, no model download. Field names were read from vLLM's source: `GET /version` answers `{"version": ...}`
(vllm/entrypoints/serve/instrumentator/basic.py), `GET /v1/models` answers `id`, `root` and `max_model_len` per
model (vllm/entrypoints/openai/models/serving.py), and `/metrics` carries the counter
`vllm:diffusion_num_denoising_steps_total` for a diffusion model (vllm/v1/spec_decode/metrics.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import engines  # noqa: E402
import eval_engine  # noqa: E402
import eval_gsm8k  # noqa: E402
from engines import EngineError, GenConfig  # noqa: E402
from engines import vllm as vllm_engine  # noqa: E402
from engines import _http  # noqa: E402
from fake_vllm_server import ANSWER, FakeServer  # noqa: E402

RECORD_KEYS = {"index", "gold", "pred", "correct", "forward_passes", "generate_s", "stopped_early", "text",
               "flags", "version", "model", "power_retries", "on_battery", "power_source", "isolation", "schema", "steps",
               "completion_tokens", "finish_reason"}
CHECKPOINT = "google/diffusiongemma-26B-A4B-it"
MODEL = CHECKPOINT


class Rig(SimpleNamespace):
    def argv(self, *extra: str, arm: str = "arm", limit: int = 2, quant: str | None = "fp8-dynamic",
             served: str | None = "dgemma", model: str = MODEL, seqs: str | None = "1") -> list[str]:
        argv = ["run", "--engine", "vllm", "--model", model, "--arm", arm, "--device", "cpu",
                "--gen-length", "256", "--limit", str(limit), "--allow-battery", "--base-url", self.server.url]
        if served is not None:
            argv += ["--served-model", served]
        if quant is not None:
            argv += ["--quantization", quant]
        if seqs is not None:
            argv += ["--max-num-seqs", seqs]
        return argv + list(extra)

    def run(self, *extra: str, **kw) -> int:
        return eval_engine.main(self.argv(*extra, **kw))

    def rows(self, arm: str = "arm") -> list[dict]:
        path = self.runs / f"{arm}.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def engine(self, **kw) -> vllm_engine.VLLM:
        args = dict(base_url=self.server.url, served_model="dgemma", checkpoint=None, quantization="fp8-dynamic",
                    attention_backend=None, chat_template=None, use_env_proxy=False, max_num_seqs=1)
        args.update(kw)
        return vllm_engine.VLLM(SimpleNamespace(**args))

    def cfg(self, **kw) -> GenConfig:
        return GenConfig(**{"model": MODEL, "device": "cpu", "gen_length": 256, "steps": 256, **kw})

    def ready(self, **kw):
        """An engine that has passed `check`, and its config."""
        e, cfg = self.engine(**kw), self.cfg()
        e.check(cfg)
        return e, cfg


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runs = tmp_path / "runs"
    (tmp_path / "bench" / "eval").mkdir(parents=True)
    monkeypatch.setattr(eval_gsm8k, "RUNS", runs)
    monkeypatch.setattr(eval_gsm8k, "PAUSE", runs / "PAUSE")
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    with FakeServer() as server:
        yield Rig(server=server, runs=runs, root=tmp_path)


def refused(rig: Rig, fragment: str, *extra: str, **kw) -> None:
    with pytest.raises(SystemExit) as e:
        rig.run(*extra, **kw)
    assert fragment in str(e.value), str(e.value)
    assert rig.rows() == []


# registry


def test_vllm_is_registered_and_satisfies_the_protocol(rig: Rig) -> None:
    assert engines.get("vllm") is vllm_engine.VLLM
    assert "vllm" in engines.names()
    assert isinstance(rig.engine(), engines.Engine)


# the request


def test_the_request_body_has_the_chat_shape_at_temperature_zero(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    post = rig.server.state.posts()[0]
    assert post["path"] == "/v1/chat/completions"
    body = json.loads(post["body"])
    first = json.loads(eval_gsm8k.DATA.read_text().splitlines()[0])
    assert body == {"model": "dgemma", "temperature": 0, "max_tokens": 256, "stream": False,
                    "messages": [{"role": "user", "content": eval_gsm8k.PROMPT.format(question=first["question"])}]}


def test_max_tokens_is_the_generation_length(rig: Rig) -> None:
    e, cfg = rig.ready()
    e.generate("q", rig.cfg(gen_length=64, steps=64))
    assert json.loads(rig.server.state.posts()[0]["body"])["max_tokens"] == 64


def test_the_temperature_is_exactly_zero(rig: Rig) -> None:
    e, cfg = rig.ready()
    e.generate("q", cfg)
    t = json.loads(rig.server.state.posts()[0]["body"])["temperature"]
    assert t == 0 and not isinstance(t, bool)


def test_no_authorization_header_is_sent_to_any_endpoint(rig: Rig) -> None:
    rig.run(limit=1)
    assert rig.server.state.requests
    for r in rig.server.state.requests:
        assert not {k.lower() for k in r["headers"]} & {"authorization", "cookie", "x-api-key"}


# the record


def test_the_record_has_exactly_the_keys_of_the_other_arms(rig: Rig) -> None:
    assert rig.run() == 0
    rows = rig.rows()
    assert len(rows) == 2 and all(set(r) == RECORD_KEYS for r in rows)


def test_the_record_values(rig: Rig) -> None:
    rig.run(limit=1)
    r = rig.rows()[0]
    assert r["text"] == ANSWER and r["pred"] == "18" and r["model"] == "diffusiongemma-26B-A4B-it|fp8-dynamic"
    assert isinstance(r["generate_s"], float) and r["generate_s"] > 0


def test_the_model_name_carries_the_quantization(rig: Rig) -> None:
    assert rig.engine(quantization="fp8-dynamic").model_name(rig.cfg()) == "diffusiongemma-26B-A4B-it|fp8-dynamic"
    assert rig.engine(quantization="bf16").model_name(rig.cfg()) == "diffusiongemma-26B-A4B-it|bf16"


def test_arms_of_two_quantizations_differ_in_weights_for_compare(rig: Rig) -> None:
    rig.run(arm="fp8", quant="fp8-dynamic")
    rig.run(arm="bf16", quant="bf16")
    assert {r["model"] for r in rig.rows("fp8")} != {r["model"] for r in rig.rows("bf16")}
    with pytest.raises(SystemExit, match="--across-weights"):
        eval_gsm8k.cmd_compare("fp8", "bf16")
    assert eval_gsm8k.cmd_compare("fp8", "bf16", across_weights=True) == 0
    doc = json.loads(next((rig.root / "bench" / "eval").glob("*.json")).read_text())
    assert doc["weights_differ"] is True


def test_a_resume_with_another_quantization_is_refused(rig: Rig) -> None:
    rig.run(limit=1)
    with pytest.raises(SystemExit, match="Use a new arm name"):
        rig.run(limit=2, quant="bf16")


def test_the_engine_needs_no_model_file(rig: Rig) -> None:
    # --model is a name only: nothing is read from disk and the preflight is not sized from a file
    assert not Path(MODEL).exists()
    assert rig.run(limit=1) == 0


def test_the_isolation_preflight_asks_for_a_small_fixed_memory(rig: Rig) -> None:
    need = rig.engine().ram_gb_after_load(rig.cfg())
    assert isinstance(need, float) and 0 < need <= 1.0


# version and flags


def test_version_records_the_release_the_server_reports_and_the_declared_fields(rig: Rig) -> None:
    e, _ = rig.ready(checkpoint=CHECKPOINT, attention_backend="TRITON_ATTN")
    assert json.loads(e.version()) == {"vllm": "0.24.0", "attention_backend": "TRITON_ATTN",
                                       "checkpoint": CHECKPOINT, "quantization": "fp8-dynamic"}


def test_version_records_null_for_an_attention_backend_nobody_declared(rig: Rig) -> None:
    e, _ = rig.ready()
    v = json.loads(e.version())
    assert v["attention_backend"] is None and v["checkpoint"] == "google/diffusiongemma-26B-A4B-it"


def test_flags_say_remote_and_which_fields_are_declared_and_not_verified(rig: Rig) -> None:
    e, cfg = rig.ready(attention_backend="FLASH_ATTN")
    f = e.flags(cfg)
    assert f["engine"] == "vllm" and f["server"] == "remote"
    assert f["declared_not_verified"] == ["attention_backend", "max_num_seqs", "quantization"]
    assert f["max_num_seqs"] == 1
    assert f["temperature"] == 0.0 and f["gen_length"] == 256 and f["device"] == "cpu"
    assert f["served_model"] == "dgemma" and f["quantization"] == "fp8-dynamic"
    assert f["chat_template"] == {"applied_by": "server", "declared": None, "verified": False}


def test_flags_leave_the_server_address_out(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert "127.0.0.1" not in json.dumps(e.flags(cfg)) and "127.0.0.1" not in e.version()


def test_a_declared_chat_template_is_recorded_as_declared(rig: Rig) -> None:
    e, cfg = rig.ready(chat_template="examples/tool_chat_template_gemma4.jinja")
    assert e.flags(cfg)["chat_template"] == {"applied_by": "server", "verified": False,
                                             "declared": "examples/tool_chat_template_gemma4.jinja"}


def test_the_checkpoint_the_server_reports_is_recorded_and_compared_with_the_declared_one(rig: Rig) -> None:
    e, cfg = rig.ready(checkpoint=CHECKPOINT)
    f = e.flags(cfg)
    assert f["checkpoint_reported_by_server"] == CHECKPOINT and f["checkpoint_matches_server"] is True


def test_a_checkpoint_that_differs_from_the_servers_root_is_refused(rig: Rig) -> None:
    refused(rig, "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic", "--checkpoint",
            "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic")


# steps


def test_forward_passes_is_null_when_the_server_has_no_metrics_endpoint(rig: Rig) -> None:
    rig.run(limit=1)
    r = rig.rows()[0]
    assert r["forward_passes"] is None
    e, cfg = rig.ready()
    assert e.flags(cfg)["forward_passes_measured"] is False


def test_forward_passes_is_null_when_the_metrics_have_no_denoising_counter(rig: Rig) -> None:
    rig.server.state.metrics = "# HELP vllm:num_requests_running x\nvllm:num_requests_running 0\n"
    rig.run(limit=1)
    assert rig.rows()[0]["forward_passes"] is None


def test_forward_passes_is_the_counter_delta_when_the_server_exposes_it(rig: Rig) -> None:
    rig.server.state.metrics = "live"       # the counter starts at 100 and grows by 7 per request
    e, cfg = rig.ready()
    assert e.flags(cfg)["forward_passes_measured"] is True
    for _ in range(2):
        g = e.generate("q", cfg)
        assert g["forward_passes"] == 7 and g["steps"] == 7


def test_the_delta_reaches_the_record(rig: Rig) -> None:
    rig.server.state.metrics = "live"
    rig.run(limit=2)
    assert [r["forward_passes"] for r in rig.rows()] == [7, 7]


def test_the_counter_is_summed_over_label_sets(rig: Rig) -> None:
    text = ('vllm:diffusion_num_denoising_steps_total{engine="0"} 40.0\n'
            'vllm:diffusion_num_denoising_steps_total{engine="1"} 2.0\n'
            'vllm:diffusion_num_denoising_steps_created{engine="0"} 1.7e9\n')
    assert vllm_engine.parse_steps_total(text) == 42


def test_a_counter_that_does_not_move_is_refused(rig: Rig) -> None:
    rig.server.state.metrics = "live"
    rig.server.state.mode = "steps_stuck"
    e, cfg = rig.ready()
    with pytest.raises(EngineError, match="denoising"):
        e.generate("q", cfg)


# refusals at check


def test_a_server_that_reports_another_model_is_refused_before_any_request_for_a_completion(rig: Rig) -> None:
    refused(rig, "dgemma", served="not-dgemma")
    assert rig.server.state.posts() == []


def test_the_refusal_names_the_models_the_server_reports(rig: Rig) -> None:
    rig.server.state.models = [{"id": "llama-3", "root": "x"}, {"id": "mistral", "root": "y"}]
    refused(rig, "llama-3, mistral")


def test_a_served_model_among_several_is_accepted(rig: Rig) -> None:
    rig.server.state.models = [{"id": "other", "root": "x"}, {"id": "dgemma", "root": CHECKPOINT}]
    assert rig.run(limit=1) == 0


def test_a_server_that_lists_no_models_is_refused(rig: Rig) -> None:
    rig.server.state.models = []
    refused(rig, "no models")


def test_a_malformed_model_list_is_refused(rig: Rig) -> None:
    rig.server.state.models = [{"nope": 1}]
    refused(rig, "malformed")


def test_the_served_model_option_is_required(rig: Rig) -> None:
    refused(rig, "--served-model", served=None)


def test_the_quantization_option_is_required(rig: Rig) -> None:
    refused(rig, "--quantization", quant=None)


def test_an_http_error_from_the_model_list_is_refused_with_the_servers_message(rig: Rig) -> None:
    rig.server.state.models_status = 503
    refused(rig, "models are down")


def test_a_server_that_cannot_report_its_version_is_refused(rig: Rig) -> None:
    rig.server.state.version = 404
    refused(rig, "/version")


def test_a_version_answer_without_a_version_is_refused(rig: Rig) -> None:
    rig.server.state.version = {"detail": "x"}
    refused(rig, "/version")


def test_an_unreachable_server_is_refused_at_check(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="127.0.0.1:9"):
        eval_engine.main(rig.argv() + ["--base-url", "http://127.0.0.1:9"])


def test_a_base_url_with_credentials_is_refused(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="credentials"):
        eval_engine.main(rig.argv() + ["--base-url", "http://user:pw@127.0.0.1:8000"])


def test_extra_engine_flags_are_refused(rig: Rig) -> None:
    refused(rig, "extra engine flags", "--", "--foo")


def test_a_device_other_than_cpu_or_gpu_is_refused(rig: Rig) -> None:
    e = rig.engine()
    cfg = rig.cfg()
    object.__setattr__(cfg, "device", "tpu")   # GenConfig refuses it at construction; the adapter's own guard stays
    with pytest.raises(SystemExit, match="cpu or gpu"):
        e.check(cfg)


# refusals at generate


def test_an_http_error_stops_the_run_with_the_servers_message_and_stores_nothing(rig: Rig) -> None:
    rig.server.state.mode = "http500"
    refused(rig, "CUDA out of memory in the denoiser")


def test_an_http_error_is_not_retried(rig: Rig) -> None:
    rig.server.state.mode = "http500"
    with pytest.raises(SystemExit):
        rig.run()
    assert len(rig.server.state.posts()) == 1


@pytest.mark.parametrize("mode", ["empty", "blank", "null_content"])
def test_an_empty_completion_is_refused_by_the_engine(rig: Rig, mode: str) -> None:
    rig.server.state.mode = mode
    e, cfg = rig.ready()
    with pytest.raises(EngineError, match="empty"):
        e.generate("q", cfg)


def test_an_empty_completion_stops_a_run_and_stores_nothing(rig: Rig) -> None:
    rig.server.state.mode = "empty"
    refused(rig, "empty")


def test_a_response_that_echoes_another_model_is_refused(rig: Rig) -> None:
    rig.server.state.mode = "wrong_echo"
    e, cfg = rig.ready()
    with pytest.raises(EngineError, match="some-other-model"):
        e.generate("q", cfg)


@pytest.mark.parametrize("mode", ["no_choices", "no_usage", "zero_tokens"])
def test_a_response_without_choices_or_token_counts_is_refused(rig: Rig, mode: str) -> None:
    rig.server.state.mode = mode
    e, cfg = rig.ready()
    with pytest.raises(EngineError):
        e.generate("q", cfg)


def test_the_token_counts_are_read_from_the_response(rig: Rig) -> None:
    e, cfg = rig.ready()
    g = e.generate("q", cfg)
    assert g["completion_tokens"] == 9 and g["prompt_tokens"] == 41 and g["text"] == ANSWER


def test_a_response_over_the_size_bound_is_refused(rig: Rig) -> None:
    rig.server.state.mode = "huge"
    e, cfg = rig.ready()
    with pytest.raises(EngineError, match="larger than"):
        e.generate("q", cfg)


def test_a_slow_generation_stops_the_run_and_names_the_timeout(rig: Rig) -> None:
    rig.server.state.mode = "sleep"
    refused(rig, "timed out", "--timeout-s", "0.4")


def test_generate_s_covers_the_request_only(rig: Rig) -> None:
    e, cfg = rig.ready()
    g = e.generate("q", cfg)
    assert 0 < g["generate_s"] < 5


def test_a_step_count_the_server_cannot_take_is_refused(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="--steps"):
        rig.engine().check(rig.cfg(steps=48))


# review fixes: one client, sanitised server text, label against root, the metrics status, the version string


def test_steps_are_refused_without_a_declared_max_num_seqs_of_one(rig: Rig) -> None:
    rig.server.state.metrics = "live"
    refused(rig, "--max-num-seqs 1", seqs=None)
    assert rig.server.state.posts() == []


def test_steps_are_refused_when_the_declared_max_num_seqs_is_four(rig: Rig) -> None:
    rig.server.state.metrics = "live"
    refused(rig, "--max-num-seqs 1", seqs="4")


def test_the_declared_max_num_seqs_is_not_needed_when_the_server_has_no_counter(rig: Rig) -> None:
    assert rig.run(limit=1, seqs=None) == 0
    assert rig.rows()[0]["forward_passes"] is None


def test_max_num_seqs_is_recorded_in_the_flags_when_declared(rig: Rig) -> None:
    rig.server.state.metrics = "live"
    rig.run(limit=1)
    f = rig.rows()[0]["flags"]
    assert f["max_num_seqs"] == 1 and "max_num_seqs" in f["declared_not_verified"]


def test_a_non_integer_max_num_seqs_is_rejected_by_the_parser(rig: Rig) -> None:
    with pytest.raises(SystemExit):
        rig.run("--max-num-seqs", "many", seqs=None)


def test_the_counter_is_read_under_its_source_name_and_its_total_name() -> None:
    assert vllm_engine.parse_steps_total('vllm:diffusion_num_denoising_steps{engine="0"} 5.0\n') == 5
    assert vllm_engine.parse_steps_total('vllm:diffusion_num_denoising_steps_total{engine="0"} 6.0\n') == 6


def test_both_names_of_one_counter_are_not_added_together() -> None:
    text = "vllm:diffusion_num_denoising_steps_total 6.0\nvllm:diffusion_num_denoising_steps 6.0\n"
    assert vllm_engine.parse_steps_total(text) == 6


def test_a_page_without_the_counter_gives_none() -> None:
    assert vllm_engine.parse_steps_total("vllm:diffusion_num_denoising_steps_created 1.7e9\n") is None


@pytest.mark.parametrize("status", [403, 500, 401])
def test_a_metrics_error_other_than_404_is_refused_not_read_as_an_absent_counter(rig: Rig, status: int) -> None:
    rig.server.state.metrics_status = status
    refused(rig, f"HTTP {status}")


def test_an_empty_version_string_is_refused(rig: Rig) -> None:
    rig.server.state.version = {"version": ""}
    refused(rig, "/version")


def test_a_model_label_that_is_not_the_servers_root_or_id_is_refused(rig: Rig) -> None:
    refused(rig, "google/diffusiongemma-26B-A4B-it", model="RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic")
    assert rig.server.state.posts() == []


def test_the_message_for_a_label_mismatch_names_the_label_and_the_root(rig: Rig) -> None:
    with pytest.raises(SystemExit) as e:
        rig.run(model="other/model")
    assert "other/model" in str(e.value) and CHECKPOINT in str(e.value) and "--model" in str(e.value)


def test_a_model_label_equal_to_the_servers_id_is_accepted(rig: Rig) -> None:
    assert rig.run(limit=1, model="dgemma") == 0
    assert rig.rows()[0]["model"] == "dgemma|fp8-dynamic"


def test_flags_record_that_the_label_matches_the_server(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert e.flags(cfg)["model_matches_server"] is True


def test_a_server_without_a_root_cannot_confirm_the_label(rig: Rig) -> None:
    rig.server.state.models = [{"id": "dgemma"}]
    e, cfg = rig.ready()
    assert e.flags(cfg)["model_matches_server"] is None


def test_model_ids_the_server_reports_cannot_inject_escapes(rig: Rig) -> None:
    rig.server.state.models = [{"id": "a\x1b[31mred", "root": "x"}, {"id": "b\rc", "root": "y"}]
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "\x1b" not in str(e.value) and "\r" not in str(e.value) and "a[31mred" in str(e.value)


def test_an_http_error_message_with_escapes_is_clean_through_the_runner(rig: Rig) -> None:
    rig.server.state.mode = "escape"
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "\x1b" not in str(e.value) and "\x07" not in str(e.value)


def test_a_wrong_echoed_model_name_is_cleaned_and_capped(rig: Rig) -> None:
    rig.server.state.mode = "wrong_echo"
    e, cfg = rig.ready()
    with pytest.raises(EngineError) as ex:
        e.generate("q", cfg)
    assert len(str(ex.value)) < 300


def test_the_http_helper_is_shared(rig: Rig) -> None:
    assert vllm_engine.HttpClient is _http.HttpClient
