"""bench/engines/sglang.py: a client for a RUNNING SGLang server's OpenAI-compatible endpoint, as a GSM8K engine.

The server is the fake in fake_sglang_server.py (standard library, 127.0.0.1, ephemeral port). No SGLang, no GPU, no
weights, no model download. Field names were read from SGLang's source (main at
dfaca5681ddc20e31d50f2dfe5dfada269f12bbd and the release v0.5.23, 2026-11-09): `GET /v1/models` answers `id` and
`root`, both the served name; `GET /model_info` answers `model_path`; `GET /server_info` answers the resolved
server arguments plus `version`; there is no `GET /version`; and `Gemma4Renoise.validate_request` answers HTTP 400
to a request that carries logprobs, a grammar, a penalty, a seed or a logit bias.
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
from engines import sglang as sglang_engine  # noqa: E402
from engines import _http  # noqa: E402
from fake_sglang_server import ANSWER, CHECKPOINT, COUNTER_LOOKALIKE, FakeServer, rejected_fields  # noqa: E402

RECORD_KEYS = {"index", "gold", "pred", "correct", "forward_passes", "generate_s", "stopped_early", "text",
               "flags", "version", "model", "power_retries", "on_battery", "power_source", "isolation", "schema", "steps",
               "completion_tokens", "finish_reason"}
MODEL = CHECKPOINT

# every request-level field the server rejects or ignores for Gemma4Renoise, under the names a chat request and
# the scheduler use for them: the adapter must send none of them
DENIED = {"logprobs", "top_logprobs", "return_logprob", "top_logprobs_num", "token_ids_logprob",
          "return_flat_raw_top_logprobs", "return_flat_raw_output_top_logprobs", "response_format", "json_schema",
          "regex", "ebnf", "grammar", "structured_outputs", "structural_tag", "logit_bias", "frequency_penalty",
          "presence_penalty", "repetition_penalty", "min_p", "min_tokens", "min_new_tokens", "seed", "sampling_seed",
          "custom_logit_processor", "return_sampling_mask", "tools", "tool_choice"}
SAMPLING_CONTROLS = {"temperature", "top_p", "top_k"}      # accepted by the server and without effect, so not sent


class Rig(SimpleNamespace):
    def argv(self, *extra: str, arm: str = "arm", limit: int = 2, dtype: str | None = "bfloat16",
             served: str | None = None, model: str = MODEL, mrr: str | None = None, algorithm: str | None = None,
             checkpoint: str | None = None) -> list[str]:
        argv = ["run", "--engine", "sglang", "--model", model, "--arm", arm, "--device", "cpu",
                "--gen-length", "256", "--limit", str(limit), "--allow-battery", "--base-url", self.server.url]
        for flag, value in (("--served-model", served), ("--dtype", dtype), ("--max-running-requests", mrr),
                            ("--algorithm", algorithm), ("--checkpoint", checkpoint)):
            if value is not None:
                argv += [flag, value]
        return argv + list(extra)

    def run(self, *extra: str, **kw) -> int:
        return eval_engine.main(self.argv(*extra, **kw))

    def rows(self, arm: str = "arm") -> list[dict]:
        path = self.runs / f"{arm}.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def engine(self, **kw) -> sglang_engine.SGLang:
        args = dict(base_url=self.server.url, served_model=None, checkpoint=None, algorithm=None, dtype="bfloat16",
                    max_running_requests=None, chat_template=None, use_env_proxy=False)
        args.update(kw)
        return sglang_engine.SGLang(SimpleNamespace(**args))

    def cfg(self, **kw) -> GenConfig:
        return GenConfig(**{"model": MODEL, "device": "cpu", "gen_length": 256, "steps": 256, **kw})

    def ready(self, **kw):
        """An engine that has passed `check`, and its config."""
        e, cfg = self.engine(**kw), self.cfg()
        e.check(cfg)
        return e, cfg

    def info(self, **kw) -> None:
        """Change what GET /server_info answers; a value of `...` removes the key."""
        doc = dict(self.server.state.server_info)
        for k, v in kw.items():
            if v is ...:
                doc.pop(k, None)
            else:
                doc[k] = v
        self.server.state.server_info = doc


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


def keys(obj: object):
    """Every key of every mapping inside a decoded JSON document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from keys(v)


# registry


def test_sglang_is_registered_and_satisfies_the_protocol(rig: Rig) -> None:
    assert engines.get("sglang") is sglang_engine.SGLang
    assert "sglang" in engines.names()
    assert isinstance(rig.engine(), engines.Engine)


def test_the_unknown_engine_message_lists_sglang() -> None:
    with pytest.raises(SystemExit) as e:
        engines.get("nosuchengine")
    assert "sglang" in str(e.value)


def test_the_http_helper_is_shared_with_the_vllm_engine(rig: Rig) -> None:
    assert sglang_engine.HttpClient is _http.HttpClient


# the request


def test_the_request_body_has_exactly_these_fields(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    post = rig.server.state.posts()[0]
    assert post["path"] == "/v1/chat/completions"
    first = json.loads(eval_gsm8k.DATA.read_text().splitlines()[0])
    assert json.loads(post["body"]) == {
        "model": CHECKPOINT, "stream": False, "max_completion_tokens": 256,
        "messages": [{"role": "user", "content": eval_gsm8k.PROMPT.format(question=first["question"])}]}


def test_max_completion_tokens_is_the_generation_length(rig: Rig) -> None:
    e, _ = rig.ready()
    e.generate("q", rig.cfg(gen_length=64, steps=64))
    assert json.loads(rig.server.state.posts()[0]["body"])["max_completion_tokens"] == 64


def test_no_logprobs_or_grammar_field_is_ever_sent(rig: Rig) -> None:
    assert rig.run(limit=2) == 0
    posts = rig.server.state.posts()
    assert len(posts) == 2
    for p in posts:
        assert not DENIED & set(keys(json.loads(p["body"]))), p["body"]
    assert rig.server.state.rejects == []


def test_no_sampling_control_is_sent_because_the_renoise_schedule_governs_sampling(rig: Rig) -> None:
    rig.run(limit=1)
    assert not SAMPLING_CONTROLS & set(keys(json.loads(rig.server.state.posts()[0]["body"])))


@pytest.mark.parametrize("field,value", [("logprobs", True), ("top_logprobs", 3), ("response_format", {"type": "json_object"}),
                                         ("json_schema", {"x": 1}), ("regex", "a+"), ("ebnf", "root ::= x"),
                                         ("grammar", "g"), ("structured_outputs", {"regex": "a"}),
                                         ("logit_bias", {"1": 2.0}), ("frequency_penalty", 0.5), ("seed", 1)])
def test_the_fake_server_answers_400_to_each_field_the_real_one_rejects(rig: Rig, field: str, value: object) -> None:
    # the control for the two tests above: a request that carries the field is refused, so a client that sent it
    # would stop with the server's message
    client = _http.HttpClient(rig.server.url, timeout_s=5)
    with pytest.raises(_http.HttpError) as e:
        client.post_json("/v1/chat/completions", {"model": CHECKPOINT, "messages": [], field: value})
    assert e.value.status == 400 and field in str(e.value)


def test_the_fake_server_accepts_a_field_at_its_default_value() -> None:
    assert rejected_fields({"logprobs": False, "frequency_penalty": 0.0, "repetition_penalty": 1.0, "min_p": 0.0,
                            "seed": None}) == []


def test_a_server_that_rejects_the_request_stops_the_run_with_its_own_message(rig: Rig) -> None:
    rig.server.state.mode = "reject_all"
    refused(rig, "does not support request-level repetition_penalty; sampling is governed by the renoise schedule")
    assert len(rig.server.state.posts()) == 1


def test_the_servers_error_message_is_the_text_alone(rig: Rig) -> None:
    client = _http.HttpClient(rig.server.url, timeout_s=5)
    with pytest.raises(_http.HttpError) as e:
        client.post_json("/v1/chat/completions", {"model": CHECKPOINT, "messages": [], "logprobs": True})
    assert str(e.value) == (f"POST {rig.server.url}/v1/chat/completions answered HTTP 400: DiffusionGemma with "
                            "Gemma4Renoise does not support request-level logprobs; sampling is governed by the "
                            "renoise schedule.")


def test_no_authorization_header_is_sent_to_any_endpoint(rig: Rig) -> None:
    rig.run(limit=1)
    assert rig.server.state.requests
    for r in rig.server.state.requests:
        assert not {k.lower() for k in r["headers"]} & {"authorization", "cookie", "x-api-key"}


def test_the_engine_reads_three_endpoints_and_posts_chat_completions_only(rig: Rig) -> None:
    rig.run(limit=2)
    seen = sorted({(r["method"], r["path"]) for r in rig.server.state.requests})
    assert seen == [("GET", "/model_info"), ("GET", "/server_info"), ("GET", "/v1/models"),
                    ("POST", "/v1/chat/completions")]


# steps: SGLang has no request-level counter on main, so the engine records none and never infers one


def test_forward_passes_is_null_in_every_row(rig: Rig) -> None:
    rig.run(limit=2)
    assert [r["forward_passes"] for r in rig.rows()] == [None, None]


def test_forward_passes_stays_null_when_the_server_exposes_a_pass_counter(rig: Rig) -> None:
    rig.server.state.metrics = "live"
    rig.run(limit=2)
    assert [r["forward_passes"] for r in rig.rows()] == [None, None]
    assert COUNTER_LOOKALIKE in rig.server.state.metrics_text()
    assert all(r["path"] != "/metrics" for r in rig.server.state.requests)


def test_generate_returns_no_step_count(rig: Rig) -> None:
    e, cfg = rig.ready()
    g = e.generate("q", cfg)
    assert g["forward_passes"] is None and g["steps"] is None


def test_flags_say_that_forward_passes_are_not_measured(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert e.flags(cfg)["forward_passes_measured"] is False


# the record


def test_the_record_has_exactly_the_keys_of_the_other_arms(rig: Rig) -> None:
    assert rig.run() == 0
    rows = rig.rows()
    assert len(rows) == 2 and all(set(r) == RECORD_KEYS for r in rows)


def test_the_record_values(rig: Rig) -> None:
    rig.run(limit=1)
    r = rig.rows()[0]
    assert r["text"] == ANSWER and r["pred"] == "18" and r["model"] == "diffusiongemma-26B-A4B-it|bfloat16"
    assert isinstance(r["generate_s"], float) and r["generate_s"] > 0


def test_the_model_name_carries_the_dtype(rig: Rig) -> None:
    assert rig.engine(dtype="bfloat16").model_name(rig.cfg()) == "diffusiongemma-26B-A4B-it|bfloat16"
    assert rig.engine(dtype="float16").model_name(rig.cfg()) == "diffusiongemma-26B-A4B-it|float16"


def test_arms_of_two_dtypes_differ_in_weights_for_compare(rig: Rig) -> None:
    rig.run(arm="bf16", dtype="bfloat16")
    rig.run(arm="fp16", dtype="float16")
    assert {r["model"] for r in rig.rows("bf16")} != {r["model"] for r in rig.rows("fp16")}
    with pytest.raises(SystemExit, match="--across-weights"):
        eval_gsm8k.cmd_compare("bf16", "fp16")
    assert eval_gsm8k.cmd_compare("bf16", "fp16", across_weights=True) == 0
    doc = json.loads(next((rig.root / "bench" / "eval").glob("*.json")).read_text())
    assert doc["weights_differ"] is True


def test_a_resume_with_another_dtype_is_refused(rig: Rig) -> None:
    rig.run(limit=1)
    with pytest.raises(SystemExit, match="Use a new arm name"):
        rig.run(limit=2, dtype="float16")


def test_the_engine_needs_no_model_file(rig: Rig) -> None:
    assert not Path(MODEL).exists()
    assert rig.run(limit=1) == 0


def test_the_isolation_preflight_asks_for_a_small_fixed_memory(rig: Rig) -> None:
    need = rig.engine().ram_gb_after_load(rig.cfg())
    assert isinstance(need, float) and 0 < need <= 1.0


# version and flags


def test_version_records_the_release_the_server_reports_and_the_checkpoint_algorithm_and_dtype(rig: Rig) -> None:
    e, _ = rig.ready()
    assert json.loads(e.version()) == {"sglang": "0.5.23", "algorithm": "Gemma4Renoise", "checkpoint": CHECKPOINT,
                                       "dtype": "bfloat16"}


def test_version_is_read_from_server_info_not_from_a_version_route(rig: Rig) -> None:
    rig.ready()
    paths = [r["path"] for r in rig.server.state.requests]
    assert "/server_info" in paths and "/version" not in paths


def test_a_dtype_the_server_reports_is_used_and_needs_no_declaration(rig: Rig) -> None:
    rig.info(dtype="bfloat16")
    e, cfg = rig.ready(dtype=None)
    assert json.loads(e.version())["dtype"] == "bfloat16"
    assert "dtype" not in e.flags(cfg)["declared_not_verified"]
    assert rig.run(limit=1, dtype=None) == 0 and rig.rows()[0]["model"].endswith("|bfloat16")


def test_flags_say_remote_and_name_what_is_declared_and_not_verified(rig: Rig) -> None:
    e, cfg = rig.ready()
    f = e.flags(cfg)
    assert f["engine"] == "sglang" and f["server"] == "remote"
    assert f["declared_not_verified"] == ["dtype"]          # the server reports "auto", so bfloat16 is the operator's word
    assert f["dtype"] == "bfloat16" and f["algorithm"] == "Gemma4Renoise" and f["max_running_requests"] == 1
    assert f["gen_length"] == 256 and f["device"] == "cpu" and f["served_model"] == CHECKPOINT


def test_flags_list_every_declared_value_the_server_could_not_confirm(rig: Rig) -> None:
    rig.info(dllm_algorithm=..., max_running_requests=None, chat_template=...)
    rig.server.state.model_info = {"served_model_name": CHECKPOINT}
    e, cfg = rig.ready(algorithm="Gemma4Renoise", max_running_requests=1, checkpoint=CHECKPOINT,
                       chat_template="gemma4.jinja")
    assert e.flags(cfg)["declared_not_verified"] == ["algorithm", "chat_template", "checkpoint", "dtype",
                                                      "max_running_requests"]


def test_flags_record_what_the_server_reports_without_a_declaration(rig: Rig) -> None:
    rig.info(dtype="bfloat16")
    e, cfg = rig.ready(dtype=None)
    assert e.flags(cfg)["declared_not_verified"] == []


def test_flags_leave_the_server_address_and_the_launch_command_out(rig: Rig) -> None:
    e, cfg = rig.ready()
    text = json.dumps(e.flags(cfg)) + e.version()
    assert "127.0.0.1" not in text and "private-cache" not in text and "api-key" not in text


def test_the_record_holds_no_server_private_field(rig: Rig) -> None:
    rig.run(limit=1)
    r = rig.rows()[0]
    text = json.dumps({k: r[k] for k in ("flags", "version", "model")})
    assert "private-cache" not in text and "launch_command" not in text and "internal_states" not in text


def test_flags_say_that_sampling_is_the_servers(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert e.flags(cfg)["sampling"] == "set by the server's diffusion algorithm; temperature, top_p and top_k are not sent"


def test_flags_say_whether_a_custom_algorithm_config_is_set(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert e.flags(cfg)["algorithm_config_set"] is False
    rig.info(dllm_algorithm_config="/root/renoise.yaml")
    e, cfg = rig.ready()
    assert e.flags(cfg)["algorithm_config_set"] is True
    assert "renoise.yaml" not in json.dumps(e.flags(cfg))
    rig.info(dllm_algorithm_config=...)
    e, cfg = rig.ready()
    assert e.flags(cfg)["algorithm_config_set"] is None


def test_the_chat_template_is_the_servers_and_is_recorded_as_reported_and_declared(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert e.flags(cfg)["chat_template"] == {"applied_by": "server", "reported": None, "declared": None,
                                             "verified": False}
    rig.info(chat_template="gemma4.jinja")
    e, cfg = rig.ready(chat_template="gemma4.jinja")
    assert e.flags(cfg)["chat_template"] == {"applied_by": "server", "reported": "gemma4.jinja",
                                             "declared": "gemma4.jinja", "verified": False}


def test_a_declared_chat_template_the_server_denies_is_refused(rig: Rig) -> None:
    refused(rig, "--chat-template", "--chat-template", "gemma4.jinja")


def test_a_chat_template_the_server_reports_is_recorded_without_a_declaration(rig: Rig) -> None:
    rig.info(chat_template="other.jinja")
    e, cfg = rig.ready()
    assert e.flags(cfg)["chat_template"]["reported"] == "other.jinja"


# the model identity


def test_the_checkpoint_the_server_reports_is_recorded_and_compared_with_the_declared_one(rig: Rig) -> None:
    e, cfg = rig.ready(checkpoint=CHECKPOINT)
    f = e.flags(cfg)
    assert f["checkpoint_reported_by_server"] == CHECKPOINT and f["checkpoint_matches_server"] is True
    assert "checkpoint" not in f["declared_not_verified"]


def test_a_checkpoint_that_differs_from_the_reported_model_path_is_refused(rig: Rig) -> None:
    refused(rig, "other/weights", checkpoint="other/weights")


def test_the_model_path_is_the_checkpoint_when_none_is_declared(rig: Rig) -> None:
    e, _ = rig.ready()
    assert json.loads(e.version())["checkpoint"] == CHECKPOINT


def test_a_server_that_reports_no_model_path_cannot_confirm_the_label_or_the_checkpoint(rig: Rig) -> None:
    rig.server.state.model_info = {"served_model_name": CHECKPOINT}
    e, cfg = rig.ready(checkpoint=CHECKPOINT)
    f = e.flags(cfg)
    assert f["model_matches_server"] is None and f["checkpoint_matches_server"] is None
    assert "checkpoint" in f["declared_not_verified"]


def test_flags_record_that_the_label_matches_the_server(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert e.flags(cfg)["model_matches_server"] is True


def test_a_model_label_that_is_neither_the_model_path_nor_the_served_name_is_refused(rig: Rig) -> None:
    with pytest.raises(SystemExit) as e:
        rig.run(model="other/model", served=CHECKPOINT)
    text = str(e.value)
    assert "other/model" in text and CHECKPOINT in text and "--model" in text
    assert rig.server.state.posts() == [] and rig.rows() == []


def test_a_model_label_equal_to_the_served_name_is_accepted_and_flagged_as_an_alias(rig: Rig) -> None:
    rig.server.state.models = [{"id": "dgemma", "root": "dgemma", "object": "model"}]
    assert rig.run(limit=1, model="dgemma") == 0
    assert rig.rows()[0]["model"] == "dgemma|bfloat16"
    assert rig.rows()[0]["flags"]["model_matches_server"] is False       # the checkpoint is CHECKPOINT, not "dgemma"
    assert json.loads(rig.server.state.posts()[0]["body"])["model"] == "dgemma"


def test_an_alias_label_does_not_claim_to_match_the_checkpoint_the_server_reports(rig: Rig) -> None:
    # the review's probe: a server started with --served-model-name myalias, and `--model myalias --served-model myalias`
    rig.server.state.models = [{"id": "myalias", "root": "myalias", "object": "model"}]
    rig.server.state.model_info = {"model_path": CHECKPOINT, "served_model_name": "myalias"}
    e = rig.engine(served_model="myalias")
    cfg = rig.cfg(model="myalias")
    e.check(cfg)
    f = e.flags(cfg)
    assert e.model_name(cfg) == "myalias|bfloat16"
    assert f["model_matches_server"] is False
    assert f["checkpoint_reported_by_server"] == CHECKPOINT
    assert json.loads(e.version())["checkpoint"] == CHECKPOINT


def test_the_flag_is_true_only_when_the_label_is_the_reported_checkpoint(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert cfg.model == CHECKPOINT and e.flags(cfg)["model_matches_server"] is True


def test_a_served_name_given_with_served_model_is_the_one_requested(rig: Rig) -> None:
    rig.server.state.models = [{"id": "dgemma", "root": "dgemma", "object": "model"}]
    assert rig.run(limit=1, served="dgemma") == 0
    assert json.loads(rig.server.state.posts()[0]["body"])["model"] == "dgemma"
    assert rig.rows()[0]["flags"]["served_model"] == "dgemma"


def test_a_served_name_the_server_does_not_list_is_refused_before_any_request_for_a_completion(rig: Rig) -> None:
    refused(rig, CHECKPOINT, served="not-served")
    assert rig.server.state.posts() == []


def test_the_refusal_names_the_models_the_server_reports(rig: Rig) -> None:
    rig.server.state.models = [{"id": "llama-3", "root": "llama-3"}, {"id": "mistral", "root": "mistral"}]
    refused(rig, "llama-3, mistral")


def test_a_served_model_among_several_is_accepted_and_is_the_one_requested(rig: Rig) -> None:
    rig.server.state.models = [{"id": "other", "root": "other"}, {"id": CHECKPOINT, "root": CHECKPOINT}]
    assert rig.run(limit=1) == 0
    assert json.loads(rig.server.state.posts()[0]["body"])["model"] == CHECKPOINT
    assert rig.rows()[0]["flags"]["served_model"] == CHECKPOINT


def test_a_server_that_lists_no_models_is_refused(rig: Rig) -> None:
    rig.server.state.models = []
    refused(rig, "no models")


def test_a_malformed_model_list_is_refused(rig: Rig) -> None:
    rig.server.state.models = [{"nope": 1}]
    refused(rig, "malformed")


def test_an_http_error_from_the_model_list_is_refused_with_the_servers_message(rig: Rig) -> None:
    rig.server.state.models_status = 503
    refused(rig, "models are down")


# the dtype


@pytest.mark.parametrize("declared,reported,name", [("half", "float16", "float16"), ("float16", "half", "float16"),
                                                    ("float", "float32", "float32"), ("bfloat16", "bfloat16", "bfloat16")])
def test_dtype_aliases_are_one_dtype(rig: Rig, declared: str, reported: str, name: str) -> None:
    rig.info(dtype=reported)
    e, cfg = rig.ready(dtype=declared)
    assert json.loads(e.version())["dtype"] == name and e.model_name(cfg).endswith("|" + name)


def test_a_declared_dtype_that_differs_from_the_reported_one_is_refused(rig: Rig) -> None:
    rig.info(dtype="float16")
    refused(rig, "float16", dtype="bfloat16")


def test_a_dtype_the_server_cannot_report_must_be_declared(rig: Rig) -> None:
    refused(rig, "--dtype", dtype=None)


def test_a_server_that_reports_no_dtype_field_needs_a_declaration(rig: Rig) -> None:
    rig.info(dtype=...)
    refused(rig, "--dtype", dtype=None)


def test_a_server_that_reports_a_null_dtype_needs_a_declaration(rig: Rig) -> None:
    rig.info(dtype=None)
    refused(rig, "--dtype", dtype=None)


def test_a_declared_dtype_of_auto_is_refused(rig: Rig) -> None:
    refused(rig, "auto", dtype="auto")


def test_a_declared_dtype_outside_the_servers_choices_is_refused(rig: Rig) -> None:
    refused(rig, "bf16", dtype="bf16")


def test_a_quantized_server_is_refused_because_the_arm_name_carries_a_dtype_only(rig: Rig) -> None:
    rig.info(quantization="fp8")
    refused(rig, "quantization")


def test_an_unquantized_server_with_a_null_quantization_is_accepted(rig: Rig) -> None:
    rig.info(quantization=None)
    assert rig.run(limit=1) == 0


# the algorithm


def test_a_server_that_runs_no_diffusion_algorithm_is_refused(rig: Rig) -> None:
    rig.info(dllm_algorithm=None)
    refused(rig, "no diffusion algorithm")


def test_a_declared_algorithm_is_refused_when_the_server_runs_none(rig: Rig) -> None:
    rig.info(dllm_algorithm=None)
    refused(rig, "no diffusion algorithm", algorithm="Gemma4Renoise")


def test_a_declared_algorithm_that_differs_from_the_reported_one_is_refused(rig: Rig) -> None:
    refused(rig, "LowConfidence", algorithm="LowConfidence")


def test_a_server_that_cannot_report_its_algorithm_needs_a_declaration(rig: Rig) -> None:
    rig.info(dllm_algorithm=...)
    refused(rig, "--algorithm")


def test_a_declared_algorithm_is_accepted_when_the_server_cannot_report_one(rig: Rig) -> None:
    rig.info(dllm_algorithm=...)
    e, cfg = rig.ready(algorithm="Gemma4Renoise")
    assert json.loads(e.version())["algorithm"] == "Gemma4Renoise"
    assert "algorithm" in e.flags(cfg)["declared_not_verified"]


# batch size one


def test_a_server_that_reports_a_batch_size_over_one_is_refused(rig: Rig) -> None:
    rig.info(max_running_requests=4)
    refused(rig, "--max-running-requests 1")
    assert rig.server.state.posts() == []


def test_a_server_that_cannot_report_its_batch_size_needs_it_declared_as_one(rig: Rig) -> None:
    rig.info(max_running_requests=None)
    refused(rig, "--max-running-requests 1")
    assert rig.server.state.posts() == []


def test_a_server_without_the_field_needs_it_declared_as_one(rig: Rig) -> None:
    rig.info(max_running_requests=...)
    refused(rig, "--max-running-requests 1")


def test_a_declared_batch_size_of_one_is_accepted_when_the_server_cannot_report_one(rig: Rig) -> None:
    rig.info(max_running_requests=None)
    e, cfg = rig.ready(max_running_requests=1)
    f = e.flags(cfg)
    assert f["max_running_requests"] == 1 and "max_running_requests" in f["declared_not_verified"]


def test_a_declared_batch_size_other_than_one_is_refused(rig: Rig) -> None:
    rig.info(max_running_requests=None)
    refused(rig, "--max-running-requests 1", mrr="4")


def test_a_declared_batch_size_that_differs_from_the_reported_one_is_refused(rig: Rig) -> None:
    rig.info(max_running_requests=4)
    refused(rig, "4", mrr="1")


def test_a_reported_batch_size_of_one_needs_no_declaration(rig: Rig) -> None:
    e, cfg = rig.ready()
    assert "max_running_requests" not in e.flags(cfg)["declared_not_verified"]


def test_a_non_integer_batch_size_option_is_rejected_by_the_parser(rig: Rig) -> None:
    with pytest.raises(SystemExit) as e:
        rig.run("--max-running-requests", "many")
    assert e.value.code == 2                       # argparse's usage error, not a refusal message


# the version


def test_a_server_info_without_a_version_is_refused(rig: Rig) -> None:
    rig.info(version=...)
    refused(rig, "/server_info")


@pytest.mark.parametrize("version", ["", "   ", None, 5, ["0.5.23"]])
def test_an_empty_or_malformed_version_is_refused(rig: Rig, version: object) -> None:
    rig.info(version=version)
    refused(rig, "/server_info")


def test_the_development_fallback_version_is_refused(rig: Rig) -> None:
    rig.info(version="0.0.0.dev0")
    refused(rig, "0.0.0.dev0")


def test_a_development_version_that_names_a_commit_is_accepted(rig: Rig) -> None:
    rig.info(version="0.5.24.dev12+gabc1234")
    e, _ = rig.ready()
    assert json.loads(e.version())["sglang"] == "0.5.24.dev12+gabc1234"


@pytest.mark.parametrize("path,attr", [("/server_info", "server_info_status"), ("/model_info", "model_info_status")])
@pytest.mark.parametrize("status", [404, 401, 403, 500])
def test_an_error_on_a_metadata_endpoint_is_refused_with_the_status(rig: Rig, path: str, attr: str, status: int) -> None:
    setattr(rig.server.state, attr, status)
    refused(rig, f"HTTP {status}")
    assert rig.server.state.posts() == []


@pytest.mark.parametrize("attr", ["server_info", "model_info"])
def test_a_metadata_answer_that_is_not_an_object_is_refused(rig: Rig, attr: str) -> None:
    setattr(rig.server.state, attr, ["not", "an", "object"])
    refused(rig, "not a JSON object")


# server text is cleaned


def test_model_ids_the_server_reports_cannot_inject_escapes(rig: Rig) -> None:
    rig.server.state.models = [{"id": "a\x1b[31mred", "root": "x"}, {"id": "b\rc", "root": "y"}]
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "\x1b" not in str(e.value) and "\r" not in str(e.value) and "a[31mred" in str(e.value)


def test_a_model_path_with_escapes_is_cleaned_in_the_label_refusal(rig: Rig) -> None:
    rig.server.state.models = [{"id": "dgemma", "root": "dgemma", "object": "model"}]
    rig.server.state.model_info = {"model_path": "evil\x1b[2Jpath", "served_model_name": "dgemma"}
    with pytest.raises(SystemExit) as e:
        rig.run(served="dgemma")
    assert "\x1b" not in str(e.value) and "evil[2Jpath" in str(e.value)


def test_a_checkpoint_conflict_names_both_values_cleanly(rig: Rig) -> None:
    with pytest.raises(SystemExit) as e:
        rig.run(checkpoint="other\x1b[31m/weights")
    assert CHECKPOINT in str(e.value) and "--checkpoint" in str(e.value) and "\x1b" not in str(e.value)


def test_a_version_with_escapes_is_cleaned_before_it_is_recorded(rig: Rig) -> None:
    rig.info(version="0.5.23\x1b[31m-x")
    e, _ = rig.ready()
    assert "\x1b" not in e.version() and "0.5.23[31m-x" in e.version()


def test_a_reported_dtype_and_algorithm_with_escapes_are_cleaned(rig: Rig) -> None:
    rig.info(dllm_algorithm="Gemma4\x1b[31mRenoise", dtype="bfloat16")
    e, cfg = rig.ready()
    assert "\x1b" not in e.version() and "\x1b" not in json.dumps(e.flags(cfg))


def test_a_conflicting_algorithm_message_is_cleaned_and_capped(rig: Rig) -> None:
    rig.info(dllm_algorithm="\x1b[31m" + "x" * 5000)
    with pytest.raises(SystemExit) as e:
        rig.run(algorithm="LowConfidence")
    assert "\x1b" not in str(e.value) and len(str(e.value)) < 900 and "--algorithm" in str(e.value)


def test_an_http_error_message_with_escapes_is_clean_through_the_runner(rig: Rig) -> None:
    rig.server.state.mode = "escape"
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "answered HTTP 500" in str(e.value) and "\x1b" not in str(e.value) and "\x07" not in str(e.value)


def test_a_wrong_echoed_model_name_is_cleaned_and_capped(rig: Rig) -> None:
    rig.server.state.mode = "wrong_echo"
    e, cfg = rig.ready()
    with pytest.raises(EngineError) as ex:
        e.generate("q", cfg)
    assert len(str(ex.value)) < 300 and "some-other-model" in str(ex.value)


# the base URL, the client and the other options


def test_an_unreachable_server_is_refused_at_check(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="127.0.0.1:9"):
        eval_engine.main(rig.argv() + ["--base-url", "http://127.0.0.1:9"])


def test_a_base_url_with_credentials_is_refused(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="credentials"):
        eval_engine.main(rig.argv() + ["--base-url", "http://user:pw@127.0.0.1:8000"])


@pytest.mark.parametrize("suffix", ["/?x=1", "#frag", "?x=1"])
def test_a_base_url_with_a_query_or_fragment_is_refused(rig: Rig, suffix: str) -> None:
    with pytest.raises(SystemExit, match="query or fragment"):
        eval_engine.main(rig.argv() + ["--base-url", rig.server.url + suffix])
    assert rig.server.state.requests == []


def test_a_trailing_slash_on_the_base_url_does_not_double_the_slash(rig: Rig) -> None:
    eval_engine.main(rig.argv(limit=1) + ["--base-url", rig.server.url + "/"])
    assert all(not r["path"].startswith("//") for r in rig.server.state.requests)


def test_environment_proxies_are_ignored_unless_asked(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(k, "http://127.0.0.1:9")          # nothing listens there
    assert rig.run(limit=1) == 0


def test_use_env_proxy_sends_every_request_through_the_proxy(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(k, "http://127.0.0.1:9")
    with pytest.raises(SystemExit, match="cannot reach"):
        rig.run("--use-env-proxy", limit=1)
    assert rig.server.state.requests == []                   # the fake server saw nothing: the proxy was tried first


@pytest.mark.parametrize("suffix", ["/sglang", "/sglang/", "/a/b"])
def test_a_base_url_path_prefix_stays_in_front_of_every_request(rig: Rig, suffix: str) -> None:
    prefix = suffix.rstrip("/")
    rig.server.state.prefix = prefix                         # the server answers only below this path
    assert eval_engine.main(rig.argv(limit=1) + ["--base-url", rig.server.url + suffix]) == 0
    seen = sorted({(r["method"], r["path"]) for r in rig.server.state.requests})
    assert seen == [("GET", prefix + "/model_info"), ("GET", prefix + "/server_info"), ("GET", prefix + "/v1/models"),
                    ("POST", prefix + "/v1/chat/completions")]
    assert len(rig.rows()) == 1


def test_the_fake_server_with_a_prefix_answers_404_outside_it(rig: Rig) -> None:
    # the control for the test above: a client that dropped the prefix would be refused, not served
    rig.server.state.prefix = "/sglang"
    with pytest.raises(SystemExit, match="HTTP 404"):
        rig.run(limit=1)
    assert rig.rows() == [] and rig.server.state.posts() == []


def test_extra_engine_flags_are_refused(rig: Rig) -> None:
    refused(rig, "extra engine flags", "--", "--foo")


def test_a_device_other_than_cpu_or_gpu_is_refused(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="cpu or gpu"):
        rig.engine().check(rig.cfg(device="tpu"))


def test_a_step_count_the_server_cannot_take_is_refused(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="--steps"):
        rig.engine().check(rig.cfg(steps=48))


def test_a_refused_check_sends_no_completion_request(rig: Rig) -> None:
    rig.info(version=...)
    refused(rig, "/server_info")
    assert rig.server.state.posts() == []


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
    assert g["finish_reason"] == "stop"


def test_a_response_over_the_size_bound_is_refused(rig: Rig) -> None:
    rig.server.state.mode = "huge"
    e, cfg = rig.ready()
    with pytest.raises(EngineError, match="larger than"):
        e.generate("q", cfg)


def test_generate_s_covers_the_request_only(rig: Rig) -> None:
    e, cfg = rig.ready()
    g = e.generate("q", cfg)
    assert 0 < g["generate_s"] < 5


# one overall deadline per request, enforced through the engine


def test_a_slow_generation_stops_the_run_and_names_the_timeout(rig: Rig) -> None:
    rig.server.state.mode = "sleep"
    refused(rig, "timed out", "--timeout-s", "0.4")


def test_a_response_trickled_past_the_deadline_stops_the_run(rig: Rig) -> None:
    rig.server.state.mode = "trickle"          # one byte every 0.25 s, 3 s in all
    refused(rig, "timed out", "--timeout-s", "0.8")


def test_headers_trickled_past_the_deadline_stop_the_run(rig: Rig) -> None:
    rig.server.state.mode = "header_trickle"
    refused(rig, "timed out", "--timeout-s", "0.8")
