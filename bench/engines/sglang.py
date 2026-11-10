"""A running SGLang server as a GSM8K engine (issue #180): a client for its OpenAI-compatible endpoint.

The engine never starts or stops a server and loads no weights. It asks a server somebody else started, over HTTP
(`--base-url`), to answer each problem with `POST /v1/chat/completions`: one user message holding the same prompt text
the other engines use, `max_completion_tokens` equal to the generation length and `stream` false. The server applies
the model's chat template, so the template is the server's and `flags` says so.

Field names and rules were read from SGLang's source on 2026-11-09, not from memory. `main` was at
dfaca5681ddc20e31d50f2dfe5dfada269f12bbd. The release v0.5.23 (2026-11-02) was compared for `available_models` and
`server_info` (identical), `model_info` (the same keys, built inline) and `gemma4_renoise.py` (one import line differs);
every other file below was read at `main` only:

- `GET /v1/models` answers one card whose `id` and `root` are both the served model name, and `max_model_len`
  (python/sglang/srt/entrypoints/http_server.py, `available_models`). Unlike vLLM, `root` is not the model path. The
  served name defaults to `model_path` (python/sglang/srt/arg_groups/serving_hook.py), so `--served-model` defaults to `--model`.
- `GET /model_info` answers `model_path` (the checkpoint) and `served_model_name`; `GET /server_info` answers the
  resolved server arguments (`dllm_algorithm`, `dtype`, `quantization`, `max_running_requests`, `chat_template`, ...)
  and `version` (http_server.py, `model_info` and `server_info`; the arguments come from `ServerArgs.resolved_dict`).
  SGLang has no `GET /version` route. `/get_model_info` and `/get_server_info` are the deprecated names of the two
  and are not used. `server_info` also holds `launch_command` and the scheduler's internal state: the engine copies
  the named fields it needs and nothing else, so a path or a flag of the server never reaches an arm file.
- `Gemma4Renoise.validate_request` (python/sglang/srt/dllm/algorithm/gemma4_renoise.py) answers HTTP 400 to a request
  with logprobs, a grammar (`json_schema`, `regex`, `ebnf`, `structural_tag`), a penalty, `min_new_tokens`, `min_p`, a
  seed, a logit bias, a custom logit processor or a sampling mask. The engine sends none of them, and the tests scan
  every request body for them. `temperature`, `top_p` and `top_k` are accepted and have no effect: the renoise
  schedule (`--dllm-algorithm-config`, defaults from the checkpoint's `generation_config.json`) governs sampling, so
  the engine does not send them and `flags.sampling` says so.
- `Gemma4Renoise.configure_server_args` refuses a quantized checkpoint ("DiffusionGemma does not support quantized
  loading"), so the weights are bf16 or another dtype and the arm's `model` carries the dtype, not a quantization.
- Non-streaming: `_wait_one_response` (python/sglang/srt/managers/tokenizer_manager.py) yields an unfinished output only
  when the request streams, and the cookbook says streaming is block-level, one denoised canvas per chunk. The engine
  sends `stream: false` and reads one completion.
- An error body is `{"object": "error", "message": ..., "type": ..., "code": ...}` (serving_base.py,
  `create_error_response`); `_http` reads the top-level `message`.

Denoising steps. SGLang on `main` and in v0.5.23 exposes no per-request step or forward-pass count: the Prometheus
metrics (python/sglang/srt/observability/metrics_collector.py) have no dLLM counter, and the only pass counter,
`sglang:cuda_graph_passes_total`, is server-wide and counts passes by CUDA-graph mode, so a change of it across a
request is not that request's denoising steps. Pull request sgl-project/sglang#48675 ("Expose cumulative
request-level forward metrics", `meta_info.dllm_metrics` on `/generate`) was open and unmerged on 2026-11-09, and
#23722, which the roadmap issue #39499 cites for forward counts, was closed without a merge. SGLang's own
benchmark/dllm/bench_diffusion_gemma.py records `--denoising-steps` as metadata only. So `forward_passes` is null in
every row, `flags.forward_passes_measured` is false, and the engine never reads `/metrics` and never invents a count.

What the server can tell the client is read and checked. What it cannot is declared with an option and listed in
`flags.declared_not_verified`. For each of the checkpoint, the diffusion algorithm, the dtype and the batch size:
a value the server reports is the value; a declared value that differs from a reported one is refused; with neither
the run is refused (the checkpoint is the one that may stay unknown, which leaves the label unconfirmed). A reported
`dtype` of `auto` and a reported `max_running_requests` of null count as "cannot say". A server that reports no
`dllm_algorithm` is not serving a diffusion model and is refused. The batch size must be 1, reported or declared:
`generate_s` is the wall time of one request, and a second client would add its queue time to it.

The isolation checks (`bench/isolation.py`) run on the CLIENT machine only. A remote server's GPU, clocks and
neighbours are not observed, and `flags` says `"server": "remote"` so nobody reads them as the server's.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from . import EngineError, GenConfig, GenResult, register
from ._http import HttpClient, clean
from ._openai import model_cards, parse_chat

MAX_RESPONSE_BYTES = 4 * 1024 * 1024    # a 256-token completion is a few KB; /server_info is the largest answer, well under this
CLIENT_RAM_GB = 0.5         # my estimate for the Python loop of a client that holds no model; not a measurement
DEV_FALLBACK_VERSION = "0.0.0.dev0"     # sglang/version.py: "Fallback for development without build"
DTYPES = ("half", "float16", "bfloat16", "float", "float32")     # `--dtype` choices except "auto" (arguments in server_args)
_DTYPE_ALIASES = {"half": "float16", "float": "float32"}         # the help text: "float16" is the same as "half"
SAMPLING_NOTE = "set by the server's diffusion algorithm; temperature, top_p and top_k are not sent"


def _canon_dtype(name: str) -> str:
    return _DTYPE_ALIASES.get(name, name)


@register
class SGLang:
    name = "sglang"

    def __init__(self, args: argparse.Namespace) -> None:
        self.base_url = args.base_url
        self.served_arg = args.served_model
        self.checkpoint = args.checkpoint
        self.algorithm_arg = args.algorithm
        self.dtype_arg = args.dtype
        self.max_running_requests_arg = args.max_running_requests
        self.chat_template = args.chat_template
        self.use_env_proxy = args.use_env_proxy
        self.client: HttpClient | None = None
        self.served_model: str | None = args.served_model
        self._version: str | None = None
        self._model_path: str | None = None
        self._checkpoint: str | None = None
        self._algorithm: str | None = None
        self._algorithm_config_set: bool | None = None
        self._dtype: str | None = None
        self._max_running_requests: int | None = None
        self._template_reported: str | None = None
        self._label_matches: bool | None = None
        self._declared_only: list[str] = []

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--base-url", default="http://127.0.0.1:30000",
                            help="the running SGLang server (default %(default)s). Never put credentials in it")
        parser.add_argument("--served-model", default=None,
                            help="the model name the server serves (`--served-model-name`; SGLang serves the model path "
                                 "when it is not given). Defaults to --model; the run is refused when /v1/models does "
                                 "not list it")
        parser.add_argument("--dtype", default=None,
                            help="the --dtype you started the server with: half, float16, bfloat16, float or float32. "
                                 "Needed when /server_info reports auto, and refused when it differs from a dtype the "
                                 "server reports. It goes into the arm's model name so compare sees different weights")
        parser.add_argument("--algorithm", default=None,
                            help="the --dllm-algorithm you started the server with (Gemma4Renoise); needed only when "
                                 "/server_info does not report it. Recorded as declared")
        parser.add_argument("--checkpoint", default=None,
                            help="the --model-path you started the server with; refused when it differs from the "
                                 "model_path the server reports")
        parser.add_argument("--max-running-requests", type=int, default=None,
                            help="the --max-running-requests you started the server with. The evaluation is batch size 1, "
                                 "so the value must be 1; it is read from /server_info and needed only when the server "
                                 "reports null. Recorded as declared in that case")
        parser.add_argument("--chat-template", default=None,
                            help="the --chat-template you started the server with, if any; refused when /server_info "
                                 "reports another. The server applies it and the client cannot read its text")
        parser.add_argument("--use-env-proxy", action="store_true",
                            help="honour http_proxy and friends; off by default so no proxy sits in the timed path")

    def model_name(self, cfg: GenConfig) -> str:
        """The `model` of every row: what `compare` reads to decide whether the weights differ, so the dtype is part
        of it."""
        dtype = self._dtype or (_canon_dtype(self.dtype_arg) if self.dtype_arg else None)
        return f"{pathlib.Path(cfg.model).name}|{dtype}"

    def ram_gb_after_load(self, cfg: GenConfig) -> float:
        """The preflight sizes memory from a model file; this client has none. It asks for a small fixed amount."""
        return CLIENT_RAM_GB

    def check(self, cfg: GenConfig) -> None:
        if cfg.device not in ("cpu", "gpu"):
            raise SystemExit(f"sglang runs with --device cpu or gpu (the isolation checks of THIS machine), not {cfg.device!r}")
        if cfg.extra:
            raise SystemExit(f"sglang takes no extra engine flags; got {' '.join(cfg.extra)}")
        if cfg.steps != cfg.gen_length:
            raise SystemExit("sglang takes its step rule from the server (`--dllm-algorithm-config` at serve time); "
                             f"--steps {cfg.steps} would be ignored, so it is refused")
        if self.dtype_arg is not None and self.dtype_arg not in DTYPES:
            raise SystemExit(f"--dtype {self.dtype_arg!r} names no dtype: choose one of {', '.join(DTYPES)} "
                             "(auto names nothing, so it is refused)")
        self.client = HttpClient(self.base_url, timeout_s=cfg.timeout_s, max_bytes=MAX_RESPONSE_BYTES,
                                 use_env_proxy=self.use_env_proxy)
        served = self.served_arg or cfg.model
        ids = [c["id"] for c in model_cards(self.client)]
        if served not in ids:
            raise EngineError(f"the server reports the models {clean(', '.join(ids))}; the served name is "
                              f"{served!r} (--served-model, else --model)")
        self.served_model = served
        info = self._object("/model_info")
        sinfo = self._object("/server_info")
        declared_only: list[str] = []

        path = info.get("model_path")
        path = path if isinstance(path, str) and path else None
        if path is not None and cfg.model not in (path, served):
            raise EngineError(f"--model {cfg.model!r} is neither the model path {clean(path)!r} nor the served name "
                              f"{served!r} the server reports: the arm would be labelled with weights it did not run "
                              "(pass the checkpoint you served as --model)")
        self._model_path = clean(path) if path is not None else None
        # True only when the label IS the checkpoint the server reports; a label that is just the served name (an alias
        # for other weights) is accepted and recorded as False. Without a reported model_path nothing can confirm it.
        self._label_matches = (cfg.model == path) if path is not None else None
        self._checkpoint = self._settle("checkpoint", "--checkpoint", self.checkpoint, path, declared_only, required=False)

        version = sinfo.get("version")
        if not isinstance(version, str) or not version.strip():
            raise EngineError(f"GET /server_info did not report a version string: {clean(repr(version))}")
        self._version = clean(version)
        if self._version == DEV_FALLBACK_VERSION:
            raise EngineError(f"GET /server_info reports the development fallback version {DEV_FALLBACK_VERSION}, which "
                              "names no release or commit: run a release, or a checkout with its git history")

        algorithm = sinfo.get("dllm_algorithm") if "dllm_algorithm" in sinfo else None
        if "dllm_algorithm" in sinfo and not (isinstance(algorithm, str) and algorithm.strip()):
            raise EngineError("the server reports no diffusion algorithm (dllm_algorithm is null in /server_info): it is "
                              "not serving a diffusion model, so the arm would not be a diffusion arm")
        self._algorithm = self._settle("diffusion algorithm", "--algorithm", self.algorithm_arg, algorithm, declared_only)
        config = sinfo.get("dllm_algorithm_config") if "dllm_algorithm_config" in sinfo else ...
        self._algorithm_config_set = None if config is ... else config not in (None, "")

        quantization = sinfo.get("quantization")
        if quantization not in (None, ""):
            raise EngineError(f"the server reports quantization {clean(repr(quantization))}: the arm's model name "
                              "carries a dtype only, so a quantized run would be mislabelled")
        dtype = sinfo.get("dtype")
        reported = _canon_dtype(dtype) if isinstance(dtype, str) and dtype not in ("", "auto") else None
        declared = _canon_dtype(self.dtype_arg) if self.dtype_arg is not None else None
        self._dtype = self._settle("dtype", "--dtype", declared, reported, declared_only)

        limit = sinfo.get("max_running_requests")
        limit = limit if isinstance(limit, int) and not isinstance(limit, bool) else None
        self._max_running_requests = self._settle("max_running_requests", "--max-running-requests",
                                                  self.max_running_requests_arg, limit, declared_only,
                                                  need="--max-running-requests 1")
        if self._max_running_requests != 1:
            raise EngineError(f"the server runs with max_running_requests {self._max_running_requests}: the evaluation is "
                              "batch size 1, and a second client would add its queue time to generate_s. Start it with "
                              "`--max-running-requests 1`")

        if "chat_template" in sinfo:
            rep = sinfo["chat_template"]
            self._template_reported = clean(rep) if isinstance(rep, str) and rep else None
            if self.chat_template is not None and self._template_reported != self.chat_template:
                raise EngineError(f"--chat-template is {self.chat_template!r} but the server reports chat_template "
                                  f"{self._template_reported!r} (None means the checkpoint's own template)")
        elif self.chat_template is not None:
            declared_only.append("chat_template")
        self._declared_only = sorted(declared_only)

    def _object(self, path: str) -> dict:
        doc = self.client.get_json(path)                             # type: ignore[union-attr]
        if not isinstance(doc, dict):
            raise EngineError(f"GET {path} answered something that is not a JSON object: {clean(repr(doc))}")
        return doc

    @staticmethod
    def _settle(label: str, option: str, declared: object, reported: object, declared_only: list[str],
                *, required: bool = True, need: str | None = None) -> object:
        """The value of one fact: the server's when it reports one, else the operator's declaration (and then
        `declared_only` gets the name). A declaration that contradicts the report is refused."""
        if isinstance(reported, str):
            reported = clean(reported)
        if declared is not None and reported is not None and declared != reported:
            raise EngineError(f"{option} is {declared!r} but the server reports {label} {reported!r}: the arm would "
                              "be labelled with what it did not run")
        if reported is not None:
            return reported
        if declared is None:
            if required:
                raise EngineError(f"the server does not report its {label} in /server_info; pass {need or option}")
            return None
        declared_only.append(option.lstrip("-").replace("-", "_"))
        return declared

    def version(self) -> str:
        """JSON: the release the server reports, then the checkpoint, algorithm and dtype (reported when the server
        says, else declared; `flags.declared_not_verified` names the declared ones)."""
        return json.dumps({"sglang": self._version, "algorithm": self._algorithm, "checkpoint": self._checkpoint,
                           "dtype": self._dtype}, sort_keys=True)

    def flags(self, cfg: GenConfig) -> dict:
        return {"engine": self.name, "server": "remote", "device": cfg.device, "gen_length": cfg.gen_length,
                "sampling": SAMPLING_NOTE, "served_model": self.served_model, "algorithm": self._algorithm,
                "algorithm_config_set": self._algorithm_config_set, "dtype": self._dtype,
                "max_running_requests": self._max_running_requests, "declared_not_verified": self._declared_only,
                "model_matches_server": self._label_matches,
                "checkpoint_reported_by_server": self._model_path,
                "checkpoint_matches_server": (self.checkpoint == self._model_path) if self.checkpoint and self._model_path else None,
                "chat_template": {"applied_by": "server", "reported": self._template_reported,
                                  "declared": self.chat_template, "verified": False},
                "forward_passes_measured": False}

    def generate(self, prompt: str, cfg: GenConfig) -> dict:
        # The fields are exactly these four. The server answers HTTP 400 to logprobs, grammar and penalty fields, and
        # accepts temperature, top_p and top_k without using them (see the module docstring), so none is sent.
        body = {"model": self.served_model, "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": cfg.gen_length, "stream": False}
        start = time.perf_counter()
        out = self.client.post_json("/v1/chat/completions", body)    # type: ignore[union-attr]
        generate_s = time.perf_counter() - start
        text, prompt_tokens, completion_tokens, finish = parse_chat(out, self.served_model)    # type: ignore[arg-type]
        # SGLang echoes the request's `model`, so the echo check in parse_chat only catches a gateway that rewrites it;
        # the identity of the weights comes from /model_info, read in `check`.
        return GenResult(text=text, generate_s=generate_s, forward_passes=None, steps=None,
                         prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                         finish_reason=finish)
