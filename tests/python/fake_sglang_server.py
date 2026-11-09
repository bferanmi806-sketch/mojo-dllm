"""A stand-in for an SGLang server running DiffusionGemma, for tests: standard library `http.server` on 127.0.0.1,
ephemeral port. It reuses the transport misbehaviours of fake_vllm_server (a 500, an escape sequence in an error,
trickled bytes, a redirect, an oversize body) and replaces what is SGLang's own.

Shapes follow SGLang's source, read on 2026-11-09 (main at dfaca5681ddc20e31d50f2dfe5dfada269f12bbd, release v0.5.23):

- `GET /v1/models` answers one card whose `id` and `root` are both the served model name
  (python/sglang/srt/entrypoints/http_server.py, `available_models`), not the model path.
- `GET /model_info` answers `model_path` and `served_model_name`; `GET /server_info` answers the resolved server
  arguments (`dllm_algorithm`, `dtype`, `quantization`, `max_running_requests`, `chat_template`, ...) with
  `version` and `launch_command` beside them. There is no `GET /version`.
- An error body is `{"object": "error", "message": ..., "type": ..., "param": ..., "code": ...}`
  (python/sglang/srt/entrypoints/openai/serving_base.py, `create_error_response`).
- A request that carries logprobs, a grammar, a penalty, a seed or a logit bias is rejected with HTTP 400 by
  `Gemma4Renoise.validate_request` (python/sglang/srt/dllm/algorithm/gemma4_renoise.py). `rejected_fields` follows that
  list and goes further (any `response_format`, `grammar` or `structured_outputs` is refused too), so a stray field
  fails loudly. The response echoes the request's `model` (`ChatCompletionResponse(model=request.model)`).

It logs every request (method, raw target, headers, body) so a test can pin what the client sent.
"""

from __future__ import annotations

import io
import json
import threading

from fake_vllm_server import ANSWER, Handler as VllmHandler, QuietServer, State as VllmState

CHECKPOINT = "google/diffusiongemma-26B-A4B-it"
COUNTER_LOOKALIKE = "sglang:cuda_graph_passes_total"     # a real SGLang metric that counts passes, not denoising steps

# Chat-request fields that Gemma4Renoise.validate_request rejects, with the value that is the default (not rejected).
_PENALTY_DEFAULTS = {"frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "min_tokens": 0,
                     "min_p": 0.0}
_REJECTED_WHEN_SET = ("logit_bias", "logprobs", "top_logprobs", "json_schema", "regex", "ebnf", "structural_tag",
                      "grammar", "structured_outputs", "response_format", "return_sampling_mask",
                      "return_flat_raw_output_top_logprobs", "custom_logit_processor", "token_ids_logprob")


def rejected_fields(req: dict) -> list[str]:
    """The request-level fields the server would answer with HTTP 400 (a falsy value is the default and passes)."""
    bad = [k for k, default in _PENALTY_DEFAULTS.items() if k in req and req[k] != default]
    bad += [k for k in _REJECTED_WHEN_SET if req.get(k)]
    if req.get("seed") is not None:
        bad.append("seed")
    return bad


class State(VllmState):
    def __init__(self) -> None:
        super().__init__()
        self.models = [{"id": CHECKPOINT, "object": "model", "owned_by": "sglang", "root": CHECKPOINT,
                        "max_model_len": 262144}]
        self.model_info: dict | int = {"model_path": CHECKPOINT, "served_model_name": CHECKPOINT,
                                       "is_generation": True, "weight_version": "default",
                                       "architectures": ["DiffusionGemmaForBlockDiffusion"]}
        # the cookbook's launch plus `--max-running-requests 1`: dtype stays "auto" because nobody passed --dtype
        self.server_info: dict | int = {
            "version": "0.5.23", "model_path": CHECKPOINT, "served_model_name": CHECKPOINT,
            "dllm_algorithm": "Gemma4Renoise", "dllm_algorithm_config": None, "dtype": "auto",
            "quantization": None, "max_running_requests": 1, "chat_template": None, "enable_metrics": False,
            "launch_command": "sglang serve --model-path google/diffusiongemma-26B-A4B-it --host 0.0.0.0 "
                              "--api-key [redacted] --download-dir /root/private-cache",
            "internal_states": [{"last_gen_throughput": 0.0}], "frontend": "python"}
        self.model_info_status = 200
        self.server_info_status = 200
        self.rejects: list[list[str]] = []                  # what `rejected_fields` found, one list per rejected POST
        self.prefix = ""                                    # a path prefix the server sits behind (a reverse proxy); "" for none

    def metrics_text(self) -> str:
        return (f"# HELP {COUNTER_LOOKALIKE} Total number of forward passes categorized by CUDA graph.\n"
                f"# TYPE {COUNTER_LOOKALIKE} counter\n"
                f'{COUNTER_LOOKALIKE}{{mode="decode"}} {self.steps_total}\n')


class Handler(VllmHandler):
    @property
    def st(self) -> State:
        return self.server.state       # type: ignore[attr-defined]

    def _without_prefix(self, path: str) -> str | None:
        """The path below the server's prefix, or None when the request is not under it (the log keeps the raw target)."""
        prefix = self.st.prefix
        if not prefix:
            return path
        if path == prefix or path.startswith(prefix + "/"):
            return path[len(prefix):] or "/"
        return None

    def do_GET(self) -> None:  # noqa: N802
        self._log()
        st = self.st
        route = self._without_prefix(self.path.split("?", 1)[0])
        if route is None:
            return self._send(404, {"detail": "Not Found"})
        if route == "/v1/models":
            if st.models_status != 200:
                return self._send(st.models_status, {"object": "error", "message": "models are down",
                                                     "type": "InternalServerError", "param": None,
                                                     "code": st.models_status})
            return self._send(200, {"object": "list", "data": st.models})
        for path, doc, status in (("/model_info", st.model_info, st.model_info_status),
                                  ("/server_info", st.server_info, st.server_info_status)):
            if route == path:
                if status != 200:
                    return self._send(status, {"detail": f"{path} is down"})
                return self._send(200, doc)
        if route == "/metrics" and st.metrics is not None:
            if st.metrics_status != 200:
                return self._send(st.metrics_status, {"detail": "metrics are down"})
            text = st.metrics_text() if st.metrics == "live" else st.metrics
            return self._send(200, text.encode(), ctype="text/plain; version=0.0.4")
        self._send(404, {"detail": "Not Found"})          # including /version: SGLang has no such route

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        below = self._without_prefix(self.path)
        if below is None:
            self._log(body)
            return self._send(404, {"detail": "Not Found"})
        self.path = below
        if self.path == "/v1/chat/completions":
            try:
                req = json.loads(body)
            except ValueError:
                req = {}
            bad = rejected_fields(req) if isinstance(req, dict) else []
            if self.st.mode == "reject_all":          # a server whose generation_config or settings make every request invalid
                bad = ["repetition_penalty"]
            if bad:
                self._log(body)
                with self.st.lock:
                    self.st.rejects.append(bad)
                return self._send(400, {"object": "error", "type": "BadRequestError", "param": None, "code": 400,
                                        "message": "DiffusionGemma with Gemma4Renoise does not support "
                                                   "request-level " + ", ".join(bad) +
                                                   "; sampling is governed by the renoise schedule."})
        self.rfile = io.BytesIO(body)         # the transport modes below read the body again
        super().do_POST()


class FakeServer:
    def __init__(self) -> None:
        self.state = State()
        self.httpd = QuietServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.state = self.state            # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> "FakeServer":
        self.thread.start()
        return self

    def __exit__(self, *a: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


__all__ = ["ANSWER", "CHECKPOINT", "COUNTER_LOOKALIKE", "FakeServer", "State", "rejected_fields"]
