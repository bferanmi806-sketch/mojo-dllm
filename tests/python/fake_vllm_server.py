"""A stand-in for a vLLM OpenAI server, for tests: standard library `http.server` on 127.0.0.1, ephemeral port.

It answers the four paths the vLLM engine reads (`GET /v1/models`, `GET /version`, `GET /metrics`,
`POST /v1/chat/completions`) with canned bodies and logs every request it receives (method, path, headers, body)
so a test can pin what the client sent. `mode` makes the chat endpoint misbehave in one named way.
"""

from __future__ import annotations

import json
import threading
import time
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = "The answer is 18."
STEPS_METRIC = "vllm:diffusion_num_denoising_steps_total"


class State:
    def __init__(self) -> None:
        self.models: list[dict] = [{"id": "dgemma", "object": "model", "root": "google/diffusiongemma-26B-A4B-it",
                                    "max_model_len": 262144}]
        self.version: dict | int = {"version": "0.24.0"}     # an int is an HTTP status to answer instead
        self.metrics: str | None = None                     # None: 404; "live": the counter text, current value
        self.steps_total = 100.0                            # the counter's value before the first request
        self.steps_per_request = 7.0
        self.mode = "ok"
        self.answer = ANSWER
        self.models_status = 200
        self.metrics_status = 200                           # a status other than 200 answers /metrics with an error
        self.requests: list[dict] = []
        self.lock = threading.Lock()

    def posts(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]

    def metrics_text(self) -> str:
        return (f"# HELP {STEPS_METRIC} Number of denoising steps.\n# TYPE {STEPS_METRIC} counter\n"
                f'{STEPS_METRIC}{{engine="0",model_name="dgemma"}} {self.steps_total}\n'
                f'{STEPS_METRIC.replace("_total", "_created")}{{engine="0"}} 1.7e9\n')


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # no keep-alive, no chunking: a body without Content-Length ends at close

    def log_message(self, *a: object) -> None:      # silence the default stderr line per request
        pass

    @property
    def st(self) -> State:
        return self.server.state       # type: ignore[attr-defined]

    def _log(self, body: bytes = b"") -> None:
        # the raw request target: http.server rewrites a leading "//" in `self.path`, which would hide a doubled slash
        with self.st.lock:
            self.st.requests.append({"method": self.command, "path": self.raw_requestline.decode("latin-1").split()[1], "headers": dict(self.headers),
                                     "body": body})

    def _send(self, status: int, payload: object, *, length: bool = True, ctype: str = "application/json") -> None:
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if length:
            self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self._log()
        st = self.st
        route = urlsplit(self.path).path          # a proxied request carries the absolute URL
        if route == "/v1/models":
            if st.models_status != 200:
                return self._send(st.models_status, {"error": {"message": "models are down", "code": st.models_status}})
            return self._send(200, {"object": "list", "data": st.models})
        if route == "/version":
            if isinstance(st.version, int):
                return self._send(st.version, {"detail": "Not Found"})
            return self._send(200, st.version)
        if route == "/metrics":
            if st.metrics_status != 200:
                return self._send(st.metrics_status, {"error": {"message": "metrics are down"}})
            if st.metrics is None:
                return self._send(404, {"detail": "Not Found"})
            text = st.metrics_text() if st.metrics == "live" else st.metrics
            return self._send(200, text.encode(), ctype="text/plain; version=0.0.4")
        self._send(404, {"detail": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        self._log(body)
        st = self.st
        if self.path != "/v1/chat/completions":
            return self._send(404, {"detail": "Not Found"})
        req = json.loads(body)
        mode = st.mode
        if mode == "http500":
            return self._send(500, {"error": {"message": "CUDA out of memory in the denoiser", "type": "InternalServerError"}})
        if mode == "http400":
            return self._send(400, {"error": {"message": "max_tokens is too large: 9999", "type": "BadRequestError"}})
        if mode == "escape":
            nasty = "\x1b[2J\x1b]0;owned\x07bad\rline\nnext " + "y" * 2000
            return self._send(500, {"error": {"message": nasty}})
        if mode == "header_trickle":
            try:
                self.wfile.write(b"HTTP/1.0 200 OK\r\n")
                for _ in range(12):
                    self.wfile.write(b"X-Slow: 1\r\n")
                    self.wfile.flush()
                    time.sleep(0.25)
            except OSError:
                pass
            return
        if mode == "http503esc":
            return self._send(503, b"down\x1b[31m!", ctype="text/plain")
        if mode == "http503text":
            return self._send(503, b"upstream is restarting", ctype="text/plain")
        if mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()
            return
        if mode == "sleep":
            time.sleep(1.5)
        if mode == "trickle":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                for _ in range(12):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.25)
            except OSError:
                pass
            return
        if mode == "huge":
            return self._send(200, b"x" * (6 * 1024 * 1024), length=False)
        if mode == "notjson":
            return self._send(200, b"<html>gateway\x1b[31m</html>", ctype="text/html")
        if mode == "cookie":
            self.send_response(200)
            self.send_header("Set-Cookie", "session=abc123")
            body_out = json.dumps(self.completion(req, st.answer)).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_out)))
            self.end_headers()
            self.wfile.write(body_out)
            return
        if mode != "steps_stuck":
            with st.lock:
                st.steps_total += st.steps_per_request
        text = {"empty": "", "blank": "  \n "}.get(mode, st.answer)
        out = self.completion(req, text)
        if mode == "wrong_echo":
            out["model"] = "some-other-model"
        elif mode == "no_choices":
            out["choices"] = []
        elif mode == "no_usage":
            del out["usage"]
        elif mode == "zero_tokens":
            out["usage"]["completion_tokens"] = 0
        elif mode == "null_content":
            out["choices"][0]["message"]["content"] = None
        self._send(200, out)

    @staticmethod
    def completion(req: dict, text: str | None) -> dict:
        return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": req.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 41, "completion_tokens": 9, "total_tokens": 50}}


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:   # a client that hangs up is expected
        pass


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
