"""bench/engines/_http.py: the small standard-library HTTP client the vLLM (and later SGLang) engines share.

Every test talks to the fake server in fake_vllm_server.py on 127.0.0.1. Nothing leaves the machine.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from engines import EngineError  # noqa: E402
from engines import _http  # noqa: E402
from fake_vllm_server import FakeServer  # noqa: E402


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch):
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    with FakeServer() as s:
        yield s


def client(server: FakeServer, **kw) -> _http.HttpClient:
    kw.setdefault("timeout_s", 5.0)
    return _http.HttpClient(server.url, **kw)


def test_post_json_sends_a_json_body_with_the_content_type(server: FakeServer) -> None:
    out = client(server).post_json("/v1/chat/completions", {"model": "dgemma", "messages": []})
    assert out["model"] == "dgemma"
    req = server.state.posts()[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Content-Type"] == "application/json"
    assert req["body"] == b'{"model": "dgemma", "messages": []}'


def test_no_credentials_and_no_cookies_are_sent(server: FakeServer) -> None:
    c = client(server)
    c.get_json("/v1/models")
    c.post_json("/v1/chat/completions", {"model": "dgemma"})
    for r in server.state.requests:
        names = {k.lower() for k in r["headers"]}
        assert not names & {"authorization", "cookie", "proxy-authorization"}, names


def test_a_cookie_the_server_sets_is_not_sent_back(server: FakeServer) -> None:
    server.state.mode = "cookie"
    c = client(server)
    c.post_json("/v1/chat/completions", {"model": "dgemma"})
    c.post_json("/v1/chat/completions", {"model": "dgemma"})
    assert all("cookie" not in {k.lower() for k in r["headers"]} for r in server.state.requests)


def test_a_url_with_a_password_is_refused() -> None:
    with pytest.raises(EngineError, match="credentials"):
        _http.HttpClient("http://user:secret@127.0.0.1:8000", timeout_s=1)


def test_a_url_with_a_username_only_is_refused() -> None:
    with pytest.raises(EngineError, match="credentials"):
        _http.HttpClient("http://token@127.0.0.1:8000", timeout_s=1)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1/x", "127.0.0.1:8000", ""])
def test_only_http_and_https_urls_are_accepted(url: str) -> None:
    with pytest.raises(EngineError, match="http"):
        _http.HttpClient(url, timeout_s=1)


def test_a_server_error_carries_the_status_and_the_servers_message(server: FakeServer) -> None:
    server.state.mode = "http500"
    with pytest.raises(_http.HttpError) as e:
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert e.value.status == 500
    assert "500" in str(e.value) and "CUDA out of memory in the denoiser" in str(e.value)


def test_a_client_error_carries_the_status_and_the_servers_message(server: FakeServer) -> None:
    server.state.mode = "http400"
    with pytest.raises(_http.HttpError, match="400.*max_tokens is too large: 9999"):
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})


def test_a_plain_text_error_body_is_quoted(server: FakeServer) -> None:
    server.state.mode = "http503text"
    with pytest.raises(_http.HttpError, match="503.*upstream is restarting"):
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})


def test_a_404_is_an_http_error_with_its_status(server: FakeServer) -> None:
    with pytest.raises(_http.HttpError) as e:
        client(server).get_text("/metrics")
    assert e.value.status == 404


def test_an_http_error_is_not_retried(server: FakeServer) -> None:
    server.state.mode = "http500"
    with pytest.raises(_http.HttpError):
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert len(server.state.posts()) == 1


def test_a_response_over_the_size_bound_is_refused(server: FakeServer) -> None:
    server.state.mode = "huge"            # 6 MiB with no Content-Length
    with pytest.raises(EngineError, match="larger than"):
        client(server, max_bytes=1024 * 1024).post_json("/v1/chat/completions", {"model": "dgemma"})


def test_a_declared_length_over_the_bound_is_refused_before_reading(server: FakeServer) -> None:
    server.state.metrics = "x" * 5000
    with pytest.raises(EngineError, match="declares a body of 5000 bytes, larger than"):
        client(server, max_bytes=1000).get_text("/metrics")


def test_a_response_exactly_at_the_bound_is_accepted(server: FakeServer) -> None:
    server.state.metrics = "x" * 1000
    assert client(server, max_bytes=1000).get_text("/metrics") == "x" * 1000


def test_a_response_one_byte_over_the_bound_is_refused(server: FakeServer) -> None:
    server.state.metrics = "x" * 1001
    with pytest.raises(EngineError, match="larger than"):
        client(server, max_bytes=1000).get_text("/metrics")


def test_a_slow_response_times_out_and_says_so(server: FakeServer) -> None:
    server.state.mode = "sleep"           # the server waits 1.5 s
    with pytest.raises(EngineError, match="timed out"):
        client(server, timeout_s=0.4).post_json("/v1/chat/completions", {"model": "dgemma"})


def test_a_response_trickled_past_the_deadline_times_out(server: FakeServer) -> None:
    # one byte every 0.25 s: no single socket read waits 0.6 s, the whole answer takes 3 s
    server.state.mode = "trickle"
    t0 = time.monotonic()
    with pytest.raises(EngineError, match="timed out"):
        client(server, timeout_s=0.6).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert time.monotonic() - t0 < 2.5


def test_a_refused_connection_names_the_url() -> None:
    with pytest.raises(EngineError, match="127.0.0.1:9"):
        _http.HttpClient("http://127.0.0.1:9", timeout_s=2).get_json("/v1/models")


def test_a_body_that_is_not_json_is_refused(server: FakeServer) -> None:
    server.state.mode = "notjson"
    with pytest.raises(EngineError, match="not JSON"):
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})


def test_a_redirect_is_refused_and_not_followed(server: FakeServer) -> None:
    server.state.mode = "redirect"
    with pytest.raises(EngineError, match="redirect"):
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert [r["path"] for r in server.state.requests] == ["/v1/chat/completions"]


def test_environment_proxies_are_ignored_by_default(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(k, "http://127.0.0.1:9")          # nothing listens there
    assert client(server).get_json("/v1/models")["object"] == "list"


def test_environment_proxies_are_used_when_asked(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(k, server.url)                    # the fake server plays the proxy
    c = _http.HttpClient("http://server.invalid:8000", timeout_s=5, use_env_proxy=True)
    assert c.get_json("/v1/models")["object"] == "list"
    assert server.state.requests[0]["path"] == "http://server.invalid:8000/v1/models"


def test_get_text_returns_the_body_as_text(server: FakeServer) -> None:
    server.state.metrics = "a 1\nb 2\n"
    assert client(server).get_text("/metrics") == "a 1\nb 2\n"


def test_the_servers_message_is_the_error_text_alone(server: FakeServer) -> None:
    server.state.mode = "http400"
    with pytest.raises(_http.HttpError) as e:
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert str(e.value) == f"POST {server.url}/v1/chat/completions answered HTTP 400: max_tokens is too large: 9999"


def test_a_trailing_slash_on_the_base_url_does_not_double_the_slash(server: FakeServer) -> None:
    _http.HttpClient(server.url + "/", timeout_s=5).get_json("/v1/models")
    assert [r["path"] for r in server.state.requests] == ["/v1/models"]


@pytest.mark.parametrize("suffix", ["/?x=1", "#frag", "?x=1"])
def test_a_base_url_with_a_query_or_fragment_is_refused(server: FakeServer, suffix: str) -> None:
    with pytest.raises(EngineError, match="query or fragment"):
        _http.HttpClient(server.url + suffix, timeout_s=1)


def test_clean_drops_control_characters_and_keeps_printable_text_and_newlines() -> None:
    assert _http.clean("a\x1b[2Jb\x07c\rd\ne\tf\u202eg h\u00e9") == "a[2Jbcd\nefg h\u00e9"


def test_clean_caps_the_length() -> None:
    out = _http.clean("z" * 1000)
    assert out == "z" * 400 + "..." and _http.clean("z" * 400) == "z" * 400


def test_a_servers_error_text_cannot_inject_terminal_escapes_and_is_capped(server: FakeServer) -> None:
    server.state.mode = "escape"
    with pytest.raises(_http.HttpError) as e:
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})
    msg = str(e.value)
    assert not {"\x1b", "\x07", "\r"} & set(msg)
    assert "bad" in msg and "\nnext" in msg
    assert len(msg.split("HTTP 500: ", 1)[1]) <= 403


def test_a_plain_text_error_body_cannot_inject_escapes_either(server: FakeServer) -> None:
    server.state.mode = "http503esc"
    with pytest.raises(_http.HttpError) as e:
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert "\x1b" not in str(e.value) and "down[31m!" in str(e.value)


def test_a_non_json_body_is_quoted_without_raw_control_characters(server: FakeServer) -> None:
    server.state.mode = "notjson"
    with pytest.raises(EngineError) as e:
        client(server).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert "\x1b" not in str(e.value)


def test_headers_trickled_past_the_deadline_time_out(server: FakeServer) -> None:
    # a header line every 0.25 s and no end of headers: no socket read waits 0.6 s, the answer never completes
    server.state.mode = "header_trickle"
    t0 = time.monotonic()
    with pytest.raises(EngineError, match="timed out"):
        client(server, timeout_s=0.6).post_json("/v1/chat/completions", {"model": "dgemma"})
    assert time.monotonic() - t0 < 2.0


def test_a_server_that_accepts_and_never_answers_times_out() -> None:
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        url = f"http://127.0.0.1:{srv.getsockname()[1]}"
        t0 = time.monotonic()
        with pytest.raises(EngineError, match="timed out"):
            _http.HttpClient(url, timeout_s=0.5).get_json("/v1/models")
        assert time.monotonic() - t0 < 2.0
