"""A small HTTP client for engines that talk to a server someone else started (the vLLM and SGLang engines).

Standard library only (`urllib`), so the frozen environment needs nothing new. What it decides, and why:

- It sends no credentials. There is no Authorization header, no cookie jar (a `Set-Cookie` is dropped, never sent
  back), no proxy credentials, and a base URL that carries `user:password@` is refused. A benchmark client has no
  business holding a secret, and the arm files record nothing that could leak one.
- It ignores the environment's proxy variables unless `use_env_proxy` is set. `urllib` honours `http_proxy` by
  default; a proxy in the path changes the timing a benchmark reads and sends prompts through a third machine.
- It never retries. A failed request raises; a retry could hide a failure, and a resent generation would be timed
  twice. It never follows a redirect: the target would be a second server nobody named.
- One deadline (`timeout_s`) covers the whole exchange: connect, headers and body. The request runs in a daemon
  thread that the caller stops waiting for at the deadline (urllib's own timeout restarts at every socket read, so a
  server that trickles bytes could outlast it). A body over `max_bytes` is refused, so a hung or hostile server
  cannot stall the run or fill the memory.
- Errors carry what the server said: the status and its message (`error.message` of an OpenAI-style body, else the
  start of the text), so a run that stops says why. Server text reaches the terminal only through `clean`: control
  characters are dropped (a terminal escape cannot ride in an error) and the length is capped. The text of a
  connection or transport error (a refused connection, a reset, a proxy's tunnel failure) goes through `clean` as well:
  a proxy in the path writes the status text of its own reply.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.client import HTTPException
from urllib.parse import urlsplit

from . import EngineError

DEFAULT_MAX_BYTES = 8 * 1024 * 1024     # a 256-token completion is a few KB; the metrics page is the largest answer
ERROR_BODY_BYTES = 4096
CLEAN_LIMIT = 400
CHUNK = 64 * 1024
USER_AGENT = "mojo-dllm-bench"


def clean(text: str, limit: int = CLEAN_LIMIT) -> str:
    """Server text made safe to print: only printable characters and newlines survive, and the length is capped."""
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    return text if len(text) <= limit else text[:limit] + "..."


class HttpError(EngineError):
    """The server answered with a 4xx or 5xx status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):  # type: ignore[no-untyped-def]
        return None


def _opener(use_env_proxy: bool) -> urllib.request.OpenerDirector:
    """An opener with only the handlers a plain request needs: no cookies, no auth, no FTP/file, no redirects."""
    opener = urllib.request.OpenerDirector()
    for h in (urllib.request.ProxyHandler() if use_env_proxy else urllib.request.ProxyHandler({}),
              urllib.request.HTTPHandler(), urllib.request.HTTPSHandler(), urllib.request.HTTPDefaultErrorHandler(),
              urllib.request.HTTPErrorProcessor(), _NoRedirect()):
        opener.add_handler(h)
    return opener


def _server_message(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        doc = json.loads(text)
    except ValueError:
        return text[:500]
    if isinstance(doc, dict):
        err = doc.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        for key in ("message", "detail", "error"):
            if isinstance(doc.get(key), str):
                return doc[key]
    return text[:500]


class HttpClient:
    def __init__(self, base_url: str, *, timeout_s: float, max_bytes: int = DEFAULT_MAX_BYTES,
                 use_env_proxy: bool = False) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise EngineError(f"base URL {base_url!r} must start with http:// or https:// and name a host")
        if parts.username is not None or parts.password is not None:
            raise EngineError("the base URL carries credentials (user:password@): this client sends none, "
                              "remove them from the URL")
        if parts.query or parts.fragment:
            raise EngineError("the base URL has a query or fragment, which the path join would break: "
                              "give scheme, host, port and an optional path prefix only")
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self._opener = _opener(use_env_proxy)

    def get_json(self, path: str) -> object:
        return self._json(self._request("GET", path, None), path)

    def post_json(self, path: str, payload: object) -> object:
        return self._json(self._request("POST", path, json.dumps(payload).encode()), path)

    def get_text(self, path: str) -> str:
        return self._request("GET", path, None).decode("utf-8", errors="replace")

    def _json(self, raw: bytes, path: str) -> object:
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as e:      # UnicodeDecodeError is a ValueError
            raise EngineError(f"{self.base}{path} answered with a body that is not JSON: "
                              f"{clean(repr(raw[:200].decode('utf-8', errors='replace')))}") from e

    def _request(self, method: str, path: str, body: bytes | None) -> bytes:
        url = self.base + path
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        timed_out = f"{method} {url} timed out after {self.timeout_s:g} s"
        result: dict = {}

        def work() -> None:
            try:
                result["body"] = self._exchange(req, method, url, timed_out, result)
            except BaseException as e:  # noqa: BLE001  (handed to the caller's thread below)
                result["error"] = e

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(self.timeout_s)
        if worker.is_alive():
            resp = result.get("resp")
            if resp is not None:
                try:
                    resp.close()
                except Exception:  # noqa: BLE001  (the worker may be reading it)
                    pass
            raise EngineError(timed_out)
        if "error" in result:
            raise result["error"]
        return result["body"]

    def _exchange(self, req: urllib.request.Request, method: str, url: str, timed_out: str, result: dict) -> bytes:
        try:
            resp = self._opener.open(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                raise EngineError(f"{method} {url} answered with a redirect ({e.code}) to "
                                  f"{clean(repr(e.headers.get('Location')))}: redirects are not followed") from None
            msg = clean(_server_message(e.read(ERROR_BODY_BYTES)))
            raise HttpError(e.code, f"{method} {url} answered HTTP {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise EngineError(timed_out) from None
            raise EngineError(f"cannot reach {url}: {clean(str(e.reason))}") from None
        except TimeoutError:
            raise EngineError(timed_out) from None
        except (OSError, HTTPException) as e:
            raise EngineError(f"{method} {url} failed: {type(e).__name__}: {clean(str(e))}") from None
        result["resp"] = resp
        with resp:
            declared = resp.headers.get("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
                raise EngineError(f"{url} declares a body of {declared} bytes, larger than the {self.max_bytes} "
                                  "byte bound: refused unread")
            chunks: list[bytes] = []
            size = 0
            try:
                while True:
                    piece = resp.read1(CHUNK)
                    if not piece:
                        break
                    size += len(piece)
                    if size > self.max_bytes:
                        raise EngineError(f"{url} sent a body larger than the {self.max_bytes} byte bound: refused")
                    chunks.append(piece)
            except TimeoutError:
                raise EngineError(timed_out) from None
            except (OSError, HTTPException) as e:
                raise EngineError(f"{method} {url} failed while reading the body: {type(e).__name__}: {clean(str(e))}") from None
        return b"".join(chunks)
