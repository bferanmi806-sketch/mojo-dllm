"""The two checks of an OpenAI-compatible server that the vLLM and SGLang engines share.

`model_cards` reads `GET /v1/models` and refuses a malformed list. `parse_chat` reads one chat completion and refuses
what cannot be scored: a response that names another model, no choices, an empty message, or no token counts.
Server text reaches an error message only through `_http.clean`.
"""

from __future__ import annotations

from . import EngineError
from ._http import HttpClient, clean


def model_cards(client: HttpClient) -> list[dict]:
    """The model cards of `GET /v1/models`; each has a string `id`. Refuses a malformed or empty list."""
    doc = client.get_json("/v1/models")
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list):
        raise EngineError(f"GET /v1/models answered a malformed model list: {clean(repr(doc))}")
    if not data:
        raise EngineError("the server reports no models")
    if not all(isinstance(c, dict) and isinstance(c.get("id"), str) for c in data):
        raise EngineError(f"GET /v1/models answered a malformed model list: {clean(repr(doc))}")
    return data


def parse_chat(out: object, served_model: str) -> tuple[str, int, int, object]:
    """`(text, prompt_tokens, completion_tokens, finish_reason)` of one chat completion, or an EngineError."""
    if not isinstance(out, dict):
        raise EngineError(f"the completion is not a JSON object: {clean(repr(out))}")
    if out.get("model") != served_model:
        raise EngineError(f"the response names the model {clean(repr(out.get('model')))}, not {served_model!r}")
    choices = out.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise EngineError(f"the completion has no choices: {clean(repr(out))}")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise EngineError(f"the server returned an empty completion (finish_reason {clean(repr(choices[0].get('finish_reason')))})")
    usage = out.get("usage")
    counts = [usage.get(k) if isinstance(usage, dict) else None for k in ("prompt_tokens", "completion_tokens")]
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in counts) or counts[1] <= 0:
        raise EngineError(f"the completion carries no usable token counts: {clean(repr(usage))}")
    return content, counts[0], counts[1], choices[0].get("finish_reason")
