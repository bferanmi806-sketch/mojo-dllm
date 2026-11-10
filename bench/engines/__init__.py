"""Engines the GSM8K runner (bench/eval_engine.py) can drive besides mojo-dllm.

An engine answers one prompt and reports the text, the generation time and what it knows about the work done.
`eval_engine.py` does everything else (the prompt, scoring, the arm file, resuming, power), so an engine is
only the part that differs: how to start it and how to read what it printed.

To add one, write a class with the members of `Engine`, decorate it with `@register`, and import its module
at the bottom of this file.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, fields
from typing import Protocol, runtime_checkable


class EngineError(RuntimeError):
    """The engine ran and its answer cannot be scored. The runner stops and names the problem."""


@dataclass(frozen=True)
class GenConfig:
    """What one generation is asked to do; the same for every problem of an arm."""

    model: str
    device: str = "gpu"
    gen_length: int = 256
    steps: int = 256
    block_length: int = 32
    schedule: str = "timestep"
    extra: tuple[str, ...] = field(default_factory=tuple)
    timeout_s: float = 1800.0      # per generation; a timeout fails the run, it never scores a row


def _count(name: str, value: object, minimum: int, optional: bool = True) -> None:
    if value is None and optional:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise EngineError(f"{name} must be an integer of at least {minimum}{' or None' if optional else ''}, "
                          f"got {value!r}")


@dataclass(frozen=True)
class GenResult:
    """What an engine reports for one prompt, checked when it is built.

    `forward_passes` is the number of full-canvas passes the engine itself counted, or None when it does not
    count them. It is never the configured number: an engine that does not measure puts the configured number
    in `steps` and leaves `forward_passes` None, and `flags.forward_passes_measured` says which kind the arm has.
    `finish_reason` is the server's (`stop`, `length`), None for an engine that has none; a row that ended on
    `length` was cut off by the token limit.

    Reading by key (`g["text"]`) works for the fields, as it did when the result was a dict.
    """

    text: str
    generate_s: float
    forward_passes: int | None = None
    steps: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    stopped_early: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise EngineError("the engine returned no text; not scored")
        g = self.generate_s
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) or g <= 0:
            raise EngineError(f"generate_s must be a finite number above 0, got {g!r}: a timing that is zero, "
                              "negative or NaN would enter every mean")
        _count("forward_passes", self.forward_passes, 1)
        _count("steps", self.steps, 1)
        _count("prompt_tokens", self.prompt_tokens, 0)
        _count("completion_tokens", self.completion_tokens, 0)
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise EngineError(f"finish_reason must be a string or None, got {self.finish_reason!r}")
        if self.stopped_early is not None and not isinstance(self.stopped_early, bool):
            raise EngineError(f"stopped_early must be a bool or None, got {self.stopped_early!r}")

    def __getitem__(self, key: str) -> object:
        if key not in {f.name for f in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default) if key in {f.name for f in fields(self)} else default


@runtime_checkable
class Engine(Protocol):
    name: str

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Options only this engine has (paths, threads)."""

    def version(self) -> str:
        """What was run, recorded next to every answer: a commit or a package version."""

    def check(self, cfg: GenConfig) -> None:
        """Refuse before the first problem: a missing binary or model, a device this engine cannot use.

        Raises SystemExit with a message that names the thing."""

    def flags(self, cfg: GenConfig) -> dict:
        """The configuration an arm keeps, recorded in every row. It decides what a resume may change."""

    def generate(self, prompt: str, cfg: GenConfig) -> GenResult:
        """Answer one prompt. `forward_passes` is the engine's own count or None, `steps` the configured (or, for
        an engine that measures, the measured) step count. Raises EngineError when the run is not a valid one."""


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    if cls.name in _REGISTRY:
        raise ValueError(f"engine {cls.name!r} is registered twice")
    _REGISTRY[cls.name] = cls
    return cls


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> type:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown engine {name!r}; known engines: {', '.join(names())}")
    return _REGISTRY[name]


from . import llamacpp  # noqa: E402,F401  (registers itself)
from . import hf  # noqa: E402,F401  (registers itself)
from . import vllm  # noqa: E402,F401  (registers itself)
from . import sglang  # noqa: E402,F401  (registers itself)
