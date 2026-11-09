"""Engines the GSM8K runner (bench/eval_engine.py) can drive besides mojo-dllm.

An engine answers one prompt and reports the text, the generation time and what it knows about the work done.
`eval_engine.py` does everything else (the prompt, scoring, the arm file, resuming, power), so an engine is
only the part that differs: how to start it and how to read what it printed.

To add one, write a class with the members of `Engine`, decorate it with `@register`, and import its module
at the bottom of this file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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

    def generate(self, prompt: str, cfg: GenConfig) -> dict:
        """Answer one prompt. Returns `text`, `generate_s`, `forward_passes` (int, or None when the engine
        cannot count them) and `steps`. Raises EngineError when the run is not a valid one."""


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
