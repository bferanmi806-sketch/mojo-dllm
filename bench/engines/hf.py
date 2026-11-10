"""The LLaDA and Dream authors' own sampling code as a GSM8K engine (issue #178).

The adapter calls the entry points the authors publish and adds no sampler of its own:

- LLaDA: the standalone `generate(model, prompt, attention_mask, steps, gen_length, block_length, temperature,
  cfg_scale, remasking, mask_id, ...)` of `generate.py` in a checkout of github.com/ML-GSAI/LLaDA (`--llada-repo`).
  That repository carries no license file, so the adapter imports the file from the checkout and never copies it.
  Read at commit 9182493720ed723ef8031210d85959364e51cbe0. Its `main()` is the template for the prompt: the chat
  template with `add_generation_prompt=True`, `tokenize=False`, then the tokenizer with
  `add_special_tokens=False`, left padding, and a pad id that must differ from the mask id 126336. The function
  returns the whole sequence (prompt plus generation), so the answer is `out[:, prompt_length:]`.
- Dream: `model.diffusion_generate(input_ids, attention_mask=..., max_new_tokens=..., steps=..., temperature=...,
  alg=..., alg_temp=..., return_dict_in_generate=True)` of `generation_utils.py` in the remote code of
  huggingface.co/Dream-org/Dream-v0-Instruct-7B, read at revision 05334cb9faaf763692dcf9d8737c642be2b2a6ae, and
  documented in github.com/DreamLM/Dream (README read at 31f94a60d187e3fd481fee3bbc2c732eb94a879c). The result's
  `sequences` holds prompt plus generation; the README cuts the text at the end-of-text token.

Settings for a like-for-like run against the llama.cpp and mojo-dllm arms: temperature 0, steps equal to the
generation length, block length 32 and low-confidence remasking for LLaDA, entropy remasking for Dream (which has
no blocks). The Dream README example differs (temperature 0.2, top_p 0.95, 512 steps; its library default
algorithm is `origin`), and `flags` says so.

`torch`, `transformers` and `bitsandbytes` are imported only by `load_model`, which runs after every cheap check:
this module imports in the repository's frozen environment, and each model runs from its own virtual environment
(docs/benchmarks.md). Nothing here is a speed baseline: the sampling loop is Python.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from . import EngineError, GenConfig, GenResult, register

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import isolation  # noqa: E402  (bench/isolation.py: its /proc/meminfo reader)
import spec  # noqa: E402  (bench/spec.py)

SHORT_TIMEOUT_S = 60           # git rev-parse
LLADA_MASK_ID = spec.ARCH_IDS["llada"]["mask"]   # generate.py's default and the id its README assigns to [MASK]
LLADA_PARAMS = ("attention_mask", "steps", "gen_length", "block_length", "temperature", "cfg_scale", "remasking")
README_EXAMPLE = {"temperature": 0.2, "top_p": 0.95, "steps": 512, "alg": "entropy", "alg_temp": 0.0}
SHA_RE = re.compile(r"[0-9a-f]{40}")
GIB = 1.073741824              # GB per GiB
# Sum of the .safetensors of each repository on the Hugging Face API, 2026-11-09 (issue #178), in GiB.
WEIGHTS_GIB = {"GSAI-ML/LLaDA-8B-Instruct": 14.93, "Dream-org/Dream-v0-Instruct-7B": 14.19}
WORKING_GB = 3.0               # an estimate, not measured: the Python loop's working memory (LLaDA's float64 Gumbel noise is about 0.5 GB a tensor)
BF16_RESIDENT = 1.25           # an estimate, not measured: resident set of a bf16 load over the weights' size
NF4_RATIO = 3.5                # an estimate, not measured: bf16 weights over their 4-bit size


@dataclass(frozen=True)
class Spec:
    """What `load_model` needs: the weights to load and where they go."""

    family: str                   # "llada" or "dream"
    model_id: str                 # a Hugging Face repository id or a local directory
    revision: str | None          # the model repository's commit sha, or None when nobody gave one
    quant: str                    # "none" (bf16) or "4bit" (bitsandbytes nf4)
    device: str                   # "cpu" or "gpu"


def package_version(name: str) -> str | None:
    """The installed version of a package, read from its metadata so the package is not imported."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def mem_available_gb(machine=None) -> float | None:
    """MemAvailable in GB (1e9 bytes) from /proc/meminfo, read the way the isolation preflight reads it, or None
    when it cannot be read."""
    kv = isolation.read_meminfo(machine or isolation.Machine())
    return None if kv is None else kv["MemAvailable"] * 1024 / 1e9


def family_of(model_id: str) -> str:
    lower = model_id.lower()
    for family in ("llada", "dream"):
        if family in lower:
            return family
    raise SystemExit(f"hf runs LLaDA or Dream, and {model_id!r} names neither")


def load_model(spec: Spec):
    """Load the weights and the tokenizer; the only place torch and transformers are imported.

    Returns (model, tokenizer). bf16 goes to the device with `.to`; a 4-bit model is placed by bitsandbytes on GPU 0
    and cannot be moved. The revision pins the weights and the remote code together (`trust_remote_code=True`
    executes the repository's Python at that revision; with no revision it is whatever the default branch holds)."""
    import torch
    import transformers

    if spec.device == "gpu" and not torch.cuda.is_available():
        raise EngineError("--device gpu but torch.cuda.is_available() is false: this is not a CUDA run")
    kw: dict = dict(trust_remote_code=True, revision=spec.revision, low_cpu_mem_usage=True)
    if spec.quant == "4bit":
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = transformers.AutoModel.from_pretrained(spec.model_id, **kw)
    if spec.quant != "4bit":
        model = model.to("cuda" if spec.device == "gpu" else "cpu")
    model = model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True, revision=spec.revision)
    return model, tokenizer


def device_error(model, device: str) -> str | None:
    """Why the model does not sit on the requested device, or None when every parameter does.

    A model that fell back to the CPU still answers, slowly, and would be labelled a GPU run. A model with no
    parameters is refused too: nothing was checked."""
    want = "cuda" if device == "gpu" else "cpu"
    seen = {p.device.type for p in model.parameters()}
    if seen != {want}:
        return f"the model is not on the {'GPU' if device == 'gpu' else 'CPU'}: its parameters sit on {sorted(seen) or 'nothing'}"
    return None


def import_llada_generate(repo: pathlib.Path):
    """`generate` from the checkout's generate.py, imported in place. The file is never copied (no license)."""
    path = repo / "generate.py"
    if not path.is_file():
        raise SystemExit(f"{repo} has no generate.py: --llada-repo must be a checkout of github.com/ML-GSAI/LLaDA")
    spec = importlib.util.spec_from_file_location("llada_generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "generate", None)
    if not callable(fn):
        raise SystemExit(f"{path} defines no generate()")
    missing = [p for p in LLADA_PARAMS if p not in inspect.signature(fn).parameters]
    if missing:
        raise SystemExit(f"{path}: generate() has no parameter {', '.join(missing)}; update the LLaDA checkout")
    return fn


def _call_with_timeout(fn, timeout_s: float, what: str):
    """Run `fn` in a thread; a timeout or an exception becomes an EngineError. A thread cannot be killed, so the
    caller stops the process (the runner exits on EngineError); the thread is a daemon and dies with it."""
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except Exception as e:  # noqa: BLE001  (anything the model code raises ends the run)
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise EngineError(f"{what} timed out after {timeout_s:g} s")
    if "error" in box:
        e = box["error"]
        raise EngineError(f"{what} raised {type(e).__name__}: {e}") from e
    return box["value"]


class LLaDABackend:
    def __init__(self, model, tokenizer, generate, device: str) -> None:
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"          # what the authors' main() does
        if tokenizer.pad_token_id == LLADA_MASK_ID:
            raise EngineError(f"the tokenizer's pad token is the mask token ({LLADA_MASK_ID}); generate() would break")
        self.model, self.tokenizer, self.generate = model, tokenizer, generate
        self.device = "cuda" if device == "gpu" else "cpu"

    def answer(self, prompt: str, cfg: GenConfig) -> str:
        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  add_generation_prompt=True, tokenize=False)
        enc = self.tokenizer(chat, add_special_tokens=False, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out = self.generate(self.model, ids, mask, steps=cfg.steps, gen_length=cfg.gen_length,
                            block_length=cfg.block_length, temperature=0., cfg_scale=0., remasking="low_confidence")
        return self.tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]


class DreamBackend:
    def __init__(self, model, tokenizer, device: str) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.device = "cuda" if device == "gpu" else "cpu"

    def answer(self, prompt: str, cfg: GenConfig) -> str:
        enc = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt",
                                                 return_dict=True, add_generation_prompt=True)
        ids = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device)
        out = self.model.diffusion_generate(ids, attention_mask=mask, max_new_tokens=cfg.gen_length,
                                            return_dict_in_generate=True, steps=cfg.steps, temperature=0.,
                                            alg="entropy", alg_temp=0.)
        text = self.tokenizer.decode(out.sequences[0][ids.shape[1]:].tolist())
        if not self.tokenizer.eos_token:
            raise EngineError("the Dream tokenizer has no end-of-text token to cut the answer at")
        return text.split(self.tokenizer.eos_token)[0]


def _git_env() -> dict[str, str]:
    """The environment without GIT_*: a pre-commit or pre-push hook exports GIT_DIR of its own repository."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


@register
class HF:
    name = "hf"

    def __init__(self, args: argparse.Namespace) -> None:
        self.hf_model = args.hf_model
        self.llada_repo = pathlib.Path(args.llada_repo).expanduser() if args.llada_repo else None
        self.quant = args.quant
        self.revision = args.model_revision
        self.allow_unpinned = args.allow_unpinned
        self.family: str | None = None
        self.backend = None
        self._local: pathlib.Path | None = None

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hf-model", default=None,
                            help="where to load the weights from when it differs from --model: a Hugging Face "
                                 "repository id or a local directory (--model names the arm's model in the records)")
        parser.add_argument("--llada-repo", default=None,
                            help="a checkout of github.com/ML-GSAI/LLaDA; its generate.py is imported, never copied")
        parser.add_argument("--quant", choices=["none", "4bit"], default="none",
                            help="none: bf16; 4bit: bitsandbytes nf4 with bf16 compute (GPU only)")
        parser.add_argument("--allow-unpinned", action="store_true",
                            help="run a Hugging Face repository without --model-revision. `trust_remote_code` then "
                                 "executes whatever Python the default branch holds today; the rows say `unpinned`")
        parser.add_argument("--model-revision", default=None,
                            help="the model repository's 40-character commit sha; recorded in `version`. With none "
                                 "the weights and remote code are unpinned and `version` records null")

    def _model_id(self, cfg: GenConfig) -> str:
        return self.hf_model or cfg.model

    def _local_dir(self, cfg: GenConfig) -> pathlib.Path | None:
        path = pathlib.Path(self._model_id(cfg)).expanduser()
        return path if path.is_dir() else None

    def _unpinned(self, cfg: GenConfig) -> bool:
        """A hub repository loaded without a revision. A local directory is not on the hub: nothing to pin."""
        return self.revision is None and self._local_dir(cfg) is None

    def model_name(self, cfg: GenConfig) -> str:
        """The `model` of every row: what `compare` reads to decide whether the weights differ, so the
        quantization is part of it."""
        name = pathlib.Path(cfg.model).name
        return name + "|4bit" if self.quant == "4bit" else name

    def check(self, cfg: GenConfig) -> None:
        if cfg.device not in ("cpu", "gpu"):
            raise SystemExit(f"hf runs on --device cpu or gpu, not {cfg.device!r}")
        if self.quant == "4bit" and cfg.device != "gpu":
            raise SystemExit("--quant 4bit needs --device gpu (bitsandbytes runs on CUDA)")
        if self.revision is not None and not SHA_RE.fullmatch(self.revision):
            raise SystemExit(f"--model-revision must be a 40-character lowercase hex commit sha, not {self.revision!r}: "
                             "a branch or tag name moves and pins nothing")
        if self._unpinned(cfg) and not self.allow_unpinned:
            raise SystemExit("--model-revision is required for a Hugging Face repository: loading it runs the "
                             "repository's own Python (trust_remote_code) on this machine, and only a commit sha "
                             "says which code. Pass the 40-character sha, or --allow-unpinned to accept the "
                             "default branch")
        if cfg.extra:
            raise SystemExit(f"hf takes no extra engine flags; got {' '.join(cfg.extra)}")
        self.family = family_of(self._model_id(cfg))
        self._local = self._local_dir(cfg)
        generate = None
        if self.family == "llada":
            if self.llada_repo is None:
                raise SystemExit("LLaDA needs --llada-repo: a checkout of github.com/ML-GSAI/LLaDA")
            if cfg.block_length <= 0 or cfg.gen_length % cfg.block_length != 0:
                raise SystemExit(f"LLaDA's generate() needs the generation length ({cfg.gen_length}) to be a "
                                 f"multiple of the block length ({cfg.block_length})")
            num_blocks = cfg.gen_length // cfg.block_length
            if cfg.steps % num_blocks != 0:
                raise SystemExit(f"LLaDA's generate() needs steps ({cfg.steps}) to be a multiple of the number of "
                                 f"blocks ({num_blocks})")
            generate = import_llada_generate(self.llada_repo)
            self.llada_commit()                      # refuse a checkout whose commit cannot be read, before loading
        if self.quant == "4bit" and package_version("bitsandbytes") is None:
            raise SystemExit("--quant 4bit needs bitsandbytes in this environment (pip install bitsandbytes)")
        if self._unpinned(cfg):
            print("hf: --allow-unpinned: the weights and the remote code are not pinned and `version` records a "
                  "null model revision", file=sys.stderr)
        spec = Spec(self.family, self._model_id(cfg), self.revision, self.quant, cfg.device)
        self._guard_memory(cfg)
        try:
            model, tokenizer = load_model(spec)
        except EngineError:
            raise
        except Exception as e:  # noqa: BLE001  (an out-of-memory or a missing file ends the run before problem 1)
            raise EngineError(f"loading {spec.model_id} raised {type(e).__name__}: {e}") from e
        bad = device_error(model, cfg.device)
        if bad:
            raise EngineError(bad)
        self.backend = (LLaDABackend(model, tokenizer, generate, cfg.device) if self.family == "llada"
                        else DreamBackend(model, tokenizer, cfg.device))

    def llada_commit(self) -> str:
        """The checkout's HEAD, with `-dirty` when a tracked file differs from it (a local edit of generate.py
        would otherwise be recorded under the upstream commit)."""
        def git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(["git", "-C", str(self.llada_repo), *args], capture_output=True, text=True,
                                  errors="replace", timeout=SHORT_TIMEOUT_S, env=_git_env())

        p = git("rev-parse", "HEAD")
        sha = p.stdout.strip()
        if p.returncode != 0 or not SHA_RE.fullmatch(sha):
            raise EngineError(f"cannot read the commit of the LLaDA checkout {self.llada_repo}: "
                              f"{p.stderr.strip() or 'no output'}")
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode != 0:
            raise EngineError(f"cannot read the status of the LLaDA checkout {self.llada_repo}: {st.stderr.strip()}")
        return sha + "-dirty" if st.stdout.strip() else sha

    def version(self) -> str:
        """JSON: the model repository's sha (null when not given), the LLaDA checkout's commit (null for Dream) and
        the three library versions (null when not installed), read from package metadata."""
        local = self._local
        return json.dumps({"model_revision": self.revision, "model_source": "local" if local else "hub",
                           "model_dir": str(local) if local else None,
                           "model_revision_verified": self.revision is not None and not local,
                           "llada_commit": self.llada_commit() if self.family == "llada" else None,
                           "transformers": package_version("transformers"), "torch": package_version("torch"),
                           "bitsandbytes": package_version("bitsandbytes")}, sort_keys=True)

    def load_need_gb(self, cfg: GenConfig) -> float | None:
        """Free RAM the load itself needs, an estimate and not a measurement: bf16 is 1.25 times the weights plus
        3 GB of working memory, 4-bit is the weights over 3.5 plus 3 GB. The weights' size comes from the table for
        the two known repositories, else from the *.safetensors of a local directory, else None (unknown)."""
        local = self._local_dir(cfg)
        if local is not None:
            total = sum(f.stat().st_size for f in local.glob("*.safetensors") if f.is_file()) / 1e9
            gb = total or None
        else:
            gib = WEIGHTS_GIB.get(self._model_id(cfg))
            gb = gib * GIB if gib else None
        if gb is None:
            return None
        return (gb * BF16_RESIDENT if self.quant == "none" else gb / NF4_RATIO) + WORKING_GB

    def ram_gb_after_load(self, cfg: GenConfig) -> float | None:
        """Free RAM the preflight asks for. `check` has loaded the weights by then, so MemAvailable already excludes
        them and only the working memory is still needed. None (unknown weights size) fails the preflight."""
        return WORKING_GB if self.load_need_gb(cfg) is not None else None

    def _guard_memory(self, cfg: GenConfig) -> None:
        """Refuse before `from_pretrained` when the load would not fit. Fails closed: an unknown weights size or an
        unreadable /proc/meminfo refuses too."""
        need = self.load_need_gb(cfg)
        if need is None:
            raise EngineError(f"cannot size the memory for {self._model_id(cfg)}: the weights' size is unknown "
                              "(not one of the two known repositories, and no *.safetensors in a local directory)")
        avail = mem_available_gb()
        if avail is None:
            raise EngineError("cannot read MemAvailable from /proc/meminfo, so the load is not started")
        if avail < need:
            raise EngineError(f"MemAvailable is {avail:.1f} GB and loading {self._model_id(cfg)} needs about "
                              f"{need:.1f} GB (an estimate): refused before loading")

    def flags(self, cfg: GenConfig) -> dict:
        base = {"engine": self.name, "family": self.family, "hf_model": self._model_id(cfg), "quant": self.quant,
                "weights": "bf16" if self.quant == "none" else "bnb-nf4", "device": cfg.device,
                "unpinned": self._unpinned(cfg),
                "gen_length": cfg.gen_length, "steps": cfg.steps, "temperature": 0.0,
                "forward_passes_measured": False}
        if self.family == "llada":
            return {**base, "block_length": cfg.block_length, "remasking": "low_confidence", "cfg_scale": 0.0,
                    "mask_id": LLADA_MASK_ID}
        mine = {"temperature": 0.0, "top_p": None, "steps": cfg.steps, "alg": "entropy", "alg_temp": 0.0}
        return {**base, "alg": "entropy", "alg_temp": 0.0, "top_p": None, "top_k": None,
                "readme_example": README_EXAMPLE, "library_default_alg": "origin",
                "departs_from_readme": sorted(k for k in mine if mine[k] != README_EXAMPLE[k])}

    def generate(self, prompt: str, cfg: GenConfig) -> dict:
        # generate_s covers tokenizing, the denoising loop and decoding, all in this process; decoding copies the
        # result to the host, so the GPU work has finished when the clock stops.
        start = time.perf_counter()
        text = _call_with_timeout(lambda: self.backend.answer(prompt, cfg), cfg.timeout_s, f"{self.family} generation")
        # The authors' loop is not instrumented: the configured steps go in `steps`, `forward_passes` stays None.
        return GenResult(text=text, generate_s=time.perf_counter() - start, forward_passes=None, steps=cfg.steps)
