"""A/B evidence from two builds refuses identical, stale or unrecorded binaries (#246), and Dream gets no --block-length (#206)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from array import array
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "scripts"))
import ab_evidence  # noqa: E402  (bench/ab_evidence.py)


def load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    assert s is not None and s.loader is not None
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


dispatch = load("check_gpu_dispatch_ab", ROOT / "bench/check_gpu_dispatch.py")
logits = load("compare_logits_ab", ROOT / "scripts/compare_logits.py")
profile = load("gpu_profile_ab", ROOT / "scripts/gpu_profile.py")

MASK, VOCAB = 9, 4


class Rig:
    """Two checkouts with a binary each, and a fake `run` that answers git, tokenize, logits and run."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp, self.calls = tmp, []
        self.same_head = self.dirty_new = False
        self.same_rows = True
        for name in ("base", "new"):
            (tmp / name / "build").mkdir(parents=True)
            (tmp / name / "build/mojo-dllm").write_bytes(f"binary of {name}".encode())
            (tmp / name / "bench").mkdir()
            (tmp / name / "bench/config-gpu.json").write_text(json.dumps({"prompts": ["p0", "p1", "p2"]}))
        (tmp / "m.gguf").write_bytes(b"model")
        monkeypatch.setattr(dispatch, "ARCH_IDS", {"llada": dict(vocab=VOCAB, mask=MASK), "dream": dict(vocab=VOCAB, mask=MASK)})

    def head_of(self, root: str) -> str:
        return "one-commit" if self.same_head else "head-of-" + Path(root).name

    def run(self, cmd: list[str]) -> str:
        self.calls.append(cmd)
        if cmd[0] == "git":
            root = cmd[2]
            if "status" in cmd:
                return " M src/main.mojo\n" if self.dirty_new and Path(root).name == "new" else ""
            return self.head_of(root) + "\n"
        exe = Path(cmd[0]).parent.parent.name
        if cmd[1] == "tokenize":
            return "log\n[1, 2]\n"
        if cmd[1] == "logits":
            ids = cmd[cmd.index("--tokens") + 1].split(",")
            Path(cmd[cmd.index("--out") + 1]).write_bytes(array("f", [1.0 + (0 if self.same_rows else len(exe))] * (len(ids) * VOCAB)).tobytes())
            return ""
        if cmd[1] == "run":
            return "noise\n" + json.dumps({"tokens": [1, 2], "forward_passes": 32}) + "\n"
        raise AssertionError(cmd)

    def spawned(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] != "git"]

    def main(self, monkeypatch: pytest.MonkeyPatch, arch: str = "llada", **replace: str) -> dict:
        monkeypatch.setattr(dispatch, "run", self.run)
        args = ["check_gpu_dispatch", "--base", str(self.tmp / "base"), "--new", str(self.tmp / "new"),
                "--model", str(self.tmp / "m.gguf"), "--arch", arch, "--out", str(self.tmp / "o/ev.json")]
        for flag, value in replace.items():
            args[args.index("--" + flag) + 1] = value
        monkeypatch.setattr(sys, "argv", args)
        dispatch.main()
        return json.loads((self.tmp / "o/ev.json").read_text())


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    return Rig(tmp_path, monkeypatch)


def refused(rig: Rig, monkeypatch: pytest.MonkeyPatch, match: str, **replace: str) -> None:
    with pytest.raises(SystemExit, match=match):
        rig.main(monkeypatch, **replace)
    assert rig.spawned() == [] and not (rig.tmp / "o").exists(), "a model run started before the refusal"


def test_dispatch_the_same_checkout_twice_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    refused(rig, monkeypatch, "same checkout", base=str(rig.tmp / "new"))


def test_dispatch_one_clean_commit_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.same_head = True
    refused(rig, monkeypatch, "same commit")


def test_dispatch_identical_binaries_are_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    (rig.tmp / "base/build/mojo-dllm").write_bytes(b"binary of new")
    refused(rig, monkeypatch, "byte-identical")


def test_dispatch_a_missing_binary_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    (rig.tmp / "new/build/mojo-dllm").unlink()
    refused(rig, monkeypatch, "does not exist")


@pytest.mark.parametrize("prompts", [[], [""], None])
def test_dispatch_an_empty_prompt_list_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch, prompts: object) -> None:
    (rig.tmp / "new/bench/config-gpu.json").write_text(json.dumps({"prompts": prompts}))
    refused(rig, monkeypatch, "no usable prompts")


def test_dispatch_evidence_records_binaries_dirty_flag_and_cuda_selection(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    rig.same_head, rig.dirty_new = True, True
    doc = rig.main(monkeypatch)
    assert doc["versions"] == {"base": "one-commit", "new": "one-commit-dirty"}
    assert doc["binary_sha256"] == {k: hashlib.sha256(f"binary of {k}".encode()).hexdigest() for k in ("base", "new")}
    assert doc["cuda"] == {"visible_devices": "1", "device_order": "PCI_BUS_ID"}
    assert len(doc["logits"]) == 8 and all(r["bytes_equal"] for r in doc["logits"]) and len(doc["generation"]) == 3


def test_dispatch_a_run_without_cuda_selection_records_none(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    assert rig.main(monkeypatch)["cuda"] == {"visible_devices": None, "device_order": None}


def test_dispatch_dream_gets_no_block_length_and_llada_does(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.main(monkeypatch, arch="llada")
    runs = [c for c in rig.calls if len(c) > 1 and c[1] == "run"]
    assert runs and all("--block-length" in c for c in runs)
    rig.calls.clear()
    (rig.tmp / "o/ev.json").unlink()
    rig.main(monkeypatch, arch="dream")
    runs = [c for c in rig.calls if len(c) > 1 and c[1] == "run"]
    assert runs and all("--block-length" not in c for c in runs) and all("--steps" in c for c in runs)


def test_dispatch_different_logits_still_fail_the_check(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.same_rows = False
    with pytest.raises(RuntimeError, match="GPU logits differ"):
        rig.main(monkeypatch)


# compare_logits -----------------------------------------------------------------------------------------

def logits_rig(tmp: Path, monkeypatch: pytest.MonkeyPatch, same_head: bool = False) -> list[list[str]]:
    for name in ("base", "new"):
        (tmp / name / "build").mkdir(parents=True)
        (tmp / name / "build/mojo-dllm").write_bytes(f"binary of {name}".encode())
    calls: list[list[str]] = []

    def run(cmd: list[str]) -> str:
        calls.append(cmd)
        if cmd[0] == "git":
            return "" if "status" in cmd else ("one-commit\n" if same_head else "head-of-" + Path(cmd[2]).name + "\n")
        if cmd[1] == "tokenize":
            return "[5, 6]\n"
        Path(cmd[cmd.index("--out") + 1]).write_bytes(array("f", [1.0, 2.0] * 4).tobytes())
        return ""

    monkeypatch.setattr(logits, "run", run)
    return calls


def logits_main(tmp: Path, monkeypatch: pytest.MonkeyPatch, **replace: str) -> dict:
    args = ["compare_logits", "--model", "m.gguf", "--base", str(tmp / "base"), "--new", str(tmp / "new"),
            "--out", str(tmp / "r.json"), "--masked", "2"]
    for flag, value in replace.items():
        args[args.index("--" + flag) + 1] = value
    monkeypatch.setattr(sys, "argv", args)
    logits.main()
    return json.loads((tmp / "r.json").read_text())


def test_logits_records_the_binary_of_each_side(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logits_rig(tmp_path, monkeypatch)
    doc = logits_main(tmp_path, monkeypatch)
    for k in ("base", "new"):
        assert doc[k]["binary_sha256"] == hashlib.sha256(f"binary of {k}".encode()).hexdigest()
        assert doc[k]["version"] == f"head-of-{k}"


@pytest.mark.parametrize("what,match", [("path", "same checkout"), ("head", "same commit"), ("bytes", "byte-identical"),
                                        ("missing", "does not exist")])
def test_logits_refuses_a_comparison_of_one_build_with_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              what: str, match: str) -> None:
    calls = logits_rig(tmp_path, monkeypatch, same_head=what == "head")
    extra: dict[str, str] = {}
    if what == "path":
        extra = {"base": str(tmp_path / "new")}
    elif what == "bytes":
        (tmp_path / "base/build/mojo-dllm").write_bytes(b"binary of new")
    elif what == "missing":
        (tmp_path / "base/build/mojo-dllm").unlink()
    with pytest.raises(SystemExit, match=match):
        logits_main(tmp_path, monkeypatch, **extra)
    assert [c for c in calls if c[0] != "git"] == [] and not (tmp_path / "r.json").exists()


# gpu_profile --------------------------------------------------------------------------------------------

def test_gpu_profile_leaves_block_length_out_for_dream_only() -> None:
    llada = profile.mojo_command(Path("/r"), "~/m.gguf", "p", 128, 4, 32, "llada")
    dream = profile.mojo_command(Path("/r"), "~/m.gguf", "p", 128, 4, 32, "dream")
    assert llada[llada.index("--block-length") + 1] == "32"
    assert "--block-length" not in dream and "--steps" in dream and dream[-2:] == ["--json", "--no-early-stop"]
    assert dream[:4] == ["/r/build/mojo-dllm", "run", "--model", os.path.expanduser("~/m.gguf")]
    assert [x for x in llada if x != "--block-length" and x != "32"] == dream


# run_bench and eval_gsm8k record the binary too ---------------------------------------------------------

rb = load("run_bench_ab", ROOT / "bench/run_bench.py")


def test_run_bench_versions_carry_the_hash_of_the_binary_that_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / "build/mojo-dllm").write_bytes(b"built")
    monkeypatch.setattr(rb, "ROOT", tmp_path)
    monkeypatch.setattr(rb, "sh", lambda cmd: "" if "status" in cmd else "abc1234")
    monkeypatch.setattr(rb, "mojo_version", lambda: "Mojo 1.1")
    c = {"llama_cpp": "/l", "diffuse_cpp": "/d"}
    v = rb.collect_versions(c, ["mojo-dllm", "llama.cpp"])
    assert v["mojo-dllm-binary-sha256"] == hashlib.sha256(b"built").hexdigest() and v["mojo-dllm"] == "abc1234"
    assert "mojo-dllm-binary-sha256" not in rb.collect_versions(c, ["llama.cpp", "diffuse-cpp"])
    (tmp_path / "build/mojo-dllm").unlink()
    assert rb.collect_versions(c, ["mojo-dllm-gpu"])["mojo-dllm-binary-sha256"] is None


def test_eval_gsm8k_passes_the_binary_hash_to_the_arm_record() -> None:
    text = (ROOT / "bench/eval_gsm8k.py").read_text()
    assert '"mojo-dllm-binary-sha256": ab_evidence.binary_sha256_or_none(ROOT)' in text
