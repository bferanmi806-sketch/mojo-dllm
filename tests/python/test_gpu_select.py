"""`--device auto` must name the GPU the binary will use: the binary opens CUDA's default device."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from launcher import runcmd
from launcher.gpu import Gpu, choose_device, parse_nvidia_smi

OLD_UUID = "GPU-11111111-2222-3333-4444-555555555555"
NEW_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
# nvidia-smi lists cards in PCI order: the old card is index 0 and CUDA's default device may differ.
TWO_GPUS = (
    f"NVIDIA GeForce GTX 1080, 6.1, 7900, 0, {OLD_UUID}\n"
    f"NVIDIA GeForce RTX 3090, 8.6, 24000, 1, {NEW_UUID}\n"
)
ONE_OLD_GPU = f"NVIDIA T4, 7.5, 15000, 0, {OLD_UUID}\n"


def test_the_listing_carries_the_index_and_uuid() -> None:
    assert parse_nvidia_smi(TWO_GPUS) == [
        Gpu("NVIDIA GeForce GTX 1080", 6.1, 7900, 0, OLD_UUID),
        Gpu("NVIDIA GeForce RTX 3090", 8.6, 24000, 1, NEW_UUID),
    ]


def test_a_name_with_a_comma_still_parses_with_the_new_columns() -> None:
    assert parse_nvidia_smi(f"Tesla, Inc. T4, 7.5, 15000, 0, {OLD_UUID}\n") == [
        Gpu("Tesla, Inc. T4", 7.5, 15000, 0, OLD_UUID)
    ]


def test_auto_pins_the_card_it_named() -> None:
    d = choose_device("auto", parse_nvidia_smi(TWO_GPUS))
    assert d.kind == "gpu" and "RTX 3090" in d.line
    assert d.env == {"CUDA_VISIBLE_DEVICES": NEW_UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_auto_on_a_listing_without_uuids_pins_nothing() -> None:
    d = choose_device("auto", parse_nvidia_smi("NVIDIA A100, 8.0, 40000\n"))
    assert d.kind == "gpu" and d.env == {}


def test_cpu_pins_nothing() -> None:
    assert choose_device("auto", [Gpu("T4", 7.5, 1, 0, OLD_UUID)]).env == {}
    assert choose_device("cpu", parse_nvidia_smi(TWO_GPUS)).env == {}


def test_an_explicit_gpu_with_a_qualifying_card_pins_it_without_a_warning() -> None:
    d = choose_device("gpu", parse_nvidia_smi(TWO_GPUS))
    assert d.kind == "gpu" and d.line.startswith("device: gpu (requested") and "RTX 3090" in d.line
    assert d.env["CUDA_VISIBLE_DEVICES"] == NEW_UUID and d.warning is None


def test_an_explicit_gpu_on_a_card_below_8_0_warns() -> None:
    d = choose_device("gpu", parse_nvidia_smi(ONE_OLD_GPU))
    assert d.kind == "gpu" and d.env == {}
    assert d.warning is not None and "7.5" in d.warning and "8.0" in d.warning


def test_an_explicit_gpu_with_no_card_or_a_failed_nvidia_smi_warns() -> None:
    assert "no NVIDIA GPU found" in str(choose_device("gpu", []).warning)
    assert "exit 9: driver mismatch" in str(choose_device("gpu", [], "exit 9: driver mismatch").warning)


class _Exec:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, smi_text: str | None) -> None:
        self.argv: list[str] | None = None
        self.environ: dict[str, str] = {}
        self.smi_calls = 0
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        monkeypatch.setattr(runcmd, "ensure_built", lambda: False)
        monkeypatch.setattr(runcmd, "ensure_model", lambda *a, **k: tmp_path / "m.gguf")
        monkeypatch.setattr(runcmd, "BINARY", tmp_path / "mojo-dllm")
        monkeypatch.setattr(runcmd.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

        def smi(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.smi_calls += 1
            return subprocess.CompletedProcess(cmd, 0, smi_text or "", "")

        monkeypatch.setattr(runcmd.subprocess, "run", smi)

        def execv(path: str, argv: list[str]) -> None:
            self.argv, self.environ = argv, dict(os.environ)

        monkeypatch.setattr(os, "execv", execv)


def test_the_environment_passed_to_execv_carries_the_uuid_of_the_card_that_qualifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _Exec(monkeypatch, tmp_path, TWO_GPUS)
    runcmd.cmd_run(["hi"])
    assert seen.argv is not None and seen.argv[seen.argv.index("--device") + 1] == "gpu"
    assert seen.environ["CUDA_VISIBLE_DEVICES"] == NEW_UUID and seen.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "RTX 3090" in capsys.readouterr().err


def test_cpu_leaves_the_cuda_environment_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _Exec(monkeypatch, tmp_path, TWO_GPUS)
    runcmd.cmd_run(["hi", "--device", "cpu"])
    assert "CUDA_VISIBLE_DEVICES" not in seen.environ and seen.smi_calls == 0


def test_an_explicit_gpu_on_a_7_5_card_warns_on_stderr_and_still_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _Exec(monkeypatch, tmp_path, ONE_OLD_GPU)
    runcmd.cmd_run(["hi", "--device", "gpu"])
    captured = capsys.readouterr()
    assert "warning:" in captured.err and "7.5" in captured.err and captured.out == ""
    assert seen.argv is not None and "CUDA_VISIBLE_DEVICES" not in seen.environ


def test_an_explicit_gpu_survives_an_nvidia_smi_listing_it_cannot_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _Exec(monkeypatch, tmp_path, "garbage\n")
    runcmd.cmd_run(["hi", "--device", "gpu"])
    assert seen.argv is not None and "warning:" in capsys.readouterr().err
