"""scripts/build.sh says which GPU architecture it chose and why (#248).

The compiler is a stub that records its arguments, and PATH holds only the tools the script needs, so no
real nvidia-smi and no real build runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build.sh"

FAKE_MOJO = "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$FAKE_ARGS\"\n"


def run_build(tmp: Path, nvidia: str | None, **env: str) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run build.sh with a stub compiler and, when `nvidia` is a script body, a stub nvidia-smi."""
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "dirname", "mkdir"):
        (bin_dir / tool).symlink_to(shutil.which(tool) or tool)
    mojo = tmp / "fake-mojo"
    mojo.write_text(FAKE_MOJO)
    mojo.chmod(0o755)
    if nvidia is not None:
        smi = bin_dir / "nvidia-smi"
        smi.write_text("#!/usr/bin/env bash\n" + nvidia)
        smi.chmod(0o755)
    args = tmp / "args"
    full = {"PATH": str(bin_dir), "MOJO": str(mojo), "FAKE_ARGS": str(args), **env}
    p = subprocess.run(["bash", str(SCRIPT), "src/main.mojo", str(tmp / "out/bin")], env=full, capture_output=True,
                       text=True, timeout=60)
    return p, args.read_text() if args.exists() else ""


def test_the_architecture_of_the_local_gpu_is_named_with_its_source(tmp_path: Path) -> None:
    p, args = run_build(tmp_path, "echo 8.6\n")
    assert p.returncode == 0, p.stderr
    assert "--target-accelerator sm_86" in args
    assert "build.sh: GPU architecture sm_86 (nvidia-smi reports compute capability 8.6)" in p.stderr
    assert "WARNING" not in p.stderr


def test_the_first_of_several_cards_is_used(tmp_path: Path) -> None:
    p, args = run_build(tmp_path, "printf '8.9\\n7.5\\n'\n")
    assert "--target-accelerator sm_89" in args and "compute capability 8.9" in p.stderr


def test_a_failing_nvidia_smi_prints_the_fallback_and_why(tmp_path: Path) -> None:
    p, args = run_build(tmp_path, 'echo "Failed to initialize NVML: Driver/library version mismatch" >&2\nexit 18\n')
    assert p.returncode == 0
    assert "--target-accelerator sm_80" in args
    assert "GPU architecture sm_80" in p.stderr and "WARNING" in p.stderr
    assert "installed but failed" in p.stderr and "version mismatch" in p.stderr


@pytest.mark.parametrize("answer", ["", "N/A", "garbage 8.6 x"])
def test_an_answer_that_is_no_compute_capability_prints_the_fallback(tmp_path: Path, answer: str) -> None:
    p, args = run_build(tmp_path, f"echo '{answer}'\n")
    assert "--target-accelerator sm_80" in args
    assert "WARNING: nvidia-smi printed no compute capability" in p.stderr


def test_no_nvidia_smi_is_a_plain_cpu_host_and_not_a_warning(tmp_path: Path) -> None:
    p, args = run_build(tmp_path, None)
    assert p.returncode == 0 and "--target-accelerator sm_80" in args
    assert "GPU architecture sm_80 (no nvidia-smi on PATH, so no GPU to match)" in p.stderr
    assert "WARNING" not in p.stderr


def test_the_override_wins_and_is_named(tmp_path: Path) -> None:
    called = tmp_path / "smi-called"
    p, args = run_build(tmp_path, f"touch {called}\necho 8.6\n", MOJO_DLLM_GPU_ARCH="sm_90")
    assert "--target-accelerator sm_90" in args and "(MOJO_DLLM_GPU_ARCH)" in p.stderr
    assert not called.exists()


def test_the_message_goes_to_stderr_only(tmp_path: Path) -> None:
    p, _ = run_build(tmp_path, "echo 8.6\n")
    assert p.stdout == "" and os.path.basename(str(SCRIPT)) == "build.sh"
