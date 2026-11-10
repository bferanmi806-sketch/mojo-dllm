from __future__ import annotations

import io
from pathlib import Path

from launcher.doctor import preflight, run_doctor
from launcher.ui import Style

from conftest import GPU_TEXT, MEMINFO, make_probes

UV_FIX = "curl -LsSf https://astral.sh/uv/install.sh | sh"


def doctor(**overrides: object) -> tuple[int, list[str]]:
    out = io.StringIO()
    code = run_doctor(make_probes(**overrides), out, Style(False))
    return code, out.getvalue().splitlines()


def marks(lines: list[str]) -> list[str]:
    return [line[0] for line in lines if line[0] in "✓!✗"]


def test_all_green() -> None:
    code, lines = doctor()
    assert code == 0
    assert marks(lines) == ["✓"] * 8
    labels = [line.split()[1] for line in lines]
    assert labels == ["platform", "glibc", "cpu", "memory", "toolchain", "binary", "gpu", "model"]
    gpu = next(line for line in lines if " gpu " in line)
    assert "NVIDIA GeForce RTX 3050 6GB Laptop GPU" in gpu
    assert "compute 8.6" in gpu and "driver 580.65.06" in gpu and "5120 MiB free" in gpu


def test_missing_uv_fails_with_the_install_command() -> None:
    code, lines = doctor(uv_path=None, mojo_version_output=None)
    assert code == 1
    i = next(i for i, line in enumerate(lines) if line.startswith("✗") and "toolchain" in line)
    assert lines[i + 1] == f"  → {UV_FIX}"


def test_toolchain_not_installed_points_at_setup() -> None:
    code, lines = doctor(mojo_version_output=None)
    assert code == 1
    assert any(line == "  → ./mojo-dllm setup" for line in lines)


def test_toolchain_version_mismatch_names_both_versions() -> None:
    code, lines = doctor(mojo_version_output="Mojo 1.0.0 (x)\n")
    line = next(line for line in lines if "toolchain" in line)
    assert line.startswith("✗") and "1.0.0" in line and "1.1.0" in line and code == 1


def test_old_glibc_fails() -> None:
    code, lines = doctor(glibc="2.31")
    line = next(line for line in lines if "glibc" in line)
    assert code == 1 and line.startswith("✗") and "2.31" in line and "2.34" in line


def test_no_glibc_fails() -> None:
    code, lines = doctor(glibc="")
    assert code == 1 and next(line for line in lines if "glibc" in line).startswith("✗")


def test_wrong_platform_fails() -> None:
    assert doctor(system="Darwin")[0] == 1
    assert doctor(machine="aarch64")[0] == 1


def test_no_avx2_fails() -> None:
    code, lines = doctor(cpuinfo="flags : fpu sse4_2 avx\n")
    assert code == 1 and next(line for line in lines if " cpu " in line).startswith("✗")


def test_low_memory_only_warns() -> None:
    code, lines = doctor(meminfo=MEMINFO.replace("12000000", "4000000"))
    assert code == 0 and next(line for line in lines if "memory" in line).startswith("!")


def test_no_gpu_is_a_warning_that_says_cpu_will_be_used() -> None:
    code, lines = doctor(gpu_text=None, driver_text=None)
    assert code == 0
    i = next(i for i, line in enumerate(lines) if line.startswith("!") and " gpu " in line)
    assert lines[i + 1] == "  → device: cpu will be used"


def test_old_gpu_is_a_warning() -> None:
    code, lines = doctor(gpu_text="GTX 1650, 7.5, 3900\n")
    line = next(line for line in lines if " gpu " in line)
    assert code == 0 and line.startswith("!") and "7.5" in line


def test_unreadable_nvidia_smi_is_reported_not_hidden() -> None:
    code, lines = doctor(gpu_text="garbage\n")
    line = next(line for line in lines if " gpu " in line)
    assert line.startswith("!") and "nvidia-smi" in line


def test_missing_binary_fails_and_stale_binary_warns() -> None:
    code, lines = doctor(binary_mtime=None)
    assert code == 1 and next(line for line in lines if "binary" in line).startswith("✗")
    code, lines = doctor(binary_mtime=120.0)
    assert code == 0 and next(line for line in lines if "binary" in line).startswith("!")
    assert "  → ./mojo-dllm setup" in lines


def test_a_stale_stamp_warns_that_the_binary_needs_a_rebuild() -> None:
    code, lines = doctor(stamp_problem="built for sm_80, this machine now builds for sm_86")
    line = next(line for line in lines if " binary " in line)
    assert code == 0 and line.startswith("!") and "sm_80" in line
    assert lines[lines.index(line) + 1] == "  → ./mojo-dllm setup"


def test_a_tree_with_no_sources_fails_the_binary_check() -> None:
    code, lines = doctor(source_mtimes=())
    line = next(line for line in lines if " binary " in line)
    assert code == 1 and line.startswith("✗") and "no sources" in line


def test_missing_model_warns_with_the_pull_command() -> None:
    code, lines = doctor(cached_model=None)
    assert code == 0
    assert "  → ./mojo-dllm pull" in lines


def test_an_unverified_cached_model_warns_and_names_the_pull_command() -> None:
    code, lines = doctor(verified=False)
    line = next(line for line in lines if " model " in line)
    assert code == 0 and line.startswith("!") and "/cache/llada.gguf" in line and "not verified" in line
    assert lines[lines.index(line) + 1] == "  → ./mojo-dllm pull"


def test_a_verified_cached_model_stays_green() -> None:
    _, lines = doctor(verified=True)
    line = next(line for line in lines if " model " in line)
    assert line.startswith("✓") and "not verified" not in line


def test_every_failure_is_listed_before_exit() -> None:
    code, lines = doctor(glibc="2.31", cpuinfo="", uv_path=None, mojo_version_output=None)
    failed = [line.split()[1] for line in lines if line.startswith("✗")]
    assert code == 1 and failed == ["glibc", "cpu", "toolchain"]


def test_colour_only_when_enabled() -> None:
    out = io.StringIO()
    run_doctor(make_probes(), out, Style(True))
    assert "\x1b[" in out.getvalue()
    assert "\x1b[" not in "\n".join(doctor()[1])


def test_preflight_is_the_hard_subset() -> None:
    names = [c.label for c in preflight(make_probes())]
    assert names == ["platform", "glibc", "cpu"]
    assert all(c.status == "ok" for c in preflight(make_probes()))
    assert [c.status for c in preflight(make_probes(glibc="2.31"))] == ["ok", "fail", "ok"]
