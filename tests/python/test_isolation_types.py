"""Typed status, validated thresholds, one power reading and a window record that matches its type (#243)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import isolation  # noqa: E402
import power  # noqa: E402
from conftest import FakeMachine  # noqa: E402

# the status


def test_a_check_status_must_be_one_of_the_three() -> None:
    for ok in ("PASS", "WARN", "FAIL"):
        assert isolation.Check("x", ok, "d").status == ok  # type: ignore[arg-type]
    for bad in ("fail", "Fail", "FAILED", "", None, 1):
        with pytest.raises(ValueError) as e:
            isolation.Check("cpu-busy", bad, "d")  # type: ignore[arg-type]
        assert "cpu-busy" in str(e.value) and "PASS" in str(e.value)


def test_a_typo_in_a_status_cannot_pass_as_not_failed() -> None:
    """`Report.failed` compares with FAIL; a misspelling used to fall through it. Now it cannot be built."""
    with pytest.raises(ValueError):
        isolation.Report("cpu", [isolation.Check("a", "FAIL ", "d")], {}, set(), [], 1.0)  # type: ignore[arg-type]
    rep = isolation.Report("cpu", [isolation.Check("a", "FAIL", "d")], {}, set(), [], 1.0)
    assert rep.failed and not rep.ok


# thresholds


def test_the_defaults_are_valid() -> None:
    isolation.Thresholds()


@pytest.mark.parametrize("name", [f.name for f in __import__("dataclasses").fields(isolation.Thresholds)])
@pytest.mark.parametrize("bad", [-0.1, -1, math.nan, math.inf, "0.1", None, True])
def test_a_negative_nan_or_non_number_threshold_is_refused(name: str, bad: object) -> None:
    with pytest.raises(ValueError) as e:
        isolation.Thresholds(**{name: bad})  # type: ignore[arg-type]
    assert name in str(e.value)


@pytest.mark.parametrize("name,bad", [
    ("cpu_busy_max", 1.5), ("other_cpu_mean_max", 2.0), ("other_cpu_sample_max", 1.01),
    ("psi_avg10_max", 100.5), ("gpu_util_max", 101),
    ("swap_pages_max", 1.5), ("thermal_events_max", 0.5), ("foreign_gpu_pids_max", 2.0),
])
def test_a_threshold_outside_its_unit_is_refused(name: str, bad: float) -> None:
    with pytest.raises(ValueError) as e:
        isolation.Thresholds(**{name: bad})  # type: ignore[arg-type]
    assert name in str(e.value)


def test_boundary_values_are_accepted() -> None:
    isolation.Thresholds(cpu_busy_max=0.0, other_cpu_mean_max=1.0, psi_avg10_max=100, gpu_util_max=0,
                         swap_pages_max=0, thermal_events_max=3, foreign_gpu_pids_max=1, gpu_mem_used_max_mib=0)


# the window record


def test_the_window_result_has_exactly_the_keys_of_its_type(fake_machine: FakeMachine) -> None:
    from test_bench_isolation import window

    for dirty in (False, True):
        fake_machine.swap_rate = 50.0 if dirty else 0.0
        res = window(fake_machine)
        assert set(res) == set(isolation.WindowResult.__annotations__)
        assert res["clean"] is (not dirty)


class Scripted:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def __call__(self, **kw: object) -> "Scripted":
        self.result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return self

    def __enter__(self) -> "Scripted":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.mark.parametrize("clean", [True, False])
def test_the_isolation_record_has_the_keys_of_its_type_and_noisy_is_not_clean(
    fake_machine: FakeMachine, clean: bool
) -> None:
    from test_bench_isolation import win

    _, info = isolation.run_isolated(lambda: 1, device="cpu", machine=fake_machine, window=Scripted([win(clean, "r")]),
                                     retries=0, allow_battery=True, log=lambda m: None, quiet_sample_s=0.1)
    iso = info["isolation"]
    expected = set(isolation.IsolationRecord.__annotations__)
    # `win` is a three-key stand-in for a window; the record adds its own keys on top of whatever the window gave
    assert {"retries", "noisy", "cpu_busy_before", "waited_s", "gate_ok", "session", "allow_noisy",
            "preflight_failed_allowed"} <= set(iso) and set(iso) <= expected
    assert iso["noisy"] is (not iso["clean"])


# power: one reading


@pytest.mark.parametrize("on", [True, False])
@pytest.mark.parametrize("state", [True, False, None])
def test_the_power_pair_is_never_contradictory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: bool, state: bool | None
) -> None:
    monkeypatch.setattr(power, "on_battery", lambda root: state)
    monkeypatch.setattr(power, "snapshot", lambda root: {})
    info = power._info(tmp_path, 0.0, 0, on)
    assert info["on_battery"] is (info["power_source"] == "battery")
    assert info["power_source"] in ("battery", "mains", "none")


def test_a_battery_event_between_the_run_and_the_second_read_gives_battery_both_ways(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(power, "on_battery", lambda root: True)       # the second read sees the charger gone
    monkeypatch.setattr(power, "snapshot", lambda root: {})
    info = power._info(tmp_path, 0.0, 0, False)                       # the run itself saw mains
    assert info["power_source"] == "battery" and info["on_battery"] is True


def test_the_power_source_table() -> None:
    assert power.power_source(True, None) == "battery" and power.power_source(True, False) == "battery"
    assert power.power_source(False, True) == "battery"
    assert power.power_source(False, False) == "mains" and power.power_source(False, None) == "none"


def test_a_machine_with_no_battery_says_none_and_not_on_battery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(power, "on_battery", lambda root: None)
    monkeypatch.setattr(power, "snapshot", lambda root: {})
    info = power._info(tmp_path, 0.0, 0, False)
    assert info["power_source"] == "none" and info["on_battery"] is False
