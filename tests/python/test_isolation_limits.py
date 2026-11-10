"""A value exactly at an isolation limit is clean and one unit over is not (#250).

The other isolation tests sit far from the limits, so turning `>` into `>=` (or `<` into `<=`) in bench/isolation.py
turned nothing red. Each case here sets the machine to the limit and then one unit past it, on the fake machine of
conftest.py, and the mutation table in the pull request says which comparison each pair pins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import isolation  # noqa: E402
from conftest import FakeMachine  # noqa: E402


@pytest.fixture(autouse=True)
def by_hand(fake_machine: FakeMachine) -> None:
    fake_machine.auto = False


def pre(fm: FakeMachine, device: str = "cpu", need: float = 4.0, th: isolation.Thresholds | None = None):
    fm.flush()
    return isolation.preflight(device, need, machine=fm, sample_s=2.0, thresholds=th)


def failed(rep) -> list[str]:
    return [c.name for c in rep.failed]


def window(fm: FakeMachine, steps: int = 10, *, each=None, th: isolation.Thresholds | None = None, device: str = "cpu") -> dict:
    w = isolation.Window(pid=4242, device=device, machine=fm, thread=False, thresholds=th)
    with w:
        for i in range(steps):
            if each:
                each(i)
            fm.advance(0.5)
            w.tick()
    return w.result


# preflight ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("busy_per_s,fails", [(40, False), (41, True)])
def test_cpu_busy_exactly_at_ten_percent_passes(fake_machine: FakeMachine, busy_per_s: int, fails: bool) -> None:
    fake_machine.extra_busy = busy_per_s                  # of 400 ticks a second on 4 CPUs: 10.0% and 10.25%
    assert (failed(pre(fake_machine)) == ["cpu-busy"]) is fails


@pytest.mark.parametrize("avg10,fails", [(5.0, False), (5.01, True)])
def test_psi_avg10_exactly_at_the_limit_passes(fake_machine: FakeMachine, avg10: float, fails: bool) -> None:
    fake_machine.psi_avg10 = avg10
    assert (failed(pre(fake_machine)) == ["psi-cpu"]) is fails


@pytest.mark.parametrize("limit,fails", [(2, False), (1, True)])
def test_swap_pages_exactly_at_the_limit_pass(fake_machine: FakeMachine, limit: int, fails: bool) -> None:
    fake_machine.swap_rate = 1                            # one page in each of the two one-second halves
    rep = pre(fake_machine, th=isolation.Thresholds(swap_pages_max=limit))
    assert "2 pages swapped" in next(c.detail for c in rep.checks if c.name == "swap-movement")
    assert (failed(rep) == ["swap-movement"]) is fails


@pytest.mark.parametrize("available_kb,fails", [(4 * 1048576, False), (4 * 1048576 - 1, True)])
def test_available_memory_exactly_at_the_need_passes(fake_machine: FakeMachine, available_kb: int, fails: bool) -> None:
    fake_machine.mem_available_kb = available_kb
    assert (failed(pre(fake_machine, need=4.0)) == ["memory"]) is fails


@pytest.mark.parametrize("limit,fails", [(6, False), (5, True)])
def test_thermal_steps_exactly_at_the_limit_pass(fake_machine: FakeMachine, limit: int, fails: bool) -> None:
    fake_machine.sleep = lambda s: (fake_machine.advance(s), fake_machine.thermal.update(core=9))[0]     # 3 to 9: six steps
    assert (failed(pre(fake_machine, th=isolation.Thresholds(thermal_events_max=limit))) == ["thermal"]) is fails


@pytest.mark.parametrize("mem,fails", [(300, False), (301, True)])
def test_gpu_memory_exactly_at_the_limit_passes(fake_machine: FakeMachine, mem: int, fails: bool) -> None:
    fake_machine.gpu["mem"] = mem
    assert (failed(pre(fake_machine, "gpu")) == ["gpu-memory"]) is fails


@pytest.mark.parametrize("util,fails", [(5, False), (6, True)])
def test_gpu_utilization_exactly_at_the_limit_passes(fake_machine: FakeMachine, util: int, fails: bool) -> None:
    fake_machine.gpu["util"] = util
    assert (failed(pre(fake_machine, "gpu")) == ["gpu-utilization"]) is fails


def test_the_limits_these_cases_sit_on_are_the_documented_defaults() -> None:
    th = isolation.Thresholds()
    assert (th.cpu_busy_max, th.psi_avg10_max, th.swap_pages_max, th.thermal_events_max) == (0.10, 5.0, 0, 0)
    assert (th.gpu_mem_used_max_mib, th.gpu_util_max) == (300, 5.0)
    assert (th.other_cpu_mean_max, th.other_cpu_sample_max) == (0.05, 0.25)


# window ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("extra,clean", [(20, True), (22, False)])
def test_other_cpu_mean_exactly_at_five_percent_is_clean(fake_machine: FakeMachine, extra: int, clean: bool) -> None:
    fake_machine.extra_busy = extra                       # 10 of 200 ticks a sample: 5.0%, then 5.5%
    res = window(fake_machine, th=isolation.Thresholds(other_cpu_sample_max=1.0))
    assert res["clean"] is clean
    assert res["other_cpu_share_mean"] == pytest.approx(0.05 if clean else 0.055)


@pytest.mark.parametrize("spike,clean", [(100, True), (102, False)])
def test_other_cpu_peak_exactly_at_twenty_five_percent_is_clean(fake_machine: FakeMachine, spike: int, clean: bool) -> None:
    def each(i: int) -> None:
        fake_machine.extra_busy = spike if i == 4 else 0

    res = window(fake_machine, each=each)
    assert res["clean"] is clean
    assert res["other_cpu_share_max"] == pytest.approx(0.25 if clean else 0.255)
    assert res["other_cpu_share_mean"] < 0.05


@pytest.mark.parametrize("limit,clean", [(2, True), (1, False)])
def test_window_swap_pages_exactly_at_the_limit_are_clean(fake_machine: FakeMachine, limit: int, clean: bool) -> None:
    fake_machine.swap_rate = 2                            # one page per half second
    res = window(fake_machine, steps=2, th=isolation.Thresholds(swap_pages_max=limit))
    assert res["swap_pages"] == 2 and res["clean"] is clean


@pytest.mark.parametrize("limit,clean", [(3, True), (2, False)])
def test_window_thermal_steps_exactly_at_the_limit_are_clean(fake_machine: FakeMachine, limit: int, clean: bool) -> None:
    def each(i: int) -> None:
        fake_machine.thermal.update(core=fake_machine.thermal["core"] + 1)

    res = window(fake_machine, steps=3, each=each, th=isolation.Thresholds(thermal_events_max=limit))
    assert res["thermal_events"] == 3 and res["clean"] is clean


@pytest.mark.parametrize("apps,clean", [([(31337, 300)], True), ([(31337, 300), (31338, 100)], False)])
def test_foreign_gpu_pids_exactly_at_the_limit_are_clean(fake_machine: FakeMachine, apps: list, clean: bool) -> None:
    fake_machine.gpu_apps = apps
    res = window(fake_machine, device="gpu", th=isolation.Thresholds(foreign_gpu_pids_max=1))
    assert res["clean"] is clean and len(res["gpu_foreign_pids"]) == len(apps)
