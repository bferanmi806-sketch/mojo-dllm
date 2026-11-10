"""A GPU engine that loaded its weights in this process is not contention for its own preflight (#240).

The fake machine's own pid is 4242 (conftest.FakeMachine). Nothing here touches the GPU: nvidia-smi is the fake.
"""

from __future__ import annotations

import pytest

import isolation
from conftest import FakeMachine


def pre(fm: FakeMachine) -> isolation.Report:
    fm.flush()
    return isolation.preflight("gpu", 4.0, machine=fm, sample_s=2.0)


def gate(fm: FakeMachine) -> dict:
    return isolation.quiet_gate("gpu", machine=fm, sample_s=1.0, max_wait_s=0)


def names(rep: isolation.Report, status: str) -> list[str]:
    return [c.name for c in rep.checks if c.status == status]


def detail(rep: isolation.Report, name: str) -> str:
    return next(c.detail for c in rep.checks if c.name == name)


def test_a_gpu_process_that_is_this_pid_passes_the_preflight_and_the_gate(fake_machine: FakeMachine) -> None:
    fake_machine.gpu_apps = [(4242, 5400)]
    fake_machine.gpu["mem"] = 5400
    rep = pre(fake_machine)
    assert names(rep, "FAIL") == [], [(c.name, c.detail) for c in rep.failed]
    assert "this benchmark" in detail(rep, "gpu-processes") and "excluded" in detail(rep, "gpu-memory")
    g = gate(fake_machine)
    assert g["ok"] is True and g["reasons"] == []


def test_a_child_of_this_pid_is_ours_too(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(5000, 4242, "python3", 0)
    fake_machine.gpu_apps = [(5000, 5400)]
    fake_machine.gpu["mem"] = 5400
    assert names(pre(fake_machine), "FAIL") == []
    assert gate(fake_machine)["ok"] is True


def test_a_foreign_pid_still_fails_both_and_only_it_is_named(fake_machine: FakeMachine) -> None:
    fake_machine.gpu_apps = [(4242, 5400), (31337, 50)]
    fake_machine.gpu["mem"] = 5450
    rep = pre(fake_machine)
    assert names(rep, "FAIL") == ["gpu-processes"]
    assert "31337" in detail(rep, "gpu-processes") and "4242" not in detail(rep, "gpu-processes")
    g = gate(fake_machine)
    assert g["ok"] is False and any("31337" in r and "4242" not in r for r in g["reasons"])


def test_memory_held_by_someone_else_still_counts(fake_machine: FakeMachine) -> None:
    fake_machine.gpu_apps = [(4242, 5400)]
    fake_machine.gpu["mem"] = 5400 + 450          # 450 MiB beyond ours, over the 300 MiB limit
    assert names(pre(fake_machine), "FAIL") == ["gpu-memory"]


def test_memory_with_no_process_of_ours_is_compared_in_full(fake_machine: FakeMachine) -> None:
    fake_machine.gpu["mem"] = 450
    assert names(pre(fake_machine), "FAIL") == ["gpu-memory"]


def test_an_unreadable_process_table_still_excludes_this_pid(
    fake_machine: FakeMachine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolation, "process_table", lambda m: None)
    fake_machine.gpu_apps = [(4242, 5400)]
    fake_machine.gpu["mem"] = 5400
    assert names(pre(fake_machine), "FAIL") == []
    assert gate(fake_machine)["ok"] is True


def test_the_split_by_hand(fake_machine: FakeMachine) -> None:
    table = {4242: (1, 0, "python3", 0), 5000: (4242, 0, "child", 0), 31337: (1, 0, "other", 0)}
    foreign, held = isolation.split_gpu_apps(fake_machine, [(4242, 100.0), (5000, 20.0), (31337, 7.0)], table)
    assert foreign == [(31337, 7.0)] and held == 120.0
