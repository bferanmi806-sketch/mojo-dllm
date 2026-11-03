"""Timed runs happen on mains power only.

A laptop caps CPU and GPU power when unplugged, so a number measured on battery
describes the battery. `run_on_mains` waits while the machine is on battery,
then runs the timed function with a watcher that samples the power state; if
the charger went at any point during the run, the result is thrown away and
the run repeated once power is back, up to `max_retries` times.

The check fails closed. A supply file that cannot be read raises
`PowerReadError` instead of reading as mains, a battery with the adapter
offline counts as battery whatever its status says (`Not charging` and
`Unknown` are real values), and every run records where its power came from:
`power_source` is `mains`, `battery` or `none` (a machine that exposes no
battery, such as a desktop or a VM), next to a snapshot of what was read.

MOJO_DLLM_POWER_SUPPLY points the check at another directory laid out like
/sys/class/power_supply, so a test can unplug a fake charger mid-run. A run
that used another root records it as `power_root`.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import threading
import time
from typing import Callable, TypeVar

DEFAULT_ROOT = pathlib.Path("/sys/class/power_supply")
POWER_SUPPLY = pathlib.Path(os.environ.get("MOJO_DLLM_POWER_SUPPLY", str(DEFAULT_ROOT)))
# Supply types that are not an external source: a battery, and a UPS, which is a
# battery in a box. Every other type (Mains, USB, USB_PD, Wireless, ...) is a charger.
NOT_EXTERNAL = ("Battery", "UPS")
T = TypeVar("T")


class PowerError(RuntimeError):
    """The power state stops the run."""


class PowerReadError(PowerError):
    """A power supply file exists but cannot be read, so the state is unknown."""


def _read(d: pathlib.Path, name: str) -> str:
    try:
        return (d / name).read_text().strip()
    except FileNotFoundError:
        return ""
    except OSError as e:
        raise PowerReadError(f"cannot read {d.name}/{name} ({e.strerror or e}): the power state is unknown") from e


def _supplies(root: pathlib.Path) -> list[dict[str, str]]:
    if not root.is_dir():
        return []
    try:
        dirs = sorted(d for d in root.iterdir() if d.is_dir())
    except OSError as e:
        raise PowerReadError(f"cannot list {root} ({e.strerror or e}): the power state is unknown") from e
    return [{"name": d.name, "type": _read(d, "type"), "online": _read(d, "online"), "status": _read(d, "status")}
            for d in dirs]


def on_battery(root: pathlib.Path = POWER_SUPPLY) -> bool | None:
    """True when no external supply is online and the machine has a battery,
    whatever the battery's status; False when a charger is online; None when the
    machine exposes no battery and no charger is online (a desktop, a VM)."""
    supplies = _supplies(root)
    if any(s["type"] not in NOT_EXTERNAL and s["type"] and s["online"] == "1" for s in supplies):
        return False
    if any(s["type"] == "Battery" for s in supplies):
        return True
    return None


def power_source(root: pathlib.Path = POWER_SUPPLY) -> str:
    state = on_battery(root)
    return "none" if state is None else ("battery" if state else "mains")


def snapshot(root: pathlib.Path = POWER_SUPPLY) -> dict[str, str]:
    """What the check read, for the result file: {"ADP1": "Mains online=1", "BAT0": "Battery Full"}."""
    out: dict[str, str] = {}
    for s in _supplies(root):
        if s["type"] == "Battery":
            out[s["name"]] = f"Battery {s['status'] or '?'}"
        elif s["type"]:
            out[s["name"]] = f"{s['type']} online={s['online'] or '?'}"
    return out


def wait_for_mains(root: pathlib.Path = POWER_SUPPLY, *, poll_s: float = 5.0, allow_battery: bool = False,
                   sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print,
                   on_pause: Callable[[], None] | None = None,
                   on_resume: Callable[[], None] | None = None) -> float:
    """Block while on battery and return the seconds waited (0 on mains)."""
    if allow_battery or not on_battery(root):
        return 0
    log(f"on battery since {dt.datetime.now():%H:%M:%S}: paused, plug in the charger to continue")
    if on_pause:
        on_pause()
    waited = 0.0
    while on_battery(root):
        sleep(poll_s)
        waited += poll_s
    log(f"charger back after {waited:.0f} s: resuming")
    if on_resume:
        on_resume()
    return waited


class _Watch:
    """Samples the power state every `sample_s` seconds until stopped. A failed
    read in the sampler ends the run with that error, not with a clean record."""

    def __init__(self, root: pathlib.Path, sample_s: float) -> None:
        self.root, self.sample_s = root, sample_s
        self.saw_battery = False
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if on_battery(self.root):
                    self.saw_battery = True
                self._stop.wait(self.sample_s)
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread in __exit__
            self.error = e

    def __enter__(self) -> "_Watch":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        if exc_type is not None:
            return
        if self.error is not None:
            raise self.error
        if on_battery(self.root):   # the last moment, which a sample may have missed
            self.saw_battery = True


def _info(root: pathlib.Path, waited: float, retries: int, on: bool) -> dict:
    state = on_battery(root)
    info = {"power_waited_s": waited, "power_retries": retries, "on_battery": on,
            "power_source": "battery" if on else ("none" if state is None else "mains"),
            "power_supplies": snapshot(root)}
    if root != DEFAULT_ROOT:
        info["power_root"] = str(root)
    return info


def run_on_mains(fn: Callable[[], T], *, root: pathlib.Path = POWER_SUPPLY, allow_battery: bool = False,
                 poll_s: float = 5.0, sample_s: float = 0.1, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print, on_pause: Callable[[], None] | None = None,
                 on_resume: Callable[[], None] | None = None, max_retries: int = 5) -> tuple[T, dict]:
    """Run `fn` on mains power, repeating it when the charger went mid-run.

    Returns fn's result and what it took: seconds paused, runs thrown away, and
    whether the kept run was on battery (only possible with allow_battery; the
    state is read before the run and sampled during it), with `power_source`
    and a snapshot of the supplies. Raises PowerError after `max_retries`
    repeats, and PowerReadError when the state cannot be read.
    """
    if allow_battery:
        before = on_battery(root)
        with _Watch(root, sample_s) as watch:
            result = fn()
        return result, _info(root, 0, 0, bool(before) or watch.saw_battery)
    waited, retries = 0.0, 0
    while True:
        waited += wait_for_mains(root, poll_s=poll_s, sleep=sleep, log=log, on_pause=on_pause, on_resume=on_resume)
        with _Watch(root, sample_s) as watch:
            result = fn()
        if not watch.saw_battery:
            return result, _info(root, waited, retries, False)
        retries += 1
        if retries > max_retries:
            raise PowerError(f"gave up after {max_retries} runs thrown away because the charger went; "
                             "check the cable, or measure with --allow-battery")
        log("the charger went during this run: its result is discarded, and it will repeat on mains")
