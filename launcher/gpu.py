"""NVIDIA GPU detection and the cpu/gpu choice."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from launcher.errors import LauncherError

# The int8 tensor-core kernels need Ampere or newer.
MIN_COMPUTE_CAP = 8.0
# index and uuid come last: a GPU name may contain a comma, and the uuid is how the row is told from a
# listing without them.
QUERY = "name,compute_cap,memory.free,index,uuid"
_UUID = re.compile(r"^(GPU|MIG)-[0-9A-Fa-f-]+$")


@dataclass(frozen=True)
class Gpu:
    name: str
    compute_cap: float
    free_mib: int
    index: int | None = None
    uuid: str = ""


@dataclass(frozen=True)
class Device:
    kind: str
    detail: str
    env: Mapping[str, str] = field(default_factory=dict)  # set before the binary starts, to pin the card
    warning: str | None = None

    @property
    def line(self) -> str:
        return f"device: {self.kind} ({self.detail})"


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Parse `nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # rsplit: a GPU name may contain a comma. A row that ends in a uuid has five columns.
        has_ids = bool(_UUID.match(line.rsplit(",", 1)[-1].strip()))
        parts = [p.strip() for p in line.rsplit(",", 4 if has_ids else 2)]
        try:
            if len(parts) != (5 if has_ids else 3) or not parts[0]:
                raise ValueError
            # float() accepts "nan" and "inf" ("inf" would pass the capability check), int() accepts "-5".
            cap = float(parts[1])
            if not math.isfinite(cap) or cap <= 0 or int(parts[2]) < 0:
                raise ValueError
            if has_ids:
                gpus.append(Gpu(parts[0], float(parts[1]), int(parts[2]), int(parts[3]), parts[4]))
            else:
                gpus.append(Gpu(parts[0], float(parts[1]), int(parts[2])))
        except ValueError:
            raise LauncherError(
                f"nvidia-smi printed a line the launcher cannot read: {line!r}\n"
                "Run nvidia-smi to check the driver, or pass --device cpu to skip GPU detection."
            ) from None
    return gpus


def pin(gpu: Gpu) -> dict[str, str]:
    """Make the binary's CUDA default device the card that was chosen.

    nvidia-smi lists cards in PCI order and CUDA defaults to the fastest first, so the card at index 0 of
    the listing is not always the one the binary opens. A uuid names the card in both.
    """
    if not gpu.uuid:
        return {}
    return {"CUDA_VISIBLE_DEVICES": gpu.uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def _qualifying(gpus: Sequence[Gpu]) -> Gpu | None:
    return next((g for g in gpus if g.compute_cap >= MIN_COMPUTE_CAP), None)


def _gpu_detail(gpu: Gpu) -> str:
    return f"{gpu.name}, compute {gpu.compute_cap:.1f}, {gpu.free_mib} MiB free"


def choose_device(requested: str, gpus: Sequence[Gpu], smi_failure: str | None = None) -> Device:
    if requested == "cpu":
        return Device("cpu", "requested")
    if requested == "gpu":
        # An explicit request is honoured, and the user hears when the binary will probably refuse it.
        gpu = _qualifying(gpus)
        if gpu is not None:
            return Device("gpu", f"requested; {_gpu_detail(gpu)}", pin(gpu))
        if smi_failure:
            warning = f"nvidia-smi failed ({smi_failure}), so --device gpu could not be checked"
        elif not gpus:
            warning = "no NVIDIA GPU found, so the binary will probably refuse --device gpu"
        else:
            best = max(g.compute_cap for g in gpus)
            warning = (f"the GPU's compute capability {best:.1f} is below {MIN_COMPUTE_CAP:.1f}, "
                       "so the binary will probably refuse --device gpu")
        return Device("gpu", "requested", warning=warning)
    if not gpus:
        return Device("cpu", f"nvidia-smi failed: {smi_failure}" if smi_failure else "no NVIDIA GPU found")
    gpu = _qualifying(gpus)
    if gpu is not None:
        return Device("gpu", _gpu_detail(gpu), pin(gpu))
    best = max(g.compute_cap for g in gpus)
    return Device("cpu", f"GPU compute capability {best:.1f} is below {MIN_COMPUTE_CAP:.1f}")
