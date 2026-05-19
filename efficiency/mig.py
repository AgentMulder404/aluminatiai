"""
MIG (Multi-Instance GPU) power splitting.

Detects NVIDIA MIG mode on supported GPUs (A100, A30, H100, H200),
enumerates MIG instances, and splits the parent GPU's total power
proportionally by compute slice count.

Example: A100 in 3g.20gb + 2g.10gb + 2g.10gb mode:
  - 3g instance gets 3/7 of total power
  - Each 2g instance gets 2/7 of total power
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML = True
except ImportError:
    _NVML = False

# Compute slice counts per MIG profile (GPU Instance size)
# A100 = 7 slices total, H100 = 7 slices total
MIG_SLICE_COUNTS: dict[int, int] = {
    0: 1,   # 1g profiles
    1: 1,
    2: 2,   # 2g profiles
    3: 2,
    4: 3,   # 3g profiles
    5: 4,   # 4g profiles
    6: 7,   # 7g (full GPU)
}


@dataclass
class MigInstance:
    index: int
    gpu_instance_id: int
    compute_instance_id: int
    profile_id: int
    slice_count: int
    memory_mb: int
    power_fraction: float


@dataclass
class MigInfo:
    enabled: bool
    instances: list[MigInstance]
    total_slices: int


def detect_mig(gpu_index: int) -> MigInfo:
    """Detect MIG configuration on a GPU.

    Returns MigInfo with enabled=False if MIG is not active or NVML unavailable.
    """
    if not _NVML:
        return MigInfo(enabled=False, instances=[], total_slices=0)

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except pynvml.NVMLError:
        return MigInfo(enabled=False, instances=[], total_slices=0)

    try:
        current_mode, pending_mode = pynvml.nvmlDeviceGetMigMode(handle)
        if current_mode != pynvml.NVML_DEVICE_MIG_ENABLE:
            return MigInfo(enabled=False, instances=[], total_slices=0)
    except (pynvml.NVMLError, AttributeError):
        return MigInfo(enabled=False, instances=[], total_slices=0)

    instances: list[MigInstance] = []
    total_slices = 0

    try:
        max_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
    except (pynvml.NVMLError, AttributeError):
        max_count = 8

    for i in range(max_count):
        try:
            mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, i)
            attrs = pynvml.nvmlDeviceGetAttributes(mig_handle)

            profile_id = getattr(attrs, 'gpuInstanceSliceCount', 1)
            slice_count = profile_id if profile_id > 0 else 1
            memory_mb = getattr(attrs, 'memorySizeMB', 0) or 0

            instances.append(MigInstance(
                index=i,
                gpu_instance_id=i,
                compute_instance_id=0,
                profile_id=profile_id,
                slice_count=slice_count,
                memory_mb=memory_mb,
                power_fraction=0.0,
            ))
            total_slices += slice_count
        except (pynvml.NVMLError, AttributeError):
            break

    # Calculate power fractions
    if total_slices > 0:
        for inst in instances:
            inst.power_fraction = inst.slice_count / total_slices

    logger.info("GPU %d MIG enabled: %d instances, %d total slices",
                gpu_index, len(instances), total_slices)

    return MigInfo(enabled=True, instances=instances, total_slices=total_slices)


def split_power_by_mig(
    total_power_w: float,
    mig_info: MigInfo,
) -> list[tuple[int, float]]:
    """Split total GPU power across MIG instances.

    Returns list of (instance_index, power_watts) tuples.
    """
    if not mig_info.enabled or not mig_info.instances:
        return [(0, total_power_w)]

    return [
        (inst.index, total_power_w * inst.power_fraction)
        for inst in mig_info.instances
    ]
