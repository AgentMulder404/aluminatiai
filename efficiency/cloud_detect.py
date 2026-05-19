"""
Cloud provider auto-detection via instance metadata (IMDS).

Detects AWS, GCP, and Azure instances and maps them to GPU hourly costs.
Safe to run on non-cloud machines — fails gracefully with "unknown" provider.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# GPU hourly rates (USD) by instance type — representative pricing as of 2026 Q1
# Source: public pricing pages for on-demand instances
GPU_COST_TABLE: dict[str, float] = {
    # AWS
    "p5.48xlarge": 98.32,     # 8x H100
    "p4d.24xlarge": 32.77,    # 8x A100 40GB
    "p4de.24xlarge": 40.97,   # 8x A100 80GB
    "g5.xlarge": 1.006,       # 1x A10G
    "g5.2xlarge": 1.212,
    "g5.4xlarge": 1.624,
    "g5.48xlarge": 16.29,     # 8x A10G
    "g6.xlarge": 0.8048,      # 1x L4
    # GCP
    "a2-highgpu-1g": 3.67,    # 1x A100 40GB
    "a2-highgpu-8g": 29.39,   # 8x A100 40GB
    "a2-ultragpu-8g": 40.22,  # 8x A100 80GB
    "a3-highgpu-8g": 101.22,  # 8x H100
    "g2-standard-4": 0.84,    # 1x L4
    # Azure
    "Standard_NC24ads_A100_v4": 3.67,   # 1x A100 80GB
    "Standard_ND96asr_v4": 27.20,       # 8x A100 40GB
    "Standard_ND96amsr_A100_v4": 32.77, # 8x A100 80GB
    "Standard_NC6s_v3": 3.06,           # 1x V100
}


GPU_HOURLY_RATES: dict[str, float] = {
    "H100-SXM5-80GB": 3.99,
    "H100-PCIe-80GB": 2.49,
    "H200-SXM-141GB": 4.99,
    "A100-SXM4-80GB": 1.89,
    "A100-SXM4-40GB": 1.49,
    "A100-PCIe-80GB": 1.59,
    "A100-PCIe-40GB": 1.29,
    "RTX 4090": 0.59,
    "L40S": 1.29,
    "L40": 0.99,
    "A10G": 0.50,
    "T4": 0.37,
    "V100-SXM2-32GB": 0.80,
    "V100-SXM2-16GB": 0.70,
    "AMD MI210": 1.10,
    "AMD MI250X": 2.50,
    "AMD MI300X": 3.50,
    "AMD MI325X": 4.20,
    "AMD RX 7900 XTX": 0.44,
    "Intel Gaudi2": 1.80,
    "Intel Gaudi3": 3.40,
    "Intel Data Center GPU Max 1550": 3.20,
    "Intel Data Center GPU Max 1100": 1.60,
}


@dataclass
class CloudInfo:
    provider: str          # "aws", "gcp", "azure", "unknown"
    instance_type: str
    region: str
    gpu_cost_per_hour: float
    gpu_count_detected: int


def detect() -> CloudInfo:
    """Auto-detect cloud provider and instance metadata.

    Uses IMDS endpoints with short timeouts. Safe on non-cloud machines.
    """
    for detector in [_detect_aws, _detect_gcp, _detect_azure]:
        try:
            result = detector()
            if result:
                return result
        except Exception:
            continue

    return CloudInfo(
        provider="unknown",
        instance_type="unknown",
        region="unknown",
        gpu_cost_per_hour=0.0,
        gpu_count_detected=0,
    )


def _detect_aws() -> Optional[CloudInfo]:
    import requests
    try:
        token_resp = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=1,
        )
        token = token_resp.text
        headers = {"X-aws-ec2-metadata-token": token}
    except Exception:
        headers = {}

    try:
        resp = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers=headers,
            timeout=1,
        )
        if resp.status_code != 200:
            return None
        instance_type = resp.text.strip()

        region_resp = requests.get(
            "http://169.254.169.254/latest/meta-data/placement/region",
            headers=headers,
            timeout=1,
        )
        region = region_resp.text.strip() if region_resp.status_code == 200 else "unknown"

        cost = GPU_COST_TABLE.get(instance_type, 0.0)
        return CloudInfo(
            provider="aws",
            instance_type=instance_type,
            region=region,
            gpu_cost_per_hour=cost,
            gpu_count_detected=0,
        )
    except Exception:
        return None


def _detect_gcp() -> Optional[CloudInfo]:
    import requests
    headers = {"Metadata-Flavor": "Google"}
    try:
        resp = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
            headers=headers,
            timeout=1,
        )
        if resp.status_code != 200:
            return None
        machine_type = resp.text.strip().split("/")[-1]

        zone_resp = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers=headers,
            timeout=1,
        )
        zone = zone_resp.text.strip().split("/")[-1] if zone_resp.status_code == 200 else "unknown"
        region = "-".join(zone.split("-")[:-1]) if zone != "unknown" else "unknown"

        cost = GPU_COST_TABLE.get(machine_type, 0.0)
        return CloudInfo(
            provider="gcp",
            instance_type=machine_type,
            region=region,
            gpu_cost_per_hour=cost,
            gpu_count_detected=0,
        )
    except Exception:
        return None


def _detect_azure() -> Optional[CloudInfo]:
    import requests
    try:
        resp = requests.get(
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            headers={"Metadata": "true"},
            timeout=1,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        compute = data.get("compute", {})
        vm_size = compute.get("vmSize", "unknown")
        location = compute.get("location", "unknown")

        cost = GPU_COST_TABLE.get(vm_size, 0.0)
        return CloudInfo(
            provider="azure",
            instance_type=vm_size,
            region=location,
            gpu_cost_per_hour=cost,
            gpu_count_detected=0,
        )
    except Exception:
        return None
