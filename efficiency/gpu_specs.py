# Copyright 2026 Kevin (AluminatiAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AluminatiAI — https://github.com/AgentMulder404/AluminatAI
"""
GPU architecture specifications and ML model profiles.

This module is the single source of truth for hardware specs used in
efficiency calculations. Values mirror the gpu_architectures and
model_profiles tables seeded in migration 007.

Roofline Model:
  For a given workload with arithmetic intensity I (FLOP/byte):
    Attainable TFLOPS = min(peak_tflops, memory_bw * I)

  Memory-bound workloads (low I): limited by bandwidth, compute underutilized
  Compute-bound workloads (high I): limited by TFLOPS, bandwidth sufficient
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    """Immutable GPU architecture specification."""

    name: str
    family: str
    tdp_w: float               # Thermal Design Power (Watts)
    fp16_tflops: float          # Peak FP16 throughput
    fp32_tflops: float          # Peak FP32 throughput
    bf16_tflops: float          # Peak BF16 throughput (0 if unsupported)
    memory_gb: float
    memory_bw_gbps: float       # Peak memory bandwidth (GB/s)
    has_transformer_engine: bool

    @property
    def idle_power_w(self) -> float:
        """Estimated idle power (~30% of TDP based on empirical data)."""
        return self.tdp_w * 0.30

    def peak_tflops_for_precision(self, precision: str) -> float:
        """Return peak TFLOPS for a given precision."""
        mapping = {
            'fp16': self.fp16_tflops,
            'bf16': self.bf16_tflops if self.bf16_tflops > 0 else self.fp16_tflops,
            'fp32': self.fp32_tflops,
            'fp8': self.fp16_tflops * 2.0,  # FP8 ~2x FP16 on Hopper
            'int8': self.fp16_tflops * 2.0,
        }
        return mapping.get(precision, self.fp16_tflops)

    def estimated_power_at_utilization(self, util_frac: float) -> float:
        """
        Power draw at a given utilization fraction (0.0 - 1.0).

        Uses linear interpolation between idle power and TDP.
        Validated against NVIDIA datacenter power curves:
          P(u) = P_idle + (P_tdp - P_idle) * u
        """
        return self.idle_power_w + (self.tdp_w - self.idle_power_w) * util_frac

    def roofline_tflops(
        self,
        math_intensity: float,
        util_frac: float,
        precision: str = 'fp16',
    ) -> float:
        """
        Roofline model: attainable TFLOPS given math intensity and utilization.

        Args:
            math_intensity: FLOP/byte of the workload
            util_frac: GPU utilization fraction (0.0 - 1.0)
            precision: Compute precision ('fp16', 'bf16', 'fp32')

        Returns:
            Effective TFLOPS this GPU can deliver for this workload.
        """
        peak = self.peak_tflops_for_precision(precision) * util_frac

        # Bandwidth ceiling: memory_bw (GB/s) * math_intensity (FLOP/byte) / 1e3 → TFLOPS
        bandwidth_ceiling = (self.memory_bw_gbps * math_intensity) / 1000.0

        return min(peak, bandwidth_ceiling)

    def joules_per_tflop(
        self,
        math_intensity: float,
        util_frac: float,
        precision: str = 'fp16',
    ) -> float:
        """
        Core efficiency metric: energy cost per unit of useful compute.

        Lower is better. Returns float('inf') if no useful work is done.
        """
        effective_tflops = self.roofline_tflops(math_intensity, util_frac, precision)
        if effective_tflops <= 0:
            return float('inf')

        power_w = self.estimated_power_at_utilization(util_frac)
        return power_w / effective_tflops


@dataclass(frozen=True)
class ModelProfile:
    """ML model workload characteristics for hardware matching."""

    tag: str
    family: str
    math_intensity: float       # FLOP/byte — the key roofline parameter
    precision: str              # Dominant compute precision
    is_memory_bound: bool       # True if math_intensity < ~50 FLOP/byte
    typical_util_min: int       # Expected utilization range lower bound
    typical_util_max: int       # Expected utilization range upper bound

    @property
    def typical_util_mid(self) -> float:
        """Midpoint of expected utilization range as fraction."""
        return (self.typical_util_min + self.typical_util_max) / 200.0


# ═══════════════════════════════════════════════════════════════════════
# GPU ARCHITECTURES — mirrors migration 007 seed data
# ═══════════════════════════════════════════════════════════════════════

GPU_ARCHITECTURES: dict[str, ArchSpec] = {
    spec.name: spec for spec in [
        # Ampere
        ArchSpec('A100-SXM4-80GB',  'Ampere',        400,  312,  19.5,  312,  80, 2039, False),
        ArchSpec('A100-SXM4-40GB',  'Ampere',        400,  312,  19.5,  312,  40, 1555, False),
        ArchSpec('A100-PCIe-80GB',  'Ampere',        300,  312,  19.5,  312,  80, 2039, False),
        ArchSpec('A100-PCIe-40GB',  'Ampere',        250,  312,  19.5,  312,  40, 1555, False),
        # Hopper
        ArchSpec('H100-SXM5-80GB',  'Hopper',        700,  989,  67.0,  989,  80, 3350, True),
        ArchSpec('H100-PCIe-80GB',  'Hopper',        350,  756,  51.0,  756,  80, 2039, True),
        ArchSpec('H200-SXM-141GB',  'Hopper',        700,  989,  67.0,  989, 141, 4800, True),
        # Ada Lovelace
        ArchSpec('RTX 4090',        'Ada Lovelace',  450,  165.2, 82.6, 165.2, 24, 1008, False),
        ArchSpec('L40S',            'Ada Lovelace',  350,  362,  91.6,  362,  48,  864, False),
        ArchSpec('L40',             'Ada Lovelace',  300,  181,  90.5,  181,  48,  864, False),
        # Lower-tier
        ArchSpec('A10G',            'Ampere',        150,   70,  31.2,   70,  24,  600, False),
        ArchSpec('T4',              'Turing',         70,   65,   8.1,    0,  16,  300, False),
        # Volta
        ArchSpec('V100-SXM2-32GB',  'Volta',        300,  125,  15.7,    0,  32,  900, False),
        ArchSpec('V100-SXM2-16GB',  'Volta',        300,  125,  15.7,    0,  16,  900, False),
        # Apple Silicon (unified memory — memory_gb is total system RAM share)
        ArchSpec('Apple M1 GPU',        'Apple Silicon',  10,   2.6,  1.3,  2.6,   8, 68,  False),
        ArchSpec('Apple M1 Pro GPU',    'Apple Silicon',  20,   5.2,  2.6,  5.2,  16, 200, False),
        ArchSpec('Apple M1 Max GPU',    'Apple Silicon',  40,  10.4,  5.2, 10.4,  32, 400, False),
        ArchSpec('Apple M1 Ultra GPU',  'Apple Silicon',  60,  20.8, 10.4, 20.8,  64, 800, False),
        ArchSpec('Apple M2 GPU',        'Apple Silicon',  12,   3.6,  1.8,  3.6,   8, 100, False),
        ArchSpec('Apple M2 Pro GPU',    'Apple Silicon',  22,   6.8,  3.4,  6.8,  16, 200, False),
        ArchSpec('Apple M2 Max GPU',    'Apple Silicon',  45,  13.6,  6.8, 13.6,  48, 400, False),
        ArchSpec('Apple M2 Ultra GPU',  'Apple Silicon',  75,  27.2, 13.6, 27.2, 192, 800, False),
        ArchSpec('Apple M3 GPU',        'Apple Silicon',  12,   4.1,  2.1,  4.1,   8, 100, False),
        ArchSpec('Apple M3 Pro GPU',    'Apple Silicon',  22,   7.0,  3.5,  7.0,  18, 150, False),
        ArchSpec('Apple M3 Max GPU',    'Apple Silicon',  45,  14.2,  7.1, 14.2,  48, 400, False),
        ArchSpec('Apple M3 Ultra GPU',  'Apple Silicon',  75,  28.4, 14.2, 28.4, 192, 800, False),
        ArchSpec('Apple M4 GPU',        'Apple Silicon',  14,   4.6,  2.3,  4.6,  16, 120, False),
        ArchSpec('Apple M4 Pro GPU',    'Apple Silicon',  25,   8.4,  4.2,  8.4,  24, 273, False),
        ArchSpec('Apple M4 Max GPU',    'Apple Silicon',  50,  16.7,  8.4, 16.7,  64, 546, False),
        ArchSpec('Apple M4 Ultra GPU',  'Apple Silicon',  80,  33.5, 16.7, 33.5, 256, 819, False),
        ArchSpec('Apple M5 GPU',        'Apple Silicon',  15,   5.0,  2.5,  5.0,  16, 120, False),
        ArchSpec('Apple M5 Pro GPU',    'Apple Silicon',  28,   9.5,  4.8,  9.5,  24, 273, False),
        ArchSpec('Apple M5 Max GPU',    'Apple Silicon',  55,  18.0,  9.0, 18.0,  64, 546, False),
        ArchSpec('Apple M5 Ultra GPU',  'Apple Silicon',  85,  36.0, 18.0, 36.0, 256, 819, False),
        # Intel Gaudi AI Accelerators
        ArchSpec('Intel Gaudi',   'Gaudi',  300,  140,  17.5,  140,  32, 1000, False),
        ArchSpec('Intel Gaudi2',  'Gaudi',  600,  420,  52.5,  420,  96, 2460, True),
        ArchSpec('Intel Gaudi3',  'Gaudi',  900,  900, 112.5,  900, 128, 3700, True),
        # Intel Arc / Data Center GPUs (Alchemist / Battlemage / Ponte Vecchio)
        ArchSpec('Intel Arc A770',  'Arc',  225,  39.3, 19.7, 39.3, 16, 560, False),
        ArchSpec('Intel Arc A750',  'Arc',  225,  34.4, 17.2, 34.4,  8, 512, False),
        ArchSpec('Intel Arc A580',  'Arc',  185,  24.6, 12.3, 24.6,  8, 512, False),
        ArchSpec('Intel Arc B580',  'Arc',  190,  27.3, 13.7, 27.3, 12, 456, False),
        ArchSpec('Intel Data Center GPU Flex 170',   'Arc',  150,  24.0, 12.0, 24.0, 16, 560, False),
        ArchSpec('Intel Data Center GPU Flex 140',   'Arc',   75,  12.0,  6.0, 12.0,  8, 280, False),
        ArchSpec('Intel Data Center GPU Max 1550',   'Ponte Vecchio', 600, 419.4, 52.4, 419.4, 128, 3276, True),
        ArchSpec('Intel Data Center GPU Max 1100',   'Ponte Vecchio', 300, 209.7, 26.2, 209.7,  48, 1228, True),
        # AMD Instinct (CDNA)
        ArchSpec('AMD MI210',   'CDNA2', 300,  181,  22.6,  181,   64, 1638, False),
        ArchSpec('AMD MI250X',  'CDNA2', 500,  383,  47.9,  383,  128, 3277, False),
        ArchSpec('AMD MI300X',  'CDNA3', 750, 1307, 163.4, 1307,  192, 5300, True),
        ArchSpec('AMD MI325X',  'CDNA3', 750, 1307, 163.4, 1307,  256, 6000, True),
        # AMD Radeon (RDNA3)
        ArchSpec('AMD RX 7900 XTX',  'RDNA3', 355, 61.4, 30.7, 61.4, 24, 960, False),
    ]
}


# ═══════════════════════════════════════════════════════════════════════
# MODEL PROFILES — mirrors migration 007 seed data
# ═══════════════════════════════════════════════════════════════════════

MODEL_PROFILES: dict[str, ModelProfile] = {
    p.tag: p for p in [
        ModelProfile('bert-base',       'BERT',        10,  'fp16',  True,  35, 65),
        ModelProfile('bert-large',      'BERT',        12,  'fp16',  True,  40, 70),
        ModelProfile('llama-3-8b',      'Llama',       95,  'bf16',  False, 60, 85),
        ModelProfile('llama-3-70b',     'Llama',      185,  'bf16',  False, 75, 95),
        ModelProfile('llama-3-405b',    'Llama',      220,  'bf16',  False, 80, 95),
        ModelProfile('mistral-7b',      'Mistral',     90,  'bf16',  False, 55, 80),
        ModelProfile('mixtral-8x7b',    'Mistral',    110,  'bf16',  False, 60, 85),
        ModelProfile('gpt-neox-20b',    'GPT-NeoX',   130,  'fp16',  False, 65, 90),
        ModelProfile('sdxl',            'Diffusion',   48,  'fp16',  True,  50, 80),
        ModelProfile('sd-3',            'Diffusion',   55,  'fp16',  False, 50, 80),
        ModelProfile('whisper-large',   'Whisper',     28,  'fp16',  True,  35, 65),
        ModelProfile('whisper-medium',  'Whisper',     22,  'fp16',  True,  30, 60),
        ModelProfile('vit-large',       'ViT',         35,  'fp16',  True,  40, 70),
        ModelProfile('t5-xxl',          'T5',         100,  'bf16',  False, 55, 85),
        ModelProfile('falcon-40b',      'Falcon',     140,  'bf16',  False, 65, 90),
        ModelProfile('deepseek-v3',     'DeepSeek',   200,  'bf16',  False, 75, 95),
    ]
}


WORKLOAD_ARCHETYPES: dict[str, ModelProfile] = {
    p.tag: p for p in [
        ModelProfile('llm-inference',   'LLM',        120, 'bf16', False, 30, 60),
        ModelProfile('llm-training',    'LLM',        200, 'bf16', False, 75, 95),
        ModelProfile('vision-training', 'Vision',      40, 'fp16', True,  50, 80),
        ModelProfile('rendering',       'Rendering',   25, 'fp32', True,  60, 90),
        ModelProfile('batch-inference', 'Inference',    60, 'fp16', False, 40, 70),
    ]
}


def resolve_arch(gpu_name: str) -> ArchSpec | None:
    """
    Match a gpu_name string to a known architecture.

    Handles NVML names ("NVIDIA A100-SXM4-80GB", "Tesla T4") and Apple
    Silicon names ("Apple M5 GPU", "Apple M4 Max GPU (40-core)").
    """
    # Exact match
    if gpu_name in GPU_ARCHITECTURES:
        return GPU_ARCHITECTURES[gpu_name]

    # Strip common prefixes
    cleaned = gpu_name.replace('NVIDIA ', '').replace('Tesla ', '')
    if cleaned in GPU_ARCHITECTURES:
        return GPU_ARCHITECTURES[cleaned]

    # Apple Silicon: strip core count suffix, e.g. "Apple M5 GPU (10-core)" -> "Apple M5 GPU"
    if "Apple M" in gpu_name:
        apple_clean = re.sub(r'\s*\(\d+-core\)', '', gpu_name)
        if apple_clean in GPU_ARCHITECTURES:
            return GPU_ARCHITECTURES[apple_clean]
        for arch_name, spec in GPU_ARCHITECTURES.items():
            if spec.family == 'Apple Silicon' and arch_name in apple_clean:
                return spec

    # Intel Gaudi: match "Intel Gaudi2", "Intel Gaudi3", etc.
    if "Gaudi" in gpu_name:
        for arch_name, spec in GPU_ARCHITECTURES.items():
            if spec.family == 'Gaudi' and arch_name in gpu_name:
                return spec

    # Intel Arc / Data Center GPU: match "Intel Arc A770", "Intel Data Center GPU Max 1550", etc.
    if "Intel" in gpu_name and ("Arc" in gpu_name or "Data Center" in gpu_name):
        for arch_name, spec in GPU_ARCHITECTURES.items():
            if spec.family in ('Arc', 'Ponte Vecchio') and arch_name in gpu_name:
                return spec

    # AMD Instinct / Radeon: match "AMD MI300X", "AMD RX 7900 XTX", etc.
    if "AMD" in gpu_name or "MI2" in gpu_name or "MI3" in gpu_name or "Radeon" in gpu_name:
        amd_clean = gpu_name.replace('AMD Instinct ', 'AMD ').replace('Radeon ', 'AMD RX ')
        for arch_name, spec in GPU_ARCHITECTURES.items():
            if spec.family in ('CDNA2', 'CDNA3', 'RDNA3') and arch_name in amd_clean:
                return spec

    # Substring match (e.g., "A100" matches "A100-SXM4-80GB")
    for arch_name, spec in GPU_ARCHITECTURES.items():
        if arch_name in cleaned or cleaned in arch_name:
            return spec

    # Family match (e.g., "A100" anywhere in the string)
    for arch_name, spec in GPU_ARCHITECTURES.items():
        base = arch_name.split('-')[0]
        if base in gpu_name:
            return spec

    return None
