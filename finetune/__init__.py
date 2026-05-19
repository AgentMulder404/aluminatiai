# AluminatiAI GreenTune — energy-efficient fine-tuning on AMD ROCm
#
# pip install aluminatiai                — base GPU monitoring
# pip install aluminatiai[finetune]      — training + energy tracking
# pip install aluminatiai[greentune]     — everything
"""GreenTune: energy-efficient LLM fine-tuning with real GPU telemetry.

Quick start::

    from aluminatiai.finetune import EnergyCallback

    # Drop into any HuggingFace Trainer
    trainer = SFTTrainer(..., callbacks=[EnergyCallback(gpu_index=0)])

    # Or use the swarm optimizer (no API key needed)
    from aluminatiai.finetune import GreenTuneSwarm
    swarm = GreenTuneSwarm()
    result = swarm.optimize("Minimize J/token for Qwen2.5-7B on 500 samples")

CLI::

    aluminatiai train --hermes-only --hermes-max 500 --batch-size 4
    aluminatiai swarm --max-samples 500
"""


def __getattr__(name: str):
    if name in ("PowerSample", "EnergyAccumulator", "AMDPowerMonitor", "PowerSamplerThread"):
        from . import rocm_power
        return getattr(rocm_power, name)

    if name in ("EnergyCallback", "StepMetrics"):
        from . import energy_callback
        return getattr(energy_callback, name)

    if name == "GreenTuneSwarm":
        from .greentune_swarm import GreenTuneSwarm
        return GreenTuneSwarm

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PowerSample",
    "EnergyAccumulator",
    "AMDPowerMonitor",
    "PowerSamplerThread",
    "EnergyCallback",
    "StepMetrics",
    "GreenTuneSwarm",
]
