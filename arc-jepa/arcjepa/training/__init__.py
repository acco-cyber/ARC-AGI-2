"""ARC-JEPA training: stage A (synthetic JEPA), B (real adaptation), C (program alignment), D (hard negatives),
the time-boxed ``train_all`` driver and the offline package ``export``."""
from arcjepa.training.common import (STAGE_NAMES, STAGES, Checkpointer, DistInfo, EpochBatchSampler, MetricsLogger,
                                     RealEpisodeDataset, StageSpec, SynthEpisodeDataset, SynthStore, TimeBudget,
                                     TrainContext, build_model, build_optimizer, build_target, deep_merge,
                                     encode_item, init_distributed, load_config, loss_weights, lr_multiplier,
                                     model_config_from, prepare_synthetic, retrieval_at_k, rule_latents, run_stage,
                                     seed_for, set_seed, synth_episode)
from arcjepa.training.export import (build_program_memory, export_package, load_program_memory, pseudo_label,
                                     save_weights)
from arcjepa.training.pretrain_jepa import build_stage_a, run_stage_a
from arcjepa.training.train_program_encoder import build_program_stage, run_stage_c, run_stage_d
from arcjepa.training.train_real import family_weights, load_real_data, run_stage_b, stream_episodes

_LAZY_TRAIN_ALL = ("main", "memory_sources", "stage_budget", "train")


def __getattr__(name: str):
    """Lazy access to ``train_all`` names (keeps ``python -m arcjepa.training.train_all`` free of runpy warnings)."""
    if name in _LAZY_TRAIN_ALL:
        from arcjepa.training import train_all

        return getattr(train_all, name)
    raise AttributeError(f"module 'arcjepa.training' has no attribute {name!r}")

__all__ = [
    "STAGES", "STAGE_NAMES", "Checkpointer", "DistInfo", "EpochBatchSampler", "MetricsLogger", "RealEpisodeDataset",
    "StageSpec", "SynthEpisodeDataset", "SynthStore", "TimeBudget", "TrainContext", "build_model", "build_optimizer",
    "build_target", "deep_merge", "encode_item", "init_distributed", "load_config", "loss_weights", "lr_multiplier",
    "model_config_from", "prepare_synthetic", "retrieval_at_k", "rule_latents", "run_stage", "seed_for", "set_seed",
    "synth_episode", "build_program_memory", "export_package", "load_program_memory", "pseudo_label",
    "save_weights", "build_stage_a", "run_stage_a", "build_program_stage", "run_stage_c", "run_stage_d",
    "family_weights", "load_real_data", "run_stage_b", "stream_episodes", "main", "memory_sources", "stage_budget",
    "train",
]
