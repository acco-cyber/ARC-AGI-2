"""Data module: HF-mirror loaders, family heuristics, 700/150/150 re-split and tensorisation."""
from .families import FAMILIES, TaskFeatures, family_from_features, family_histogram, family_of, task_features
from .hf_loader import (ALL_CONFIGS, CARD_COUNTS, DEFAULT_LOCAL_ROOT, EPISODE_CONFIGS, EVAL_PUBLIC, HF_SPLITS,
                        KAGGLE_ROOT, PACKAGE_DATA_DIR, RESPLIT_NAMES, RESPLIT_SEED, RESPLIT_SIZES, SPLITS_FILENAME,
                        DataRootNotFound, EvalPublicGuardError, ensure_resplit, episodes_from_task,
                        filter_episodes_by_tasks, has_local_root, load_counterfactuals, load_episodes,
                        load_eval_public_tasks, load_resplit, load_rows, load_rule_programs, load_task_rows,
                        load_tasks, load_training_tasks, read_hf_splits, read_splits, resolve_root,
                        resplit_700_150_150, splits_payload, task_splits, training_task_ids, write_splits)
from .tensorize import (DEFAULT_MAX_CTX, MAX_OBJECTS, OBJ_FEAT_DIM, REL_FEAT_DIM, EpisodeDataset, collate,
                        encode_episode, encode_objects, grid_mask, grid_to_tensor, mask_to_shape, pair_to_tensors,
                        resolve_parser, tensor_to_grid, zero_parser)

__all__ = [
    "ALL_CONFIGS", "CARD_COUNTS", "DEFAULT_LOCAL_ROOT", "DEFAULT_MAX_CTX", "EPISODE_CONFIGS", "EVAL_PUBLIC",
    "FAMILIES", "HF_SPLITS", "KAGGLE_ROOT", "MAX_OBJECTS", "OBJ_FEAT_DIM", "PACKAGE_DATA_DIR", "REL_FEAT_DIM",
    "RESPLIT_NAMES", "RESPLIT_SEED", "RESPLIT_SIZES", "SPLITS_FILENAME", "DataRootNotFound", "EpisodeDataset",
    "EvalPublicGuardError", "TaskFeatures", "collate", "encode_episode", "encode_objects", "ensure_resplit",
    "episodes_from_task", "family_from_features", "family_histogram", "family_of", "filter_episodes_by_tasks",
    "grid_mask", "grid_to_tensor", "has_local_root", "load_counterfactuals", "load_episodes",
    "load_eval_public_tasks", "load_resplit", "load_rows", "load_rule_programs", "load_task_rows", "load_tasks",
    "load_training_tasks", "mask_to_shape", "pair_to_tensors", "read_hf_splits", "read_splits", "resolve_parser",
    "resolve_root", "resplit_700_150_150", "splits_payload", "task_features", "task_splits", "tensor_to_grid",
    "training_task_ids", "write_splits", "zero_parser",
]
