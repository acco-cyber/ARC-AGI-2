"""Synthetic program-generated ARC tasks: input grids, program sampling, perturbations, JSONL dataset writer."""
from arcjepa.synthetic.generators import STYLES, STYLE_WEIGHTS, random_input_grid, random_palette
from arcjepa.synthetic.program_sampler import (CATEGORIES, CATEGORY_MIX, DEPTH_MIX, sample_category, sample_depth,
                                               sample_program)
from arcjepa.synthetic.dataset import (HELDOUT_COMPOSITIONS, SynthTask, degeneracy_reason, difficulty_score,
                                       generate, iter_rows, load_jsonl, make_task, make_task_with_reason,
                                       row_to_task, split_of, task_to_row, to_arc_task)
from arcjepa.synthetic.perturbations import add_distractors, ambiguous_segmentation

__all__ = [
    "STYLES", "STYLE_WEIGHTS", "random_input_grid", "random_palette",
    "DEPTH_MIX", "CATEGORY_MIX", "CATEGORIES", "sample_depth", "sample_category", "sample_program",
    "SynthTask", "HELDOUT_COMPOSITIONS", "make_task", "make_task_with_reason", "generate", "split_of",
    "difficulty_score", "degeneracy_reason", "task_to_row", "row_to_task", "iter_rows", "load_jsonl",
    "to_arc_task", "add_distractors", "ambiguous_segmentation",
]
