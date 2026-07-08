"""
Centralized configuration for the Grounded Few-Shot English Learner project.

All hyperparameters that were previously hard-coded inline (epochs=400,
epochs=300, lr=2e-3, batch_size=16, etc. scattered across train_stageN()
functions) are collected here. Import this module instead of redefining
constants locally.
"""

import torch

# ── Device ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ─────────────────────────────────────
SEED = 42

# ── World vocabulary (base 4x4x2x4 world used in Stages 1-12) ──
SHAPES = ["circle", "square", "triangle", "star"]
COLORS = ["red", "blue", "green", "yellow"]
SIZES = ["small", "large"]
RELATIONS = ["above", "below", "left of", "right of"]

# ── Model dimensions ────────────────────────────────────
HIDDEN_DIM = 128        # used by Stage 1 (factored classification heads)
D = 128                 # shared embedding/hidden dim for Stages 2-13

# ── Optimization ────────────────────────────────────────
LR = 2e-3
BATCH_SIZE = 16
GRAD_CLIP_NORM = 1.0

# ── Epoch budgets per stage ─────────────────────────────
EPOCHS_STAGE1 = 350
EPOCHS_STAGE2 = 350
EPOCHS_STAGE3 = 400
EPOCHS_STAGE4 = 300
EPOCHS_STAGE5 = 400
EPOCHS_STAGE6 = 400          # single-split compositionality test
EPOCHS_STAGE7_BASELINE = 400
EPOCHS_STAGE8_MULTISPLIT = 300
EPOCHS_STAGE9_ABLATION = 400
EPOCHS_STAGE10_SEED_S3 = 400
EPOCHS_STAGE10_SEED_S5 = 400
EPOCHS_STAGE10_SEED_S6 = 300
EPOCHS_STAGE11_TRANSFORMER = 400
EPOCHS_STAGE13_WORLD_SCALING = 250

# ── Dataset sizing ──────────────────────────────────────
N_PER_TRAIN_COMBO = 3
N_PER_HELDOUT_COMBO = 2
N_TRAIN_PAIRS_BASE = 90
N_HELD_PAIRS_BASE = 15

# Stage 5 / paragraph sampling
N_TRAIN_PARAGRAPHS = 150
N_HELD_PARAGRAPHS = 25

# Stage 6 / compositionality split
COMPOSITIONALITY_N_TRAIN_PAIRS = 100
COMPOSITIONALITY_N_TEST_PAIRS = 20

# Stage 10 / multi-seed benchmark
MULTI_SEED_LIST = (0, 1, 2, 3, 4)

# Stage 13 / world scaling
WORLD_SCALING_CONFIGS = [
    {"n_shapes": 4, "n_colors": 4, "n_sizes": 2, "n_relations": 4},
    {"n_shapes": 25, "n_colors": 25, "n_sizes": 20, "n_relations": 25},
]
WORLD_SCALING_EPOCHS = 250
WORLD_SCALING_BATCH_SIZE = 64
WORLD_SCALING_N_TRAIN_PAIRS = 500
WORLD_SCALING_N_TEST_PAIRS = 60
WORLD_SCALING_N_TRAIN_PARAGRAPHS = 10000
WORLD_SCALING_N_TEST_PARAGRAPHS = 25


def set_all_seeds(seed: int = SEED) -> None:
    """Seed every RNG the project touches, including CUDA."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
