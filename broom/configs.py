"""Hyperparameters and grid metadata for the four APS07 configurations.

All knobs that vary between configs or grid sizes live here, so that
train.py / inference.py / plot.py can stay declarative.
"""

from typing import Literal


ConfigName = Literal[
    "baseline",
    "curriculum",
    "curriculum_enriched",
    "curriculum_recurrent",
    "curriculum_recurrent_v2",
    "mapcnn_bc_pbrs",
    "maskable_v3",
    "maskable_bc_kl",
]
GridSize = Literal[5, 10, 20]


SEEDS: tuple[int, int, int] = (0, 1, 2)


PHASE_TIMESTEPS: dict[int, int] = {
    5: 300_000,
    10: 800_000,
    20: 2_000_000,
}

PHASE_OBSTACLES: dict[int, int] = {
    5: 3,
    10: 12,
    20: 48,
}

PHASE_MAX_STEPS: dict[int, int] = {
    5: 200,
    10: 500,
    20: 1000,
}


def get_phase_n_envs(config: ConfigName, size: GridSize) -> int:
    """Number of parallel envs for a (config, grid) combination.

    RecurrentPPO with CPU is heavier in RAM due to LSTM hidden state, so it
    caps at 2 everywhere. The GPU variant (`curriculum_recurrent_v2`) frees
    up RAM (LSTM lives on the GPU), so it uses 4/4/2 like the MLP configs.
    Plain PPO uses 4 for 5x5 and 10x10, dropping to 2 for 20x20.
    """
    if config == "curriculum_recurrent":
        return 2
    if size == 20:
        return 2
    return 4


# PPO hyperparameters (mirrored from upstream train_grid_world_cpp.py)
PPO_HYPERPARAMS = {
    "ent_coef": 0.05,
    "device": "cpu",
}

RECURRENT_HYPERPARAMS = {
    "ent_coef": 0.05,
    "device": "cpu",
    "policy_kwargs": {
        "lstm_hidden_size": 64,
        "n_lstm_layers": 1,
    },
}

# Second-attempt recurrent config (GPU + larger LSTM + longer rollouts).
# Combines three changes versus RECURRENT_HYPERPARAMS so the LSTM has a real
# chance to learn temporal dependencies on a 20x20 grid:
#   - device GPU (compute can grow without slowing wall-clock)
#   - lstm_hidden_size 256 (4x capacity to encode visited cells)
#   - n_steps 512 (4x default rollout length so the LSTM sees longer
#     sequences per PPO update)
# Note: `n_steps` is a top-level RecurrentPPO constructor argument (not a
# `policy_kwargs` entry); SB3 accepts it via **kwargs at construction time.
RECURRENT_V2_HYPERPARAMS = {
    "ent_coef": 0.05,
    "device": "cuda",
    "n_steps": 512,
    "policy_kwargs": {
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
    },
}


# Map-CNN + BC + PBRS (Epic 7) — uses the egocentric accumulated map env,
# a CNN feature extractor (auto-routed by SB3 MultiInputPolicy on the 3D Box
# `ego_map` channel), BC warm-start from the FrontierAgent expert, and
# potential-based reward shaping with potential = coverage_ratio at training
# time only. Long-horizon-friendly hyperparams: gamma=0.999, n_steps=1024.
MAPCNN_BC_PBRS_HYPERPARAMS = {
    "ent_coef": 0.05,
    "device": "cuda",
    "n_steps": 1024,
    "gamma": 0.999,
    "gae_lambda": 0.95,
}

# Path of the BC-trained checkpoint that warm-starts the first phase
# (5x5) of the curriculum for `mapcnn_bc_pbrs`. Generated once by
# `python -m broom.bc_pipeline` and reused across seeds.
BC_WARMSTART_PATH = "results/models/bc_warmstart.zip"

# Discount used inside the PBRS shaping wrapper. Matches the PPO gamma above.
PBRS_GAMMA = 0.999


# Maskable PPO + reward redesign (Epic 8). Bundle of changes targeting the
# closing-cell ceiling at 77% on 10x10:
#   * action masking via sb3-contrib MaskablePPO (Huang & Ontanon arXiv 2006.14171)
#   * reward redesign: terminal +60, truncation 0, step penalty gated on coverage>=0.80
#     (calibrated from Theile et al. arXiv 2309.03157)
#   * gamma 0.999 for long-horizon credit assignment
#   * entropy schedule from 0.02 -> 0.001 (linear, via SB3 Schedule callable)
#   * larger MLP head (256x256) since the closing decision likely needs
#     more capacity than the default (64x64).
def _maskable_v3_entropy_schedule(progress_remaining: float) -> float:
    """Linear entropy anneal from 0.02 (start) to 0.001 (end).

    Applied via a callback that mutates `model.ent_coef` between rollouts —
    SB3's MaskablePPO/PPO don't support a callable `ent_coef` natively.
    """
    progress = 1.0 - progress_remaining
    return max(0.001, 0.02 * (1.0 - progress))


# Initial ent_coef passed to the constructor; the callback overrides it at
# every rollout start based on `_maskable_v3_entropy_schedule`.
MASKABLE_V3_HYPERPARAMS = {
    "ent_coef": 0.02,
    "device": "cuda",
    "n_steps": 1024,
    "gamma": 0.999,
    "gae_lambda": 0.97,
    "learning_rate": 3e-4,
    "policy_kwargs": {"net_arch": [256, 256]},
}


# Maskable PPO + BC + KL anchor (Epic 9). Same env (V3) and reward redesign as
# `maskable_v3`, plus a frozen BC reference (trained on FrontierAgent rollouts
# of the V3 env) used as a KL anchor inside the PPO loss. Lambda for the KL
# term decays linearly across the cumulative curriculum timesteps so the
# policy starts close to BC and is allowed to drift further as RL training
# progresses (Rajeswaran et al. DAPG 2018; Zhao et al. 2022 adaptive BC).
MASKABLE_BC_KL_HYPERPARAMS = {
    "ent_coef": 0.02,
    "device": "cuda",
    "n_steps": 1024,
    "gamma": 0.999,
    "gae_lambda": 0.97,
    "learning_rate": 1e-4,  # smaller LR for warm-started PPO (Cal-QL-style)
    "policy_kwargs": {"net_arch": [256, 256]},
}

# BC warm-start checkpoint for the V3 env. Generated by
# `python -m broom.bc_v3_pipeline` and reused across seeds.
BC_V3_WARMSTART_PATH = "results/models/bc_warmstart_v3.zip"

# KL anchor schedule: lambda decays linearly from 1.0 to 0.05 over the full
# curriculum timesteps (~3.1M total: 300k 5x5 + 800k 10x10 + 2M 20x20).
KL_LAMBDA_INITIAL = 1.0
KL_LAMBDA_FINAL = 0.05
KL_LAMBDA_DECAY_TIMESTEPS = 3_100_000


# Curriculum chain: each entry is (size, init_from_size or None)
CURRICULUM_CHAIN: list[tuple[int, int | None]] = [
    (5, None),    # train from scratch
    (10, 5),      # warm-start from 5x5 model
    (20, 10),     # warm-start from 10x10 model
]
