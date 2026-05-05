"""Hyperparameters and grid metadata for the four APS06 configurations.

All knobs that vary between configs or grid sizes live here, so that
train.py / inference.py / plot.py can stay declarative.
"""

from typing import Literal


ConfigName = Literal[
    "baseline",
    "curriculum",
    "curriculum_enriched",
    "curriculum_recurrent",
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

    RecurrentPPO is heavier in RAM due to LSTM hidden state, so we cap at 2
    everywhere. Plain PPO uses 4 for 5x5 and 10x10, dropping to 2 for 20x20
    to stay under 8GB RAM.
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


# Curriculum chain: each entry is (size, init_from_size or None)
CURRICULUM_CHAIN: list[tuple[int, int | None]] = [
    (5, None),    # train from scratch
    (10, 5),      # warm-start from 5x5 model
    (20, 10),     # warm-start from 10x10 model
]
