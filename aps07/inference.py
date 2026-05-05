"""Evaluate a trained model over N episodes on a target grid size.

Records per-episode coverage/steps to a CSV and returns aggregate metrics.
"""

import csv
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from aps07.configs import PHASE_MAX_STEPS, PHASE_OBSTACLES, ConfigName, GridSize
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


def _results_dir() -> Path:
    return Path(os.environ.get("APS07_RESULTS_DIR", "results"))


def _register_envs():
    if "gymnasium_env/GridWorldCPP-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPP-v0", entry_point=GridWorldCPPEnv)
    try:
        from gymnasium_env.grid_world_cpp_enriched import GridWorldCPPEnrichedEnv
        if "gymnasium_env/GridWorldCPPEnriched-v0" not in gym.envs.registry:
            gym.register(id="gymnasium_env/GridWorldCPPEnriched-v0", entry_point=GridWorldCPPEnrichedEnv)
    except ImportError:
        pass


def _env_id_for_config(config_name: ConfigName) -> str:
    if config_name == "curriculum_enriched":
        return "gymnasium_env/GridWorldCPPEnriched-v0"
    return "gymnasium_env/GridWorldCPP-v0"


def _load_model(model_path: str, config_name: ConfigName, env: gym.Env):
    if config_name == "curriculum_recurrent":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(model_path, env=env, device="cpu")
    return PPO.load(model_path, env=env, device="cpu")


def evaluate(
    model_path: str,
    config_name: ConfigName,
    seed: int,
    eval_size: GridSize,
    n_episodes: int = 100,
) -> dict:
    """Run `n_episodes` of the loaded model on a grid of `eval_size` and report metrics."""
    _register_envs()
    env_id = _env_id_for_config(config_name)
    obstacles = PHASE_OBSTACLES[eval_size]
    max_steps = PHASE_MAX_STEPS[eval_size]

    env = gym.make(env_id, size=eval_size, obs_quantity=obstacles, max_steps=max_steps)
    model = _load_model(model_path, config_name, env)

    rows: list[tuple[int, float, int, bool]] = []
    full_coverage_count = 0
    coverages: list[float] = []
    steps_list: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        lstm_state = None
        episode_starts = np.ones((1,), dtype=bool)
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated):
            if config_name == "curriculum_recurrent":
                action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_starts, deterministic=False)
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1

        coverage = info["coverage"]
        rows.append((ep, coverage, steps, terminated))
        coverages.append(coverage)
        steps_list.append(steps)
        if terminated and not truncated:
            full_coverage_count += 1

    env.close()

    csv_path = _results_dir() / "inference" / f"{config_name}_seed{seed}_eval_{eval_size}x{eval_size}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "coverage", "steps", "terminated"])
        w.writerows(rows)

    return {
        "config_name": config_name,
        "seed": seed,
        "eval_size": eval_size,
        "full_coverage_rate": full_coverage_count / n_episodes,
        "avg_coverage": float(np.mean(coverages)),
        "std_coverage": float(np.std(coverages)),
        "avg_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "csv_path": str(csv_path),
    }
