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

from broom.configs import PHASE_MAX_STEPS, PHASE_OBSTACLES, ConfigName, GridSize
from broom.solvability import count_reachable_cells
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
    try:
        from gymnasium_env.grid_world_cpp_mapobs import GridWorldCPPMapObsEnv
        if "gymnasium_env/GridWorldCPPMapObs-v0" not in gym.envs.registry:
            gym.register(id="gymnasium_env/GridWorldCPPMapObs-v0", entry_point=GridWorldCPPMapObsEnv)
    except ImportError:
        pass
    try:
        from gymnasium_env.grid_world_cpp_v3 import GridWorldCPPV3Env
        if "gymnasium_env/GridWorldCPPV3-v0" not in gym.envs.registry:
            gym.register(id="gymnasium_env/GridWorldCPPV3-v0", entry_point=GridWorldCPPV3Env)
    except ImportError:
        pass
    try:
        from gymnasium_env.grid_world_cpp_v4 import GridWorldCPPV4Env
        if "gymnasium_env/GridWorldCPPV4-v0" not in gym.envs.registry:
            gym.register(id="gymnasium_env/GridWorldCPPV4-v0", entry_point=GridWorldCPPV4Env)
    except ImportError:
        pass


def _env_id_for_config(config_name: ConfigName) -> str:
    if config_name == "curriculum_enriched":
        return "gymnasium_env/GridWorldCPPEnriched-v0"
    if config_name == "mapcnn_bc_pbrs":
        return "gymnasium_env/GridWorldCPPMapObs-v0"
    if config_name in ("maskable_v3", "maskable_bc_kl"):
        return "gymnasium_env/GridWorldCPPV3-v0"
    if config_name == "maskable_frontier_pbrs":
        return "gymnasium_env/GridWorldCPPV4-v0"
    return "gymnasium_env/GridWorldCPP-v0"


def _load_model(model_path: str, config_name: ConfigName, env: gym.Env):
    if config_name in ("curriculum_recurrent", "curriculum_recurrent_v2"):
        from sb3_contrib import RecurrentPPO
        # Inference always on CPU: this avoids holding the GPU during the
        # 100-episode evaluation loop, which doesn't benefit from GPU much
        # (one env, no batching).
        return RecurrentPPO.load(model_path, env=env, device="cpu")
    if config_name == "maskable_v3":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path, env=env, device="cpu")
    if config_name == "maskable_bc_kl":
        from sb3_contrib import MaskablePPO
        # At inference we don't need the KL anchor; load as plain MaskablePPO
        # since the saved policy weights are state-dict compatible.
        return MaskablePPO.load(model_path, env=env, device="cpu")
    if config_name == "maskable_frontier_pbrs":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path, env=env, device="cpu")
    return PPO.load(model_path, env=env, device="cpu")


def evaluate(
    model_path: str,
    config_name: ConfigName,
    seed: int,
    eval_size: GridSize,
    train_size: Optional[GridSize] = None,
    n_episodes: int = 100,
) -> dict:
    """Run `n_episodes` of the loaded model on a grid of `eval_size` and report metrics.

    `train_size` is included in the output CSV filename so we don't lose track of
    which model produced the results when the same seed has multiple trained models
    (which happens for the baseline config).
    """
    _register_envs()
    env_id = _env_id_for_config(config_name)
    obstacles = PHASE_OBSTACLES[eval_size]
    max_steps = PHASE_MAX_STEPS[eval_size]

    env = gym.make(env_id, size=eval_size, obs_quantity=obstacles, max_steps=max_steps)
    model = _load_model(model_path, config_name, env)

    rows: list[tuple[int, float, int, bool, bool]] = []
    full_coverage_count = 0
    solvable_count = 0
    full_coverage_count_solvable = 0
    coverages: list[float] = []
    steps_list: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        # Compute whether this map is solvable (all free cells reachable from start)
        # right after reset, before the agent has moved. Stored per-episode so the
        # filtered metric can be recomputed offline.
        reachable = count_reachable_cells(env.unwrapped)
        solvable = reachable >= env.unwrapped.total_free_cells

        lstm_state = None
        episode_starts = np.ones((1,), dtype=bool)
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated):
            if config_name in ("curriculum_recurrent", "curriculum_recurrent_v2"):
                action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_starts, deterministic=False)
                episode_starts = np.zeros((1,), dtype=bool)
            elif config_name in ("maskable_v3", "maskable_bc_kl", "maskable_frontier_pbrs"):
                masks = env.unwrapped.action_masks()
                action, _ = model.predict(obs, deterministic=False, action_masks=masks)
            else:
                action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1

        coverage = info["coverage"]
        rows.append((ep, coverage, steps, terminated, solvable))
        coverages.append(coverage)
        steps_list.append(steps)
        if terminated and not truncated:
            full_coverage_count += 1
            if solvable:
                full_coverage_count_solvable += 1
        if solvable:
            solvable_count += 1

    env.close()

    train_tag = f"_train{train_size}x{train_size}" if train_size is not None else ""
    csv_path = (
        _results_dir() / "inference"
        / f"{config_name}_seed{seed}{train_tag}_eval_{eval_size}x{eval_size}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "coverage", "steps", "terminated", "solvable"])
        w.writerows(rows)

    return {
        "config_name": config_name,
        "seed": seed,
        "eval_size": eval_size,
        "full_coverage_rate": full_coverage_count / n_episodes,
        "full_coverage_rate_solvable": (
            full_coverage_count_solvable / solvable_count if solvable_count > 0 else 0.0
        ),
        "n_solvable": solvable_count,
        "avg_coverage": float(np.mean(coverages)),
        "std_coverage": float(np.std(coverages)),
        "avg_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "csv_path": str(csv_path),
    }


def evaluate_scripted(
    agent,
    algo_name: str,
    seed: int,
    eval_size: GridSize,
    n_episodes: int = 100,
) -> dict:
    """Evaluate a non-learning scripted agent on the upstream env.

    Mirrors `evaluate()` but uses `agent.act((x, y), neighbors)` instead of
    `model.predict(...)`. The agent is reset at the start of each episode.
    """
    _register_envs()
    env_id = "gymnasium_env/GridWorldCPP-v0"
    obstacles = PHASE_OBSTACLES[eval_size]
    max_steps = PHASE_MAX_STEPS[eval_size]

    env = gym.make(env_id, size=eval_size, obs_quantity=obstacles, max_steps=max_steps)

    rows: list[tuple[int, float, int, bool, bool]] = []
    full_coverage_count = 0
    solvable_count = 0
    full_coverage_count_solvable = 0
    coverages: list[float] = []
    steps_list: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        reachable = count_reachable_cells(env.unwrapped)
        solvable = reachable >= env.unwrapped.total_free_cells
        agent.reset()
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated):
            ax = max(0, min(eval_size - 1, int(round(obs["agent"][0] * eval_size))))
            ay = max(0, min(eval_size - 1, int(round(obs["agent"][1] * eval_size))))
            action = agent.act((ax, ay), obs["neighbors"])
            obs, _, terminated, truncated, info = env.step(int(action))
            steps += 1

        coverage = info["coverage"]
        rows.append((ep, coverage, steps, terminated, solvable))
        coverages.append(coverage)
        steps_list.append(steps)
        if terminated and not truncated:
            full_coverage_count += 1
            if solvable:
                full_coverage_count_solvable += 1
        if solvable:
            solvable_count += 1

    env.close()

    csv_path = (
        _results_dir() / "inference"
        / f"scripted_{algo_name}_seed{seed}_eval_{eval_size}x{eval_size}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "coverage", "steps", "terminated", "solvable"])
        w.writerows(rows)

    return {
        "algo_name": algo_name,
        "seed": seed,
        "eval_size": eval_size,
        "full_coverage_rate": full_coverage_count / n_episodes,
        "full_coverage_rate_solvable": (
            full_coverage_count_solvable / solvable_count if solvable_count > 0 else 0.0
        ),
        "n_solvable": solvable_count,
        "avg_coverage": float(np.mean(coverages)),
        "std_coverage": float(np.std(coverages)),
        "avg_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "csv_path": str(csv_path),
    }
