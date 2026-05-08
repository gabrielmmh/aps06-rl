"""Inference-time mixture evaluation: PPO model + scripted FrontierAgent.

At each step, with probability `p_model` use the trained model's action,
with probability `(1 - p_model)` use the FrontierAgent's action. Lets us
test residual-policy-style hybrids without retraining.

Usage:
    python -u -m broom.eval_mixture \\
        --model results/models/maskable_v3_seed0_10x10.zip \\
        --config maskable_v3 \\
        --eval-size 20 \\
        --seeds 0,1,2 \\
        --p-models 0.0,0.1,0.5,1.0 \\
        --n-episodes 100
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from broom.baselines.frontier import FrontierAgent
from broom.configs import PHASE_MAX_STEPS, PHASE_OBSTACLES
from broom.inference import _env_id_for_config, _load_model, _register_envs
from broom.solvability import count_reachable_cells


def _extract_3x3(base_env) -> np.ndarray:
    ax = int(base_env._agent_location[0])
    ay = int(base_env._agent_location[1])
    matrix = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            nx = ax + (j - 1)
            ny = ay + (i - 1)
            if not (0 <= nx < base_env.size and 0 <= ny < base_env.size):
                matrix[i][j] = 1
            elif any(np.array_equal(np.array([nx, ny]), loc) for loc in base_env.obstacles_locations):
                matrix[i][j] = 1
            elif (nx, ny) in base_env.visited:
                matrix[i][j] = 2
    return matrix


def evaluate_mixture(model, config_name: str, eval_size: int, seed: int, p_model: float,
                    n_episodes: int = 100, rng_seed: int = 42) -> dict:
    env_id = _env_id_for_config(config_name)
    env = gym.make(env_id, size=eval_size, obs_quantity=PHASE_OBSTACLES[eval_size],
                   max_steps=PHASE_MAX_STEPS[eval_size])

    rng = np.random.default_rng(rng_seed)
    full = 0
    full_solv = 0
    n_solv = 0
    cov_sum = 0.0
    cov_solv_sum = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        base_env = env.unwrapped
        reachable = count_reachable_cells(base_env)
        solvable = reachable >= base_env.total_free_cells
        agent = FrontierAgent(size=eval_size)
        agent.reset()
        term = trunc = False
        while not (term or trunc):
            use_model = rng.random() < p_model
            if use_model:
                masks = base_env.action_masks() if hasattr(base_env, "action_masks") else None
                if masks is not None:
                    action, _ = model.predict(obs, deterministic=False, action_masks=masks)
                else:
                    action, _ = model.predict(obs, deterministic=False)
                action = int(action)
            else:
                ax = int(base_env._agent_location[0])
                ay = int(base_env._agent_location[1])
                neighbors_3x3 = _extract_3x3(base_env)
                action = agent.act((ax, ay), neighbors_3x3)
            obs, _, term, trunc, info = env.step(action)
        cov = info["coverage"]
        cov_sum += cov
        if term and not trunc:
            full += 1
            if solvable:
                full_solv += 1
        if solvable:
            n_solv += 1
            cov_solv_sum += cov

    env.close()
    return {
        "p_model": p_model,
        "seed": seed,
        "eval_size": eval_size,
        "full_raw": full / n_episodes,
        "full_solvable": full_solv / n_solv if n_solv else 0.0,
        "avg_raw": cov_sum / n_episodes,
        "avg_solvable": cov_solv_sum / n_solv if n_solv else 0.0,
        "n_solvable": n_solv,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--config", required=True, help="Config name (e.g. maskable_v3)")
    parser.add_argument("--eval-size", type=int, required=True)
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds")
    parser.add_argument("--p-models", default="0.0,0.1,0.5,1.0", help="Comma-separated p_model values")
    parser.add_argument("--n-episodes", type=int, default=100)
    args = parser.parse_args()

    _register_envs()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    p_models = [float(p) for p in args.p_models.split(",") if p.strip()]

    # Load once with the size-appropriate env
    env_id = _env_id_for_config(args.config)
    env = gym.make(env_id, size=args.eval_size, obs_quantity=PHASE_OBSTACLES[args.eval_size],
                   max_steps=PHASE_MAX_STEPS[args.eval_size])
    model = _load_model(args.model, args.config, env)

    print(f"Mixture eval | model={args.model} | eval_size={args.eval_size}", flush=True)
    print(f"{'p_model':>7s} | {'seed':>4s} | {'full_raw':>8s} | {'full_solv':>9s} | {'avg_raw':>7s} | {'avg_solv':>8s}", flush=True)
    print("-" * 65, flush=True)
    for p in p_models:
        for seed in seeds:
            m = evaluate_mixture(
                model, config_name=args.config, eval_size=args.eval_size, seed=seed,
                p_model=p, n_episodes=args.n_episodes,
            )
            print(f"{p:7.2f} | {seed:4d} | {m['full_raw']*100:7.1f}% | {m['full_solvable']*100:8.1f}% | "
                  f"{m['avg_raw']*100:6.1f}% | {m['avg_solvable']*100:7.1f}%", flush=True)


if __name__ == "__main__":
    main()
