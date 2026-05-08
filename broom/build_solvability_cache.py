"""Build a JSON cache of (eval_size, seed, ep) -> solvable bool.

Reads no model output. Just regenerates the random map for each evaluation
seed and runs BFS reachability check from the agent's start position.

Used to retro-fit the "filtered" metric onto inference CSVs that were
generated before solvability was tracked per-episode.

Usage:
    python -m broom.build_solvability_cache
    -> writes results/solvability_cache.json
"""

import json
from pathlib import Path

import gymnasium as gym

from broom.configs import PHASE_MAX_STEPS, PHASE_OBSTACLES
from broom.inference import _register_envs
from broom.solvability import count_reachable_cells


def main():
    _register_envs()
    cache: dict[str, bool] = {}

    for eval_size in (5, 10, 20):
        env = gym.make(
            "gymnasium_env/GridWorldCPP-v0",
            size=eval_size,
            obs_quantity=PHASE_OBSTACLES[eval_size],
            max_steps=PHASE_MAX_STEPS[eval_size],
        )
        for seed in (0, 1, 2):
            for ep in range(100):
                env.reset(seed=seed * 1000 + ep)
                solvable = count_reachable_cells(env.unwrapped) >= env.unwrapped.total_free_cells
                key = f"{eval_size}_{seed}_{ep}"
                cache[key] = bool(solvable)
        env.close()

    out_path = Path("results/solvability_cache.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=0))
    print(f"wrote {out_path} with {len(cache)} entries")
    # Sanity report
    by_size = {5: 0, 10: 0, 20: 0}
    counts = {5: 0, 10: 0, 20: 0}
    for k, v in cache.items():
        sz = int(k.split("_")[0])
        counts[sz] += 1
        if v:
            by_size[sz] += 1
    for sz in (5, 10, 20):
        print(f"  {sz}x{sz}: {by_size[sz]}/{counts[sz]} solvable ({100*by_size[sz]/counts[sz]:.1f}%)")


if __name__ == "__main__":
    main()
