"""End-to-end BC pre-flight: collect expert data, train BC, evaluate.

Usage:
    python -m broom.bc_pipeline
"""

import time

import gymnasium as gym

from broom.bc import collect_expert_trajectories, train_bc
from gymnasium_env.grid_world_cpp_mapobs import GridWorldCPPMapObsEnv


def evaluate_bc(model_path: str, eval_size: int, n_episodes: int = 50) -> dict:
    from stable_baselines3 import PPO

    if "gymnasium_env/GridWorldCPPMapObs-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPMapObs-v0", entry_point=GridWorldCPPMapObsEnv)

    obstacles = {5: 3, 10: 12, 20: 48}[eval_size]
    max_steps = {5: 200, 10: 500, 20: 1000}[eval_size]

    env = gym.make(
        "gymnasium_env/GridWorldCPPMapObs-v0",
        size=eval_size,
        obs_quantity=obstacles,
        max_steps=max_steps,
    )
    model = PPO.load(model_path, env=env)

    full = 0
    cov_sum = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        term = trunc = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(int(action))
        if term and not trunc:
            full += 1
        cov_sum += info["coverage"]
    env.close()
    return {
        "full_coverage_rate": full / n_episodes,
        "avg_coverage": cov_sum / n_episodes,
        "eval_size": eval_size,
        "n_episodes": n_episodes,
    }


def main():
    print("=" * 60)
    print("Phase 1: collect expert trajectories")
    print("=" * 60)
    t0 = time.time()
    # 100 episodes per size yields ~75k samples (5x5 + 10x10 small, 20x20 dominant).
    # 200 was tipping the 8 GB host into OOM during Phase 2 tensorization;
    # frontier coverage is deterministic enough that 75k samples cover the
    # state distribution well.
    samples = collect_expert_trajectories(
        grid_sizes=(5, 10, 20),
        n_episodes_per_size=100,
    )
    print(f"  total {len(samples)} samples in {time.time()-t0:.1f}s")

    print()
    print("=" * 60)
    print("Phase 2: BC train")
    print("=" * 60)
    t0 = time.time()
    save_path = "results/models/bc_warmstart.zip"
    train_bc(
        samples,
        save_path=save_path,
        n_epochs=10,
        batch_size=512,
        lr=3e-4,
        smoke_env_size=20,
    )
    print(f"  BC training done in {time.time()-t0:.1f}s")

    print()
    print("=" * 60)
    print("Phase 3: evaluate BC alone")
    print("=" * 60)
    for sz in (5, 10, 20):
        m = evaluate_bc(save_path, eval_size=sz, n_episodes=50)
        print(
            f"  bc-only {sz}x{sz}: full_coverage={m['full_coverage_rate']:.1%} "
            f"avg={m['avg_coverage']:.1%}"
        )


if __name__ == "__main__":
    main()
