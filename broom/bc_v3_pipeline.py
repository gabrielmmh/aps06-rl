"""End-to-end BC pre-flight for maskable_bc_kl: collect, train, evaluate."""

import time

import gymnasium as gym

from broom.bc_v3 import collect_expert_trajectories, train_bc_v3
from gymnasium_env.grid_world_cpp_v3 import GridWorldCPPV3Env


def evaluate_bc_v3(model_path: str, eval_size: int, n_episodes: int = 50) -> dict:
    from sb3_contrib import MaskablePPO

    if "gymnasium_env/GridWorldCPPV3-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPV3-v0", entry_point=GridWorldCPPV3Env)

    obstacles = {5: 3, 10: 12, 20: 48}[eval_size]
    max_steps = {5: 200, 10: 500, 20: 1000}[eval_size]

    env = gym.make(
        "gymnasium_env/GridWorldCPPV3-v0",
        size=eval_size, obs_quantity=obstacles, max_steps=max_steps,
    )
    model = MaskablePPO.load(model_path, env=env)

    full = 0
    cov_sum = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        term = trunc = False
        while not (term or trunc):
            masks = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=False, action_masks=masks)
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
    print("Phase 1: collect expert trajectories on V3 env")
    print("=" * 60)
    t0 = time.time()
    samples = collect_expert_trajectories(grid_sizes=(5, 10, 20), n_episodes_per_size=100)
    print(f"  total {len(samples)} samples in {time.time()-t0:.1f}s")

    print()
    print("=" * 60)
    print("Phase 2: BC train (MaskablePPO architecture)")
    print("=" * 60)
    t0 = time.time()
    save_path = "results/models/bc_warmstart_v3.zip"
    train_bc_v3(samples, save_path=save_path, n_epochs=10, batch_size=512, lr=3e-4, smoke_env_size=20)
    print(f"  BC training done in {time.time()-t0:.1f}s")

    print()
    print("=" * 60)
    print("Phase 3: evaluate BC alone (sanity check)")
    print("=" * 60)
    for sz in (5, 10, 20):
        m = evaluate_bc_v3(save_path, eval_size=sz, n_episodes=50)
        print(f"  bc_v3-only {sz}x{sz}: full_coverage={m['full_coverage_rate']:.1%} avg={m['avg_coverage']:.1%}")


if __name__ == "__main__":
    main()
