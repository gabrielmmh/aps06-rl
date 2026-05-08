"""Behavioral cloning warm-start for maskable_bc_kl on the V3 env.

Same shape as Epic 7's `broom/bc.py` but adapted for:
  * V3 env: enriched obs (5x5 + direction + distance) + action_masks
  * MaskablePPO architecture: trained model is MaskablePPO (state_dict
    compatible at the policy level so the KL anchor can load it cleanly).

Pipeline:
  1. Roll out FrontierAgent across 5x5/10x10/20x20 grids on the V3 env.
     Record (obs, action, action_masks) per step.
  2. Train a MaskablePPO MultiInputPolicy via cross-entropy on the (obs, action)
     pairs. Action masks are passed at the forward pass but not used in the
     loss (they only zero out illegal logits in the distribution).
  3. Save the MaskablePPO checkpoint at `results/models/bc_warmstart_v3.zip`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch

from broom.baselines.frontier import FrontierAgent
from gymnasium_env.grid_world_cpp_v3 import GridWorldCPPV3Env


def _extract_3x3_neighbors_from_env(env_unwrapped) -> np.ndarray:
    """3x3 window around the agent in the FrontierAgent encoding.

    Same as Epic 7's helper but separated here for clarity. 0=free, 1=wall/oob,
    2=visited. Center cell [1][1] is the agent's position.
    """
    ax, ay = int(env_unwrapped._agent_location[0]), int(env_unwrapped._agent_location[1])
    matrix = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            nx = ax + (j - 1)
            ny = ay + (i - 1)
            if not (0 <= nx < env_unwrapped.size and 0 <= ny < env_unwrapped.size):
                matrix[i][j] = 1
            elif any(np.array_equal(np.array([nx, ny]), loc) for loc in env_unwrapped.obstacles_locations):
                matrix[i][j] = 1
            elif (nx, ny) in env_unwrapped.visited:
                matrix[i][j] = 2
    return matrix


def collect_expert_trajectories(
    grid_sizes: Iterable[int] = (5, 10, 20),
    obstacles_per_size: dict[int, int] = None,
    max_steps_per_size: dict[int, int] = None,
    n_episodes_per_size: int = 100,
    seed_offset: int = 0,
) -> list[tuple[dict, int]]:
    """Roll out FrontierAgent on the V3 env across sizes; collect (obs, action) pairs."""
    if obstacles_per_size is None:
        obstacles_per_size = {5: 3, 10: 12, 20: 48}
    if max_steps_per_size is None:
        max_steps_per_size = {5: 200, 10: 500, 20: 1000}

    if "gymnasium_env/GridWorldCPPV3-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPV3-v0", entry_point=GridWorldCPPV3Env)

    samples: list[tuple[dict, int]] = []
    for size in grid_sizes:
        env = gym.make(
            "gymnasium_env/GridWorldCPPV3-v0",
            size=size, obs_quantity=obstacles_per_size[size], max_steps=max_steps_per_size[size],
        )
        for ep in range(n_episodes_per_size):
            obs, _ = env.reset(seed=seed_offset + ep + size * 10000)
            agent = FrontierAgent(size=size)
            agent.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                base_env = env.unwrapped
                ax = int(base_env._agent_location[0])
                ay = int(base_env._agent_location[1])
                neighbors_3x3 = _extract_3x3_neighbors_from_env(base_env)
                action = agent.act((ax, ay), neighbors_3x3)
                samples.append(({k: v.copy() for k, v in obs.items()}, int(action)))
                obs, _, terminated, truncated, _ = env.step(int(action))
        env.close()
    return samples


def _build_maskable_ppo(env: gym.Env, seed: int = 0):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [256, 256]},
    )


def train_bc_v3(
    samples: list[tuple[dict, int]],
    save_path: str,
    n_epochs: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    seed: int = 0,
    smoke_env_size: int = 5,
) -> str:
    """Train MaskablePPO MultiInputPolicy by CE on (obs, action) pairs."""
    if "gymnasium_env/GridWorldCPPV3-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPV3-v0", entry_point=GridWorldCPPV3Env)

    from sb3_contrib.common.wrappers import ActionMasker

    base_env = gym.make(
        "gymnasium_env/GridWorldCPPV3-v0",
        size=smoke_env_size, obs_quantity=3, max_steps=200,
    )
    env = ActionMasker(base_env, lambda e: e.unwrapped.action_masks())
    model = _build_maskable_ppo(env, seed=seed)
    device = next(model.policy.parameters()).device

    n = len(samples)
    print(f"  bc_v3: {n} samples", flush=True)
    keys = list(samples[0][0].keys())
    shapes = {k: samples[0][0][k].shape for k in keys}
    obs_cpu = {k: torch.empty((n,) + shapes[k], dtype=torch.float32) for k in keys}
    actions_cpu = torch.empty(n, dtype=torch.int64)
    while samples:
        obs_d, action = samples.pop()
        i = len(samples)
        for k in keys:
            obs_cpu[k][i] = torch.from_numpy(obs_d[k])
        actions_cpu[i] = action

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    for epoch in range(n_epochs):
        rng.shuffle(indices)
        epoch_loss = 0.0
        epoch_correct = 0
        steps = 0
        for start in range(0, n, batch_size):
            batch_idx = torch.from_numpy(indices[start:start + batch_size])
            batch_obs = {k: v[batch_idx].to(device, non_blocking=True) for k, v in obs_cpu.items()}
            batch_actions = actions_cpu[batch_idx].to(device, non_blocking=True)
            features = model.policy.extract_features(batch_obs)
            if isinstance(features, tuple):
                pi_features, _ = features
            else:
                pi_features = features
            latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
            logits = model.policy.action_net(latent_pi)
            loss = loss_fn(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=-1) == batch_actions).sum().item()
            steps += 1
        acc = epoch_correct / n
        print(f"  bc_v3 epoch {epoch+1}/{n_epochs}: loss={epoch_loss/steps:.4f} acc={acc:.3f}", flush=True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    env.close()
    print(f"  bc_v3 model saved to {save_path}")
    return save_path
