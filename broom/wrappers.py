"""Reward shaping wrappers for training-time only.

Evaluation must always use the upstream reward (no shaping) so cross-config
comparisons remain valid. These wrappers are added on top of the env during
training and removed at eval time.
"""

import gymnasium as gym


class PBRSCoverageWrapper(gym.Wrapper):
    """Potential-based reward shaping with phi(s) = coverage_ratio(s).

    Adds F = gamma * phi(s') - phi(s) to the per-step reward during training.
    By the Ng/Harada/Russell 1999 theorem this is policy-invariant: the
    optimal policy is unchanged. The reward signal is denser, especially
    near 100% coverage where phi has the steepest gradient.

    The original (unshaped) reward is preserved in `info["r_eval"]` so the
    train loop can log it and the eval loop can avoid double-counting.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.999, weight: float = 1.0):
        super().__init__(env)
        self.gamma = gamma
        self.weight = weight
        self._prev_phi = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = float(self.env.unwrapped.coverage_ratio)
        return obs, info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        phi = float(self.env.unwrapped.coverage_ratio)
        f = self.gamma * phi - self._prev_phi
        self._prev_phi = phi
        info["r_eval"] = r_env
        return obs, r_env + self.weight * f, terminated, truncated, info


class PBRSFrontierDistanceWrapper(gym.Wrapper):
    """Potential-based shaping with phi(s) = -bfs_distance_to_frontier / diameter.

    The previous PBRS wrapper used phi = coverage_ratio, whose per-step magnitude
    is ~1/N_free (≈ 0.001 on 20x20). That is drowned by the upstream step
    penalty (-0.1) and effectively invisible to PPO advantage estimation.

    This wrapper uses the BFS distance to the nearest frontier cell on the
    agent's known terrain (the same BFS that powers the V4 env's frontier
    feature), normalized by the grid diameter. Each step toward the frontier
    drops d by 1, giving F ≈ +1/diameter ≈ +0.05 on 20x20 — comparable in
    magnitude to the +1 new-cell reward, dense across the entire trajectory,
    and theoretically policy-invariant (Ng-Harada-Russell 1999).

    Optimism under uncertainty: the BFS treats never-observed cells as
    potentially-free, so partial observability is preserved (only the agent's
    seen obstacles block the BFS).

    The unshaped reward is preserved in info["r_eval"].
    """

    def __init__(self, env: gym.Env, gamma: float = 0.995, weight: float = 1.0):
        super().__init__(env)
        self.gamma = gamma
        self.weight = weight
        self._prev_phi = 0.0

    def _phi(self) -> float:
        base = self.env.unwrapped
        d = base.bfs_frontier_distance()
        if d == float("inf") or d <= 0.0:
            # No frontier reachable (covered everything reachable) -> phi = 0
            return 0.0
        diameter = base._diameter
        return -float(d) / max(1.0, diameter)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        phi = self._phi()
        # Ng-Harada-Russell: terminal Φ should be 0 for episodic tasks.
        if terminated or truncated:
            phi = 0.0
        f = self.gamma * phi - self._prev_phi
        self._prev_phi = phi
        info["r_eval"] = r_env
        return obs, r_env + self.weight * f, terminated, truncated, info
