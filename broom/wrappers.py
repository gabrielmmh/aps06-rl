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
