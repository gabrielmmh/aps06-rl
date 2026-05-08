"""MaskablePPO with KL anchor toward a frozen BC reference policy.

Why: the bundle (Epic 7) showed PPO drift erases BC initialization on 20x20.
DAPG (Rajeswaran et al. 2018), AWAC (Nair et al. 2020), and Zhao et al. 2022
all add an auxiliary loss that pulls the trained policy back toward the BC
reference, preventing the drift while still allowing RL refinement.

Loss = PPO_loss + lambda_bc * KL(pi || pi_BC_frozen)

lambda_bc is annealed from a high initial value (close to BC) to a low value
(mostly RL refining) over the cumulative training timesteps. Schedule applied
across the entire curriculum (5x5 -> 10x10 -> 20x20) using `num_timesteps`,
which carries over phases when `reset_num_timesteps=False`.

The override points the same MaskablePPO.train() with one inserted block
that adds the KL term right before the optimizer step. Other PPO mechanics
(advantage normalisation, GAE, clip range, value loss, entropy bonus) are
untouched.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class MaskablePPOWithKLAnchor(MaskablePPO):
    """MaskablePPO with an auxiliary KL-to-frozen-BC loss term."""

    def __init__(
        self,
        *args,
        bc_policy_path: Optional[str] = None,
        kl_lambda_schedule: Optional[Callable[[int], float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._bc_policy = None
        self._bc_policy_path = bc_policy_path
        self.kl_lambda_schedule = kl_lambda_schedule or (lambda step: 0.0)
        if bc_policy_path is not None:
            self._load_bc_policy(bc_policy_path)

    def _load_bc_policy(self, path: str) -> None:
        bc_model = MaskablePPO.load(path, device=self.device)
        bc_policy = bc_model.policy
        for p in bc_policy.parameters():
            p.requires_grad = False
        bc_policy.set_training_mode(False)
        self._bc_policy = bc_policy

    def train(self) -> None:
        # Verbatim from MaskablePPO.train (sb3-contrib 2.8) with one inserted
        # block (annotated below) that adds lambda_bc * KL(pi || pi_BC) to loss.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        kl_to_bc_values: list[float] = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # ---- BC KL anchor (added) ----
                if self._bc_policy is not None:
                    with th.no_grad():
                        _, bc_log_prob, _ = self._bc_policy.evaluate_actions(
                            rollout_data.observations,
                            actions,
                            action_masks=rollout_data.action_masks,
                        )
                    # KL(pi || pi_BC) approx via sample (Schulman): log_prob - bc_log_prob
                    kl_to_bc = th.mean(log_prob - bc_log_prob)
                    lambda_bc = float(self.kl_lambda_schedule(int(self.num_timesteps)))
                    loss = loss + lambda_bc * kl_to_bc
                    kl_to_bc_values.append(kl_to_bc.item())
                # ---- end BC KL anchor ----

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self._bc_policy is not None and kl_to_bc_values:
            self.logger.record("train/kl_to_bc", float(np.mean(kl_to_bc_values)))
            self.logger.record("train/lambda_bc", float(self.kl_lambda_schedule(int(self.num_timesteps))))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def make_kl_lambda_schedule(
    initial: float = 1.0,
    final: float = 0.05,
    decay_over_timesteps: int = 3_000_000,
) -> Callable[[int], float]:
    """Linear decay from `initial` to `final` over `decay_over_timesteps`.

    Uses cumulative timesteps so the schedule is correct across curriculum
    phases (when `reset_num_timesteps=False` is passed to model.learn()).
    """

    def schedule(num_timesteps: int) -> float:
        progress = min(1.0, max(0.0, num_timesteps / decay_over_timesteps))
        return final + (initial - final) * (1.0 - progress)

    return schedule
