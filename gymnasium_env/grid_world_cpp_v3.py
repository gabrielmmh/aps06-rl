"""Coverage Path Planning environment v3 — for MaskablePPO + reward redesign.

Inherits the enriched env (5x5 window + direction/distance features that already
got us to 77% on 10x10 native) and changes two things:

  * Reward redesign: the +10 terminal bonus and -5 truncation penalty in the
    upstream reward make "stop with the wall" trajectories competitive with
    closing the cover. Theile et al. (arXiv 2309.03157) calibration argues B
    must dominate the cumulative step penalty over max_steps; with gamma=0.999
    and a closing horizon of ~100 steps, B >= (0.1*500 + 5)/0.95 ~= 60.
    We also gate the per-step penalty on coverage_ratio < 0.80 to relax time
    pressure in the closing regime, where the agent has to wander a bit to
    find the last 1-3 cells.

  * action_masks(): returns a bool array of legal moves (no walls/obstacles).
    Used by sb3-contrib's MaskablePPO; per Huang & Ontanon (arXiv 2006.14171)
    masking outperforms invalid-action penalties, especially as grid size grows.
"""

import gymnasium as gym
import numpy as np

from gymnasium_env.grid_world_cpp_enriched import GridWorldCPPEnrichedEnv


COVERAGE_THRESHOLD_FOR_STEP_PENALTY = 0.80
TERMINAL_FULL_COVERAGE_BONUS = 60.0


class GridWorldCPPV3Env(GridWorldCPPEnrichedEnv):
    def step(self, action):
        direction = self._action_to_direction[int(action)]
        old_location = self._agent_location.copy()

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if any(np.array_equal(self._agent_location, loc) for loc in self.obstacles_locations):
            self._agent_location = old_location

        self.set_neighbors(self.obstacles_locations)
        self.count_steps += 1

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        # Coverage-gated step penalty: once the agent is in the closing regime
        # the time pressure is relaxed so it can wander to find the last cells.
        step_penalty = -0.1 if self.coverage_ratio < COVERAGE_THRESHOLD_FOR_STEP_PENALTY else 0.0
        reward = step_penalty

        if stayed_in_place:
            reward -= 0.5
        elif is_new_cell:
            reward += 1.0
            self.visited.add(current_pos)
        else:
            reward -= 0.3

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage

        if full_coverage:
            reward += TERMINAL_FULL_COVERAGE_BONUS

        # No truncation penalty: a -5 truncation reward actively teaches the
        # agent to "exit" a failed episode rather than persist to closure.
        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
        else:
            truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Bool array of legal moves: True = won't hit wall/obstacle."""
        masks = np.zeros(4, dtype=bool)
        for a in range(4):
            direction = self._action_to_direction[a]
            target = self._agent_location + direction
            in_bounds = (0 <= target[0] < self.size) and (0 <= target[1] < self.size)
            not_obstacle = not any(np.array_equal(target, loc) for loc in self.obstacles_locations)
            masks[a] = in_bounds and not_obstacle
        if not masks.any():
            masks[:] = True
        return masks
