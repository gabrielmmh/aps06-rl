"""Coverage Path Planning environment with enriched observation.

Differences vs. upstream `GridWorldCPPEnv`:
  * neighbors window is 5x5 (radius 2) instead of 3x3
  * adds `direction_to_nearest_unvisited` (one-hot 4-dim over right/up/left/down)
  * adds `distance_to_nearest_unvisited` (Chebyshev distance, normalized in [0, 1])

The new features are computed only from the visible 5x5 window - they don't
expose the full map, preserving partial observability.
"""

import gymnasium as gym
import numpy as np

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


class GridWorldCPPEnrichedEnv(GridWorldCPPEnv):
    NEIGHBOR_RADIUS = 2

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        super().__init__(render_mode=render_mode, size=size, obs_quantity=obs_quantity, max_steps=max_steps)

        side = 2 * self.NEIGHBOR_RADIUS + 1

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((side, side), dtype=np.float32),
                high=np.full((side, side), 2.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "direction_to_nearest_unvisited": gym.spaces.Box(
                low=np.zeros((4,), dtype=np.float32),
                high=np.ones((4,), dtype=np.float32),
                dtype=np.float32,
            ),
            "distance_to_nearest_unvisited": gym.spaces.Box(
                low=np.zeros((1,), dtype=np.float32),
                high=np.ones((1,), dtype=np.float32),
                dtype=np.float32,
            ),
        })

        self._neighbors = np.zeros((side, side), dtype=int)

    def set_neighbors(self, obstacles_locations):
        radius = self.NEIGHBOR_RADIUS
        side = 2 * radius + 1
        matrix = np.zeros((side, side), dtype=int)
        for i in range(side):
            for j in range(side):
                nx = self._agent_location[0] + (j - radius)
                ny = self._agent_location[1] + (i - radius)
                neighbor = np.array([nx, ny])
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    matrix[i][j] = 1
                elif any(np.array_equal(neighbor, loc) for loc in obstacles_locations):
                    matrix[i][j] = 1
                elif (nx, ny) in self.visited:
                    matrix[i][j] = 2
        self._neighbors = matrix

    def _direction_and_distance_to_nearest_unvisited(self) -> tuple[np.ndarray, float]:
        radius = self.NEIGHBOR_RADIUS
        side = 2 * radius + 1
        best = None
        best_dist = None
        for i in range(side):
            for j in range(side):
                if self._neighbors[i][j] != 0:
                    continue
                if i == radius and j == radius:
                    continue
                dx = j - radius
                dy = i - radius
                dist = max(abs(dx), abs(dy))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = (dx, dy)

        direction = np.zeros((4,), dtype=np.float32)
        if best is None:
            return direction, 1.0

        dx, dy = best
        if abs(dx) >= abs(dy):
            if dx > 0:
                direction[0] = 1.0
            elif dx < 0:
                direction[2] = 1.0
        else:
            if dy < 0:
                direction[1] = 1.0
            elif dy > 0:
                direction[3] = 1.0

        normalized_dist = best_dist / radius
        return direction, float(normalized_dist)

    def _get_obs(self):
        direction, dist = self._direction_and_distance_to_nearest_unvisited()
        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
            ], dtype=np.float32),
            "neighbors": self._neighbors.astype(np.float32),
            "direction_to_nearest_unvisited": direction,
            "distance_to_nearest_unvisited": np.array([dist], dtype=np.float32),
        }
