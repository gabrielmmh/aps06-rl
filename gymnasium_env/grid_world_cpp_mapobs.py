"""Coverage Path Planning environment with an egocentric accumulated map observation.

This is the third env subclass in the project (after `GridWorldCPPEnv` and
`GridWorldCPPEnrichedEnv`). It exposes the agent's **internal accumulated map**
of what it has observed so far, in agent-centered coordinates, so that a CNN
policy can learn to plan over a 2D representation rather than compress the
history into an LSTM hidden state.

Partial observability is preserved by construction: the internal map is
filled cell-by-cell only from the 5x5 windows the agent has actually
observed during the episode. Cells the agent has never observed remain 0
in every channel of `ego_map`.

Observation (`Dict`):
- `agent`: Box(3,)            — (x/N, y/N, coverage_ratio), same as upstream
- `ego_map`: Box(0, 1, shape=(3, K, K))  — agent-centered binary map,
   K = 2 * MAX_GRID - 1 = 39, where channels are:
     0: visited cells
     1: known walls / obstacles / out-of-bounds inside the explored region
     2: known free unvisited cells (frontiers + interior unvisited that has
        been seen but not stepped on)

Note on the fixed K=39: it is sized for the largest grid we train on (20x20),
so the same observation shape is shared by 5x5, 10x10, and 20x20. Smaller grids
have most of the K x K window outside the world, which is treated as
"never observed" (zeros) — the CNN learns to ignore that padded region.
"""

from typing import Optional

import gymnasium as gym
import numpy as np

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


MAX_GRID = 20  # largest grid we train/eval on
K = 2 * MAX_GRID - 1  # 39: covers any agent-centered window for grids up to 20x20
OBS_RADIUS = 2  # 5x5 window observed at each step


class GridWorldCPPMapObsEnv(GridWorldCPPEnv):
    def __init__(
        self,
        render_mode=None,
        size: int = 5,
        obs_quantity: int = 3,
        max_steps: int = 200,
    ):
        super().__init__(
            render_mode=render_mode,
            size=size,
            obs_quantity=obs_quantity,
            max_steps=max_steps,
        )
        self._known_world_map: np.ndarray = np.zeros((3, size, size), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "ego_map": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, K, K),
                dtype=np.float32,
            ),
        })

    def _observe_window_into_map(self):
        """Fill self._known_world_map from the 5x5 cells around the agent.

        Channel 0: visited cells inside the window.
        Channel 1: walls / obstacles inside the window.
        Channel 2: known free unvisited inside the window.
        """
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        for di in range(-OBS_RADIUS, OBS_RADIUS + 1):
            for dj in range(-OBS_RADIUS, OBS_RADIUS + 1):
                wx, wy = ax + dj, ay + di
                if not (0 <= wx < self.size and 0 <= wy < self.size):
                    continue
                pos = np.array([wx, wy])
                is_obstacle = any(np.array_equal(pos, loc) for loc in self.obstacles_locations)
                is_visited = (wx, wy) in self.visited
                if is_obstacle:
                    self._known_world_map[1, wy, wx] = 1.0
                    self._known_world_map[0, wy, wx] = 0.0
                    self._known_world_map[2, wy, wx] = 0.0
                elif is_visited:
                    self._known_world_map[0, wy, wx] = 1.0
                    self._known_world_map[2, wy, wx] = 0.0
                else:
                    self._known_world_map[2, wy, wx] = 1.0

    def _ego_map_view(self) -> np.ndarray:
        """Return a (3, K, K) array centered on the agent.

        World cell (wx, wy) maps to ego index (wy - ay + center, wx - ax + center).
        Cells outside the world are 0 (never observed).
        """
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        center = K // 2
        out = np.zeros((3, K, K), dtype=np.float32)
        for wy in range(self.size):
            for wx in range(self.size):
                ei = wy - ay + center
                ej = wx - ax + center
                if 0 <= ei < K and 0 <= ej < K:
                    out[:, ei, ej] = self._known_world_map[:, wy, wx]
        return out

    def _get_obs(self):
        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
            ], dtype=np.float32),
            "ego_map": self._ego_map_view(),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._known_world_map = np.zeros((3, self.size, self.size), dtype=np.float32)
        self._observe_window_into_map()
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._observe_window_into_map()
        return self._get_obs(), reward, terminated, truncated, info
