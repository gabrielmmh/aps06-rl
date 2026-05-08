"""Coverage Path Planning environment v4 — reward redesign + structured memory + frontier features.

Inherits the V3 env (5x5 enriched obs + action_masks + reward redesign with terminal +60,
truncation 0, step penalty gated on coverage>=0.80) and adds three pieces designed to
break the 20x20 closing-cell credit-assignment ceiling:

1. **Structured memory `visited_pooled` (2, 8, 8)**: max-pooled mask of cells the
   agent has personally visited + one-hot of current agent position, both at fixed
   resolution F=8 independent of the grid. CNN-friendly substitute for LSTM hidden
   state. Partial-observability safe — only encodes self.visited (subset of what
   `coverage_ratio` already exposes), never obstacle locations beyond what was seen.

2. **Frontier feature (3,)**: BFS over the agent's known terrain (visited ∪ NOT
   seen_obstacles) to find the nearest unvisited cell. Returns (dx, dy) direction
   normalized + BFS distance normalized by diameter. Always informative even when
   the 5x5 window has no unvisited cells. Optimism under uncertainty: never-observed
   cells treated as potentially traversable.

3. **`progress` feature (1,)**: count_steps / max_steps. Provides explicit budget
   info so the policy can balance "explore more" vs "close what's left".

Partial observability is preserved throughout: visited_pooled only encodes the
agent's trajectory, frontier BFS only uses cells the agent has actually observed,
and progress is meta-information about the agent itself.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np

from gymnasium_env.grid_world_cpp_v3 import GridWorldCPPV3Env


POOLED_RESOLUTION = 8
FRONTIER_FEATURE_DIM = 3
PROGRESS_FEATURE_DIM = 1


class GridWorldCPPV4Env(GridWorldCPPV3Env):
    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        super().__init__(render_mode=render_mode, size=size, obs_quantity=obs_quantity, max_steps=max_steps)

        # Track obstacles the agent has personally observed via its 5x5 window.
        # Used by the frontier BFS — never expose obstacles the agent hasn't seen.
        self._seen_obstacles: set[tuple[int, int]] = set()

        # Cache the diameter for normalizing the BFS distance feature.
        self._diameter: float = float(np.sqrt(2.0) * size)

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
            "visited_pooled": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2, POOLED_RESOLUTION, POOLED_RESOLUTION),
                dtype=np.float32,
            ),
            "frontier": gym.spaces.Box(
                low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "progress": gym.spaces.Box(
                low=np.zeros((PROGRESS_FEATURE_DIM,), dtype=np.float32),
                high=np.ones((PROGRESS_FEATURE_DIM,), dtype=np.float32),
                dtype=np.float32,
            ),
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._seen_obstacles = set()
        self._diameter = float(np.sqrt(2.0) * self.size)
        self._update_seen_obstacles_from_window()
        # Re-emit obs after seeing the initial window.
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._update_seen_obstacles_from_window()
        return self._get_obs(), reward, terminated, truncated, info

    def _update_seen_obstacles_from_window(self) -> None:
        """Record any obstacle/wall cell that's currently inside the agent's 5x5 window."""
        radius = self.NEIGHBOR_RADIUS
        ax = int(self._agent_location[0])
        ay = int(self._agent_location[1])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = ax + dx, ay + dy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if any(np.array_equal(np.array([nx, ny]), loc) for loc in self.obstacles_locations):
                    self._seen_obstacles.add((nx, ny))

    def _compute_visited_pooled(self) -> np.ndarray:
        """(2, F, F) — channel 0: max-pool of self.visited; channel 1: agent position one-hot."""
        F = POOLED_RESOLUTION
        out = np.zeros((2, F, F), dtype=np.float32)
        # Max-pool visited mask: each F×F bucket gets value 1 if any real cell mapped to it was visited.
        for x, y in self.visited:
            bx = min(F - 1, int(x * F / self.size))
            by = min(F - 1, int(y * F / self.size))
            out[0, by, bx] = 1.0
        # Agent position one-hot in pooled grid.
        ax = int(self._agent_location[0])
        ay = int(self._agent_location[1])
        bx = min(F - 1, int(ax * F / self.size))
        by = min(F - 1, int(ay * F / self.size))
        out[1, by, bx] = 1.0
        return out

    def _bfs_frontier(self) -> tuple[float, float, float]:
        """BFS over known terrain (visited ∪ NOT seen_obstacles) from agent.

        Returns (dx_normalized, dy_normalized, distance_normalized) toward the nearest
        unvisited cell that's reachable on the known-terrain graph. Cells the agent
        hasn't observed yet are treated as potentially-free (optimism under uncertainty).
        Returns (0.0, 0.0, 1.0) when the agent has covered everything reachable.
        """
        ax = int(self._agent_location[0])
        ay = int(self._agent_location[1])
        start = (ax, ay)

        # If everything is visited, no frontier to head to.
        if len(self.visited) >= self.total_free_cells:
            return 0.0, 0.0, 1.0

        # BFS with parent tracking so we can recover direction of first step.
        visited_bfs = {start: None}
        queue: deque[tuple[int, int]] = deque([start])
        target: Optional[tuple[int, int]] = None
        target_dist = -1
        # Distance map for the target, computed from parents.
        while queue:
            cell = queue.popleft()
            cx, cy = cell
            # Frontier check: cell is free + unvisited.
            if cell != start and cell not in self.visited and cell not in self._seen_obstacles:
                target = cell
                # Compute distance by walking parents back to start.
                d = 0
                cur = cell
                while visited_bfs[cur] is not None:
                    cur = visited_bfs[cur]
                    d += 1
                target_dist = d
                break
            for ddx, ddy in ((1, 0), (0, -1), (-1, 0), (0, 1)):
                nx, ny = cx + ddx, cy + ddy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if (nx, ny) in self._seen_obstacles:
                    continue
                if (nx, ny) in visited_bfs:
                    continue
                visited_bfs[(nx, ny)] = cell
                queue.append((nx, ny))

        if target is None:
            return 0.0, 0.0, 1.0

        # Direction vector from agent to target (raw, then normalized).
        dx = float(target[0] - ax)
        dy = float(target[1] - ay)
        norm = max(abs(dx), abs(dy), 1.0)
        dx_n = dx / norm
        dy_n = dy / norm
        d_n = min(1.0, float(target_dist) / max(1.0, self._diameter))
        return dx_n, dy_n, d_n

    def bfs_frontier_distance(self) -> float:
        """Public helper for the PBRS wrapper: raw BFS distance to nearest frontier (in cells).

        Returns float('inf') when no frontier is reachable.
        """
        ax = int(self._agent_location[0])
        ay = int(self._agent_location[1])
        start = (ax, ay)
        if len(self.visited) >= self.total_free_cells:
            return 0.0

        visited_bfs = {start}
        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        while queue:
            cell, depth = queue.popleft()
            cx, cy = cell
            if cell != start and cell not in self.visited and cell not in self._seen_obstacles:
                return float(depth)
            for ddx, ddy in ((1, 0), (0, -1), (-1, 0), (0, 1)):
                nx, ny = cx + ddx, cy + ddy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if (nx, ny) in self._seen_obstacles:
                    continue
                if (nx, ny) in visited_bfs:
                    continue
                visited_bfs.add((nx, ny))
                queue.append(((nx, ny), depth + 1))
        return float("inf")

    def _get_obs(self):
        # Defensive: at very early init (before reset) some attributes don't exist yet.
        if not hasattr(self, "_seen_obstacles"):
            self._seen_obstacles = set()
        if not hasattr(self, "_diameter"):
            self._diameter = float(np.sqrt(2.0) * self.size)

        base = super()._get_obs()
        dx_n, dy_n, d_n = self._bfs_frontier()
        base["visited_pooled"] = self._compute_visited_pooled()
        base["frontier"] = np.array([dx_n, dy_n, d_n], dtype=np.float32)
        base["progress"] = np.array([self.count_steps / max(1, self.max_steps)], dtype=np.float32)
        return base
