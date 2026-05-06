"""Frontier-based BFS exploration agent.

Maintains an internal map of what it has observed. At each step it:
  1. Updates the internal map with the current 3x3 neighborhood.
  2. Picks the closest known unvisited cell as target (BFS over known free).
  3. Walks toward it (BFS shortest path).
  4. If no known unvisited target exists, takes a step that maximizes how
     much new (unknown) territory it could observe next.

This is a non-learning baseline. Partial observability is preserved: the
internal map is built only from what the agent has seen, never from a
ground-truth oracle.
"""

from typing import Optional

import numpy as np

from broom.baselines.shared import (
    ACTION_RIGHT,
    DELTAS,
    InternalMap,
)


class FrontierAgent:
    def __init__(self, size: int):
        self.map = InternalMap(size)
        self._target: Optional[tuple[int, int]] = None

    def reset(self) -> None:
        self.map = InternalMap(self.map.size)
        self._target = None

    def act(self, agent_pos: tuple[int, int], neighbors: np.ndarray) -> int:
        self.map.update_from_obs(agent_pos, neighbors)
        if (
            self._target is None
            or self._target == agent_pos
            or self.map.is_visited(self._target)
        ):
            self._target = self.map.find_nearest_unvisited(agent_pos)
        if self._target is None:
            return self._exploration_step(agent_pos)
        return self.map.next_action_toward(agent_pos, self._target)

    def _exploration_step(self, agent_pos: tuple[int, int]) -> int:
        """Pick the action that points at the most unknown territory.

        Score per candidate action: number of unknown cells in the 3x3 around
        the candidate cell, plus a bonus if the candidate cell itself is unknown.
        Walls are skipped.
        """
        ax, ay = agent_pos
        best_action = ACTION_RIGHT
        best_score = -1
        for action, (dx, dy) in DELTAS.items():
            nx, ny = ax + dx, ay + dy
            state = self.map.cell_state(nx, ny)
            if state == 3:
                continue
            score = 0
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if self.map.cell_state(nx + j, ny + i) == 0:
                        score += 1
            if state == 0:
                score += 5
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
