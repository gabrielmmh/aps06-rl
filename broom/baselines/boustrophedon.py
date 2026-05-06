"""Boustrophedon (zigzag) sweep with frontier fallback.

Greedy systematic sweep: keep moving in the current horizontal direction
until blocked, then go down one row and reverse direction. When both the
horizontal direction and "down" are blocked, switch to a frontier-based
target and walk toward it via BFS over the internal map. Once the target
is reached (or becomes visited), resume zigzag.
"""

from typing import Optional

import numpy as np

from broom.baselines.shared import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    InternalMap,
)


_ACTION_BY_NAME = {
    "right": ACTION_RIGHT,
    "up": ACTION_UP,
    "left": ACTION_LEFT,
    "down": ACTION_DOWN,
}

_DELTA_BY_NAME = {
    "right": (1, 0),
    "up": (0, -1),
    "left": (-1, 0),
    "down": (0, 1),
}


class BoustrophedonAgent:
    def __init__(self, size: int):
        self.map = InternalMap(size)
        self._direction = "right"
        self._fallback_target: Optional[tuple[int, int]] = None

    def reset(self) -> None:
        self.map = InternalMap(self.map.size)
        self._direction = "right"
        self._fallback_target = None

    def act(self, agent_pos: tuple[int, int], neighbors: np.ndarray) -> int:
        self.map.update_from_obs(agent_pos, neighbors)

        # If we are walking toward a frontier target, stay the course until
        # we get there or it becomes visited from a different angle.
        if self._fallback_target is not None:
            if (
                self._fallback_target == agent_pos
                or self.map.is_visited(self._fallback_target)
            ):
                self._fallback_target = None
            else:
                return self.map.next_action_toward(agent_pos, self._fallback_target)

        x, y = agent_pos
        if self._can_move_to(x, y, self._direction):
            return _ACTION_BY_NAME[self._direction]
        if self._can_move_to(x, y, "down"):
            self._direction = "left" if self._direction == "right" else "right"
            return _ACTION_BY_NAME["down"]

        # Zigzag exhausted; switch to frontier mode.
        target = self.map.find_nearest_unvisited(agent_pos)
        if target is None:
            return self._exploration_step(agent_pos)
        self._fallback_target = target
        return self.map.next_action_toward(agent_pos, target)

    def _can_move_to(self, x: int, y: int, direction: str) -> bool:
        dx, dy = _DELTA_BY_NAME[direction]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.map.size and 0 <= ny < self.map.size):
            return False
        return self.map.cell_state(nx, ny) != 3

    def _exploration_step(self, agent_pos: tuple[int, int]) -> int:
        x, y = agent_pos
        for direction, action in _ACTION_BY_NAME.items():
            if self._can_move_to(x, y, direction):
                return action
        return ACTION_RIGHT
