"""Internal map data structure shared by scripted baseline agents.

Cell states (in the agent's internal map):
  0 = unknown (never observed)
  1 = free and visited
  2 = free and unvisited (seen in some neighbor window but not stepped on)
  3 = obstacle (or out-of-bounds wall)

The encoding from the env's neighbors window (upstream protocol):
  0 = free not yet visited
  1 = obstacle, wall, or out-of-bounds
  2 = already visited
"""

from collections import deque
from typing import Optional

import numpy as np


ACTION_RIGHT = 0
ACTION_UP = 1
ACTION_LEFT = 2
ACTION_DOWN = 3

DELTAS = {
    ACTION_RIGHT: (1, 0),
    ACTION_UP: (0, -1),
    ACTION_LEFT: (-1, 0),
    ACTION_DOWN: (0, 1),
}


class InternalMap:
    """Map an agent builds incrementally from partial observations.

    Coordinates are (x, y) with x increasing rightward, y increasing downward,
    matching the env's `_agent_location`.
    """

    def __init__(self, size: int):
        self.size = size
        self._grid = np.zeros((size, size), dtype=int)

    def cell_state(self, x: int, y: int) -> int:
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 3
        return int(self._grid[y][x])

    def is_visited(self, pos: tuple[int, int]) -> bool:
        return self.cell_state(*pos) == 1

    def update_from_obs(self, agent_pos: tuple[int, int], neighbors: np.ndarray) -> None:
        ax, ay = agent_pos
        if 0 <= ax < self.size and 0 <= ay < self.size:
            self._grid[ay][ax] = 1

        rows, cols = neighbors.shape
        radius = rows // 2
        for i in range(rows):
            for j in range(cols):
                if i == radius and j == radius:
                    continue
                nx = ax + (j - radius)
                ny = ay + (i - radius)
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                cur = self._grid[ny][nx]
                if cur == 1:
                    continue
                code = int(neighbors[i][j])
                if code == 1:
                    self._grid[ny][nx] = 3
                elif code == 2:
                    self._grid[ny][nx] = 1
                else:
                    if cur == 0:
                        self._grid[ny][nx] = 2

    def find_nearest_unvisited(self, agent_pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        ax, ay = agent_pos
        seen = {(ax, ay)}
        q: deque[tuple[int, int]] = deque([(ax, ay)])
        while q:
            cx, cy = q.popleft()
            for action, (dx, dy) in DELTAS.items():
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen:
                    continue
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                state = int(self._grid[ny][nx])
                if state == 2:
                    return (nx, ny)
                if state == 1:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return None

    def next_action_toward(self, agent_pos: tuple[int, int], target: tuple[int, int]) -> int:
        ax, ay = agent_pos
        if (ax, ay) == target:
            return ACTION_RIGHT
        parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        seen = {(ax, ay)}
        q: deque[tuple[int, int]] = deque([(ax, ay)])
        found = False
        while q:
            cx, cy = q.popleft()
            for action, (dx, dy) in DELTAS.items():
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen:
                    continue
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                state = int(self._grid[ny][nx])
                if state in (1, 2):
                    parent[(nx, ny)] = ((cx, cy), action)
                    if (nx, ny) == target:
                        found = True
                        break
                    seen.add((nx, ny))
                    q.append((nx, ny))
            if found:
                break

        if not found:
            tx, ty = target
            dx, dy = tx - ax, ty - ay
            if abs(dx) >= abs(dy):
                return ACTION_RIGHT if dx >= 0 else ACTION_LEFT
            return ACTION_DOWN if dy >= 0 else ACTION_UP

        cur = target
        while parent[cur][0] != (ax, ay):
            cur = parent[cur][0]
        return parent[cur][1]
