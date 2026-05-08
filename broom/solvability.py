"""Map-solvability helpers.

A map is "solvable" if all free cells are reachable from the agent's start
position via 4-connected moves through non-obstacle cells. Some randomly
generated maps have unreachable cells (the agent is sealed off from a pocket
of free cells), making 100% coverage physically impossible. In those cases
the upstream `total_free_cells` over-counts the reachable target.

This module provides:
  * `count_reachable_cells(env_unwrapped)` — BFS from agent start through
    free cells, returns the number of cells the agent can possibly visit.
  * `is_solvable(env_unwrapped)` — True iff `count_reachable_cells` equals
    the upstream `total_free_cells`.

Used both at evaluation time (to report a "filtered" metric over solvable
maps only) and in post-hoc analysis of existing inference CSVs.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def count_reachable_cells(env_unwrapped) -> int:
    """BFS from the agent's start through 4-connected non-obstacle cells.

    Returns the number of cells the agent can possibly visit (including the
    start cell itself). Note: this is partial-observability-safe — it uses
    only the env's internal obstacle layout, never feeds into the agent obs.
    """
    start = (int(env_unwrapped._agent_location[0]), int(env_unwrapped._agent_location[1]))
    obstacles = {(int(loc[0]), int(loc[1])) for loc in env_unwrapped.obstacles_locations}
    size = env_unwrapped.size

    visited = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (0, -1), (-1, 0), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in obstacles and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return len(visited)


def is_solvable(env_unwrapped) -> bool:
    """True iff every free cell is reachable from the agent's start position."""
    return count_reachable_cells(env_unwrapped) >= env_unwrapped.total_free_cells
