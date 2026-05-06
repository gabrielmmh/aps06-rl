import numpy as np

from broom.baselines.shared import InternalMap


def test_initial_map_is_unknown():
    m = InternalMap(5)
    assert m.size == 5
    assert m.cell_state(0, 0) == 0
    assert m.cell_state(4, 4) == 0


def test_update_marks_agent_position_as_visited():
    m = InternalMap(5)
    neighbors = np.zeros((3, 3), dtype=int)
    m.update_from_obs((2, 2), neighbors)
    assert m.cell_state(2, 2) == 1


def test_update_marks_walls_and_unvisited_neighbors():
    m = InternalMap(5)
    neighbors = np.zeros((3, 3), dtype=int)
    neighbors[1][2] = 1
    neighbors[0][1] = 0
    m.update_from_obs((2, 2), neighbors)
    assert m.cell_state(3, 2) == 3
    assert m.cell_state(2, 1) == 2


def test_find_nearest_unvisited_picks_closest():
    m = InternalMap(5)
    m._grid[2][2] = 1
    m._grid[2][3] = 2
    m._grid[4][4] = 2
    target = m.find_nearest_unvisited((2, 2))
    assert target == (3, 2) or target == (2, 3)


def test_find_nearest_unvisited_returns_none_when_all_known_visited():
    m = InternalMap(3)
    for x in range(3):
        for y in range(3):
            m._grid[y][x] = 1
    assert m.find_nearest_unvisited((0, 0)) is None


def test_next_action_toward_picks_first_step():
    m = InternalMap(5)
    m._grid[2][2] = 1
    m._grid[2][3] = 2
    action = m.next_action_toward((2, 2), (3, 2))
    assert action == 0
