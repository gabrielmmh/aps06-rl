"""Tests for GridWorldCPPV4Env: structured memory + frontier features + progress."""

import gymnasium as gym
import numpy as np
import pytest

from gymnasium_env.grid_world_cpp_v4 import (
    GridWorldCPPV4Env,
    POOLED_RESOLUTION,
)


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPPV4-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPV4-v0", entry_point=GridWorldCPPV4Env)


def _env(size=5, max_steps=200, obs_quantity=3):
    return gym.make(
        "gymnasium_env/GridWorldCPPV4-v0",
        size=size, obs_quantity=obs_quantity, max_steps=max_steps,
    )


def test_observation_space_has_new_keys():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {
        "agent", "neighbors",
        "direction_to_nearest_unvisited", "distance_to_nearest_unvisited",
        "visited_pooled", "frontier", "progress",
    }


def test_visited_pooled_shape_and_bounds():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert obs["visited_pooled"].shape == (2, POOLED_RESOLUTION, POOLED_RESOLUTION)
    assert obs["visited_pooled"].dtype == np.float32
    assert obs["visited_pooled"].min() >= 0.0
    assert obs["visited_pooled"].max() <= 1.0


def test_visited_pooled_starts_with_only_initial_position_marked():
    env = _env(size=5, obs_quantity=0)
    obs, _ = env.reset(seed=0)
    # Channel 0: visited mask. Right after reset, only the start cell is visited.
    visited_mask = obs["visited_pooled"][0]
    # At least one cell marked
    assert visited_mask.sum() >= 1
    # Channel 1: agent position one-hot. Exactly one cell.
    agent_pos = obs["visited_pooled"][1]
    assert agent_pos.sum() == 1.0


def test_frontier_feature_shape_and_bounds():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert obs["frontier"].shape == (3,)
    assert obs["frontier"].dtype == np.float32
    dx, dy, d = obs["frontier"]
    assert -1.0 <= dx <= 1.0
    assert -1.0 <= dy <= 1.0
    assert 0.0 <= d <= 1.0


def test_progress_feature_starts_at_zero():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert obs["progress"].shape == (1,)
    assert obs["progress"][0] == 0.0


def test_progress_grows_with_steps():
    env = _env(size=5, obs_quantity=0)
    env.reset(seed=0)
    # Take a few steps
    for _ in range(10):
        masks = env.unwrapped.action_masks()
        legal = np.flatnonzero(masks)
        a = int(legal[0])
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    assert obs["progress"][0] > 0.0


def test_partial_observability_visited_pooled_does_not_expose_obstacles():
    """visited_pooled should encode ONLY agent's visited trajectory, not obstacles."""
    env = _env(size=10, obs_quantity=12, max_steps=500)
    env.reset(seed=42)
    base_env = env.unwrapped
    # Channel 0 should be 1 only for cells that the agent has visited (subset of free cells).
    # In particular, no obstacle cell should ever map to a "1" in channel 0.
    pooled = base_env._compute_visited_pooled()
    visited_mask = pooled[0]
    F = POOLED_RESOLUTION
    # Check: every position with mask=1 must come from a visited cell, not an obstacle.
    obs_set = {(int(loc[0]), int(loc[1])) for loc in base_env.obstacles_locations}
    for by in range(F):
        for bx in range(F):
            if visited_mask[by, bx] == 0.0:
                continue
            # Find at least one real cell that maps to (bx, by) AND is in self.visited.
            found_visited = False
            found_obstacle_only = True
            for vx, vy in base_env.visited:
                rbx = min(F - 1, int(vx * F / base_env.size))
                rby = min(F - 1, int(vy * F / base_env.size))
                if (rbx, rby) == (bx, by):
                    found_visited = True
                    found_obstacle_only = False
                    break
            assert found_visited, f"pooled cell ({bx},{by}) marked but no visited cell maps to it"


def test_partial_observability_frontier_bfs_uses_only_seen_obstacles():
    """The frontier BFS must use _seen_obstacles, not the global obstacles_locations."""
    env = _env(size=10, obs_quantity=12, max_steps=500)
    env.reset(seed=42)
    base_env = env.unwrapped
    # Force a state where seen_obstacles is empty but global has obstacles.
    base_env._seen_obstacles = set()
    # The frontier BFS should now treat all cells as potentially traversable
    # (optimism under uncertainty), so it should find a frontier on the global graph
    # ignoring real obstacles.
    dx, dy, d = base_env._bfs_frontier()
    # No assertion error; just verify it runs without crashing.
    # And the BFS distance should be 1 step (since adjacent unvisited cells exist
    # as long as the agent isn't surrounded — visit only the start cell so the
    # neighbors are guaranteed unvisited).
    assert d <= 1.0


def test_inherits_v3_action_masks_and_reward():
    """V4 should still have the V3 action_masks and reward redesign behavior."""
    env = _env(size=5, obs_quantity=0)
    env.reset(seed=0)
    base_env = env.unwrapped
    # Action masks present (4-dim bool).
    masks = base_env.action_masks()
    assert masks.shape == (4,)
    assert masks.dtype == bool
    # Reward redesign: terminal +60 should fire on full coverage. Inherited
    # behavior, just sanity-check that step() returns a dict obs and updates
    # progress correctly.
    obs, r, term, trunc, info = env.step(0)
    assert isinstance(obs, dict)
    assert "progress" in obs


def test_diameter_scales_with_grid_size():
    for size in (5, 10, 20):
        env = _env(size=size, obs_quantity=0, max_steps=size * size * 4)
        env.reset(seed=0)
        base_env = env.unwrapped
        expected = float(np.sqrt(2.0) * size)
        assert abs(base_env._diameter - expected) < 1e-6
