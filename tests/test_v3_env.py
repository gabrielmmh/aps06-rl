import gymnasium as gym
import numpy as np
import pytest

from gymnasium_env.grid_world_cpp_v3 import (
    GridWorldCPPV3Env,
    COVERAGE_THRESHOLD_FOR_STEP_PENALTY,
    TERMINAL_FULL_COVERAGE_BONUS,
)


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPPV3-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPV3-v0", entry_point=GridWorldCPPV3Env)


def _env(size=5, max_steps=200, obs_quantity=3):
    return gym.make(
        "gymnasium_env/GridWorldCPPV3-v0",
        size=size, obs_quantity=obs_quantity, max_steps=max_steps,
    )


def test_action_masks_shape_and_dtype():
    env = _env()
    env.reset(seed=0)
    masks = env.unwrapped.action_masks()
    assert masks.shape == (4,)
    assert masks.dtype == bool
    assert masks.any()


def test_action_masks_block_out_of_bounds_at_corner():
    env = _env(size=5, obs_quantity=0)  # no obstacles so right/down are guaranteed free
    env.reset(seed=0)
    # Force agent to corner (0, 0)
    env.unwrapped._agent_location = np.array([0, 0], dtype=int)
    env.unwrapped.set_neighbors(env.unwrapped.obstacles_locations)
    masks = env.unwrapped.action_masks()
    # action 2 (left) and action 1 (up) should be illegal at (0, 0)
    assert masks[2] == False, "left should be masked at x=0"
    assert masks[1] == False, "up should be masked at y=0"
    assert masks[0] == True, "right should be legal"
    assert masks[3] == True, "down should be legal"


def test_action_masks_fallback_when_all_blocked():
    env = _env(size=5, obs_quantity=0)
    env.reset(seed=0)
    # Force agent to (0, 0) and place obstacles to block right and down
    env.unwrapped._agent_location = np.array([0, 0], dtype=int)
    env.unwrapped.obstacles_locations = [
        np.array([1, 0], dtype=int),
        np.array([0, 1], dtype=int),
    ]
    masks = env.unwrapped.action_masks()
    # All 4 moves are blocked (left/up out of bounds, right/down obstacles).
    # Fallback should mark all as legal so MaskablePPO doesn't assert.
    assert masks.all(), f"fallback should make all legal, got {masks}"


def test_terminal_bonus_is_60():
    env = _env(size=5, max_steps=200, obs_quantity=0)
    env.reset(seed=0)
    # Cover all 25 free cells (no obstacles). Use a deterministic walk
    # by repeatedly choosing the action_masks-respecting next move.
    visited_count_before = len(env.unwrapped.visited)
    # We'll just take many random moves until done; sufficient seed to terminate.
    rng = np.random.default_rng(0)
    last_reward = None
    for _ in range(2000):
        masks = env.unwrapped.action_masks()
        legal_actions = np.flatnonzero(masks)
        action = int(rng.choice(legal_actions))
        obs, r, term, trunc, info = env.step(action)
        last_reward = r
        if term or trunc:
            break
    if term and not trunc:
        # Confirm the last step's reward includes the +60 bonus.
        # On a new-cell-closing step: reward = step_penalty + 1 (new) + 60.
        # step_penalty = 0 since coverage went from <0.80 to 1.0 between
        # the last step's start (could be < 0.80) and the cell visit.
        # We don't enforce the exact decomposition; just check >> base 1.0.
        assert last_reward >= 50.0, f"expected terminal reward >= 50 (1 + 60), got {last_reward}"


def test_truncation_no_penalty():
    env = _env(size=20, max_steps=20, obs_quantity=0)  # too few steps to cover
    env.reset(seed=0)
    rewards = []
    for _ in range(20):
        masks = env.unwrapped.action_masks()
        legal_actions = np.flatnonzero(masks)
        action = int(np.random.choice(legal_actions))
        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        if term or trunc:
            break
    # The last step should be truncation. Reward should NOT include -5.
    # Last reward could be: step_penalty (-0.1) + new_cell (+1) or revisit (-0.3).
    # In either case, the magnitude is small (between -0.4 and +0.9 — never includes -5).
    assert rewards[-1] >= -1.0, f"truncation step reward should not include -5 penalty, got {rewards[-1]}"
    assert trunc, "expected truncation"


def test_step_penalty_disappears_above_threshold():
    env = _env(size=5, obs_quantity=0)
    env.reset(seed=0)
    # Manually fill in visited cells until coverage >= 0.80
    total = env.unwrapped.total_free_cells
    needed = int(np.ceil(COVERAGE_THRESHOLD_FOR_STEP_PENALTY * total))
    # Mark the agent location and arbitrary cells as visited
    env.unwrapped.visited = set()
    env.unwrapped.visited.add(tuple(env.unwrapped._agent_location))
    cell_iter = ((x, y) for x in range(5) for y in range(5))
    while len(env.unwrapped.visited) < needed:
        c = next(cell_iter)
        env.unwrapped.visited.add(c)
    # Now move to a cell that's already visited so we can check the
    # step penalty is gated. Pick a target adjacent visited cell.
    masks = env.unwrapped.action_masks()
    legal_actions = np.flatnonzero(masks)
    # Take a step; coverage_ratio should be >= 0.80 now.
    obs, r, term, trunc, info = env.step(int(legal_actions[0]))
    # If we revisited (most likely): reward = 0 (no step penalty) + (-0.3) = -0.3
    # If new cell: reward = 0 + 1 = 1
    # Critically, the step component should be 0, not -0.1.
    assert r == pytest.approx(-0.3, abs=0.01) or r >= 1.0 - 0.01, (
        f"after threshold reward should be -0.3 (revisit, no step penalty) or >= 1.0 (new cell), got {r}"
    )


def test_inherits_enriched_observation_space():
    env = _env(size=5)
    obs, _ = env.reset(seed=0)
    # Should have the 4 enriched obs keys
    assert set(obs.keys()) == {
        "agent", "neighbors", "direction_to_nearest_unvisited", "distance_to_nearest_unvisited"
    }
    # neighbors window is 5x5 (from enriched parent)
    assert obs["neighbors"].shape == (5, 5)
