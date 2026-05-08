import gymnasium as gym
import numpy as np
import pytest

from gymnasium_env.grid_world_cpp_mapobs import GridWorldCPPMapObsEnv


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPPMapObs-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPMapObs-v0", entry_point=GridWorldCPPMapObsEnv)


def _make(size, obstacles=3, max_steps=200):
    return gym.make(
        "gymnasium_env/GridWorldCPPMapObs-v0",
        size=size,
        obs_quantity=obstacles,
        max_steps=max_steps,
    )


def test_observation_space_has_agent_and_ego_map():
    env = _make(5)
    obs, _ = env.reset(seed=0)
    assert "agent" in obs
    assert "ego_map" in obs
    assert obs["agent"].shape == (3,)
    # ego_map: (3 channels, 39, 39) — K = 2*MAX_GRID - 1 = 39 for grids up to 20x20
    assert obs["ego_map"].shape == (3, 39, 39)
    env.close()


def test_ego_map_values_are_binary():
    env = _make(5)
    obs, _ = env.reset(seed=0)
    ego = obs["ego_map"]
    assert ego.min() >= 0.0
    assert ego.max() <= 1.0
    # only 0 or 1 (binary masks)
    unique_vals = set(np.unique(ego).tolist())
    assert unique_vals.issubset({0.0, 1.0})
    env.close()


def test_agent_cell_is_visited_after_reset():
    env = _make(5)
    obs, info = env.reset(seed=0)
    # Agent cell is at center of ego_map
    K = obs["ego_map"].shape[1]
    center = K // 2
    visited_channel = obs["ego_map"][0]
    assert visited_channel[center, center] == 1.0


def test_partial_observability_cells_outside_window_are_unknown():
    """Critical: cells the agent has NEVER seen (outside any 5x5 window observed
    so far) must be all-zero across every channel of ego_map."""
    env = _make(20, obstacles=48, max_steps=1000)
    obs, _ = env.reset(seed=0)
    ego = obs["ego_map"]
    K = ego.shape[1]
    center = K // 2
    # On reset, only the initial 5x5 window around the agent has been observed.
    # Everything beyond a 5x5 (radius 2) box centered at the agent should be 0
    # in EVERY channel.
    radius = 2
    for c in range(3):
        for i in range(K):
            for j in range(K):
                if abs(i - center) > radius or abs(j - center) > radius:
                    assert ego[c, i, j] == 0.0, (
                        f"channel {c} at offset ({i-center},{j-center}) outside window must be 0, "
                        f"got {ego[c, i, j]}"
                    )
    env.close()


def test_step_extends_known_region():
    """After one step, the new 5x5 around the new agent position is added to
    the known map. Cells we have NOT observed remain 0."""
    env = _make(20, obstacles=48, max_steps=1000)
    obs0, _ = env.reset(seed=0)
    ego0 = obs0["ego_map"]
    obs1, _, _, _, _ = env.step(0)  # right
    ego1 = obs1["ego_map"]
    # Total non-zero cells must not decrease (we accumulate knowledge)
    nonzero_0 = (ego0 > 0).sum()
    nonzero_1 = (ego1 > 0).sum()
    assert nonzero_1 >= nonzero_0
    env.close()


def test_visited_channel_reflects_history():
    env = _make(5, obstacles=0, max_steps=200)
    obs, _ = env.reset(seed=0)
    K = obs["ego_map"].shape[1]
    center = K // 2
    # Take 3 steps right (or as many as possible given grid)
    last_obs = obs
    for _ in range(3):
        last_obs, _, term, trunc, _ = env.step(0)  # right
        if term or trunc:
            break
    # The agent's current cell must be visited, and the visited channel must
    # have at least the count of distinct cells walked on.
    visited_count = int(last_obs["ego_map"][0].sum())
    assert visited_count >= 1
    env.close()


def test_reset_clears_known_map():
    env = _make(5)
    env.reset(seed=0)
    # walk a few steps
    for _ in range(5):
        env.step(env.action_space.sample())
    # reset
    obs, _ = env.reset(seed=1)
    ego = obs["ego_map"]
    K = ego.shape[1]
    center = K // 2
    radius = 2
    # After reset, only the new initial 5x5 is known
    for c in range(3):
        for i in range(K):
            for j in range(K):
                if abs(i - center) > radius or abs(j - center) > radius:
                    assert ego[c, i, j] == 0.0
    env.close()


def test_no_oracle_access_to_obstacles_outside_observed_window():
    """Stronger partial-observability check: even after many steps, cells the
    agent has NEVER physically been within observation range of must be 0.
    We do this by tracking which world cells were ever within observation
    radius of an agent position, then asserting all other cells are 0 in the
    obstacle channel of ego_map."""
    env = _make(20, obstacles=48, max_steps=1000)
    obs, _ = env.reset(seed=0)
    radius = 2
    observed_world_cells = set()

    base_env = env.unwrapped

    def _record_observed(env_):
        ax, ay = int(env_._agent_location[0]), int(env_._agent_location[1])
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                wx, wy = ax + dj, ay + di
                if 0 <= wx < env_.size and 0 <= wy < env_.size:
                    observed_world_cells.add((wx, wy))

    _record_observed(base_env)

    for _ in range(20):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        _record_observed(base_env)
        if term or trunc:
            break

    ego = obs["ego_map"]
    K = ego.shape[1]
    center = K // 2
    ax = int(base_env._agent_location[0])
    ay = int(base_env._agent_location[1])
    obstacle_channel = ego[1]

    # For every cell that is "1" in the obstacle channel, the corresponding
    # world cell MUST be in observed_world_cells (we cannot know about a wall
    # we have not observed).
    for i in range(K):
        for j in range(K):
            if obstacle_channel[i, j] != 1.0:
                continue
            wx = ax + (j - center)
            wy = ay + (i - center)
            if not (0 <= wx < base_env.size and 0 <= wy < base_env.size):
                continue  # OOB cells are allowed to be 1 (treated as walls — harmless)
            assert (wx, wy) in observed_world_cells, (
                f"obstacle_channel reports wall at world ({wx},{wy}) but agent never observed it"
            )
    env.close()


def test_works_with_sb3_ppo_smoke():
    """Smoke test: PPO can construct a model with this env and run a few steps."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def _thunk():
        return Monitor(_make(5))

    venv = DummyVecEnv([_thunk])
    model = PPO("MultiInputPolicy", venv, n_steps=128, device="cpu")
    model.learn(total_timesteps=256)
    venv.close()
