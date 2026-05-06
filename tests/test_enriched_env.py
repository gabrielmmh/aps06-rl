import gymnasium as gym
import numpy as np
import pytest

from gymnasium_env.grid_world_cpp_enriched import GridWorldCPPEnrichedEnv


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPPEnriched-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPEnriched-v0", entry_point=GridWorldCPPEnrichedEnv)


def test_observation_has_5x5_neighbors_and_extra_features():
    env = gym.make("gymnasium_env/GridWorldCPPEnriched-v0", size=5, obs_quantity=3, max_steps=50)
    obs, _ = env.reset(seed=0)
    assert obs["agent"].shape == (3,)
    assert obs["neighbors"].shape == (5, 5)
    assert obs["direction_to_nearest_unvisited"].shape == (4,)
    assert obs["distance_to_nearest_unvisited"].shape == (1,)
    env.close()


def test_neighbors_use_same_codes_as_upstream():
    env = gym.make("gymnasium_env/GridWorldCPPEnriched-v0", size=5, obs_quantity=3, max_steps=50)
    obs, _ = env.reset(seed=0)
    assert obs["neighbors"].min() >= 0
    assert obs["neighbors"].max() <= 2
    env.close()


def test_direction_one_hot_or_zero_when_no_unvisited():
    env = gym.make("gymnasium_env/GridWorldCPPEnriched-v0", size=5, obs_quantity=3, max_steps=50)
    obs, _ = env.reset(seed=0)
    direction = obs["direction_to_nearest_unvisited"]
    assert np.all((direction == 0) | (direction == 1))
    assert direction.sum() <= 1
    env.close()


def test_distance_in_unit_interval():
    env = gym.make("gymnasium_env/GridWorldCPPEnriched-v0", size=5, obs_quantity=3, max_steps=50)
    obs, _ = env.reset(seed=0)
    d = float(obs["distance_to_nearest_unvisited"][0])
    assert 0.0 <= d <= 1.0
    env.close()


def test_step_returns_enriched_observation():
    env = gym.make("gymnasium_env/GridWorldCPPEnriched-v0", size=5, obs_quantity=3, max_steps=50)
    obs, _ = env.reset(seed=0)
    obs2, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs2["neighbors"].shape == (5, 5)
    assert obs2["direction_to_nearest_unvisited"].shape == (4,)
    assert obs2["distance_to_nearest_unvisited"].shape == (1,)
    env.close()
