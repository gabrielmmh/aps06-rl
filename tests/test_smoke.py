import gymnasium as gym
import pytest

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPP-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPP-v0", entry_point=GridWorldCPPEnv)


def test_env_resets_and_steps():
    env = gym.make("gymnasium_env/GridWorldCPP-v0", size=5, obs_quantity=3, max_steps=50)
    obs, info = env.reset(seed=0)
    assert "agent" in obs and obs["agent"].shape == (3,)
    assert "neighbors" in obs and obs["neighbors"].shape == (3, 3)
    assert info["size"] == 5

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    env.close()


def test_observation_value_ranges():
    env = gym.make("gymnasium_env/GridWorldCPP-v0", size=5, obs_quantity=3, max_steps=50)
    obs, _ = env.reset(seed=0)
    assert 0.0 <= obs["agent"][0] <= 1.0
    assert 0.0 <= obs["agent"][1] <= 1.0
    assert 0.0 <= obs["agent"][2] <= 1.0
    assert obs["neighbors"].min() >= 0 and obs["neighbors"].max() <= 2
    env.close()
