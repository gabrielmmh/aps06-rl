import gymnasium as gym
import numpy as np
import pytest

from broom.baselines.frontier import FrontierAgent
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPP-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPP-v0", entry_point=GridWorldCPPEnv)


def test_frontier_agent_completes_small_grid():
    env = gym.make("gymnasium_env/GridWorldCPP-v0", size=5, obs_quantity=3, max_steps=200)
    agent = FrontierAgent(size=5)
    obs, info = env.reset(seed=0)
    agent.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        agent_pos = (int(round(obs["agent"][0] * 5)), int(round(obs["agent"][1] * 5)))
        ax = max(0, min(4, agent_pos[0]))
        ay = max(0, min(4, agent_pos[1]))
        action = agent.act((ax, ay), obs["neighbors"])
        obs, _, terminated, truncated, info = env.step(action)
    assert terminated
    env.close()


def test_frontier_agent_reset_clears_internal_map():
    agent = FrontierAgent(size=5)
    fake_neighbors = np.zeros((3, 3), dtype=int)
    agent.act((2, 2), fake_neighbors)
    agent.reset()
    assert agent.map.cell_state(2, 2) == 0
