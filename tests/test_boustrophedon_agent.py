import gymnasium as gym
import numpy as np
import pytest

from broom.baselines.boustrophedon import BoustrophedonAgent
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPP-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPP-v0", entry_point=GridWorldCPPEnv)


def test_boustrophedon_completes_or_covers_most_of_small_grid():
    env = gym.make("gymnasium_env/GridWorldCPP-v0", size=5, obs_quantity=3, max_steps=300)
    agent = BoustrophedonAgent(size=5)
    obs, info = env.reset(seed=0)
    agent.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        agent_pos = (int(round(obs["agent"][0] * 5)), int(round(obs["agent"][1] * 5)))
        ax = max(0, min(4, agent_pos[0]))
        ay = max(0, min(4, agent_pos[1]))
        action = agent.act((ax, ay), obs["neighbors"])
        obs, _, terminated, truncated, info = env.step(action)
    assert terminated or info["coverage"] > 0.85
    env.close()


def test_boustrophedon_reset_clears_state():
    agent = BoustrophedonAgent(size=5)
    fake = np.zeros((3, 3), dtype=int)
    agent.act((0, 0), fake)
    agent.reset()
    assert agent._direction == "right"
    assert agent.map.cell_state(0, 0) == 0
