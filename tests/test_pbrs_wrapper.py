import gymnasium as gym
import pytest

from broom.wrappers import PBRSCoverageWrapper
from gymnasium_env.grid_world_cpp_mapobs import GridWorldCPPMapObsEnv


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPPMapObs-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPMapObs-v0", entry_point=GridWorldCPPMapObsEnv)


def _env(size=5, max_steps=200, gamma=0.99):
    return PBRSCoverageWrapper(
        gym.make(
            "gymnasium_env/GridWorldCPPMapObs-v0",
            size=size,
            obs_quantity=3,
            max_steps=max_steps,
        ),
        gamma=gamma,
    )


def test_info_carries_unshaped_reward():
    env = _env()
    env.reset(seed=0)
    obs, r_shaped, term, trunc, info = env.step(0)
    assert "r_eval" in info
    env.close()


def test_shaping_positive_when_coverage_grows():
    """At least one step out of many should have shaped > unshaped because
    the agent visits a new cell and Phi increases."""
    env = _env(size=10, max_steps=500, gamma=0.99)
    env.reset(seed=0)
    seen_positive = False
    for _ in range(80):
        obs, r_shaped, term, trunc, info = env.step(env.action_space.sample())
        if r_shaped - info["r_eval"] > 0.001:
            seen_positive = True
            break
        if term or trunc:
            break
    assert seen_positive, "expected positive shaping when discovering new cells"
    env.close()


def test_shaping_resets_potential_on_reset():
    env = _env(size=5, max_steps=200, gamma=0.99)
    env.reset(seed=0)
    for _ in range(10):
        env.step(env.action_space.sample())
    obs, _ = env.reset(seed=1)
    obs, r_shaped, _, _, info = env.step(0)
    diff = r_shaped - info["r_eval"]
    assert -2.0 <= diff <= 2.0
    env.close()
