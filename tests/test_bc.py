import gymnasium as gym
import pytest

from broom.bc import _extract_3x3_neighbors_from_env, collect_expert_trajectories, train_bc
from gymnasium_env.grid_world_cpp_mapobs import GridWorldCPPMapObsEnv


@pytest.fixture(autouse=True, scope="session")
def register_env():
    if "gymnasium_env/GridWorldCPPMapObs-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPPMapObs-v0", entry_point=GridWorldCPPMapObsEnv)


def test_neighbors_extraction_matches_upstream_codes():
    env = gym.make(
        "gymnasium_env/GridWorldCPPMapObs-v0", size=5, obs_quantity=3, max_steps=200
    )
    env.reset(seed=0)
    n = _extract_3x3_neighbors_from_env(env.unwrapped)
    assert n.shape == (3, 3)
    assert n.min() >= 0 and n.max() <= 2
    env.close()


def test_collect_expert_trajectories_returns_obs_action_pairs():
    samples = collect_expert_trajectories(grid_sizes=(5,), n_episodes_per_size=3)
    assert len(samples) > 0
    obs, action = samples[0]
    assert "agent" in obs and "ego_map" in obs
    assert isinstance(action, int)
    assert 0 <= action < 4


def test_train_bc_smoke(tmp_path):
    samples = collect_expert_trajectories(grid_sizes=(5,), n_episodes_per_size=5)
    save_path = str(tmp_path / "bc_smoke.zip")
    train_bc(samples, save_path=save_path, n_epochs=1, batch_size=64, smoke_env_size=5)
    assert (tmp_path / "bc_smoke.zip").exists()
