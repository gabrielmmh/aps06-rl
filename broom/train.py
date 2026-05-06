"""Train one (config, seed, grid) combination.

This is the leaf function called by run_experiments.py. It produces:
  - results/models/<config>_seed<N>_<size>x<size>.zip
  - results/learning_curves/<config>_seed<N>_<size>x<size>.csv

For curriculum configs, pass `init_from=<path>` to warm-start from a prior model.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from broom.configs import (
    PHASE_MAX_STEPS,
    PHASE_OBSTACLES,
    PHASE_TIMESTEPS,
    PPO_HYPERPARAMS,
    RECURRENT_HYPERPARAMS,
    RECURRENT_V2_HYPERPARAMS,
    ConfigName,
    GridSize,
    get_phase_n_envs,
)
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


@dataclass
class TrainResult:
    config_name: str
    seed: int
    size: int
    model_path: str
    curve_path: str
    init_from: Optional[str]

    def __getitem__(self, k):
        return getattr(self, k)


def _results_dir() -> Path:
    return Path(os.environ.get("APS07_RESULTS_DIR", "results"))


def _register_envs():
    if "gymnasium_env/GridWorldCPP-v0" not in gym.envs.registry:
        gym.register(id="gymnasium_env/GridWorldCPP-v0", entry_point=GridWorldCPPEnv)
    try:
        from gymnasium_env.grid_world_cpp_enriched import GridWorldCPPEnrichedEnv
        if "gymnasium_env/GridWorldCPPEnriched-v0" not in gym.envs.registry:
            gym.register(id="gymnasium_env/GridWorldCPPEnriched-v0", entry_point=GridWorldCPPEnrichedEnv)
    except ImportError:
        pass


def _env_id_for_config(config_name: ConfigName) -> str:
    if config_name == "curriculum_enriched":
        return "gymnasium_env/GridWorldCPPEnriched-v0"
    return "gymnasium_env/GridWorldCPP-v0"


def _make_env_fn(env_id: str, size: int, max_steps: int, obstacles: int, seed: int):
    """Wraps in Monitor so SB3 populates info["episode"] = {r, l, t} on done."""
    def _thunk():
        _register_envs()
        env = gym.make(env_id, size=size, obs_quantity=obstacles, max_steps=max_steps)
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk


def _make_vec_env(env_id: str, size: int, max_steps: int, obstacles: int, seed: int, n_envs: int):
    fns = [_make_env_fn(env_id, size, max_steps, obstacles, seed + i) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


class _EpisodeLogger(BaseCallback):
    """Records (episode, reward, length, coverage) per finished episode."""

    def __init__(self, curve_path: Path):
        super().__init__()
        self.curve_path = curve_path
        self._buffer: list[tuple[int, float, int, float]] = []
        self._ep_idx = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "episode" in info:
                ep = info["episode"]
                coverage = info.get("coverage", 0.0)
                self._buffer.append((self._ep_idx, float(ep["r"]), int(ep["l"]), float(coverage)))
                self._ep_idx += 1
        return True

    def _on_training_end(self) -> None:
        self.curve_path.parent.mkdir(parents=True, exist_ok=True)
        with self.curve_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", "length", "coverage"])
            w.writerows(self._buffer)


def _is_recurrent(config_name: ConfigName) -> bool:
    return config_name in ("curriculum_recurrent", "curriculum_recurrent_v2")


def _recurrent_hyperparams(config_name: ConfigName) -> dict:
    if config_name == "curriculum_recurrent_v2":
        return RECURRENT_V2_HYPERPARAMS
    return RECURRENT_HYPERPARAMS


def train_one(
    config_name: ConfigName,
    seed: int,
    size: GridSize,
    total_timesteps: Optional[int] = None,
    init_from: Optional[str] = None,
) -> TrainResult:
    """Train a single (config, seed, size) combination.

    If `init_from` is provided, loads weights from that model into the new env
    (used for curriculum). `total_timesteps` defaults to PHASE_TIMESTEPS[size].
    """
    _register_envs()

    env_id = _env_id_for_config(config_name)
    n_envs = get_phase_n_envs(config_name, size)
    obstacles = PHASE_OBSTACLES[size]
    max_steps = PHASE_MAX_STEPS[size]
    timesteps = total_timesteps if total_timesteps is not None else PHASE_TIMESTEPS[size]

    vec_env = _make_vec_env(env_id, size, max_steps, obstacles, seed, n_envs)

    results = _results_dir()
    tag = f"{config_name}_seed{seed}_{size}x{size}"
    model_path = results / "models" / f"{tag}.zip"
    curve_path = results / "learning_curves" / f"{tag}.csv"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    callback = _EpisodeLogger(curve_path)

    # verbose=1 prints per-rollout stats (~once every n_steps timesteps).
    # Negligible cost in wall-clock; useful for diagnosing fps, reward trend,
    # and entropy/KL during long runs.
    verbose = int(os.environ.get("APS07_TRAIN_VERBOSE", "1"))

    if _is_recurrent(config_name):
        from sb3_contrib import RecurrentPPO
        hp = _recurrent_hyperparams(config_name)
        if init_from is not None:
            model = RecurrentPPO.load(init_from, env=vec_env, verbose=verbose, **hp)
        else:
            model = RecurrentPPO("MultiInputLstmPolicy", vec_env, seed=seed, verbose=verbose, **hp)
    else:
        if init_from is not None:
            model = PPO.load(init_from, env=vec_env, verbose=verbose, **PPO_HYPERPARAMS)
        else:
            model = PPO("MultiInputPolicy", vec_env, seed=seed, verbose=verbose, **PPO_HYPERPARAMS)

    model.learn(total_timesteps=timesteps, reset_num_timesteps=(init_from is None), callback=callback)
    model.save(str(model_path))
    vec_env.close()

    return TrainResult(
        config_name=config_name,
        seed=seed,
        size=size,
        model_path=str(model_path),
        curve_path=str(curve_path),
        init_from=init_from,
    )
