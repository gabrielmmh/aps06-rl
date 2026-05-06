import os
from pathlib import Path

import pytest

from broom.train import train_one


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setenv("APS07_RESULTS_DIR", str(tmp_path))
    return tmp_path


def test_baseline_train_smokes_in_minimal_steps(tmp_results):
    """Train baseline for ~256 timesteps and verify model + curve are written."""
    out = train_one(
        config_name="baseline",
        seed=0,
        size=5,
        total_timesteps=256,
        init_from=None,
    )
    assert Path(out["model_path"]).exists()
    assert Path(out["curve_path"]).exists()
    assert out["config_name"] == "baseline"
    assert out["seed"] == 0
    assert out["size"] == 5


def test_curriculum_loads_init_from(tmp_results):
    """Train baseline first, then warm-start a second run and verify it loads."""
    first = train_one(
        config_name="baseline",
        seed=0,
        size=5,
        total_timesteps=256,
        init_from=None,
    )
    second = train_one(
        config_name="curriculum",
        seed=0,
        size=10,
        total_timesteps=256,
        init_from=first["model_path"],
    )
    assert Path(second["model_path"]).exists()
    assert second["init_from"] == first["model_path"]
