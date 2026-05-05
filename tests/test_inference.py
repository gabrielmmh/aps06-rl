import os
from pathlib import Path

import pytest

from aps06.inference import evaluate
from aps06.train import train_one


@pytest.fixture
def trained_model(tmp_path, monkeypatch):
    monkeypatch.setenv("APS06_RESULTS_DIR", str(tmp_path))
    return train_one(
        config_name="baseline",
        seed=0,
        size=5,
        total_timesteps=256,
    )


def test_evaluate_writes_csv_and_returns_metrics(trained_model, tmp_path, monkeypatch):
    monkeypatch.setenv("APS06_RESULTS_DIR", str(tmp_path))
    result = evaluate(
        model_path=trained_model.model_path,
        config_name="baseline",
        seed=0,
        eval_size=5,
        n_episodes=3,
    )
    assert "full_coverage_rate" in result
    assert "avg_coverage" in result
    assert "avg_steps" in result
    assert Path(result["csv_path"]).exists()
