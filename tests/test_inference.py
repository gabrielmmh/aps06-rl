import os
from pathlib import Path

import pytest

from broom.inference import evaluate
from broom.train import train_one


@pytest.fixture
def trained_model(tmp_path, monkeypatch):
    monkeypatch.setenv("APS08_RESULTS_DIR", str(tmp_path))
    return train_one(
        config_name="baseline",
        seed=0,
        size=5,
        total_timesteps=256,
    )


def test_evaluate_writes_csv_and_returns_metrics(trained_model, tmp_path, monkeypatch):
    monkeypatch.setenv("APS08_RESULTS_DIR", str(tmp_path))
    result = evaluate(
        model_path=trained_model.model_path,
        config_name="baseline",
        seed=0,
        eval_size=5,
        train_size=5,
        n_episodes=3,
    )
    assert "full_coverage_rate" in result
    assert "avg_coverage" in result
    assert "avg_steps" in result
    csv_path = Path(result["csv_path"])
    assert csv_path.exists()
    assert "train5x5_eval_5x5" in csv_path.name


def test_evaluate_without_train_size_omits_train_tag(trained_model, tmp_path, monkeypatch):
    monkeypatch.setenv("APS08_RESULTS_DIR", str(tmp_path))
    result = evaluate(
        model_path=trained_model.model_path,
        config_name="baseline",
        seed=0,
        eval_size=5,
        n_episodes=3,
    )
    assert "train" not in Path(result["csv_path"]).name


def test_evaluate_scripted_writes_csv_and_returns_metrics(tmp_path, monkeypatch):
    from broom.baselines.frontier import FrontierAgent
    from broom.inference import evaluate_scripted

    monkeypatch.setenv("APS08_RESULTS_DIR", str(tmp_path))
    agent = FrontierAgent(size=5)
    result = evaluate_scripted(
        agent=agent,
        algo_name="frontier",
        seed=0,
        eval_size=5,
        n_episodes=3,
    )
    assert "full_coverage_rate" in result
    assert "avg_coverage" in result
    assert "avg_steps" in result
    csv_path = Path(result["csv_path"])
    assert csv_path.exists()
    assert "scripted_frontier_seed0_eval_5x5" in csv_path.name
