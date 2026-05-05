import csv
from pathlib import Path

import pytest

from aps06.plot import plot_learning_curve


@pytest.fixture
def fake_curves(tmp_path, monkeypatch):
    """Build 3 seeds × 1 grid of synthetic learning-curve CSVs."""
    monkeypatch.setenv("APS06_RESULTS_DIR", str(tmp_path))
    curves_dir = tmp_path / "learning_curves"
    curves_dir.mkdir()
    for seed in range(3):
        for size in (5,):
            path = curves_dir / f"baseline_seed{seed}_{size}x{size}.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "reward", "length", "coverage"])
                for i in range(50):
                    w.writerow([i, -10 + i * 0.1, 100 - i, min(0.05 * i, 1.0)])
    return tmp_path


def test_plot_learning_curve_writes_png(fake_curves):
    out = plot_learning_curve("baseline")
    assert Path(out).exists()
    assert out.endswith(".png")
