"""Plot generation for APS06 results.

Reads CSVs from results/learning_curves/ and results/inference/, writes PNGs
to results/plots/.
"""

import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _results_dir() -> Path:
    return Path(os.environ.get("APS06_RESULTS_DIR", "results"))


def _plots_dir() -> Path:
    out = _results_dir() / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


GRID_SIZES = (5, 10, 20)


def _load_seeds_for(config_name: str, size: int) -> list[pd.DataFrame]:
    curves_dir = _results_dir() / "learning_curves"
    pattern = f"{config_name}_seed*_{size}x{size}.csv"
    return [pd.read_csv(p) for p in sorted(curves_dir.glob(pattern))]


def _smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_learning_curve(config_name: str) -> str:
    """3 panels (5x5, 10x10, 20x20). Per panel, mean ± std over seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, size in zip(axes, GRID_SIZES):
        seeds = _load_seeds_for(config_name, size)
        if not seeds:
            ax.set_title(f"{size}x{size} (no data)")
            continue
        max_len = min(len(s) for s in seeds)
        rewards = np.stack([s["reward"].to_numpy()[:max_len] for s in seeds])
        smoothed = np.stack([_smooth(r) for r in rewards])
        x = np.arange(smoothed.shape[1])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        ax.plot(x, mean, color="tab:blue")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="tab:blue")
        ax.set_title(f"{size}x{size}")
        ax.set_xlabel("episódio")
        ax.set_ylabel("reward (smoothed)")
    fig.suptitle(f"Curva de aprendizado — {config_name}")
    fig.tight_layout()
    out = _plots_dir() / f"learning_curve_{config_name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
