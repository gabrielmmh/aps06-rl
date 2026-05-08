"""Plot generation for APS08 results.

Reads CSVs from results/learning_curves/ and results/inference/, writes PNGs
to results/plots/.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _results_dir() -> Path:
    return Path(os.environ.get("APS08_RESULTS_DIR", "results"))


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


# Aggregated plotting (Epic 5)

RL_CONFIGS = (
    "baseline",
    "curriculum",
    "curriculum_enriched",
    "curriculum_recurrent",
    "curriculum_recurrent_v2",
    "mapcnn_bc_pbrs",
    "maskable_v3",
    "maskable_bc_kl",
    "maskable_frontier_pbrs",
)
SCRIPTED_ALGOS = ("frontier", "boustrophedon")


def _load_solvability_cache() -> dict:
    """Loads the (eval_size, seed, ep) -> solvable bool cache, if present.

    Generated offline by `python -m broom.build_solvability_cache`. Allows the
    "filtered" full coverage rate (over solvable maps only) to be computed
    even for inference CSVs that were written before solvability tracking.
    """
    import json
    cache_path = _results_dir() / "solvability_cache.json"
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text())


def _filtered_full_rate(df: pd.DataFrame, eval_size: int, seed: int, cache: dict) -> float | None:
    """Full coverage rate restricted to solvable maps (where 100% is achievable).

    If the CSV already has a `solvable` column (post-Epic 9 inference output),
    use it directly. Otherwise look up `(eval_size, seed, ep)` in the cache.
    Returns None if no solvable maps were sampled (shouldn't happen with our
    seed choices, but kept for safety).
    """
    terminated = df["terminated"].astype(str).str.lower() == "true"
    if "solvable" in df.columns:
        solvable = df["solvable"].astype(str).str.lower() == "true"
    else:
        solvable_list = []
        for ep in df["episode"]:
            key = f"{eval_size}_{seed}_{int(ep)}"
            solvable_list.append(cache.get(key, True))
        solvable = pd.Series(solvable_list, index=df.index)
    n_solv = int(solvable.sum())
    if n_solv == 0:
        return None
    return float((terminated & solvable).sum()) / n_solv


def _seed_from_filename(path: Path) -> int:
    import re
    m = re.search(r"seed(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _rl_summary() -> pd.DataFrame:
    """For each RL config, returns the row of the inference matrix where the
    model was trained on the same grid as the eval (the "native" cell), since
    that's the reading the assignment cares about for the success criterion.
    Computes mean and std across the 3 seeds. Both raw (over all maps) and
    filtered (over solvable maps only) full coverage rates are reported.
    """
    inf_dir = _results_dir() / "inference"
    cache = _load_solvability_cache()
    rows = []
    for cfg in RL_CONFIGS:
        for size in GRID_SIZES:
            pattern = f"{cfg}_seed*_train{size}x{size}_eval_{size}x{size}.csv"
            files = sorted(inf_dir.glob(pattern))
            if not files:
                continue
            full_rates = []
            full_rates_solvable = []
            avg_covs = []
            for f in files:
                df = pd.read_csv(f)
                full_rates.append((df["terminated"].astype(str).str.lower() == "true").mean())
                avg_covs.append(df["coverage"].mean())
                fr_solv = _filtered_full_rate(df, size, _seed_from_filename(f), cache)
                if fr_solv is not None:
                    full_rates_solvable.append(fr_solv)
            rows.append({
                "config": cfg,
                "kind": "RL",
                "eval_size": size,
                "full_mean": float(np.mean(full_rates)),
                "full_std": float(np.std(full_rates)),
                "full_solvable_mean": float(np.mean(full_rates_solvable)) if full_rates_solvable else float("nan"),
                "full_solvable_std": float(np.std(full_rates_solvable)) if full_rates_solvable else float("nan"),
                "avg_mean": float(np.mean(avg_covs)),
                "avg_std": float(np.std(avg_covs)),
            })
    return pd.DataFrame(rows)


def _scripted_summary() -> pd.DataFrame:
    inf_dir = _results_dir() / "inference"
    cache = _load_solvability_cache()
    rows = []
    for algo in SCRIPTED_ALGOS:
        for size in GRID_SIZES:
            pattern = f"scripted_{algo}_seed*_eval_{size}x{size}.csv"
            files = sorted(inf_dir.glob(pattern))
            if not files:
                continue
            full_rates = []
            full_rates_solvable = []
            avg_covs = []
            for f in files:
                df = pd.read_csv(f)
                full_rates.append((df["terminated"].astype(str).str.lower() == "true").mean())
                avg_covs.append(df["coverage"].mean())
                fr_solv = _filtered_full_rate(df, size, _seed_from_filename(f), cache)
                if fr_solv is not None:
                    full_rates_solvable.append(fr_solv)
            rows.append({
                "config": algo,
                "kind": "scripted",
                "eval_size": size,
                "full_mean": float(np.mean(full_rates)),
                "full_std": float(np.std(full_rates)),
                "full_solvable_mean": float(np.mean(full_rates_solvable)) if full_rates_solvable else float("nan"),
                "full_solvable_std": float(np.std(full_rates_solvable)) if full_rates_solvable else float("nan"),
                "avg_mean": float(np.mean(avg_covs)),
                "avg_std": float(np.std(avg_covs)),
            })
    return pd.DataFrame(rows)


def plot_coverage_heatmap(metric: str = "full") -> str:
    """Heatmap: rows = config (RL natives + scripted), cols = eval grid."""
    df = pd.concat([_rl_summary(), _scripted_summary()], ignore_index=True)
    mean_col = f"{metric}_mean"
    configs_order = list(RL_CONFIGS) + list(SCRIPTED_ALGOS)
    pivot = df.pivot(index="config", columns="eval_size", values=mean_col).reindex(configs_order).reindex(columns=GRID_SIZES)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GRID_SIZES)))
    ax.set_xticklabels([f"{s}x{s}" for s in GRID_SIZES])
    ax.set_yticks(range(len(configs_order)))
    ax.set_yticklabels(configs_order)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label=("avg coverage" if metric == "avg" else "full coverage rate"))
    ax.set_title(f"{metric.capitalize()} coverage por config × grid (nativo)")
    fig.tight_layout()
    out = _plots_dir() / f"heatmap_native_{metric}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def plot_coverage_by_size(metric: str = "full") -> str:
    """Line plot: avg metric on native grid as a function of grid size, one
    line per config. Shows how each strategy degrades with grid size.
    """
    df = pd.concat([_rl_summary(), _scripted_summary()], ignore_index=True)
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    configs_order = list(RL_CONFIGS) + list(SCRIPTED_ALGOS)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs_order:
        sub = df[df["config"] == cfg].sort_values("eval_size")
        if sub.empty:
            continue
        ax.errorbar(sub["eval_size"], sub[mean_col], yerr=sub[std_col],
                    marker="o", capsize=3, label=cfg)
    ax.set_xlabel("grid size")
    ax.set_ylabel("avg coverage" if metric == "avg" else "full coverage rate")
    ax.set_xticks(GRID_SIZES)
    ax.set_ylim(0, 1)
    ax.set_title(f"Generalização por tamanho de grid (nativo) — {metric}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = _plots_dir() / f"coverage_by_size_{metric}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
