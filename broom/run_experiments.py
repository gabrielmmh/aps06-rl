"""Driver: runs all (config, seed, size) combinations sequentially.

Usage:
    python -m broom.run_experiments --configs baseline,curriculum,curriculum_enriched,curriculum_recurrent

Resumable: skips combinations whose model already exists.
"""

import argparse
import time
from pathlib import Path

from broom.configs import (
    CURRICULUM_CHAIN,
    SEEDS,
    ConfigName,
)
from broom.inference import evaluate
from broom.train import train_one


CURRICULUM_CONFIGS: tuple[ConfigName, ...] = (
    "curriculum",
    "curriculum_enriched",
    "curriculum_recurrent",
    "curriculum_recurrent_v2",
)


def _model_path(config_name: ConfigName, seed: int, size: int) -> Path:
    from broom.train import _results_dir  # type: ignore
    return _results_dir() / "models" / f"{config_name}_seed{seed}_{size}x{size}.zip"


def run_baseline(seed: int) -> None:
    """Baseline trains each grid from scratch (no warm start)."""
    for size, _ in CURRICULUM_CHAIN:
        target = _model_path("baseline", seed, size)
        if target.exists():
            print(f"  skip baseline seed={seed} size={size} (exists)")
            continue
        t0 = time.time()
        train_one(config_name="baseline", seed=seed, size=size, init_from=None)
        print(f"  baseline seed={seed} size={size} done in {time.time()-t0:.0f}s")


def run_curriculum(config_name: ConfigName, seed: int) -> None:
    """Curriculum-style configs warm-start each phase from the previous size's model."""
    for size, init_size in CURRICULUM_CHAIN:
        target = _model_path(config_name, seed, size)
        if target.exists():
            print(f"  skip {config_name} seed={seed} size={size} (exists)")
            continue
        init_from = (
            str(_model_path(config_name, seed, init_size)) if init_size is not None else None
        )
        t0 = time.time()
        train_one(config_name=config_name, seed=seed, size=size, init_from=init_from)
        print(f"  {config_name} seed={seed} size={size} done in {time.time()-t0:.0f}s")


def run_inference_for(config_name: ConfigName, seed: int) -> None:
    for train_size, _ in CURRICULUM_CHAIN:
        model_path = _model_path(config_name, seed, train_size)
        if not model_path.exists():
            print(f"  no model for {config_name} seed={seed} size={train_size}, skipping")
            continue
        for eval_size, _ in CURRICULUM_CHAIN:
            metrics = evaluate(
                model_path=str(model_path),
                config_name=config_name,
                seed=seed,
                eval_size=eval_size,
                train_size=train_size,
                n_episodes=100,
            )
            print(
                f"  {config_name} seed={seed} train={train_size} eval={eval_size} "
                f"full_coverage={metrics['full_coverage_rate']:.1%} avg={metrics['avg_coverage']:.1%}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="baseline,curriculum,curriculum_enriched,curriculum_recurrent",
        help="Comma-separated config names to run.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Train only; don't run evaluation.",
    )
    args = parser.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    for config_name in configs:
        print(f"\n=== {config_name} ===")
        for seed in SEEDS:
            print(f" seed {seed}")
            if config_name == "baseline":
                run_baseline(seed)
            elif config_name in CURRICULUM_CONFIGS:
                run_curriculum(config_name, seed)
            else:
                raise ValueError(f"Unknown config: {config_name}")
            if not args.skip_inference:
                run_inference_for(config_name, seed)


if __name__ == "__main__":
    main()
