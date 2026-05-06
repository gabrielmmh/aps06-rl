"""Driver para baselines clássicos (FrontierAgent e BoustrophedonAgent).

Roda 100 episódios em cada (algo, seed, grid). Persiste em
`results/inference/scripted_<algo>_seed<N>_eval_<G>x<G>.csv`.

Usage:
    python -m broom.run_scripted
    python -m broom.run_scripted --algos frontier
    python -m broom.run_scripted --algos boustrophedon --seeds 0
"""

import argparse
import time

from broom.baselines.boustrophedon import BoustrophedonAgent
from broom.baselines.frontier import FrontierAgent
from broom.configs import SEEDS
from broom.inference import evaluate_scripted

GRIDS = (5, 10, 20)
AGENTS = {
    "frontier": FrontierAgent,
    "boustrophedon": BoustrophedonAgent,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algos",
        default=",".join(AGENTS.keys()),
        help="Comma-separated agent names to run (default: all).",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help=f"Comma-separated seeds to run (default: {','.join(str(s) for s in SEEDS)}).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Episodes per (algo, seed, grid) combination.",
    )
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for algo_name in algos:
        if algo_name not in AGENTS:
            raise ValueError(f"unknown agent: {algo_name}")
        AgentCls = AGENTS[algo_name]
        print(f"\n=== {algo_name} ===")
        for seed in seeds:
            for grid in GRIDS:
                t0 = time.time()
                agent = AgentCls(size=grid)
                metrics = evaluate_scripted(
                    agent=agent,
                    algo_name=algo_name,
                    seed=seed,
                    eval_size=grid,
                    n_episodes=args.n_episodes,
                )
                print(
                    f"  {algo_name} seed={seed} {grid}x{grid}: "
                    f"full={metrics['full_coverage_rate']:.1%} "
                    f"avg={metrics['avg_coverage']:.1%} "
                    f"in {time.time()-t0:.0f}s"
                )


if __name__ == "__main__":
    main()
