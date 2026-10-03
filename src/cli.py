"""Command-line entry point: `spectra sweep ...`"""

from __future__ import annotations

import argparse

from src.sweep.batch import run_sweep
from src.sweep.grid import sample_sweep


def main() -> None:
    parser = argparse.ArgumentParser(prog="spectra")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep_parser = subparsers.add_parser(
        "sweep", help="Simulate a sweep of combustion chamber transients."
    )
    sweep_parser.add_argument("--n-cases", type=int, default=50)
    sweep_parser.add_argument("--out", type=str, default="datasets")
    sweep_parser.add_argument("--workers", type=int, default=1)
    sweep_parser.add_argument("--seed", type=int, default=0)
    sweep_parser.add_argument("--fine-dt", type=float, default=30.0, help="sample interval through transitions, seconds")
    sweep_parser.add_argument("--coarse-dt", type=float, default=300.0, help="sample interval through steady holds, seconds")
    sweep_parser.add_argument("--fine-window", type=float, default=300.0, help="settle buffer sampled at fine-dt after each transition, seconds")

    args = parser.parse_args()

    if args.command == "sweep":
        cases = sample_sweep(
            n_cases=args.n_cases,
            seed=args.seed,
            fine_dt=args.fine_dt,
            coarse_dt=args.coarse_dt,
            fine_window=args.fine_window,
        )
        run_ids = run_sweep(cases, out_dir=args.out, workers=args.workers)
        print(f"Wrote {len(run_ids)} runs to {args.out}/<run_id>/")
