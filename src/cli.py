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

    args = parser.parse_args()

    if args.command == "sweep":
        cases = sample_sweep(n_cases=args.n_cases, seed=args.seed)
        run_ids = run_sweep(cases, out_dir=args.out, workers=args.workers)
        print(f"Wrote {len(run_ids)} runs to {args.out}/<run_id>/")
