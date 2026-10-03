"""Runs a sweep of SimCases (optionally in parallel). Each run gets its own
directory: data.parquet (truth + observed trace), conditions.txt (a
human-readable summary of the operating conditions and profile), a
chamber/adiabatic flame temperature (TIT proxy) diagnostic plot, and a
NOx/CO emissions diagnostic plot."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.sim.report import plot_emissions, plot_temperature, write_conditions_txt
from src.sim.runner import SimCase, run_case
from src.sim.sensor import add_sensor_noise


def _run_and_write(case: SimCase, out_dir: Path, noise_seed: int) -> str:
    result = run_case(case)
    result.df = add_sensor_noise(result.df, seed=noise_seed)

    run_dir = out_dir / case.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result.df.to_parquet(run_dir / "data.parquet", index=False)
    write_conditions_txt(result, run_dir / "conditions.txt")
    plot_temperature(result, run_dir / "temperature.png")
    plot_emissions(result, run_dir / "emissions.png")

    return case.run_id


def run_sweep(cases: list[SimCase], out_dir: str | Path, workers: int = 1) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        return [_run_and_write(case, out_dir, noise_seed=i) for i, case in enumerate(cases)]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_and_write, case, out_dir, i) for i, case in enumerate(cases)]
        return [f.result() for f in futures]
