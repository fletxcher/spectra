"""Per-run human-readable summary and diagnostic plot, written alongside
each run's parquet trace."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.sim.regions import REGIONS
from src.sim.runner import RunResult

REFERENCE_YEAR_START = datetime.date(2024, 1, 1)  # arbitrary anchor, for human-readable dates only

_MW_NO = 30.006
_MW_NO2 = 46.005
_MW_CO = 28.010


def _date_str(day_of_year: float) -> str:
    return (REFERENCE_YEAR_START + datetime.timedelta(days=day_of_year)).isoformat()


def write_conditions_txt(result: RunResult, path: str | Path) -> None:
    case = result.case
    net = result.network
    region = REGIONS[case.region_name]
    n_segments = len(case.profile.segments)
    total_days = case.profile.total_duration / 86400.0

    lines = [
        f"run_id: {case.run_id}",
        "",
        "-- region --",
        f"region: {region.name}",
        f"hemisphere: {region.hemisphere}",
        f"notes: {region.notes}",
        f"simulated period: {_date_str(case.start_day_of_year)} to {_date_str(case.start_day_of_year + total_days)} "
        f"({total_days:.1f} days)",
        "",
        "-- site / compressor --",
        f"site pressure: {case.ambient.pressure:.0f} Pa",
        f"pressure ratio: {case.pressure_ratio:.2f}",
        f"isentropic efficiency: {case.isentropic_efficiency:.2f}",
        "",
        "-- combustor --",
        f"chamber pressure target: {net.chamber_pressure_target:.0f} Pa",
        f"volume: {case.volume:.5f} m^3",
        f"reference air mass flow (load_fraction=1.0): {net.mdot_air_ref:.3f} kg/s",
        f"estimated residence time (reference conditions): {result.residence_time_s * 1000:.1f} ms",
        "",
        "-- fuel --",
        f"H2 mole fraction: {case.fuel.h2_mole_fraction:.3f}",
        f"stoichiometric air/fuel mass ratio: {net.stoich_afr:.2f}",
        "",
        "-- dispatch / control schedule --",
        f"idle (minimum-turndown) phi: {case.idle_phi:.3f}",
        f"base-load phi: {case.base_load_phi:.3f}",
        f"minimum load fraction (turndown limit): {case.min_load_fraction:.3f}",
        f"dispatch re-evaluated every: {case.profile.segments[0].duration / 3600:.1f} h "
        f"({n_segments} dispatch intervals over {total_days:.0f} days)",
        "load/phi tracks regional grid demand (seasonal + diurnal + weekday/weekend), "
        "ramped between levels over a few minutes to tens of minutes per change (no overshoot).",
        "",
        "-- operating conditions swept over the 30-day period (daily min / mean / max) --",
        f"{'date':<12}{'ambient T [C]':<22}{'load_fraction':<22}{'phi':<22}{'adiabatic T [K]':<22}",
    ]

    segments_per_day = max(n_segments // 30, 1)
    ambient_c = np.array(case.segment_ambient_temp_k) - 273.15
    load_fraction = np.array([seg.setpoint.load_fraction for seg in case.profile.segments])
    phi = np.array([seg.setpoint.phi for seg in case.profile.segments])
    adiabatic_t = np.array([result.segment_adiabatic_temp[i] for i in range(n_segments)])

    def fmt_range(arr: np.ndarray) -> str:
        return f"{arr.min():.1f} / {arr.mean():.1f} / {arr.max():.1f}"

    for day in range(n_segments // segments_per_day):
        sl = slice(day * segments_per_day, (day + 1) * segments_per_day)
        date = _date_str(case.start_day_of_year + day)
        lines.append(
            f"{date:<12}{fmt_range(ambient_c[sl]):<22}{fmt_range(load_fraction[sl]):<22}"
            f"{fmt_range(phi[sl]):<22}{fmt_range(adiabatic_t[sl]):<22}"
        )

    Path(path).write_text("\n".join(lines) + "\n")


def plot_temperature(result: RunResult, path: str | Path) -> None:
    df = result.df
    days = df.time / 86400.0

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(days, df.temperature_truth, label="chamber temperature (TIT proxy, truth)", color="tab:red", lw=0.9)
    ax.plot(days, df.temperature, label="chamber temperature (observed)", color="tab:red", alpha=0.25, lw=0.5)
    ax.step(days, df.adiabatic_flame_temp, where="post", label="adiabatic flame temperature", color="black", ls="--", lw=0.8)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlim(0, days.max())
    ax.grid(axis="x", color="gray", ls=":", lw=0.4, alpha=0.6)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("temperature [K]")
    region_name = REGIONS[result.case.region_name].name
    ax.set_title(f"{result.case.run_id} ({region_name}): chamber / adiabatic flame temperature over 30 days")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_emissions(result: RunResult, path: str | Path) -> None:
    """NOx (NO+NO2) and CO, converted from tracked species mass fractions to
    molar ppm (wet, uncorrected -- no dry/15%-O2 correction is applied, so
    treat these as relative trends rather than compliance-grade figures)."""
    # Drop the very first sample: the combustor is initialized at full
    # chemical equilibrium (so it starts already alight), which briefly
    # drives trace species (NO, CO) to their equilibrium rather than
    # kinetically-limited level -- a one-sample artifact that would otherwise
    # dominate the axis scale.
    df = result.df.iloc[1:]
    days = df.time / 86400.0

    nox_ppm = (df.Y_NO / _MW_NO + df.Y_NO2 / _MW_NO2) * df.mean_molecular_weight * 1e6
    co_ppm = (df.Y_CO / _MW_CO) * df.mean_molecular_weight * 1e6

    fig, ax_nox = plt.subplots(figsize=(11, 4.5))
    ax_co = ax_nox.twinx()

    (line_nox,) = ax_nox.plot(days, nox_ppm, color="tab:purple", lw=0.8, label="NOx")
    ax_nox.set_ylim(0, nox_ppm.max() * 1.1)
    ax_nox.set_ylabel("NOx [ppm]", color="tab:purple")
    ax_nox.tick_params(axis="y", colors="tab:purple")
    ax_nox.grid(axis="x", color="gray", ls=":", lw=0.4, alpha=0.6)

    (line_co,) = ax_co.plot(days, co_ppm, color="tab:blue", lw=0.8, label="CO")
    ax_co.set_ylim(0, co_ppm.max() * 1.1)
    ax_co.set_ylabel("CO [ppm]", color="tab:blue")
    ax_co.tick_params(axis="y", colors="tab:blue")

    ax_nox.set_xlabel("time [days]")
    ax_nox.xaxis.set_major_locator(MultipleLocator(5))
    ax_nox.set_xlim(0, days.max())
    ax_nox.legend(handles=[line_nox, line_co], loc="upper right")

    region_name = REGIONS[result.case.region_name].name
    ax_nox.set_title(f"{result.case.run_id} ({region_name}): NOx and CO emissions over 30 days")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
