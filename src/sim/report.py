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
    profile = case.profile
    n_segments = len(profile.segments)
    total_days = profile.total_duration / 86400.0
    intervals_min = np.array([seg.duration for seg in profile.segments]) / 60.0
    ramps_min = np.array([seg.transition_duration for seg in profile.segments[1:]]) / 60.0
    kinds = [d.kind for d in profile.disturbances]

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
        f"dispatch intervals: {n_segments} over {total_days:.0f} days, "
        f"length min / median / max = {intervals_min.min():.1f} / {np.median(intervals_min):.1f} / "
        f"{intervals_min.max():.1f} min",
        f"ramp duration min / median / max = {ramps_min.min():.1f} / {np.median(ramps_min):.1f} / "
        f"{ramps_min.max():.1f} min (ramp-rate limited, no overshoot)",
        "load/phi tracks regional grid demand (seasonal + diurnal + weekday/weekend); re-dispatched "
        "frequently through the morning/evening ramps, sparsely otherwise, with occasional short "
        "real-time corrections.",
        "",
        "-- AGC (regulation wander) --",
        f"load offset std: {profile.agc_sigma * 100:.2f}% of full load, correlation time: {profile.agc_tau:.0f} s",
        f"steady tolerance: +/-{profile.steady_tolerance * 100:.2f}% of full load",
        "",
        "-- disturbances --",
        f"total: {len(kinds)} (runback: {kinds.count('runback')}, fuel_shift: {kinds.count('fuel_shift')}, "
        f"dynamics: {kinds.count('dynamics')})",
        "",
        "-- operating conditions swept over the 30-day period (daily min / mean / max) --",
        f"{'date':<12}{'ambient T [C]':<22}{'load_fraction':<22}{'phi':<22}{'adiabatic T [K]':<22}",
    ]

    seg_day = np.floor(np.array(profile.segment_starts) / 86400.0).astype(int)
    ambient_c = np.array(case.segment_ambient_temp_k) - 273.15
    load_fraction = np.array([seg.setpoint.load_fraction for seg in profile.segments])
    phi = np.array([seg.setpoint.phi for seg in profile.segments])
    adiabatic_t = np.array([result.segment_adiabatic_temp[i] for i in range(n_segments)])

    def fmt_range(arr: np.ndarray) -> str:
        return f"{arr.min():.1f} / {arr.mean():.1f} / {arr.max():.1f}"

    for day in np.unique(seg_day):
        mask = seg_day == day
        date = _date_str(case.start_day_of_year + day)
        lines.append(
            f"{date:<12}{fmt_range(ambient_c[mask]):<22}{fmt_range(load_fraction[mask]):<22}"
            f"{fmt_range(phi[mask]):<22}{fmt_range(adiabatic_t[mask]):<22}"
        )

    Path(path).write_text("\n".join(lines) + "\n")


def time_axis(ax, time_s) -> np.ndarray:
    """Puts a time axis flush against the plot edges: hours (2 h ticks) for
    windows up to 2 days, days (5-day ticks) beyond that. Returns the time
    values in the chosen unit."""
    time_s = np.asarray(time_s, dtype=float)
    span_s = time_s.max() if len(time_s) else 0.0
    if span_s <= 2 * 86400.0:
        values, unit, tick = time_s / 3600.0, "hours", 2
        end = np.ceil(span_s / 3600.0)
    else:
        values, unit, tick = time_s / 86400.0, "days", 5
        end = np.ceil(span_s / 86400.0)
    ax.xaxis.set_major_locator(MultipleLocator(tick))
    ax.set_xlim(0, end)
    ax.set_xlabel(f"time [{unit}]")
    return values


def window_label(time_s) -> str:
    span_s = float(np.max(time_s))
    if span_s <= 2 * 86400.0:
        return f"over {np.ceil(span_s / 3600.0):.0f} hours"
    return f"over {np.ceil(span_s / 86400.0):.0f} days"


def plot_temperature_window(df, run_id: str, region_name: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t = time_axis(ax, df.time)
    ax.plot(t, df.temperature_truth, label="chamber temperature (TIT proxy, truth)", color="tab:red", lw=0.9)
    ax.plot(t, df.temperature, label="chamber temperature (observed)", color="tab:red", alpha=0.25, lw=0.5)
    ax.step(t, df.adiabatic_flame_temp, where="post", label="adiabatic flame temperature", color="black", ls="--", lw=0.8)

    ax.grid(axis="x", color="gray", ls=":", lw=0.4, alpha=0.6)
    ax.set_ylabel("temperature [K]")
    ax.set_title(f"{run_id} ({region_name}): chamber / adiabatic flame temperature {window_label(df.time)}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_emissions_window(df, run_id: str, region_name: str, path: str | Path) -> None:
    """NOx (NO+NO2) and CO, converted from tracked species mass fractions to
    molar ppm (wet, uncorrected: no dry/15%-O2 correction is applied, so
    treat these as relative trends rather than compliance-grade figures)."""
    # Drop the very first sample: the combustor is initialized at full
    # chemical equilibrium (so it starts already alight), which briefly
    # drives trace species (NO, CO) to their equilibrium rather than
    # kinetically-limited level, a one-sample artifact that would otherwise
    # dominate the axis scale.
    df = df.iloc[1:]
    nox_ppm = (df.Y_NO / _MW_NO + df.Y_NO2 / _MW_NO2) * df.mean_molecular_weight * 1e6
    co_ppm = (df.Y_CO / _MW_CO) * df.mean_molecular_weight * 1e6

    fig, ax_nox = plt.subplots(figsize=(11, 4.5))
    ax_co = ax_nox.twinx()
    t = time_axis(ax_nox, df.time)

    (line_nox,) = ax_nox.plot(t, nox_ppm, color="tab:purple", lw=0.8, label="NOx")
    ax_nox.set_ylim(0, nox_ppm.max() * 1.1)
    ax_nox.set_ylabel("NOx [ppm]", color="tab:purple")
    ax_nox.tick_params(axis="y", colors="tab:purple")
    ax_nox.grid(axis="x", color="gray", ls=":", lw=0.4, alpha=0.6)

    (line_co,) = ax_co.plot(t, co_ppm, color="tab:blue", lw=0.8, label="CO")
    ax_co.set_ylim(0, co_ppm.max() * 1.1)
    ax_co.set_ylabel("CO [ppm]", color="tab:blue")
    ax_co.tick_params(axis="y", colors="tab:blue")

    ax_nox.legend(handles=[line_nox, line_co], loc="upper right")
    ax_nox.set_title(f"{run_id} ({region_name}): NOx and CO emissions {window_label(df.time)}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_temperature(result: RunResult, path: str | Path) -> None:
    plot_temperature_window(result.df, result.case.run_id, REGIONS[result.case.region_name].name, path)


def plot_emissions(result: RunResult, path: str | Path) -> None:
    plot_emissions_window(result.df, result.case.run_id, REGIONS[result.case.region_name].name, path)
