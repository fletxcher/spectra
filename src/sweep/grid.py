"""Generates a sweep of SimCases: each run simulates 30 days of a large gas
turbine following grid electricity demand in a particular world region. The
dispatched setpoint (equivalence ratio / load) tracks a demand signal built
from that region's seasonal + diurnal + weekday/weekend pattern, re-dispatched
every few hours like a real economic-dispatch schedule, with realistic
loading/unloading ramps between levels. Ambient temperature is driven by the
same region's diurnal/seasonal climate. Engine design parameters (pressure
ratio, reference mass flow, combustor volume, fuel blend, site elevation) are
Latin-hypercube sampled across runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from src.sim import demand
from src.sim.conditions import AmbientConditions
from src.sim.fuel import FuelBlend
from src.sim.profiles import Profile, Segment, Setpoint
from src.sim.regions import REGIONS
from src.sim.runner import SimCase

TOTAL_DAYS = 30
DISPATCH_INTERVAL_S = 4 * 3600.0  # re-dispatch every 4 hours, like day-ahead scheduling
N_SEGMENTS = int(TOTAL_DAYS * 86400 / DISPATCH_INTERVAL_S)


@dataclass(frozen=True)
class SweepBounds:
    """Continuous engine/site parameter ranges for the Latin-hypercube
    sample. Defaults are plausible for a single can/basket of a large
    industrial gas turbine's combustion chamber."""

    ambient_pressure_pa: tuple[float, float] = (80000.0, 101325.0)  # sea level .. ~2000m
    pressure_ratio: tuple[float, float] = (10.0, 25.0)
    mdot_air_ref_kg_s: tuple[float, float] = (5.0, 20.0)
    volume_m3: tuple[float, float] = (0.01, 0.05)
    h2_mole_fraction: tuple[float, float] = (0.0, 0.5)
    base_load_phi: tuple[float, float] = (0.6, 0.85)  # lean premixed, base-load range
    idle_phi: tuple[float, float] = (0.3, 0.4)  # low-fuel, minimum-turndown combustion
    min_load_fraction: tuple[float, float] = (0.35, 0.55)  # combustor turndown limit

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.ambient_pressure_pa,
                self.pressure_ratio,
                self.mdot_air_ref_kg_s,
                self.volume_m3,
                self.h2_mole_fraction,
            ]
        )


@dataclass(frozen=True)
class RampBounds:
    ramp_duration_s: tuple[float, float] = (300.0, 1200.0)  # 5-20 min, per load change
    demand_floor: float = 0.2  # minimum normalized demand (plant stays committed, not shut down)


def _build_dispatch_schedule(
    region_key: str, start_day_of_year: float, rng: np.random.Generator, bounds: SweepBounds, ramp_bounds: RampBounds
) -> tuple[Profile, tuple[float, ...]]:
    region = REGIONS[region_key]

    segment_mid_s = (np.arange(N_SEGMENTS) + 0.5) * DISPATCH_INTERVAL_S
    hour_of_day = (segment_mid_s / 3600.0) % 24.0
    day_index = np.floor(segment_mid_s / 86400.0)
    # Seasonal baseline is fixed at the run's start date, not left to advance
    # across the window: a calendar month's climatological mean doesn't
    # meaningfully shift over 30 days -- that's a year-timescale effect, not
    # a within-month one. Only the diurnal (hour_of_day) and weekday/weekend
    # (day_index) components should evolve across the 30 days.
    season_day_of_year = np.full(N_SEGMENTS, start_day_of_year)

    raw = demand.raw_demand(region, season_day_of_year, hour_of_day, day_index, rng)
    demand_norm = demand.normalize(raw, floor=ramp_bounds.demand_floor)
    ambient_temp_k = demand.ambient_temperature_k(region, season_day_of_year, hour_of_day, rng)

    idle_phi = rng.uniform(*bounds.idle_phi)
    base_load_phi = rng.uniform(*bounds.base_load_phi)
    min_load_fraction = rng.uniform(*bounds.min_load_fraction)

    phi = idle_phi + (base_load_phi - idle_phi) * demand_norm
    load_fraction = min_load_fraction + (1.0 - min_load_fraction) * demand_norm

    segments = []
    for j in range(N_SEGMENTS):
        setpoint = Setpoint(phi=float(phi[j]), load_fraction=float(load_fraction[j]))
        if j == 0:
            segments.append(
                Segment(setpoint=setpoint, hold_duration=DISPATCH_INTERVAL_S, transition_kind="step", settle_time=60.0)
            )
        else:
            ramp_duration = rng.uniform(*ramp_bounds.ramp_duration_s)
            segments.append(
                Segment(
                    setpoint=setpoint,
                    hold_duration=DISPATCH_INTERVAL_S - ramp_duration,
                    transition_kind="ramp",
                    transition_duration=ramp_duration,
                    settle_time=ramp_duration,
                )
            )

    profile = Profile(initial=segments[0].setpoint, segments=tuple(segments))
    return profile, tuple(float(t) for t in ambient_temp_k), idle_phi, base_load_phi, min_load_fraction


def sample_sweep(
    n_cases: int,
    seed: int = 0,
    bounds: SweepBounds = SweepBounds(),
    ramp_bounds: RampBounds = RampBounds(),
) -> list[SimCase]:
    """Latin-hypercube sample over continuous engine/site params; each case
    is independently assigned a region (cycled for even coverage) and a
    randomly timed 30-day dispatch schedule following that region's demand."""
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    unit_samples = sampler.random(n=n_cases)
    scaled = qmc.scale(unit_samples, bounds.as_array()[:, 0], bounds.as_array()[:, 1])

    region_keys = list(REGIONS.keys())
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_cases):
        ambient_pressure, pressure_ratio, mdot_air_ref, volume, h2_frac = scaled[i]
        region_key = region_keys[i % len(region_keys)]
        start_day_of_year = rng.uniform(0.0, 365.0)

        profile, segment_ambient_temp_k, idle_phi, base_load_phi, min_load_fraction = _build_dispatch_schedule(
            region_key, start_day_of_year, rng, bounds, ramp_bounds
        )

        cases.append(
            SimCase(
                run_id=f"run_{i:05d}",
                region_name=region_key,
                start_day_of_year=start_day_of_year,
                ambient=AmbientConditions(temperature=segment_ambient_temp_k[0], pressure=ambient_pressure),
                segment_ambient_temp_k=segment_ambient_temp_k,
                pressure_ratio=pressure_ratio,
                fuel=FuelBlend(h2_mole_fraction=h2_frac),
                profile=profile,
                mdot_air_ref=mdot_air_ref,
                volume=volume,
                idle_phi=idle_phi,
                base_load_phi=base_load_phi,
                min_load_fraction=min_load_fraction,
            )
        )
    return cases
