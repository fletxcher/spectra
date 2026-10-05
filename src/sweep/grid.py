"""Generates a sweep of SimCases: each run simulates 30 days of a large gas
turbine following grid electricity demand in a particular world region. The
dispatched setpoint (equivalence ratio / load) tracks a demand signal built
from that region's seasonal + diurnal + weekday/weekend pattern, re-dispatched
at irregular intervals (frequent through the morning/evening ramps, sparse
otherwise, with occasional short real-time corrections), with ramp-rate-limited
transitions between levels. Unplanned disturbances (runbacks, fuel shifts,
combustion-dynamics bursts) are injected into holds. Ambient temperature is
driven by the same region's diurnal/seasonal climate. Engine design
parameters (pressure ratio, reference mass flow, combustor volume, fuel blend,
site elevation) are Latin-hypercube sampled across runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import qmc

from src.sim import demand
from src.sim.conditions import AmbientConditions
from src.sim.fuel import FuelBlend
from src.sim.profiles import Disturbance, Profile, Segment, Setpoint
from src.sim.regions import REGIONS
from src.sim.runner import SimCase

TOTAL_DAYS = 30
RAMP_HOURS = ((6.0, 10.0), (16.0, 21.0))  # local hours when demand moves fastest


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
class DispatchBounds:
    ramp_hour_interval_s: tuple[float, float] = (15 * 60.0, 60 * 60.0)
    off_hour_interval_s: tuple[float, float] = (60 * 60.0, 4 * 3600.0)
    correction_prob: float = 0.08  # chance a given interval is a short real-time correction
    correction_interval_s: tuple[float, float] = (2 * 60.0, 10 * 60.0)
    ramp_rate_per_min: tuple[float, float] = (0.02, 0.06)  # fraction of full load per minute, per run
    min_ramp_s: float = 60.0
    max_ramp_frac: float = 0.8  # a ramp may use at most this fraction of its interval
    demand_floor: float = 0.2  # minimum normalized demand (plant stays committed, not shut down)


@dataclass(frozen=True)
class AgcBounds:
    """Per-run AGC regulation wander. sigma is the stationary std of the
    load_fraction offset; a wide range so some units barely regulate and
    others regulate hard."""

    sigma: tuple[float, float] = (0.002, 0.015)
    tau_s: tuple[float, float] = (60.0, 300.0)
    steady_tolerance: float = 0.005  # +/- load_fraction band that still counts as steady


@dataclass(frozen=True)
class DisturbanceBounds:
    rate_per_day: tuple[float, float] = (1.0, 3.0)  # sampled once per run
    kind_weights: tuple[tuple[str, float], ...] = (("runback", 0.3), ("fuel_shift", 0.35), ("dynamics", 0.35))
    lead_s: float = 120.0  # minimum settled time before an event starts
    trail_s: float = 60.0  # minimum settled time after an event's settle buffer


def _dispatch_intervals(rng: np.random.Generator, db: DispatchBounds) -> tuple[np.ndarray, np.ndarray]:
    total = TOTAL_DAYS * 86400.0
    starts, durations = [], []
    t = 0.0
    while t < total:
        hour = (t / 3600.0) % 24.0
        if rng.random() < db.correction_prob:
            lo, hi = db.correction_interval_s
        elif any(a <= hour < b for a, b in RAMP_HOURS):
            lo, hi = db.ramp_hour_interval_s
        else:
            lo, hi = db.off_hour_interval_s
        d = rng.uniform(lo, hi)
        if total - (t + d) < db.correction_interval_s[0]:
            d = total - t  # absorb a sliver at the end instead of leaving a tiny last segment
        starts.append(t)
        durations.append(d)
        t += d
    return np.array(starts), np.array(durations)


def _sample_disturbance(kind: str, rng: np.random.Generator) -> Disturbance:
    if kind == "runback":
        load_delta = -rng.uniform(0.15, 0.35)
        return Disturbance(
            kind="runback", start=0.0,
            ramp_in=rng.uniform(30.0, 90.0), hold=rng.uniform(120.0, 600.0), ramp_out=rng.uniform(180.0, 600.0),
            load_delta=load_delta, phi_delta=0.5 * load_delta,
        )
    if kind == "fuel_shift":
        return Disturbance(
            kind="fuel_shift", start=0.0,
            ramp_in=rng.uniform(60.0, 300.0), hold=rng.uniform(1200.0, 5400.0), ramp_out=rng.uniform(60.0, 300.0),
            phi_delta=rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.08),
        )
    return Disturbance(
        kind="dynamics", start=0.0,
        ramp_in=rng.uniform(30.0, 60.0), hold=rng.uniform(120.0, 480.0), ramp_out=rng.uniform(60.0, 120.0),
        osc_amplitude=rng.uniform(0.02, 0.05), osc_period=rng.uniform(60.0, 240.0),
    )


def _place_disturbances(
    segments: list[Segment], starts: np.ndarray, rng: np.random.Generator, dist: DisturbanceBounds
) -> tuple[Disturbance, ...]:
    """At most one event per segment, Poisson-thinned by hold length, placed
    so it starts and ends inside that segment's settled hold."""
    rate = rng.uniform(*dist.rate_per_day)
    kinds = [k for k, _ in dist.kind_weights]
    weights = np.array([w for _, w in dist.kind_weights])
    weights = weights / weights.sum()

    events = []
    for seg, seg_start in zip(segments, starts):
        hold_start = seg_start + seg.settle_time
        seg_end = seg_start + seg.duration
        if rng.random() >= 1.0 - math.exp(-rate * (seg_end - hold_start) / 86400.0):
            continue
        d = _sample_disturbance(str(rng.choice(kinds, p=weights)), rng)
        earliest = hold_start + dist.lead_s
        latest = seg_end - (d.end - d.start) - d.settle_buffer - dist.trail_s
        if latest <= earliest:
            continue
        events.append(replace(d, start=float(rng.uniform(earliest, latest))))
    return tuple(events)


def _build_dispatch_schedule(
    region_key: str,
    start_day_of_year: float,
    rng: np.random.Generator,
    bounds: SweepBounds,
    dispatch_bounds: DispatchBounds,
    disturbance_bounds: DisturbanceBounds,
    agc_bounds: AgcBounds,
):
    region = REGIONS[region_key]

    seg_starts, seg_durations = _dispatch_intervals(rng, dispatch_bounds)
    n = len(seg_starts)
    segment_mid_s = seg_starts + seg_durations / 2.0
    hour_of_day = (segment_mid_s / 3600.0) % 24.0
    day_index = np.floor(segment_mid_s / 86400.0)
    # Seasonal baseline is fixed at the run's start date, not left to advance
    # across the window: a calendar month's climatological mean doesn't
    # meaningfully shift over 30 days -- that's a year-timescale effect, not
    # a within-month one. Only the diurnal (hour_of_day) and weekday/weekend
    # (day_index) components should evolve across the 30 days.
    season_day_of_year = np.full(n, start_day_of_year)

    raw = demand.raw_demand(region, season_day_of_year, hour_of_day, day_index, rng)
    demand_norm = demand.normalize(raw, floor=dispatch_bounds.demand_floor)
    ambient_temp_k = demand.ambient_temperature_k(region, season_day_of_year, hour_of_day, rng)

    idle_phi = rng.uniform(*bounds.idle_phi)
    base_load_phi = rng.uniform(*bounds.base_load_phi)
    min_load_fraction = rng.uniform(*bounds.min_load_fraction)
    ramp_rate_per_s = rng.uniform(*dispatch_bounds.ramp_rate_per_min) / 60.0

    phi = idle_phi + (base_load_phi - idle_phi) * demand_norm
    load_fraction = min_load_fraction + (1.0 - min_load_fraction) * demand_norm

    segments = []
    for j in range(n):
        setpoint = Setpoint(phi=float(phi[j]), load_fraction=float(load_fraction[j]))
        if j == 0:
            segments.append(
                Segment(setpoint=setpoint, hold_duration=float(seg_durations[0]), transition_kind="step", settle_time=60.0)
            )
            continue
        ramp = max(dispatch_bounds.min_ramp_s, abs(load_fraction[j] - load_fraction[j - 1]) / ramp_rate_per_s)
        ramp = min(ramp, dispatch_bounds.max_ramp_frac * seg_durations[j])
        segments.append(
            Segment(
                setpoint=setpoint,
                hold_duration=float(seg_durations[j] - ramp),
                transition_kind="ramp",
                transition_duration=float(ramp),
                settle_time=float(ramp),
            )
        )

    disturbances = _place_disturbances(segments, seg_starts, rng, disturbance_bounds)
    profile = Profile(
        initial=segments[0].setpoint,
        segments=tuple(segments),
        disturbances=disturbances,
        agc_sigma=float(rng.uniform(*agc_bounds.sigma)),
        agc_tau=float(rng.uniform(*agc_bounds.tau_s)),
        # phi moves with load along the same mapping dispatch uses
        agc_phi_per_load=float((base_load_phi - idle_phi) / (1.0 - min_load_fraction)),
        agc_seed=int(rng.integers(2**32)),
        steady_tolerance=agc_bounds.steady_tolerance,
    )
    return profile, tuple(float(t) for t in ambient_temp_k), idle_phi, base_load_phi, min_load_fraction


def sample_sweep(
    n_cases: int,
    seed: int = 0,
    bounds: SweepBounds = SweepBounds(),
    dispatch_bounds: DispatchBounds = DispatchBounds(),
    disturbance_bounds: DisturbanceBounds = DisturbanceBounds(),
    agc_bounds: AgcBounds = AgcBounds(),
    fine_dt: float = 30.0,
    coarse_dt: float = 300.0,
    fine_window: float = 300.0,
) -> list[SimCase]:
    """Latin-hypercube sample over continuous engine/site params; each case
    is independently assigned a region (cycled for even coverage), a
    randomly timed 30-day dispatch schedule following that region's demand,
    and a set of disturbances."""
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
            region_key, start_day_of_year, rng, bounds, dispatch_bounds, disturbance_bounds, agc_bounds
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
                fine_dt=fine_dt,
                coarse_dt=coarse_dt,
                fine_window=fine_window,
            )
        )
    return cases
