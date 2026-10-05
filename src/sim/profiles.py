"""Time-varying operating profiles: sequences of held setpoints connected by
step / ramp / damped-oscillation transitions, plus unplanned disturbances
during holds, and the ground-truth labeling (segment id, steady/transient)
that detection methods are scored against.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd

TransitionKind = Literal["step", "ramp", "oscillation"]
STABILITY_WINDOW_S = 60.0
# An AGC-only excursion outside the band must persist this long to count as
# transient. Shorter ones are AGC grazing the band edge, which a hard
# threshold on a wandering signal turns into tens of thousands of ~30 s
# flickers per run that don't correspond to any real operating-point change.
MIN_AGC_TRANSIENT_S = 120.0
DisturbanceKind = Literal["runback", "fuel_shift", "dynamics"]


@dataclass(frozen=True)
class Setpoint:
    """A commanded operating point."""

    phi: float  # equivalence ratio
    load_fraction: float  # fraction of the reference total mass flow, 0-1+


@dataclass(frozen=True)
class Segment:
    """One held setpoint, reached via a transition from the previous segment.

    `transition_duration` and `settle_time` are both measured from the start
    of this segment (i.e. from when the transition into it begins).
    `settle_time` must be >= `transition_duration`; it encodes how long after
    the transition starts we consider the plant to have actually reached
    steady state (driven by residence time + chemical relaxation, estimated
    by the caller), not just how long the commanded setpoint took to move.
    """

    setpoint: Setpoint
    hold_duration: float  # seconds spent at `setpoint` after settling
    transition_kind: TransitionKind = "step"
    transition_duration: float = 0.0
    settle_time: float = 0.0
    # oscillation-only: fractional overshoot and damped-cosine frequency (Hz)
    overshoot: float = 0.15
    osc_frequency: float = 0.5

    @property
    def duration(self) -> float:
        return self.transition_duration + self.hold_duration


@dataclass(frozen=True)
class Disturbance:
    """An unplanned event during a hold, applied on top of the dispatched
    setpoint with a trapezoidal envelope (ramp in, hold, ramp out).

    - runback: fast load (and fuel) reduction, short hold, slower recovery.
    - fuel_shift: fuel heating-value drift, modeled as an effective phi
      offset that settles at a new level for tens of minutes.
    - dynamics: combustion-dynamics burst, a phi oscillation of
      `osc_amplitude` and `osc_period` under the envelope.

    Ground truth: runback/fuel_shift are transient through each ramp plus
    `settle_buffer`, and steady on their plateau (a genuine, if temporary,
    new operating point). dynamics is transient for its whole duration.
    """

    kind: DisturbanceKind
    start: float  # absolute run time, seconds
    ramp_in: float
    hold: float
    ramp_out: float
    load_delta: float = 0.0  # fractional change to load_fraction at full envelope
    phi_delta: float = 0.0  # fractional change to phi at full envelope
    osc_amplitude: float = 0.0
    osc_period: float = 0.0
    settle_buffer: float = 60.0

    @property
    def end(self) -> float:
        return self.start + self.ramp_in + self.hold + self.ramp_out

    def envelope(self, t: float) -> float:
        e = t - self.start
        if e < 0 or t >= self.end:
            return 0.0
        if e < self.ramp_in:
            return e / self.ramp_in
        if e < self.ramp_in + self.hold:
            return 1.0
        return 1.0 - (e - self.ramp_in - self.hold) / self.ramp_out

    def apply(self, sp: Setpoint, t: float) -> Setpoint:
        env = self.envelope(t)
        if env == 0.0:
            return sp
        phi_scale = 1.0 + self.phi_delta * env
        if self.kind == "dynamics":
            phi_scale += self.osc_amplitude * env * math.sin(2 * math.pi * (t - self.start) / self.osc_period)
        return Setpoint(phi=sp.phi * phi_scale, load_fraction=sp.load_fraction * (1.0 + self.load_delta * env))

    def is_transient(self, t: float) -> bool:
        if t < self.start or t >= self.end + self.settle_buffer:
            return False
        if self.kind == "dynamics":
            return True
        plateau_start = self.start + self.ramp_in + self.settle_buffer
        plateau_end = self.start + self.ramp_in + self.hold
        return not (plateau_start <= t < plateau_end)


@dataclass(frozen=True)
class Profile:
    """A full run: an initial setpoint followed by a sequence of segments,
    plus non-overlapping disturbances sorted by start time, plus AGC
    (automatic generation control) wander on top of everything.

    AGC is an Ornstein-Uhlenbeck process on load_fraction (stationary std
    `agc_sigma`, correlation time `agc_tau`), with phi moving alongside it by
    `agc_phi_per_load`. It's generated lazily from `agc_seed` so a profile
    stays small until it's actually integrated.

    Steady means: the dispatch ramp has settled, no disturbance transient is
    in progress (`label_at`), AND the commanded operating point has been
    stable over the trailing STABILITY_WINDOW_S: load stayed inside a
    +/-`steady_tolerance` band (range <= 2 * tolerance, in load_fraction
    units) and phi inside the equivalent band (`steady_mask`), unless the
    band was only left by AGC for less than MIN_AGC_TRANSIENT_S
    (`steady_labels`). Stability is judged on the signal's own recent
    history, not distance from the dispatch target, because a detector can
    only ever observe the former.

    With runs spanning many hundreds of segments (a 30-day dispatch
    schedule), lookups are done via `bisect` against precomputed start
    times rather than a linear scan.
    """

    initial: Setpoint
    segments: tuple[Segment, ...]
    disturbances: tuple[Disturbance, ...] = ()
    agc_sigma: float = 0.0
    agc_tau: float = 120.0
    agc_phi_per_load: float = 0.0
    agc_seed: int = 0
    agc_dt: float = 10.0
    steady_tolerance: float = 0.005

    @cached_property
    def _agc_grid(self) -> np.ndarray:
        n = int(self.total_duration / self.agc_dt) + 2
        if self.agc_sigma == 0.0:
            return np.zeros(n)
        rng = np.random.default_rng(self.agc_seed)
        a = math.exp(-self.agc_dt / self.agc_tau)
        shocks = rng.normal(0.0, self.agc_sigma * math.sqrt(1.0 - a * a), size=n)
        x = np.empty(n)
        x[0] = rng.normal(0.0, self.agc_sigma)
        for k in range(1, n):
            x[k] = a * x[k - 1] + shocks[k]
        return x

    def agc_offset(self, t: float) -> float:
        """Raw AGC load_fraction offset at t (linear interpolation)."""
        grid = self._agc_grid
        pos = t / self.agc_dt
        k = min(int(pos), len(grid) - 2)
        frac = pos - k
        return float(grid[k] * (1.0 - frac) + grid[k + 1] * frac)

    def _base_setpoint(self, t: float) -> Setpoint:
        """Dispatched setpoint plus any disturbance, before AGC."""
        idx = self._segment_index(t)
        seg = self.segments[idx]
        prev = self.initial if idx == 0 else self.segments[idx - 1].setpoint
        sp = _interpolate(prev, seg, t - self.segment_starts[idx])
        d = self.disturbance_at(t)
        return d.apply(sp, t) if d is not None else sp

    def _agc_delta(self, t: float, base: Setpoint) -> float:
        """AGC load offset actually applied, clipped so load never exceeds
        rated (1.0) -- a unit at full load can't regulate further up."""
        return min(base.load_fraction + self.agc_offset(t), 1.0) - base.load_fraction

    @cached_property
    def segment_starts(self) -> list[float]:
        starts = []
        t = 0.0
        for seg in self.segments:
            starts.append(t)
            t += seg.duration
        return starts

    @cached_property
    def _disturbance_starts(self) -> list[float]:
        return [d.start for d in self.disturbances]

    @cached_property
    def total_duration(self) -> float:
        return sum(seg.duration for seg in self.segments)

    def _segment_index(self, t: float) -> int:
        idx = bisect.bisect_right(self.segment_starts, t) - 1
        return min(max(idx, 0), len(self.segments) - 1)

    def disturbance_at(self, t: float) -> Disturbance | None:
        """The disturbance whose window (including its settle buffer)
        contains t, if any."""
        idx = bisect.bisect_right(self._disturbance_starts, t) - 1
        if idx < 0:
            return None
        d = self.disturbances[idx]
        return d if t < d.end + d.settle_buffer else None

    def setpoint_at(self, t: float) -> Setpoint:
        base = self._base_setpoint(t)
        delta = self._agc_delta(t, base)
        return Setpoint(phi=base.phi + self.agc_phi_per_load * delta, load_fraction=base.load_fraction + delta)

    def phi(self, t: float) -> float:
        return self.setpoint_at(t).phi

    def load_fraction(self, t: float) -> float:
        return self.setpoint_at(t).load_fraction

    def label_at(self, t: float) -> tuple[int, bool]:
        """Returns (segment_index, settled): the dispatch ramp has finished
        and no disturbance transient is in progress. Combine with
        `steady_mask` for the full steady label."""
        idx = self._segment_index(t)
        seg = self.segments[idx]
        settled = t - self.segment_starts[idx] >= seg.settle_time
        d = self.disturbance_at(t)
        return idx, settled and not (d is not None and d.is_transient(t))

    def steady_mask(self, time_s: np.ndarray, load: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """True where the commanded load and phi both stayed inside their
        +/-tolerance band over the trailing STABILITY_WINDOW_S."""
        idx = pd.to_datetime(time_s, unit="s")
        window = f"{STABILITY_WINDOW_S:g}s"

        def stable(x: np.ndarray, band: float) -> np.ndarray:
            s = pd.Series(x, index=idx)
            return ((s.rolling(window).max() - s.rolling(window).min()) <= 2.0 * band).to_numpy()

        phi_band = self.steady_tolerance * self.agc_phi_per_load
        return stable(load, self.steady_tolerance) & stable(phi, phi_band)

    def steady_labels(self, time_s: np.ndarray, load: np.ndarray, phi: np.ndarray, settled: np.ndarray) -> np.ndarray:
        """Full ground-truth steady label. `settled` is `label_at` evaluated
        at each sample (ramps and disturbance transients always count as not
        steady). On top of that, samples outside the stability band are not
        steady, except AGC-only gaps shorter than MIN_AGC_TRANSIENT_S, which
        are filled back in as steady-with-jitter."""
        steady = settled & self.steady_mask(time_s, load, phi)
        edges = np.flatnonzero(np.diff(np.concatenate(([1], steady.astype(np.int8), [1]))))
        end_time = np.append(time_s, time_s[-1] + (time_s[-1] - time_s[-2] if len(time_s) > 1 else 0.0))
        for a, b in zip(edges[0::2], edges[1::2]):
            agc_only = settled[a:b].all()
            if agc_only and end_time[b] - time_s[a] < MIN_AGC_TRANSIENT_S:
                steady[a:b] = True
        return steady

    def sample_times(self, fine_dt: float, coarse_dt: float, fine_window: float) -> np.ndarray:
        """Non-uniform output times: `fine_dt` resolution through each
        segment's transition and each disturbance, plus a settling window
        after them (so transients are well resolved), `coarse_dt` resolution
        through the remaining held/steady portion (since a flat plateau
        carries no extra information at fine resolution). Mirrors how real
        plant historians are far denser around events than during steady
        running."""
        if fine_dt >= coarse_dt:
            # uniform mode: one global grid, so irregular segment starts don't
            # leave off-grid samples at every boundary
            return np.arange(0.0, self.total_duration, fine_dt)
        times: list[float] = [0.0]
        for idx, seg in enumerate(self.segments):
            start = self.segment_starts[idx]
            fine_end = min(seg.duration, seg.transition_duration + fine_window)
            t = fine_dt
            while t < fine_end:
                times.append(start + t)
                t += fine_dt
            t = fine_end
            while t < seg.duration:
                times.append(start + t)
                t += coarse_dt
        for d in self.disturbances:
            t = d.start
            while t < d.end + d.settle_buffer + fine_window:
                times.append(t)
                t += fine_dt
        total = self.total_duration
        times = [t for t in times if t < total]
        return np.array(sorted(set(times)))


def _interpolate(prev: Setpoint, seg: Segment, elapsed: float) -> Setpoint:
    """Setpoint value at `elapsed` seconds into `seg` (elapsed may extend
    past transition_duration, in which case we're in the held period)."""
    target = seg.setpoint
    dur = seg.transition_duration
    if dur <= 0 or elapsed >= dur:
        return target

    frac = elapsed / dur

    if seg.transition_kind == "step":
        return target

    if seg.transition_kind == "ramp":
        return _lerp(prev, target, frac)

    if seg.transition_kind == "oscillation":
        decay = math.exp(-3.0 * frac)
        osc = seg.overshoot * decay * math.cos(2 * math.pi * seg.osc_frequency * elapsed)
        base = _lerp(prev, target, frac)
        return Setpoint(
            phi=base.phi + osc * target.phi,
            load_fraction=base.load_fraction + osc * target.load_fraction,
        )

    raise ValueError(f"unknown transition kind: {seg.transition_kind}")


def _lerp(a: Setpoint, b: Setpoint, frac: float) -> Setpoint:
    return Setpoint(
        phi=a.phi + (b.phi - a.phi) * frac,
        load_fraction=a.load_fraction + (b.load_fraction - a.load_fraction) * frac,
    )
