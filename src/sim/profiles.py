"""Time-varying operating profiles: sequences of held setpoints connected by
step / ramp / damped-oscillation transitions, plus the ground-truth labeling
(segment id, steady/transient) that CPD methods will eventually be scored
against.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

TransitionKind = Literal["step", "ramp", "oscillation"]


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
class Profile:
    """A full run: an initial setpoint followed by a sequence of segments.

    With runs now spanning many hundreds of segments (a 30-day dispatch
    schedule), lookups are done via `bisect` against precomputed segment
    start times rather than a linear scan.
    """

    initial: Setpoint
    segments: tuple[Segment, ...]

    @cached_property
    def _starts(self) -> list[float]:
        starts = []
        t = 0.0
        for seg in self.segments:
            starts.append(t)
            t += seg.duration
        return starts

    @cached_property
    def total_duration(self) -> float:
        return sum(seg.duration for seg in self.segments)

    def _segment_index(self, t: float) -> int:
        idx = bisect.bisect_right(self._starts, t) - 1
        return min(max(idx, 0), len(self.segments) - 1)

    def setpoint_at(self, t: float) -> Setpoint:
        idx = self._segment_index(t)
        seg = self.segments[idx]
        prev = self.initial if idx == 0 else self.segments[idx - 1].setpoint
        return _interpolate(prev, seg, t - self._starts[idx])

    def phi(self, t: float) -> float:
        return self.setpoint_at(t).phi

    def load_fraction(self, t: float) -> float:
        return self.setpoint_at(t).load_fraction

    def label_at(self, t: float) -> tuple[int, bool]:
        """Returns (segment_index, is_steady)."""
        idx = self._segment_index(t)
        seg = self.segments[idx]
        elapsed = t - self._starts[idx]
        return idx, elapsed >= seg.settle_time

    def sample_times(self, fine_dt: float, coarse_dt: float, fine_window: float) -> np.ndarray:
        """Non-uniform output times: `fine_dt` resolution through each
        segment's transition plus a settling window after it (so transients
        are well resolved), `coarse_dt` resolution through the remaining
        held/steady portion (since a flat plateau carries no extra
        information at fine resolution). Mirrors how real plant historians
        are far denser around events than during steady running."""
        times: list[float] = [0.0]
        for idx, seg in enumerate(self.segments):
            start = self._starts[idx]
            fine_end = min(seg.duration, seg.transition_duration + fine_window)
            t = fine_dt
            while t < fine_end:
                times.append(start + t)
                t += fine_dt
            t = fine_end
            while t < seg.duration:
                times.append(start + t)
                t += coarse_dt
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
