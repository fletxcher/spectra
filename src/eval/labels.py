"""Ground-truth labeling for the steady-window detection benchmark.

The simulator's raw `is_steady` column marks "the commanded setpoint has
finished moving" (see src/sim/profiles.py). That's a reasonable per-sample
proxy for "physically settled", but the benchmark's actual target is
stricter: has the plant been continuously steady for at least a full
WINDOW_SECONDS (60s by default), not just steady at this one instant. A
single steady sample sitting next to a transient one is not a usable window.
"""

from __future__ import annotations

import pandas as pd

WINDOW_SECONDS = 60
REFERENCE_EPOCH = pd.Timestamp("2024-01-01")


def to_datetime_index(time_seconds: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(REFERENCE_EPOCH + pd.to_timedelta(time_seconds, unit="s"))


def steady_window_label(is_steady: pd.Series, window_seconds: int = WINDOW_SECONDS) -> pd.Series:
    """True at t iff `is_steady` has been continuously True for every sample
    in the preceding `window_seconds` (a time-based rolling AND). Requires a
    DatetimeIndex."""
    as_int = is_steady.astype(int)
    rolling_min = as_int.rolling(f"{window_seconds}s").min()
    return rolling_min.fillna(0).astype(bool)
