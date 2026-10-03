"""Turns a clean (truth) simulation trace into a noisy 'observed' trace,
mimicking real plant data acquisition: measurement noise, sensor lag, and
occasional dropped samples. CPD methods are evaluated against the observed
columns and scored against the *_truth / segment_id / is_steady ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NOISY_COLUMNS = ("temperature", "pressure", "mdot_air", "mdot_fuel")


def add_sensor_noise(
    df: pd.DataFrame,
    snr_db: float = 40.0,
    sensor_tau: float = 0.0,
    dropout_prob: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Returns a copy of `df` with `<col>_truth` (renamed originals) and
    `<col>` (noisy, sensor-lagged, possibly dropped-out observed values) for
    each column in NOISY_COLUMNS."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    dt = df["time"].diff().median() if len(df) > 1 else 0.0

    for col in NOISY_COLUMNS:
        truth = df[col].to_numpy(dtype=float)
        out[f"{col}_truth"] = truth

        signal_power = np.mean(truth**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0.0, np.sqrt(max(noise_power, 0.0)), size=truth.shape)
        noisy = truth + noise

        if sensor_tau > 0 and dt > 0:
            alpha = dt / (sensor_tau + dt)
            lagged = np.empty_like(noisy)
            lagged[0] = noisy[0]
            for i in range(1, len(noisy)):
                lagged[i] = lagged[i - 1] + alpha * (noisy[i] - lagged[i - 1])
            noisy = lagged

        if dropout_prob > 0:
            dropped = rng.random(size=noisy.shape) < dropout_prob
            noisy = noisy.copy()
            noisy[dropped] = np.nan

        out[col] = noisy

    return out
