"""Builds a 30-day grid-demand signal and matching ambient-temperature
signal for a given Region, combining seasonal, diurnal, weekday/weekend, and
day-to-day weather-noise components. This is what drives the gas turbine's
dispatched load/equivalence-ratio setpoint over the course of a run: operators
follow grid demand, so the plant's operating point tracks it."""

from __future__ import annotations

import numpy as np

from src.sim.regions import Region

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
DAYS_PER_MONTH = 30.4368

# Hourly demand-shape templates (arbitrary units, re-scaled per region to the
# region's actual peak/trough ratio). Trough ~2-5 AM, morning ramp ~6-9 AM;
# "double" peaks morning+evening (heating-mixed climates), "single_midday"
# sustains a broad midday-to-evening AC-driven plateau.
_DOUBLE_PEAK_TEMPLATE = np.array(
    [
        0.55, 0.50, 0.45, 0.42, 0.42, 0.45, 0.55, 0.70, 0.85, 0.90, 0.88, 0.85,
        0.80, 0.78, 0.80, 0.85, 0.90, 0.97, 1.00, 0.98, 0.90, 0.80, 0.70, 0.60,
    ]
)
_SINGLE_MIDDAY_TEMPLATE = np.array(
    [
        0.50, 0.45, 0.40, 0.38, 0.38, 0.40, 0.48, 0.60, 0.72, 0.82, 0.90, 0.95,
        0.98, 1.00, 1.00, 0.99, 0.97, 0.95, 0.90, 0.85, 0.78, 0.70, 0.62, 0.55,
    ]
)


def _month_of_year(day_of_year: np.ndarray) -> np.ndarray:
    return 1.0 + (day_of_year % 365.25) / DAYS_PER_MONTH


def _circular_distance(a: np.ndarray, b: float, period: float) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def seasonal_multiplier(region: Region, day_of_year: np.ndarray) -> np.ndarray:
    month = _month_of_year(day_of_year)
    width = 1.5  # months, Gaussian half-width of each seasonal bump

    def bump(peak_month: int | None, frac: float) -> np.ndarray:
        if peak_month is None or frac == 0.0:
            return np.zeros_like(month)
        dist = _circular_distance(month, float(peak_month), 12.0)
        return frac * np.exp(-0.5 * (dist / width) ** 2)

    return (
        1.0
        + bump(region.primary_peak_month, region.primary_peak_frac)
        + bump(region.secondary_peak_month, region.secondary_peak_frac)
    )


def diurnal_multiplier(region: Region, hour_of_day: np.ndarray) -> np.ndarray:
    template = _DOUBLE_PEAK_TEMPLATE if region.diurnal_shape == "double" else _SINGLE_MIDDAY_TEMPLATE
    hours = np.arange(24)
    raw = np.interp(hour_of_day % 24.0, hours, template, period=24.0)

    ratio = region.diurnal_peak_to_trough
    target_min = 2.0 / (1.0 + ratio)
    target_max = target_min * ratio
    raw_min, raw_max = template.min(), template.max()
    return target_min + (raw - raw_min) / (raw_max - raw_min) * (target_max - target_min)


def ambient_temperature_k(
    region: Region, day_of_year: np.ndarray, hour_of_day: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    mean_c = (region.summer_high_c + region.winter_high_c) / 2.0
    amplitude_c = (region.summer_high_c - region.winter_high_c) / 2.0
    warmest_month = 7.0 if region.hemisphere == "N" else 1.0

    month = _month_of_year(day_of_year)
    seasonal = amplitude_c * np.cos(2 * np.pi * (month - warmest_month) / 12.0)
    diurnal = (region.diurnal_temp_amplitude_c / 2.0) * np.cos(2 * np.pi * (hour_of_day - 15.0) / 24.0)
    weather_noise = rng.normal(0.0, 1.5, size=np.shape(day_of_year))

    return mean_c + seasonal + diurnal + weather_noise + 273.15


def raw_demand(
    region: Region,
    day_of_year: np.ndarray,
    hour_of_day: np.ndarray,
    day_index: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unnormalized demand signal: seasonal x diurnal x weekday/weekend x
    day-to-day noise. Callers should normalize against a run's own observed
    range (see `normalize`)."""
    seasonal = seasonal_multiplier(region, day_of_year)
    diurnal = diurnal_multiplier(region, hour_of_day)
    is_weekend = (day_index.astype(int) % 7) >= 5
    weekday_factor = np.where(is_weekend, 0.92, 1.0)
    noise = 1.0 + rng.normal(0.0, 0.04, size=np.shape(day_of_year))
    return seasonal * diurnal * weekday_factor * noise


def normalize(raw: np.ndarray, floor: float = 0.2) -> np.ndarray:
    """Maps a run's raw demand trace to [floor, 1.0], peaking at exactly 1.0
    at that run's own maximum (representing the plant's rated output)."""
    normalized = raw / raw.max()
    return np.clip(normalized, floor, 1.0)
