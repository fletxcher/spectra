"""Representative regional climate and grid-demand-seasonality parameters.

Magnitudes are directionally-correct approximations assembled from EIA,
ENTSO-E, and grid-operator reporting (US summer/winter peaks, European
winter-heating peak, Gulf-state summer AC peak, Indian summer peak growth,
etc.) -- not precise forecasts, which is adequate for driving a simplified
dispatch simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    hemisphere: str  # "N" or "S" -- which calendar half holds local summer
    summer_high_c: float  # representative seasonal daytime-high temperature
    winter_high_c: float
    diurnal_temp_amplitude_c: float  # full day/night swing
    primary_peak_month: int  # 1-12, calendar month of the dominant demand peak
    primary_peak_frac: float  # fractional demand increase at peak vs. baseline
    secondary_peak_month: int | None  # a second, smaller seasonal peak, if any
    secondary_peak_frac: float
    diurnal_shape: str  # "double" (morning+evening) or "single_midday" (AC-driven)
    diurnal_peak_to_trough: float  # ratio of daily peak demand to overnight trough
    notes: str


REGIONS: dict[str, Region] = {
    "united_states": Region(
        name="United States",
        hemisphere="N",
        summer_high_c=31.0,
        winter_high_c=-2.0,
        diurnal_temp_amplitude_c=9.0,
        primary_peak_month=7,
        primary_peak_frac=0.32,
        secondary_peak_month=1,
        secondary_peak_frac=0.15,
        diurnal_shape="double",
        diurnal_peak_to_trough=1.7,
        notes="Summer AC peak dominant nationally; secondary winter heating peak in colder regions.",
    ),
    "canada": Region(
        name="Canada",
        hemisphere="N",
        summer_high_c=27.0,
        winter_high_c=-3.5,
        diurnal_temp_amplitude_c=9.0,
        primary_peak_month=1,
        primary_peak_frac=0.25,
        secondary_peak_month=7,
        secondary_peak_frac=0.10,
        diurnal_shape="double",
        diurnal_peak_to_trough=1.5,
        notes="Winter heating-driven peak dominant; smaller summer AC peak in urban Ontario/Quebec.",
    ),
    "central_america": Region(
        name="Central America",
        hemisphere="N",
        summer_high_c=32.0,
        winter_high_c=30.5,
        diurnal_temp_amplitude_c=6.0,
        primary_peak_month=2,
        primary_peak_frac=0.08,
        secondary_peak_month=None,
        secondary_peak_frac=0.0,
        diurnal_shape="double",
        diurnal_peak_to_trough=1.5,
        notes="Tropical, flat seasonality; mild dry-season (Dec-Apr) uptick in thermal dispatch.",
    ),
    "south_america": Region(
        name="South America",
        hemisphere="S",
        summer_high_c=29.0,
        winter_high_c=15.0,
        diurnal_temp_amplitude_c=8.0,
        primary_peak_month=1,  # Southern Hemisphere summer
        primary_peak_frac=0.20,
        secondary_peak_month=7,  # Southern Hemisphere winter
        secondary_peak_frac=0.18,
        diurnal_shape="double",
        diurnal_peak_to_trough=1.6,
        notes="Hemisphere-inverted; increasingly bimodal summer-AC + winter-heating peaks (southern cone).",
    ),
    "europe": Region(
        name="Europe",
        hemisphere="N",
        summer_high_c=25.0,
        winter_high_c=4.5,
        diurnal_temp_amplitude_c=8.0,
        primary_peak_month=1,
        primary_peak_frac=0.20,
        secondary_peak_month=7,
        secondary_peak_frac=0.10,
        diurnal_shape="double",
        diurnal_peak_to_trough=1.5,
        notes="Winter heating peak dominant; growing summer AC peak in southern Europe.",
    ),
    "middle_east": Region(
        name="Middle East",
        hemisphere="N",
        summer_high_c=43.5,
        winter_high_c=21.5,
        diurnal_temp_amplitude_c=13.0,
        primary_peak_month=7,
        primary_peak_frac=0.50,
        secondary_peak_month=None,
        secondary_peak_frac=0.0,
        diurnal_shape="single_midday",
        diurnal_peak_to_trough=1.9,
        notes="Extreme summer AC-driven peak, among the most pronounced globally.",
    ),
    "russia": Region(
        name="Russia",
        hemisphere="N",
        summer_high_c=23.0,
        winter_high_c=-6.5,
        diurnal_temp_amplitude_c=8.0,
        primary_peak_month=1,
        primary_peak_frac=0.30,
        secondary_peak_month=None,
        secondary_peak_frac=0.0,
        diurnal_shape="double",
        diurnal_peak_to_trough=1.4,
        notes="Strong winter heating peak; flatter diurnal curve since heating persists overnight.",
    ),
    "china": Region(
        name="China",
        hemisphere="N",
        summer_high_c=33.0,
        winter_high_c=2.5,
        diurnal_temp_amplitude_c=9.0,
        primary_peak_month=7,
        primary_peak_frac=0.25,
        secondary_peak_month=1,
        secondary_peak_frac=0.15,
        diurnal_shape="double",
        diurnal_peak_to_trough=1.6,
        notes="Increasingly bimodal: summer AC peak (south/central), growing winter heating peak (north).",
    ),
    "india": Region(
        name="India",
        hemisphere="N",
        summer_high_c=41.5,
        winter_high_c=21.0,
        diurnal_temp_amplitude_c=10.0,
        primary_peak_month=5,
        primary_peak_frac=0.35,
        secondary_peak_month=None,
        secondary_peak_frac=0.0,
        diurnal_shape="single_midday",
        diurnal_peak_to_trough=1.9,
        notes="Summer (Apr-Jun) AC-driven peak dominant and intensifying; demand dips in monsoon/winter.",
    ),
}
