# SPECTRA

**S**teady **P**oint **E**xtraction and **C**lustering for **T**ransient **R**egime **A**nalysis

A framework for combustion time-series analysis: extracting steady-state
operating points from transient data and automatically grouping them.

## Phase 1: simulation

Before evaluating any extraction/clustering technique, we need labeled
transient-to-steady time-series data. `spectra` simulates a large gas
turbine's combustion chamber as a 0-D Cantera reactor network (GRI-Mech 3.0
chemistry, CH4/H2 fuel blends) over a **30-day period**, with the dispatched
setpoint (equivalence ratio / load) tracking realistic grid electricity
demand for one of 9 world regions (United States, Canada, Central America,
South America, Europe, Middle East, Russia, China, India) — demand is
re-evaluated every 4 hours from that region's seasonal + diurnal +
weekday/weekend pattern, with realistic ramps (minutes, no overshoot) between
levels. Ambient temperature follows the same region's diurnal/seasonal
climate. Engine design parameters (pressure ratio, reference mass flow,
combustor volume, fuel blend, site elevation) are Latin-hypercube sampled
across runs.

Each run is written with both the ground truth and a noisy "observed" trace,
plus per-timestep `segment_id` / `is_steady` labels to score extraction
methods against.

```bash
uv run spectra sweep --n-cases 200 --out datasets --workers 8
```

Output: one directory per run, `datasets/<run_id>/`, containing:

- `data.parquet` — the time series (truth + noisy observed columns, plus
  per-timestep `segment_id` / `is_steady` ground truth)
- `conditions.txt` — region, simulated date range, site/engine parameters,
  control-schedule setpoints, and a daily summary table of ambient
  temperature / load / phi / adiabatic flame temperature over the 30 days
- `temperature.png` — chamber temperature (TIT proxy) vs. the equilibrium
  adiabatic flame temperature, over the full 30-day run
- `emissions.png` — NOx and CO (converted from tracked species mass
  fractions to molar ppm) over the full 30-day run

## Modeling assumptions

This is a simplified engineering model built for generating realistic
*shapes* of transient/steady behavior, not a validated performance or
emissions prediction tool. Key simplifications:

**Combustor physics**
- The chamber is a single 0-D well-stirred reactor (`IdealGasReactor`), not
  a 3-D CFD flame or a multi-stage/multi-can combustor. It represents one
  can/basket of a can-annular or annular combustor, not the full engine.
- Chemistry is GRI-Mech 3.0 (53 species), developed for natural-gas
  combustion; it also covers H2 chemistry, used here as a simplified proxy
  for CH4/H2-blended fuel. It is not a surrogate mechanism for heavier
  liquid fuels.
- The combustor starts pre-ignited: it's initialized at the full chemical
  equilibrium of the run's starting setpoint, skipping cold-start ignition
  transients (which are out of scope — this project cares about
  steady/transient *operating-point* behavior, not ignition dynamics). This
  does leave a brief, non-physical first-sample artifact in trace species
  (NO/CO) that the emissions plot deliberately excludes from its axis scale.
- Compressor discharge conditions use a simple isentropic-compression +
  fixed isentropic-efficiency relation, not a real compressor map; inlet
  pressure loss, bleed, and IGV effects are not modeled.
- Chamber pressure is held near target via a `PressureController` against a
  fixed-pressure exhaust reservoir, approximating constant turbine
  nozzle/backpressure conditions downstream.
- Site/ambient pressure (elevation) is fixed per run; only ambient
  *temperature* varies over the 30 days.

**Dispatch / control schedule**
- Equivalence ratio and load both scale linearly with a single normalized
  "demand" signal between a sampled turndown point and a sampled full-load
  point — real multi-variable combustion control (staging, pilot/main
  splits, IGV schedules) is not modeled.
- Setpoint changes ramp over minutes without overshoot, consistent with how
  real heavy-duty turbine control systems actively avoid firing-temperature
  overshoot (see Resources). Fast cold-start/ignition transients are
  excluded by design (see above).
- A run's seasonal baseline (the climatological "which month is it")
  is fixed at that run's start date — only the diurnal and weekday/weekend
  components evolve across the 30 simulated days, since a calendar month's
  climatological mean does not meaningfully drift over just 30 days; that's
  a year-timescale effect, not a within-month one.
- Grid demand seasonality, diurnal shape, and weekday/weekend variation are
  hand-parameterized per region from the directional magnitudes in the
  Resources below, not fit to measured load data.

**Emissions**
- NOx (NO+NO2) and CO are read directly from the kinetic simulation's
  species mass fractions and converted to molar ppm (wet, uncorrected — no
  dry-basis or 15%-O2 correction is applied). A single 0-D WSR tends to
  over-predict both relative to a real staged/lean-premix combustor, so
  treat these as relative trends across the sweep, not compliance-grade
  figures.

**Sensors**
- The "observed" trace adds Gaussian measurement noise (default 40 dB SNR)
  to temperature/pressure/mass-flow; optional sensor lag and dropout are
  supported but off by default.

## Resources

Assumptions about grid demand seasonality/shape and turbine startup/loading
behavior were informed by:

- [EIA, Electricity Load Shapes handbook](https://www.eia.gov/analysis/handbook/pdf/Handbook%20Section%20B3_Electricity%20Load%20Shapes.pdf)
- [EIA, Today in Energy — id=10211](https://www.eia.gov/todayinenergy/detail.php?id=10211)
- [EIA, Today in Energy — id=46117](https://www.eia.gov/todayinenergy/detail.php?id=46117)
- [EIA, Today in Energy — id=29112](https://www.eia.gov/todayinenergy/detail.php?id=29112)
- [ENTSO-E, Winter/Summer Outlook reports](https://eepublicdownloads.entsoe.eu)
- [ISO New England, System Load Graph](https://isonewswire.com/2025/06/16/system-load-graph-tracks-ebb-and-flow-of-daily-electricity-use/)
- [Green Building Advisor, duck-curve explainer](https://www.greenbuildingadvisor.com/article/an-introduction-to-the-duck-curve)
- [ORF, Energy News Monitor (Indian peak demand)](https://www.orfonline.org/research/energy-news-monitor-volume-xxii-issue-39)
- [Powerline, Demand Uptick: Key Drivers (Indian peak demand)](https://powerline.net.in/2026/07/03/demand-uptick-key-drivers-and-initiatives-to-meet-the-growing-power-requirement/)
- [Combined Cycle Journal, Turbine Tip No. 7 (startup sequencing)](https://ccj-online.com/?p=12204)
- [control.com forum, Frame 9E.03 time from barring to sync](https://control.com/forums/threads/51486)
- [control.com forum, fast load rate](https://control.com/forums/threads/fast-load-rate.47614/post-47614)
- [control.com forum, gas turbine ramp load capacity](https://control.com/forums/threads/gas-turbine-ramp-load-capacity.50333/)
- [control.com forum, start-up sequence of GT Frame 9](https://control.com/forums/threads/start-up-sequence-of-gt-frame-9.48018/post-48018)
- [US Patent 11,255,218, Method for starting up a gas turbine engine of a combined cycle power plant](https://patents.google.com/patent/US11255218B2)
- [US Patent 4,010,605, Fixed time acceleration gas turbine startup speed control](https://patents.google.com/patent/US4010605A)
- [US Patent 5,732,546, Transient turbine overtemperature control](https://patents.google.com/patent/US5732546A)

Combustion chemistry:

- [Cantera](https://cantera.org)
- [G. P. Smith et al., GRI-Mech 3.0](http://www.me.berkeley.edu/gri_mech/)
