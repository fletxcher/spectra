"""Runs a single simulation case end-to-end: builds the reactor network,
steps through the output time grid, periodically refreshing ambient-driven
inlet conditions as the simulated region's temperature moves through its
diurnal/seasonal cycle, and returns a tidy DataFrame with the physical trace
and ground-truth segment/steady labels.

Integration is quasi-steady: at each output sample the inputs (air/fuel
flow from the profile, which already includes dispatch ramps, disturbances
and AGC) are held fixed and the reactor is solved directly for its steady
state (Newton, `ReactorNet.solve_steady`), seeded from the previous sample.
The combustor's residence time is ~3-70 ms, roughly 1000x shorter than the
15 s sample interval, so the chamber is always fully settled to its current
inputs at sample resolution; this matches continuous time integration to
~0.01 K while being ~20x cheaper. If the Newton solve fails, it falls back
to relaxing by time integration for RELAX_S (many residence times)."""

from __future__ import annotations

from dataclasses import dataclass

import cantera as ct
import pandas as pd

from src.sim.conditions import AmbientConditions, compressor_discharge
from src.sim.fuel import FuelBlend
from src.sim.profiles import Profile
from src.sim.reactor import (
    CombustorNetwork,
    adiabatic_flame_temperature,
    build_combustor_network,
    set_inputs,
    update_discharge_temperature,
)

TRACKED_SPECIES = ("O2", "CO2", "CO", "H2", "CH4", "NO", "NO2")
RELAX_S = 1.0


@dataclass(frozen=True)
class SimCase:
    """Everything needed to run one simulation."""

    run_id: str
    region_name: str
    start_day_of_year: float
    ambient: AmbientConditions  # conditions at t=0; .pressure is fixed for the run
    segment_ambient_temp_k: tuple[float, ...]  # per-segment ambient temperature, K
    pressure_ratio: float
    fuel: FuelBlend
    profile: Profile
    mdot_air_ref: float
    volume: float
    idle_phi: float
    base_load_phi: float
    min_load_fraction: float
    isentropic_efficiency: float = 0.85
    fine_dt: float = 30.0  # sample interval through transitions, seconds
    coarse_dt: float = 300.0  # sample interval through steady holds, seconds
    fine_window: float = 300.0  # extra settle buffer sampled at fine_dt, seconds


@dataclass
class RunResult:
    case: SimCase
    network: CombustorNetwork
    df: pd.DataFrame
    segment_adiabatic_temp: dict[int, float]
    residence_time_s: float


def _settle(network: CombustorNetwork) -> None:
    try:
        network.net.solve_steady()
    except ct.CanteraError:
        network.net.reinitialize()
        network.net.advance(network.net.time + RELAX_S)


def run_case(case: SimCase) -> RunResult:
    network = build_combustor_network(
        ambient=case.ambient,
        pressure_ratio=case.pressure_ratio,
        fuel=case.fuel,
        profile=case.profile,
        mdot_air_ref=case.mdot_air_ref,
        volume=case.volume,
        isentropic_efficiency=case.isentropic_efficiency,
    )

    species_idx = {
        name: network.combustor.thermo.species_index(name) for name in TRACKED_SPECIES
    }

    segment_adiabatic_temp = {}
    for idx, seg in enumerate(case.profile.segments):
        discharge_temp = compressor_discharge(
            AmbientConditions(temperature=case.segment_ambient_temp_k[idx], pressure=case.ambient.pressure),
            case.pressure_ratio,
            case.isentropic_efficiency,
        ).temperature
        segment_adiabatic_temp[idx] = adiabatic_flame_temperature(
            case.fuel, seg.setpoint.phi, discharge_temp, network.chamber_pressure_target
        )

    times = case.profile.sample_times(case.fine_dt, case.coarse_dt, case.fine_window)

    rows = []
    last_segment_idx = -1
    for t in times:
        segment_id, is_steady = case.profile.label_at(t)
        if segment_id != last_segment_idx:
            update_discharge_temperature(
                network, case.segment_ambient_temp_k[segment_id], case.pressure_ratio, case.isentropic_efficiency
            )
            last_segment_idx = segment_id

        sp = case.profile.setpoint_at(t)
        set_inputs(network, sp.load_fraction, sp.phi)
        _settle(network)
        gas = network.combustor.thermo
        disturbance = case.profile.disturbance_at(t)
        rows.append(
            {
                "run_id": case.run_id,
                "time": t,
                "temperature": gas.T,
                "pressure": gas.P,
                "mdot_air": network.air_mfc.mass_flow_rate,
                "mdot_fuel": network.fuel_mfc.mass_flow_rate,
                "phi_cmd": sp.phi,
                "load_cmd": sp.load_fraction,
                "ambient_temperature": case.segment_ambient_temp_k[segment_id],
                "mean_molecular_weight": gas.mean_molecular_weight,
                **{
                    f"Y_{name}": gas.Y[idx] for name, idx in species_idx.items()
                },
                "segment_id": segment_id,
                "is_steady": is_steady,
                "disturbance": disturbance.kind if disturbance is not None else "",
                "adiabatic_flame_temp": segment_adiabatic_temp[segment_id],
            }
        )

    df = pd.DataFrame.from_records(rows)
    df["is_steady"] = case.profile.steady_labels(
        df["time"].to_numpy(), df["load_cmd"].to_numpy(), df["phi_cmd"].to_numpy(), df["is_steady"].to_numpy()
    )

    mdot_total_ref = network.mdot_air_ref + network.mdot_fuel_ref
    residence_time_s = network.initial_density * case.volume / mdot_total_ref

    return RunResult(
        case=case,
        network=network,
        df=df,
        segment_adiabatic_temp=segment_adiabatic_temp,
        residence_time_s=residence_time_s,
    )
