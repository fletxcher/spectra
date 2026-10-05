"""Builds the 0-D Cantera reactor network representing the combustion
chamber: air + fuel reservoirs feeding a well-stirred combustor through
mass-flow controllers (set per sample via `set_inputs`), exhausting through a
pressure controller that holds chamber pressure near its target."""

from __future__ import annotations

from dataclasses import dataclass

import cantera as ct

from src.sim.conditions import AmbientConditions, compressor_discharge
from src.sim.fuel import AIR_COMPOSITION, FuelBlend, stoichiometric_afr
from src.sim.profiles import Profile

MECHANISM = "gri30.yaml"
COMBUSTOR_PRESSURE_DROP_FRACTION = 0.05  # chamber P vs compressor discharge P
EXHAUST_PRESSURE_FRACTION = 0.90  # exhaust reservoir P, relative to chamber P


@dataclass
class CombustorNetwork:
    net: ct.ReactorNet
    combustor: ct.IdealGasReactor
    air_gas: ct.Solution
    air_mfc: ct.MassFlowController
    fuel_mfc: ct.MassFlowController
    exhaust_controller: ct.PressureController
    mdot_air_ref: float
    mdot_fuel_ref: float
    stoich_afr: float
    chamber_pressure_target: float
    discharge_temperature: float
    discharge_pressure: float
    initial_density: float


def build_combustor_network(
    ambient: AmbientConditions,
    pressure_ratio: float,
    fuel: FuelBlend,
    profile: Profile,
    mdot_air_ref: float,
    volume: float,
    isentropic_efficiency: float = 0.85,
) -> CombustorNetwork:
    """Assemble the reactor network. `mdot_air_ref` is the air mass flow at
    load_fraction == 1.0; `volume` is the combustor volume (sets residence
    time together with the mass flow)."""

    discharge = compressor_discharge(ambient, pressure_ratio, isentropic_efficiency)
    chamber_pressure = discharge.pressure * (1 - COMBUSTOR_PRESSURE_DROP_FRACTION)
    exhaust_pressure = chamber_pressure * EXHAUST_PRESSURE_FRACTION
    afr = stoichiometric_afr(fuel)

    air_gas = ct.Solution(MECHANISM)
    air_gas.TPX = discharge.temperature, discharge.pressure, AIR_COMPOSITION
    air_reservoir = ct.Reservoir(air_gas, name="air_supply")

    fuel_gas = ct.Solution(MECHANISM)
    fuel_gas.TPX = 300.0, discharge.pressure, fuel.composition()
    fuel_reservoir = ct.Reservoir(fuel_gas, name="fuel_supply")

    # Initialize the combustor already alight, at the equilibrium products
    # of the profile's starting setpoint, so we skip simulating cold ignition.
    combustor_gas = ct.Solution(MECHANISM)
    phi0 = profile.initial.phi
    combustor_gas.set_equivalence_ratio(phi0, fuel.composition(), AIR_COMPOSITION)
    combustor_gas.TP = discharge.temperature, chamber_pressure
    combustor_gas.equilibrate("HP")
    combustor = ct.IdealGasReactor(combustor_gas, name="combustor", volume=volume)

    exhaust_gas = ct.Solution(MECHANISM)
    exhaust_gas.TPX = combustor_gas.T, exhaust_pressure, combustor_gas.X
    exhaust_reservoir = ct.Reservoir(exhaust_gas, name="exhaust")

    air_mfc = ct.MassFlowController(air_reservoir, combustor, name="air_mfc")
    fuel_mfc = ct.MassFlowController(fuel_reservoir, combustor, name="fuel_mfc")

    mdot_fuel_ref = mdot_air_ref * profile.initial.phi / afr
    kv = (mdot_air_ref + mdot_fuel_ref) / (chamber_pressure - exhaust_pressure)
    exhaust_controller = ct.PressureController(
        combustor, exhaust_reservoir, primary=air_mfc, K=kv, name="exhaust"
    )

    net = ct.ReactorNet([combustor])

    network = CombustorNetwork(
        net=net,
        combustor=combustor,
        air_gas=air_gas,
        air_mfc=air_mfc,
        fuel_mfc=fuel_mfc,
        exhaust_controller=exhaust_controller,
        mdot_air_ref=mdot_air_ref,
        mdot_fuel_ref=mdot_fuel_ref,
        stoich_afr=afr,
        chamber_pressure_target=chamber_pressure,
        discharge_temperature=discharge.temperature,
        discharge_pressure=discharge.pressure,
        initial_density=combustor_gas.density,
    )
    set_inputs(network, profile.initial.load_fraction, profile.initial.phi)
    return network


def set_inputs(network: CombustorNetwork, load_fraction: float, phi: float) -> None:
    """Holds air/fuel mass flow constant at the given setpoint until the next
    call. The run is integrated quasi-steadily (see runner.run_case), so the
    flow controllers take plain constants rather than time callbacks."""
    mdot_air = load_fraction * network.mdot_air_ref
    network.air_mfc.mass_flow_rate = mdot_air
    network.fuel_mfc.mass_flow_rate = mdot_air * phi / network.stoich_afr


def update_discharge_temperature(
    network: CombustorNetwork, ambient_temperature_k: float, pressure_ratio: float, isentropic_efficiency: float
) -> float:
    """Re-derives compressor discharge temperature from a new ambient air
    temperature and updates the air reservoir in place (ambient pressure and
    the pressure ratio are treated as fixed for the run, so discharge
    pressure -- and hence chamber pressure -- doesn't need to move). Used to
    track a region's diurnal/seasonal temperature swing over a multi-day run
    without rebuilding the reactor network. Returns the new discharge
    temperature."""
    ambient = AmbientConditions(temperature=ambient_temperature_k, pressure=network.air_gas.P)
    discharge = compressor_discharge(ambient, pressure_ratio, isentropic_efficiency)
    network.air_gas.TP = discharge.temperature, network.discharge_pressure
    return discharge.temperature


def adiabatic_flame_temperature(
    fuel: FuelBlend, phi: float, inlet_temperature: float, pressure: float
) -> float:
    """Equilibrium (HP) adiabatic flame temperature for a given equivalence
    ratio and inlet conditions -- the theoretical ceiling the chamber
    temperature relaxes towards at each setpoint."""
    gas = ct.Solution(MECHANISM)
    gas.set_equivalence_ratio(phi, fuel.composition(), AIR_COMPOSITION)
    gas.TP = inlet_temperature, pressure
    gas.equilibrate("HP")
    return gas.T
