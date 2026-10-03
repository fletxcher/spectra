"""Fuel blend definitions: CH4/H2 mixtures, expressed as mole fractions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuelBlend:
    """A CH4/H2 fuel blend, by mole fraction of H2."""

    h2_mole_fraction: float  # 0 (pure natural-gas proxy) .. 1 (pure hydrogen)

    def composition(self) -> str:
        """Cantera 'X' composition string for this blend."""
        h2 = self.h2_mole_fraction
        ch4 = 1.0 - h2
        return f"CH4:{ch4}, H2:{h2}"

    @property
    def lower_heating_value_proxy(self) -> float:
        """Relative LHV proxy (MJ/kg-ish scale) used only to sanity-check
        sweep coverage, not for any quantitative energy balance."""
        LHV_CH4 = 50.0
        LHV_H2 = 120.0
        return (1 - self.h2_mole_fraction) * LHV_CH4 + self.h2_mole_fraction * LHV_H2


AIR_COMPOSITION = "O2:0.21, N2:0.78, AR:0.01"

_MW_CH4 = 16.04
_MW_H2 = 2.016
_MW_AIR = 0.21 * 32.00 + 0.78 * 28.01 + 0.01 * 39.95
_O2_PER_MOL_CH4 = 2.0
_O2_PER_MOL_H2 = 0.5
_O2_MOLE_FRACTION_IN_AIR = 0.21


def stoichiometric_afr(fuel: FuelBlend) -> float:
    """Stoichiometric air/fuel mass ratio for a CH4/H2 blend, from the
    O2-demand of each component (CH4 + 2 O2 -> ...; H2 + 0.5 O2 -> ...)."""
    x_h2 = fuel.h2_mole_fraction
    x_ch4 = 1.0 - x_h2
    mw_fuel = x_ch4 * _MW_CH4 + x_h2 * _MW_H2
    o2_demand = x_ch4 * _O2_PER_MOL_CH4 + x_h2 * _O2_PER_MOL_H2
    air_mol_per_fuel_mol = o2_demand / _O2_MOLE_FRACTION_IN_AIR
    return (air_mol_per_fuel_mol * _MW_AIR) / mw_fuel
