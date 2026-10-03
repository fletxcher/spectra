"""Ambient conditions and the compressor model that turns them into
combustor-inlet (compressor-discharge) conditions."""

from __future__ import annotations

from dataclasses import dataclass

GAMMA_AIR = 1.4


@dataclass(frozen=True)
class AmbientConditions:
    """Conditions at the compressor inlet (engine intake)."""

    temperature: float  # K
    pressure: float  # Pa
    relative_humidity: float = 0.0  # 0-1, reserved for future humid-air effects


@dataclass(frozen=True)
class CompressorDischarge:
    """Conditions entering the combustor, downstream of the compressor."""

    temperature: float  # K
    pressure: float  # Pa


def compressor_discharge(
    ambient: AmbientConditions,
    pressure_ratio: float,
    isentropic_efficiency: float = 0.85,
) -> CompressorDischarge:
    """Simple isentropic-compression-with-efficiency model.

    T2s / T0 = PR ** ((gamma-1)/gamma); real rise is scaled by 1/eta_c.
    """
    t0 = ambient.temperature
    p0 = ambient.pressure
    t2s = t0 * pressure_ratio ** ((GAMMA_AIR - 1) / GAMMA_AIR)
    t2 = t0 + (t2s - t0) / isentropic_efficiency
    p2 = p0 * pressure_ratio
    return CompressorDischarge(temperature=t2, pressure=p2)
