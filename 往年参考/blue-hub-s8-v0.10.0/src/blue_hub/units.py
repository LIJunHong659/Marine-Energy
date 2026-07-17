"""Small explicit unit conversion registry for model-boundary checks."""

from __future__ import annotations

_ENERGY_TO_KWH: dict[str, float] = {
    "Wh": 0.001,
    "kWh": 1.0,
    "MWh": 1000.0,
    "GWh": 1_000_000.0,
}


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """Convert energy without guessing or accepting unknown units."""
    try:
        value_kwh = value * _ENERGY_TO_KWH[from_unit]
        return value_kwh / _ENERGY_TO_KWH[to_unit]
    except KeyError as exc:
        raise ValueError(f"unsupported energy unit: {exc.args[0]}") from exc


def mwh_to_kwh(value_mwh: float) -> float:
    """Convert MWh to kWh at the hydrogen production boundary."""
    return convert_energy(value_mwh, "MWh", "kWh")
