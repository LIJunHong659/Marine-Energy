import pytest

from blue_hub.provenance import configuration_hash
from blue_hub.units import convert_energy, mwh_to_kwh


def test_mwh_to_kwh_contract() -> None:
    assert mwh_to_kwh(1.0) == 1000.0
    assert convert_energy(1.0, "GWh", "MWh") == 1000.0


def test_unknown_energy_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported energy unit"):
        convert_energy(1.0, "MW", "kWh")


def test_configuration_hash_is_order_independent() -> None:
    assert configuration_hash({"hours": 24, "scenario": "base"}) == configuration_hash(
        {"scenario": "base", "hours": 24}
    )
