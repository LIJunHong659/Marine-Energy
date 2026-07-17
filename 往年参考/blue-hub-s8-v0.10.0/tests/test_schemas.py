import pytest
from pydantic import ValidationError

from blue_hub.schemas import SystemConfiguration, TechnologyParameter


def test_parameter_bounds_are_ordered() -> None:
    with pytest.raises(ValidationError, match="low <= base <= high"):
        TechnologyParameter(
            parameter="bad",
            value_base=2.0,
            value_low=3.0,
            value_high=4.0,
            unit="fraction",
            source="test",
            source_grade="D",
        )


def test_partial_battery_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="both be zero or both be positive"):
        SystemConfiguration(
            config_id="bad",
            wind_capacity_mw=1000.0,
            pv_capacity_mw=0.0,
            tx_capacity_mw=700.0,
            battery_power_mw=100.0,
            battery_energy_mwh=0.0,
            electrolyzer_power_mw=0.0,
            hydrogen_storage_kg=0.0,
            compute_it_capacity_mw=0.0,
            fuel_cell_power_mw=0.0,
            initial_battery_soc_fraction=0.5,
            initial_hydrogen_inventory_fraction=0.0,
            tx_technology="hvdc",
            export_policy="economic",
            battery_reserve_power_mw=0.0,
            battery_reserve_duration_h=0.0,
        )
