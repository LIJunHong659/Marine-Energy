from pathlib import Path

import pytest

from blue_hub.battery import battery_spec_from_parameters
from blue_hub.battery_resilience import simulate_islanded_critical_load
from blue_hub.loaders import load_parameters, load_system_configuration

ROOT = Path(__file__).resolve().parents[1]


def test_unforeseen_outage_resilience_is_monotonic_in_initial_soc() -> None:
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    config = load_system_configuration(ROOT / "configs/s1_battery_100mw_400mwh.yaml")
    specification = battery_spec_from_parameters(config, parameters)
    critical_load = [10.0] * 24
    results = [
        simulate_islanded_critical_load(critical_load, specification, initial_soc)
        for initial_soc in (0.3, 0.5, 0.7, 0.9)
    ]
    eens = [result.kpis["eens_mwh"] for result in results]
    ride_through = [result.kpis["hours_before_first_shortfall"] for result in results]
    assert eens == sorted(eens, reverse=True)
    assert ride_through == sorted(ride_through)
    assert results[1].kpis["eens_mwh"] > 0.0
    assert results[-1].kpis["eens_mwh"] == pytest.approx(0.0)


def test_event_audit_rejects_soc_outside_operational_range() -> None:
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    config = load_system_configuration(ROOT / "configs/s1_battery_100mw_400mwh.yaml")
    specification = battery_spec_from_parameters(config, parameters)
    with pytest.raises(ValueError, match="initial SOC"):
        simulate_islanded_critical_load([10.0] * 24, specification, 0.05)
