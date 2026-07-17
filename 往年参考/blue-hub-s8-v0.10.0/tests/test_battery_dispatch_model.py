from pathlib import Path

import pytest

from blue_hub.battery_dispatch_model import run_s1_dispatch
from blue_hub.dispatch_model import run_s0_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    s0_config = load_system_configuration(ROOT / "configs/base_case.yaml")
    s1_config = load_system_configuration(ROOT / "configs/s1_battery_100mw_400mwh.yaml")
    scenarios = {
        item.scenario_id: item for item in load_scenarios(ROOT / "configs/scenario_matrix.csv")
    }
    return parameters, s0_config, s1_config, scenarios


def test_zero_battery_ablation_matches_exact_s0() -> None:
    parameters, s0_config, _, scenarios = inputs()
    frame = generate_synthetic_timeseries(24)
    s0 = run_s0_dispatch(frame, parameters, s0_config, scenarios["base"])
    ablation = run_s1_dispatch(frame, parameters, s0_config, scenarios["base"])
    assert ablation.kpis["export_land_mwh"] == pytest.approx(s0.kpis["export_land_mwh"])
    assert ablation.kpis["operating_margin_cny"] == pytest.approx(s0.kpis["operating_margin_cny"])
    assert ablation.kpis["battery_charge_mwh"] == 0.0


def test_s1_respects_soc_cyclic_state_and_exclusivity() -> None:
    parameters, _, s1_config, scenarios = inputs()
    result = run_s1_dispatch(
        generate_synthetic_timeseries(168), parameters, s1_config, scenarios["base"]
    )
    assert result.kpis["battery_min_soc"] >= parameters.value("battery_soc_min") - 1e-9
    assert result.kpis["battery_max_soc"] <= parameters.value("battery_soc_max") + 1e-9
    assert result.kpis["battery_final_soc"] == pytest.approx(s1_config.initial_battery_soc_fraction)
    assert result.kpis["battery_max_simultaneous_charge_discharge_mw"] <= parameters.value(
        "battery_simultaneous_tolerance"
    )
    assert result.kpis["max_battery_state_residual_mwh"] < parameters.value(
        "battery_state_tolerance"
    )
    assert result.kpis["max_offshore_balance_residual_mw"] < parameters.value(
        "power_balance_tolerance"
    )


def test_battery_reduces_congestion_curtailment_and_increases_margin() -> None:
    parameters, s0_config, s1_config, scenarios = inputs()
    s0_config = s0_config.model_copy(update={"tx_capacity_mw": 500.0})
    s1_config = s1_config.model_copy(update={"tx_capacity_mw": 500.0})
    frame = generate_synthetic_timeseries(168)
    without_battery = run_s0_dispatch(frame, parameters, s0_config, scenarios["base"])
    with_battery = run_s1_dispatch(frame, parameters, s1_config, scenarios["base"])
    assert with_battery.kpis["curtailment_mwh"] < without_battery.kpis["curtailment_mwh"]
    assert with_battery.kpis["operating_margin_cny"] > without_battery.kpis["operating_margin_cny"]


def test_battery_uses_negative_price_energy_and_reduces_economic_curtailment() -> None:
    parameters, s0_config, s1_config, scenarios = inputs()
    frame = generate_synthetic_timeseries(168)
    without_battery = run_s0_dispatch(frame, parameters, s0_config, scenarios["negative_price_24h"])
    with_battery = run_s1_dispatch(frame, parameters, s1_config, scenarios["negative_price_24h"])
    assert with_battery.kpis["battery_charge_mwh"] > 0.0
    assert (
        with_battery.kpis["economic_curtailment_mwh"]
        < without_battery.kpis["economic_curtailment_mwh"]
    )


def test_battery_eliminates_24_hour_wind_lull_eens_with_perfect_foresight() -> None:
    parameters, s0_config, s1_config, scenarios = inputs()
    frame = generate_synthetic_timeseries(168)
    without_battery = run_s0_dispatch(frame, parameters, s0_config, scenarios["wind_lull_24h"])
    with_battery = run_s1_dispatch(frame, parameters, s1_config, scenarios["wind_lull_24h"])
    assert without_battery.kpis["eens_mwh"] > 0.0
    assert with_battery.kpis["eens_mwh"] == pytest.approx(0.0, abs=1e-8)
    event = with_battery.hourly.iloc[72:96]
    assert event["battery_discharge_mw"].sum() > 0.0


def test_full_year_s1_regression_and_linearization_error() -> None:
    parameters, s0_config, s1_config, scenarios = inputs()
    frame = generate_synthetic_timeseries(8760)
    s0 = run_s0_dispatch(frame, parameters, s0_config, scenarios["base"])
    s1 = run_s1_dispatch(frame, parameters, s1_config, scenarios["base"])
    assert s1.kpis["operating_margin_cny"] > s0.kpis["operating_margin_cny"]
    assert 250.0 < s1.kpis["battery_efc"] < 330.0
    assert s1.kpis["tx_linearization_error_mwh"] / s1.kpis["export_land_mwh"] < 1e-4
    assert s1.kpis["max_offshore_balance_residual_mw"] < 1e-8
    assert s1.kpis["max_battery_state_residual_mwh"] < 1e-8


def test_fixed_reserve_preserves_power_and_energy_headroom() -> None:
    parameters, _, s1_config, scenarios = inputs()
    reserve_config = s1_config.model_copy(
        update={"battery_reserve_power_mw": 10.0, "battery_reserve_duration_h": 4.0}
    )
    result = run_s1_dispatch(
        generate_synthetic_timeseries(168), parameters, reserve_config, scenarios["base"]
    )
    assert result.kpis["battery_min_soc"] >= result.kpis["battery_scheduled_minimum_soc"] - 1e-9
    net_discharge = result.hourly["battery_discharge_mw"] - result.hourly["battery_charge_mw"]
    assert net_discharge.max() <= reserve_config.battery_power_mw - 10.0 + 1e-8
