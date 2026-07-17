from pathlib import Path

import numpy as np
import pytest

from blue_hub.dispatch_model import run_s0_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_s0_results
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def base_inputs():
    params = load_parameters(ROOT / "configs/technology_parameters.csv")
    config = load_system_configuration(ROOT / "configs/base_case.yaml")
    scenario = load_scenarios(ROOT / "configs/scenario_matrix.csv")[0]
    return params, config, scenario


def test_four_hour_hand_calculation_and_power_balances() -> None:
    params, config, scenario = base_inputs()
    config = config.model_copy(
        update={
            "wind_capacity_mw": 100.0,
            "tx_capacity_mw": 50.0,
            "export_policy": "must_take",
        }
    )
    frame = generate_synthetic_timeseries(4)
    frame["wind_cf"] = [0.0, 0.5, 1.0, 1.0]
    frame["wind_availability"] = 1.0
    frame["critical_load"] = 10.0
    frame["electricity_price"] = 500.0
    result = run_s0_dispatch(frame, params, config, scenario)

    np.testing.assert_allclose(result.hourly["wind_available_mw"], [0.0, 50.0, 100.0, 100.0])
    np.testing.assert_allclose(result.hourly["critical_load_served_mw"], [0.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(result.hourly["unmet_critical_load_mw"], [10.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.hourly["export_send_mw"], [0.0, 40.0, 50.0, 50.0])
    np.testing.assert_allclose(result.hourly["curtailment_mw"], [0.0, 0.0, 40.0, 40.0])
    assert result.kpis["max_offshore_balance_residual_mw"] < 1e-12
    assert result.kpis["max_land_balance_residual_mw"] < 1e-12
    expected_revenue = float((500.0 * result.hourly["export_land_mw"]).sum())
    assert result.kpis["electricity_revenue_cny"] == pytest.approx(expected_revenue)


def test_outage_forces_zero_export_and_increases_network_curtailment() -> None:
    params, config, scenario = base_inputs()
    scenario = scenario.model_copy(update={"tx_outage_start_hour": 5, "tx_outage_hours": 3})
    frame = generate_synthetic_timeseries(24)
    result = run_s0_dispatch(frame, params, config, scenario)
    outage = result.hourly.iloc[5:8]
    assert (outage["export_send_mw"] == 0.0).all()
    assert (outage["export_land_mw"] == 0.0).all()
    assert outage["network_curtailment_mw"].sum() > 0.0


def test_negative_price_causes_economic_curtailment() -> None:
    params, config, scenario = base_inputs()
    frame = generate_synthetic_timeseries(24)
    frame["electricity_price"] = -50.0
    result = run_s0_dispatch(frame, params, config, scenario)
    assert result.kpis["export_send_mwh"] == 0.0
    assert result.kpis["economic_curtailment_mwh"] > 0.0
    assert result.kpis["negative_price_export_mwh"] == 0.0


def test_full_year_solves_without_balance_or_state_errors() -> None:
    params, config, scenario = base_inputs()
    result = run_s0_dispatch(generate_synthetic_timeseries(8760), params, config, scenario)
    assert len(result.hourly) == 8760
    assert result.hourly.select_dtypes(include=[float]).notna().all().all()
    assert result.kpis["max_offshore_balance_residual_mw"] < 1e-9
    assert result.kpis["max_land_balance_residual_mw"] < 1e-9
    assert 0.0 <= result.kpis["curtailment_rate"] <= 1.0


def test_s0_rejects_future_phase_capacity() -> None:
    params, config, scenario = base_inputs()
    config = config.model_copy(update={"battery_power_mw": 10.0, "battery_energy_mwh": 20.0})
    with pytest.raises(ValueError, match="unsupported nonzero capacities"):
        run_s0_dispatch(generate_synthetic_timeseries(24), params, config, scenario)


def test_no_wind_cannot_create_energy() -> None:
    params, config, scenario = base_inputs()
    frame = generate_synthetic_timeseries(24)
    frame["wind_cf"] = 0.0
    result = run_s0_dispatch(frame, params, config, scenario)
    assert result.kpis["renewable_generation_mwh"] == 0.0
    assert result.kpis["export_send_mwh"] == 0.0
    assert result.kpis["export_land_mwh"] == 0.0
    assert result.kpis["eens_mwh"] == pytest.approx(frame["critical_load"].sum())


def test_24_hour_regression_snapshot_and_artifact_hashes(tmp_path) -> None:
    params, config, scenario = base_inputs()
    result = run_s0_dispatch(generate_synthetic_timeseries(24), params, config, scenario)
    assert result.kpis["export_land_mwh"] == pytest.approx(10986.655684087636)
    assert result.kpis["transmission_loss_mwh"] == pytest.approx(465.72834459620236)
    assert result.kpis["operating_margin_cny"] == pytest.approx(3954180.5231942614)
    paths = export_s0_results(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    manifest = paths["manifest"].read_text(encoding="utf-8")
    assert "configuration_hash" in manifest
    assert "hourly_results.csv" in manifest
