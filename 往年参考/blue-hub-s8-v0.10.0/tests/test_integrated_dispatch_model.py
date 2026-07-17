from pathlib import Path

import numpy as np
import pytest

from blue_hub.compute_dispatch_model import run_s3_dispatch
from blue_hub.hydrogen_dispatch_model import run_s2_dispatch
from blue_hub.integrated_dispatch_model import run_s4_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    scenarios = {
        item.scenario_id: item for item in load_scenarios(ROOT / "configs/scenario_matrix.csv")
    }
    s2_config = load_system_configuration(
        ROOT / "configs/s2_battery_hydrogen_100mw_400mwh_100mw_72000kg.yaml"
    )
    s3_config = load_system_configuration(
        ROOT / "configs/s3_compute_100mw_fiber_100mw_it.yaml"
    )
    s4_config = load_system_configuration(
        ROOT / "configs/s4_integrated_b100_e400_h100_s72000_c100_f100.yaml"
    )
    return parameters, scenarios, s2_config, s3_config, s4_config


def test_zero_compute_s4_ablation_matches_s2() -> None:
    parameters, scenarios, s2_config, _, _ = inputs()
    frame = generate_synthetic_timeseries(168)
    s2 = run_s2_dispatch(frame, parameters, s2_config, scenarios["base"])
    s4 = run_s4_dispatch(frame, parameters, s2_config, scenarios["base"])
    assert s4.metadata["phase"] == "S4_ablation_S2"
    assert s4.kpis["operating_margin_cny"] == pytest.approx(s2.kpis["operating_margin_cny"])
    assert s4.kpis["hydrogen_sales_kg"] == pytest.approx(s2.kpis["hydrogen_sales_kg"])


def test_zero_hydrogen_and_battery_s4_ablation_matches_s3() -> None:
    parameters, scenarios, _, s3_config, _ = inputs()
    frame = generate_synthetic_timeseries(168)
    s3 = run_s3_dispatch(frame, parameters, s3_config, scenarios["base"])
    s4 = run_s4_dispatch(frame, parameters, s3_config, scenarios["base"])
    assert s4.metadata["phase"] == "S4_ablation_S3"
    assert s4.kpis["operating_margin_cny"] == pytest.approx(s3.kpis["operating_margin_cny"])
    assert s4.kpis["compute_service_mwh_it"] == pytest.approx(s3.kpis["compute_service_mwh_it"])


def test_integrated_ledger_closes_all_physical_and_service_states() -> None:
    parameters, scenarios, _, _, config = inputs()
    result = run_s4_dispatch(
        generate_synthetic_timeseries(168), parameters, config, scenarios["base"]
    )
    hourly = result.hourly
    assert result.kpis["max_offshore_balance_residual_mw"] < parameters.value(
        "power_balance_tolerance"
    )
    assert result.kpis["max_hydrogen_state_residual_kg"] < parameters.value(
        "hydrogen_state_tolerance"
    )
    assert result.kpis["max_flex_queue_state_residual_mwh_it"] < parameters.value(
        "compute_state_tolerance"
    )
    assert result.kpis["battery_final_soc"] == pytest.approx(
        config.initial_battery_soc_fraction
    )
    assert result.kpis["hydrogen_terminal_state_error_kg"] == pytest.approx(0.0, abs=1e-8)
    assert result.kpis["flex_queue_terminal_error_mwh_it"] == pytest.approx(0.0, abs=1e-8)
    assert result.kpis["compute_service_rate"] == pytest.approx(1.0)
    assert np.allclose(
        hourly["dc_facility_power_mw"],
        hourly["it_power_mw"] * result.kpis["data_center_pue"],
        atol=1e-8,
    )


def test_fiber_outage_preserves_hydrogen_service_but_records_compute_sla_loss() -> None:
    parameters, scenarios, _, _, config = inputs()
    result = run_s4_dispatch(
        generate_synthetic_timeseries(168), parameters, config, scenarios["fiber_outage_24h"]
    )
    event = result.hourly.iloc[72:96]
    assert event["it_power_mw"].sum() == pytest.approx(0.0, abs=1e-8)
    assert event["rigid_compute_unserved_mwh_it"].sum() > 0.0
    assert result.kpis["hydrogen_sales_kg"] > 0.0
    assert result.kpis["minimum_flex_deadline_slack_mwh_it"] >= -parameters.value(
        "compute_state_tolerance"
    )
