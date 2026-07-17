from pathlib import Path

import numpy as np
import pytest

from blue_hub.compute import compute_spec_from_parameters
from blue_hub.compute_dispatch_model import run_s3_dispatch
from blue_hub.dispatch_model import run_s0_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    scenarios = {
        item.scenario_id: item for item in load_scenarios(ROOT / "configs/scenario_matrix.csv")
    }
    s0_config = load_system_configuration(ROOT / "configs/base_case.yaml")
    compute_config = load_system_configuration(
        ROOT / "configs/s3_compute_100mw_fiber_100mw_it.yaml"
    )
    return parameters, scenarios, s0_config, compute_config


def test_pue_is_facility_energy_divided_by_it_energy() -> None:
    parameters, scenarios, _, compute_config = inputs()
    base = compute_spec_from_parameters(compute_config, parameters, scenarios["base"])
    optimistic = compute_spec_from_parameters(
        compute_config, parameters, scenarios["compute_pue_optimistic"]
    )
    conservative = compute_spec_from_parameters(
        compute_config, parameters, scenarios["compute_pue_conservative"]
    )
    assert base.pue == pytest.approx(1.15)
    assert optimistic.pue == pytest.approx(1.08)
    assert conservative.pue == pytest.approx(1.25)


def test_zero_compute_capacity_is_exact_s0_ablation() -> None:
    parameters, scenarios, s0_config, _ = inputs()
    frame = generate_synthetic_timeseries(168)
    s0 = run_s0_dispatch(frame, parameters, s0_config, scenarios["base"])
    s3 = run_s3_dispatch(frame, parameters, s0_config, scenarios["base"])
    assert s3.metadata["phase"] == "S3_ablation"
    assert s3.kpis["operating_margin_cny"] == pytest.approx(s0.kpis["operating_margin_cny"])
    assert s3.kpis["export_land_mwh"] == pytest.approx(s0.kpis["export_land_mwh"])
    assert s3.kpis["compute_service_mwh_it"] == 0.0


def test_compute_workloads_close_queue_and_preserve_power_balance() -> None:
    parameters, scenarios, _, compute_config = inputs()
    result = run_s3_dispatch(
        generate_synthetic_timeseries(168), parameters, compute_config, scenarios["base"]
    )
    hourly = result.hourly
    assert result.kpis["rigid_compute_service_rate"] == pytest.approx(1.0)
    assert result.kpis["flex_compute_service_rate"] == pytest.approx(1.0)
    assert result.kpis["flex_queue_terminal_error_mwh_it"] == pytest.approx(0.0, abs=1e-8)
    assert result.kpis["max_flex_queue_state_residual_mwh_it"] < parameters.value(
        "compute_state_tolerance"
    )
    assert result.kpis["max_offshore_balance_residual_mw"] < parameters.value(
        "power_balance_tolerance"
    )
    assert np.allclose(
        hourly["dc_facility_power_mw"],
        hourly["it_power_mw"] * result.kpis["data_center_pue"],
        atol=1e-8,
    )


def test_pue_increase_requires_more_facility_energy_and_reduces_margin() -> None:
    parameters, scenarios, _, compute_config = inputs()
    frame = generate_synthetic_timeseries(168)
    base = run_s3_dispatch(frame, parameters, compute_config, scenarios["base"])
    conservative = run_s3_dispatch(
        frame, parameters, compute_config, scenarios["compute_pue_conservative"]
    )
    assert conservative.kpis["compute_service_mwh_it"] == pytest.approx(
        base.kpis["compute_service_mwh_it"]
    )
    assert conservative.hourly["dc_facility_power_mw"].sum() > base.hourly[
        "dc_facility_power_mw"
    ].sum()
    assert conservative.kpis["operating_margin_cny"] < base.kpis["operating_margin_cny"]


def test_fiber_outage_blocks_service_and_preserves_flexible_work_for_recovery() -> None:
    parameters, scenarios, _, compute_config = inputs()
    result = run_s3_dispatch(
        generate_synthetic_timeseries(168),
        parameters,
        compute_config,
        scenarios["fiber_outage_24h"],
    )
    event = result.hourly.iloc[72:96]
    assert event["it_power_mw"].sum() == pytest.approx(0.0, abs=1e-8)
    assert event["rigid_compute_unserved_mwh_it"].sum() > 0.0
    assert event["flex_queue_end_mwh_it"].iloc[-1] > 0.0
    assert result.kpis["flex_compute_service_rate"] == pytest.approx(1.0)
    assert result.kpis["minimum_flex_deadline_slack_mwh_it"] >= -parameters.value(
        "compute_state_tolerance"
    )
