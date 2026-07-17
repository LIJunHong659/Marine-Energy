from pathlib import Path

import pytest

from blue_hub.china_counterfactual_model import (
    S7Design,
    build_lagged_wave_capacity_factor,
    calibrate_landing_limit_mw,
    evaluate_s7_design,
    load_china_infrastructure_cost_cases,
)
from blue_hub.investment_planning_model import (
    PlanningLimits,
    load_investment_cost_cases,
    run_s6_investment_planning,
)
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def inputs(hours: int = 48):
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    scenario = load_scenarios(ROOT / "configs/scenario_matrix.csv")[0]
    config = load_system_configuration(
        ROOT / "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )
    flexible = load_investment_cost_cases(
        ROOT / "configs/s7_china_flexible_cost_cases.csv"
    )[1]
    common = load_china_infrastructure_cost_cases(
        ROOT / "configs/s7_china_infrastructure_cost_cases.csv"
    )[1]
    frame = generate_synthetic_timeseries(hours)
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    return parameters, scenario, config, flexible, common, frame


def test_absolute_landing_limit_does_not_expand_with_cable_capacity() -> None:
    parameters, scenario, config, flexible, _, frame = inputs()
    frame["grid_absorption_limit_mw"] = 250.0
    zero = PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    result = run_s6_investment_planning(
        frame,
        parameters,
        config.model_copy(update={"tx_capacity_mw": 1_400.0}),
        scenario,
        flexible,
        zero,
    )
    assert result.hourly["tx_available_capacity_mw"].max() == pytest.approx(250.0)


def test_only_explicit_flexible_compute_fraction_can_be_served() -> None:
    parameters, scenario, config, flexible, _, frame = inputs()
    frame["grid_absorption_limit_mw"] = 0.0
    frame["national_compute_demand_mw_it"] = 100.0
    frame["national_compute_flexible_fraction"] = 0.25
    frame["national_compute_price_cny_per_mwh_it"] = 10_000.0
    result = run_s6_investment_planning(
        frame,
        parameters,
        config,
        scenario,
        flexible,
        PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
    )
    assert result.hourly["spot_compute_completed_mwh_it"].max() <= 25.0 + 1e-8


def test_hydrogen_sales_require_endogenous_pipeline_capacity() -> None:
    parameters, scenario, config, flexible, _, frame = inputs()
    frame["grid_absorption_limit_mw"] = 0.0
    result = run_s6_investment_planning(
        frame,
        parameters,
        config,
        scenario.model_copy(update={"hydrogen_price_multiplier": 2.0}),
        flexible,
        PlanningLimits(0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 200.0),
    )
    assert result.capacities["hydrogen_export_capacity_kg_per_h"] <= 200.0 + 1e-8
    assert result.hourly["hydrogen_sale_kg"].max() <= 200.0 + 1e-8


def test_national_utilization_calibration_is_reproduced() -> None:
    *_, frame = inputs(876)
    limit = calibrate_landing_limit_mw(
        frame,
        wind_capacity_mw=1_000.0,
        pv_capacity_mw=0.0,
        tx_capacity_mw=700.0,
        target_utilization=0.959,
    )
    wind = 1_000.0 * frame["wind_cf"] * frame["wind_availability"]
    used = wind.clip(upper=frame["critical_load"] + limit)
    assert used.sum() / wind.sum() == pytest.approx(0.959, abs=1e-8)


def test_s7_full_project_cost_includes_common_and_flexible_assets() -> None:
    parameters, scenario, config, flexible, common, frame = inputs()
    frame["national_compute_demand_mw_it"] = 0.0
    design = S7Design(
        "test_direct",
        "direct",
        1_000.0,
        0.0,
        0.0,
        700.0,
        500.0,
        False,
        False,
    )
    result = evaluate_s7_design(
        frame,
        parameters,
        config,
        scenario,
        flexible,
        common,
        PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        design,
    )
    assert result.kpis["common_infrastructure_capex_cny"] > 0.0
    assert result.kpis["full_project_capex_cny"] == pytest.approx(
        result.kpis["common_infrastructure_capex_cny"]
        + result.kpis["flexible_asset_capex_cny"]
    )


def test_wave_proxy_is_bounded_and_matches_target_mean() -> None:
    _, _, _, _, _, frame = inputs(168)
    wave = build_lagged_wave_capacity_factor(frame["wind_cf"], target_mean=0.25)
    assert wave.min() >= 0.0
    assert wave.max() <= 1.0
    assert wave.mean() == pytest.approx(0.25, rel=1e-6)
