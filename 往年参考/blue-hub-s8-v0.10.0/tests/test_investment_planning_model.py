from dataclasses import replace
from pathlib import Path

import pytest

from blue_hub.investment_planning_model import (
    PlanningLimits,
    capital_recovery_factor,
    load_investment_cost_cases,
    optimal_system_configuration,
    run_s6_investment_planning,
)
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.scarcity_dispatch_model import run_s5_dispatch
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    scenario = load_scenarios(ROOT / "configs/scenario_matrix.csv")[0].model_copy(
        update={"hydrogen_price_multiplier": 0.75}
    )
    config = load_system_configuration(
        ROOT / "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )
    costs = {
        item.cost_case_id: item
        for item in load_investment_cost_cases(ROOT / "configs/s6_investment_cost_cases.csv")
    }
    frame = generate_synthetic_timeseries(48)
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    frame["national_compute_demand_mw_it"] = 100.0
    frame["national_compute_price_cny_per_mwh_it"] = 360.0
    frame["grid_absorption_factor"] = 0.32
    return parameters, scenario, config, costs, frame


def without_assets(config):
    return config.model_copy(
        update={
            "config_id": "S6_direct_test",
            "battery_power_mw": 0.0,
            "battery_energy_mwh": 0.0,
            "electrolyzer_power_mw": 0.0,
            "hydrogen_storage_kg": 0.0,
            "fuel_cell_power_mw": 0.0,
            "compute_it_capacity_mw": 0.0,
            "subsea_fiber_service_capacity_mw_it": 0.0,
            "initial_battery_soc_fraction": 0.0,
            "initial_hydrogen_inventory_fraction": 0.0,
        }
    )


def test_capital_recovery_factor_handles_zero_and_positive_rates() -> None:
    assert capital_recovery_factor(0.0, 20.0) == pytest.approx(0.05)
    assert capital_recovery_factor(0.06, 15.0) == pytest.approx(0.10296276)


def test_zero_planning_limits_reproduce_s5_direct_dispatch() -> None:
    parameters, scenario, config, costs, frame = inputs()
    zero = PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    planned = run_s6_investment_planning(
        frame, parameters, config, scenario, costs["cost_reference"], zero
    )
    replay = run_s5_dispatch(frame, parameters, without_assets(config), scenario)
    annualization = planned.kpis["annualization_factor"]
    assert planned.kpis["annualized_operating_margin_cny"] / annualization == pytest.approx(
        replay.kpis["operating_margin_cny"], rel=1e-9, abs=1e-5
    )
    assert planned.kpis["curtailment_mwh"] == pytest.approx(
        replay.kpis["curtailment_mwh"], abs=1e-7
    )


def test_compute_capacity_is_endogenous_and_respects_planning_limit() -> None:
    parameters, scenario, config, costs, frame = inputs()
    frame["national_compute_price_cny_per_mwh_it"] = 2_000.0
    free_compute = replace(
        costs["cost_reference"],
        compute_fibre_capex_cny_per_kw_it=0.0,
        compute_fibre_fixed_om_fraction=0.0,
    )
    limits = PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 50.0)
    result = run_s6_investment_planning(
        frame, parameters, config, scenario, free_compute, limits
    )
    assert result.capacities["compute_it_capacity_mw"] == pytest.approx(50.0)
    assert result.kpis["spot_compute_service_mwh_it"] > 0.0
    assert result.kpis["max_offshore_balance_residual_mw"] < parameters.value(
        "power_balance_tolerance"
    )


def test_low_cost_case_can_select_electrolyzer_without_forcing_other_assets() -> None:
    parameters, scenario, config, costs, frame = inputs()
    limits = PlanningLimits(100.0, 800.0, 150.0, 200_000.0, 30.0, 0.0)
    result = run_s6_investment_planning(
        frame, parameters, config, scenario, costs["cost_low"], limits
    )
    assert 0.0 < result.capacities["electrolyzer_power_mw"] <= 150.0
    assert result.capacities["compute_it_capacity_mw"] == pytest.approx(0.0)
    assert result.kpis["hydrogen_sales_kg"] > 0.0


def test_optimal_capacity_vector_converts_to_a_valid_s5_configuration() -> None:
    parameters, scenario, config, costs, frame = inputs()
    frame["national_compute_price_cny_per_mwh_it"] = 2_000.0
    free_compute = replace(
        costs["cost_reference"],
        compute_fibre_capex_cny_per_kw_it=0.0,
        compute_fibre_fixed_om_fraction=0.0,
    )
    result = run_s6_investment_planning(
        frame,
        parameters,
        config,
        scenario,
        free_compute,
        PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 50.0),
    )
    fixed = optimal_system_configuration(config, result)
    replay = run_s5_dispatch(frame, parameters, fixed, scenario)
    assert fixed.compute_it_capacity_mw == pytest.approx(50.0)
    assert fixed.subsea_fiber_service_capacity_mw_it == pytest.approx(50.0)
    assert replay.kpis["spot_compute_service_mwh_it"] > 0.0
