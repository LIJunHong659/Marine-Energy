from pathlib import Path

import pandas as pd
import pytest

from blue_hub.china_counterfactual_model import (
    S7Design,
    load_china_infrastructure_cost_cases,
)
from blue_hub.investment_planning_model import load_investment_cost_cases
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.risk_screening_model import (
    S8CapacityCandidate,
    S8ScenarioCase,
    evaluate_capacity_candidates,
    fixed_flexible_asset_costs,
    interpolate_required_capacity,
    weighted_lower_tail_mean,
)
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    scenario = load_scenarios(ROOT / "configs/scenario_matrix.csv")[0].model_copy(
        update={"offshore_distance_km": 200.0}
    )
    base = load_system_configuration(
        ROOT / "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )
    costs = load_investment_cost_cases(
        ROOT / "configs/s7_china_flexible_cost_cases.csv"
    )[1]
    common = load_china_infrastructure_cost_cases(
        ROOT / "configs/s7_china_infrastructure_cost_cases.csv"
    )[1]
    return parameters, scenario, base, costs, common


def _zero_assets(base, *, config_id: str):
    return base.model_copy(
        update={
            "config_id": config_id,
            "battery_power_mw": 0.0,
            "battery_energy_mwh": 0.0,
            "electrolyzer_power_mw": 0.0,
            "hydrogen_storage_kg": 0.0,
            "hydrogen_export_capacity_kg_per_h": 0.0,
            "fuel_cell_power_mw": 0.0,
            "compute_it_capacity_mw": 0.0,
            "subsea_fiber_service_capacity_mw_it": 0.0,
            "initial_battery_soc_fraction": 0.0,
            "initial_hydrogen_inventory_fraction": 0.0,
        }
    )


def _frame(price: float) -> pd.DataFrame:
    frame = generate_synthetic_timeseries(48)
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    frame["national_compute_demand_mw_it"] = 50.0
    frame["national_compute_flexible_fraction"] = 1.0
    frame["national_compute_price_cny_per_mwh_it"] = price
    frame["grid_absorption_limit_mw"] = 200.0
    return frame


def test_weighted_lower_tail_mean_uses_fractional_probability_mass() -> None:
    value = weighted_lower_tail_mean(
        [-100.0, 0.0, 100.0], [0.1, 0.2, 0.7], tail_probability=0.2
    )
    assert value == pytest.approx(-50.0)


def test_fixed_asset_costs_include_distance_dependent_pipeline() -> None:
    _, _, base, costs, _ = _inputs()
    config = _zero_assets(base, config_id="pipeline_test").model_copy(
        update={"hydrogen_export_capacity_kg_per_h": 100.0}
    )
    capex, annual = fixed_flexible_asset_costs(
        config, costs, offshore_distance_km=200.0
    )
    assert capex["hydrogen_pipeline_capex_cny"] > 0.0
    assert annual["hydrogen_pipeline_annual_cost_cny"] > 0.0


def test_fixed_candidates_are_replayed_under_every_scenario() -> None:
    parameters, scenario, base, costs, common = _inputs()
    direct_config = _zero_assets(base, config_id="risk_direct")
    compute_config = direct_config.model_copy(
        update={
            "config_id": "risk_compute",
            "compute_it_capacity_mw": 20.0,
            "subsea_fiber_service_capacity_mw_it": 20.0,
        }
    )
    direct = S8CapacityCandidate(
        "direct",
        direct_config,
        S7Design("risk_direct", "direct", 1_000.0, 0.0, 0.0, 700.0, 450.0, False, False),
    )
    compute = S8CapacityCandidate(
        "compute",
        compute_config,
        S7Design(
            "risk_compute", "integrated_hub", 1_000.0, 0.0, 0.0, 700.0, 450.0, True, True
        ),
    )
    cases = (
        S8ScenarioCase("soft_compute", _frame(0.0), scenario, 0.5),
        S8ScenarioCase("strong_compute", _frame(5_000.0), scenario, 0.5),
    )
    result = evaluate_capacity_candidates(
        (direct, compute),
        cases,
        parameters,
        costs,
        common,
        tail_probability=0.5,
        risk_aversion=0.5,
    )
    assert len(result.scenario_results) == 4
    assert set(result.candidate_summary["candidate_id"]) == {"direct", "compute"}
    assert result.selected_candidate_id in {"direct", "compute"}
    assert result.scenario_results["scenario_regret_cny"].min() >= -1e-7
    assert result.scenario_results["strategy"].notna().all()
    assert result.scenario_results["max_offshore_balance_residual_mw"].max() < 1e-6
    assert (
        result.candidate_summary["risk_adjusted_score_cny_per_year"]
        <= result.candidate_summary["expected_full_net_value_cny_per_year"] + 1e-7
    ).all()


def test_capacity_interpolation_returns_minimum_equivalent_capacity() -> None:
    capacity = interpolate_required_capacity(
        [300.0, 400.0, 500.0], [0.70, 0.85, 0.95], target=0.90
    )
    assert capacity == pytest.approx(450.0)


def test_capacity_interpolation_rejects_nonmonotone_values() -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        interpolate_required_capacity(
            [300.0, 400.0, 500.0], [0.70, 0.90, 0.85], target=0.88
        )
