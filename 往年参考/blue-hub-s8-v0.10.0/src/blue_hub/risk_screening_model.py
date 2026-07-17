"""Phase 9 / S8 multi-scenario capacity-risk screening.

S8 deliberately keeps the audited S5 hourly dispatch and the S6/S7 cost
contracts unchanged.  It evaluates one *fixed* capacity vector in multiple
out-of-design years, then compares candidate designs by expected value, lower
tail value and scenario regret.  This closes the main gap between a design
that is optimal in one synthetic year and a design that remains useful when
resource, landing capability and customer contracts change.

The module is a discrete candidate-screening layer.  It does not claim to be a
continuous two-stage stochastic programme, and operational dispatch within
each scenario still has perfect foresight.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from blue_hub.china_counterfactual_model import (
    ChinaInfrastructureCostCase,
    S7Design,
    common_infrastructure_costs,
)
from blue_hub.investment_planning_model import InvestmentCostCase, capital_recovery_factor
from blue_hub.scarcity_dispatch_model import run_s5_dispatch
from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters


@dataclass(frozen=True)
class S8ScenarioCase:
    """One weighted out-of-design operating case."""

    case_id: str
    timeseries: pd.DataFrame
    scenario: ScenarioDefinition
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be blank")
        if self.timeseries.empty:
            raise ValueError("timeseries must not be empty")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("scenario weight must be finite and positive")


@dataclass(frozen=True)
class S8CapacityCandidate:
    """A fixed capacity vector and its associated common-infrastructure design."""

    candidate_id: str
    configuration: SystemConfiguration
    design: S7Design

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be blank")
        pairs = (
            (self.configuration.wind_capacity_mw, self.design.wind_capacity_mw),
            (self.configuration.pv_capacity_mw, self.design.pv_capacity_mw),
            (self.configuration.tx_capacity_mw, self.design.tx_capacity_mw),
        )
        if any(abs(config_value - design_value) > 1e-9 for config_value, design_value in pairs):
            raise ValueError("configuration and design generation/export capacities must match")


@dataclass(frozen=True)
class S8RiskScreeningResult:
    """Candidate summary, scenario ledger and transparent selection metadata."""

    candidate_summary: pd.DataFrame
    scenario_results: pd.DataFrame
    selected_candidate_id: str
    metadata: dict[str, Any]


def _annualized_capex(capex: float, fixed_om_fraction: float, life: float, rate: float) -> float:
    return capex * (capital_recovery_factor(rate, life) + fixed_om_fraction)


def fixed_flexible_asset_costs(
    config: SystemConfiguration,
    costs: InvestmentCostCase,
    *,
    offshore_distance_km: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return CAPEX and annual fixed cost for a fixed S5 configuration."""
    if offshore_distance_km <= 0.0:
        raise ValueError("offshore_distance_km must be positive")
    capex = {
        "battery_power_capex_cny": config.battery_power_mw
        * 1_000.0
        * costs.battery_power_capex_cny_per_kw,
        "battery_energy_capex_cny": config.battery_energy_mwh
        * 1_000.0
        * costs.battery_energy_capex_cny_per_kwh,
        "electrolyzer_capex_cny": config.electrolyzer_power_mw
        * 1_000.0
        * costs.electrolyzer_capex_cny_per_kw,
        "hydrogen_storage_capex_cny": config.hydrogen_storage_kg
        * costs.hydrogen_storage_capex_cny_per_kg,
        "fuel_cell_capex_cny": config.fuel_cell_power_mw
        * 1_000.0
        * costs.fuel_cell_capex_cny_per_kw,
        "compute_fibre_capex_cny": config.compute_it_capacity_mw
        * 1_000.0
        * costs.compute_fibre_capex_cny_per_kw_it,
        "hydrogen_pipeline_capex_cny": config.hydrogen_export_capacity_kg_per_h
        * offshore_distance_km
        * costs.hydrogen_pipeline_capex_cny_per_kg_h_km,
    }
    annual = {
        "battery_power_annual_cost_cny": _annualized_capex(
            capex["battery_power_capex_cny"],
            costs.battery_fixed_om_fraction,
            costs.battery_lifetime_years,
            costs.discount_rate,
        ),
        "battery_energy_annual_cost_cny": _annualized_capex(
            capex["battery_energy_capex_cny"],
            costs.battery_fixed_om_fraction,
            costs.battery_lifetime_years,
            costs.discount_rate,
        ),
        "electrolyzer_annual_cost_cny": _annualized_capex(
            capex["electrolyzer_capex_cny"],
            costs.electrolyzer_fixed_om_fraction,
            costs.electrolyzer_lifetime_years,
            costs.discount_rate,
        ),
        "hydrogen_storage_annual_cost_cny": _annualized_capex(
            capex["hydrogen_storage_capex_cny"],
            costs.hydrogen_storage_fixed_om_fraction,
            costs.hydrogen_storage_lifetime_years,
            costs.discount_rate,
        ),
        "fuel_cell_annual_cost_cny": _annualized_capex(
            capex["fuel_cell_capex_cny"],
            costs.fuel_cell_fixed_om_fraction,
            costs.fuel_cell_lifetime_years,
            costs.discount_rate,
        ),
        "compute_fibre_annual_cost_cny": _annualized_capex(
            capex["compute_fibre_capex_cny"],
            costs.compute_fibre_fixed_om_fraction,
            costs.compute_fibre_lifetime_years,
            costs.discount_rate,
        ),
        "hydrogen_pipeline_annual_cost_cny": _annualized_capex(
            capex["hydrogen_pipeline_capex_cny"],
            costs.hydrogen_pipeline_fixed_om_fraction,
            costs.hydrogen_pipeline_lifetime_years,
            costs.discount_rate,
        ),
    }
    return capex, annual


def weighted_lower_tail_mean(
    values: Iterable[float], weights: Iterable[float], *, tail_probability: float
) -> float:
    """Return the weighted mean of the lowest ``tail_probability`` mass."""
    value_array = np.asarray(tuple(values), dtype=float)
    weight_array = np.asarray(tuple(weights), dtype=float)
    if value_array.ndim != 1 or len(value_array) == 0 or len(value_array) != len(weight_array):
        raise ValueError("values and weights must be non-empty vectors of equal length")
    if not np.isfinite(value_array).all() or not np.isfinite(weight_array).all():
        raise ValueError("values and weights must be finite")
    if (weight_array <= 0.0).any():
        raise ValueError("weights must be positive")
    if not 0.0 < tail_probability <= 1.0:
        raise ValueError("tail_probability must lie in (0, 1]")
    normalized = weight_array / weight_array.sum()
    order = np.argsort(value_array)
    remaining = tail_probability
    total = 0.0
    for index in order:
        mass = min(float(normalized[index]), remaining)
        total += mass * float(value_array[index])
        remaining -= mass
        if remaining <= 1e-15:
            break
    return total / tail_probability


def _stack_replacement_cost_per_mwh(costs: InvestmentCostCase) -> float:
    return (
        costs.electrolyzer_capex_cny_per_kw
        * costs.electrolyzer_replacement_fraction
        * 1_000.0
        / costs.electrolyzer_replacement_interval_full_load_h
    )


def _capacity_record(config: SystemConfiguration) -> dict[str, float]:
    return {
        "battery_power_mw": config.battery_power_mw,
        "battery_energy_mwh": config.battery_energy_mwh,
        "electrolyzer_power_mw": config.electrolyzer_power_mw,
        "hydrogen_storage_kg": config.hydrogen_storage_kg,
        "fuel_cell_power_mw": config.fuel_cell_power_mw,
        "compute_it_capacity_mw": config.compute_it_capacity_mw,
        "hydrogen_export_capacity_kg_per_h": config.hydrogen_export_capacity_kg_per_h,
    }


def evaluate_capacity_candidates(
    candidates: Sequence[S8CapacityCandidate],
    cases: Sequence[S8ScenarioCase],
    params: TechnologyParameters,
    flexible_costs: InvestmentCostCase,
    infrastructure_costs: ChinaInfrastructureCostCase,
    *,
    tail_probability: float = 0.20,
    risk_aversion: float = 0.35,
) -> S8RiskScreeningResult:
    """Replay fixed candidates and select the highest risk-adjusted score.

    ``risk_aversion=0`` selects expected annual value. ``risk_aversion=1``
    selects lower-tail annual value.  Scenario regret is calculated only after
    every candidate has been evaluated under the same case.
    """
    if not candidates or not cases:
        raise ValueError("at least one candidate and one scenario case are required")
    if not 0.0 <= risk_aversion <= 1.0:
        raise ValueError("risk_aversion must lie in [0, 1]")
    candidate_ids = [item.candidate_id for item in candidates]
    case_ids = [item.case_id for item in cases]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate identifiers must be unique")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("scenario case identifiers must be unique")
    distances = {round(case.scenario.offshore_distance_km, 9) for case in cases}
    if len(distances) != 1:
        raise ValueError("all risk cases must share one physical offshore distance")
    distance = cases[0].scenario.offshore_distance_km
    raw_weights = np.asarray([case.weight for case in cases], dtype=float)
    normalized_weights = raw_weights / raw_weights.sum()
    normalized_by_case = dict(zip(case_ids, normalized_weights, strict=True))

    rows: list[dict[str, Any]] = []
    stack_rate = _stack_replacement_cost_per_mwh(flexible_costs)
    for candidate in candidates:
        flexible_capex, flexible_annual = fixed_flexible_asset_costs(
            candidate.configuration,
            flexible_costs,
            offshore_distance_km=distance,
        )
        common_capex, common_annual = common_infrastructure_costs(
            candidate.design,
            infrastructure_costs,
            offshore_distance_km=distance,
        )
        flexible_capex_total = float(sum(flexible_capex.values()))
        flexible_annual_total = float(sum(flexible_annual.values()))
        common_capex_total = float(sum(common_capex.values()))
        common_annual_total = float(sum(common_annual.values()))
        for case in cases:
            dispatch = run_s5_dispatch(
                case.timeseries,
                params,
                candidate.configuration,
                case.scenario,
            )
            hours = float(dispatch.kpis["simulation_hours"])
            annualization = 8_760.0 / hours
            stack_replacement = (
                float(dispatch.kpis["hydrogen_production_kg"])
                * params.value("hydrogen_sec_system")
                / 1_000.0
                * stack_rate
                * annualization
            )
            annual_operating_margin = (
                float(dispatch.kpis["operating_margin_cny"]) * annualization
                - stack_replacement
            )
            full_net = annual_operating_margin - flexible_annual_total - common_annual_total
            gross_revenue = float(
                (
                    dispatch.hourly["electricity_revenue_cny"]
                    + dispatch.hourly["hydrogen_gross_revenue_cny"]
                    + dispatch.hourly["compute_gross_revenue_cny"]
                ).sum()
                * annualization
            )
            spot_demand = float(dispatch.kpis["spot_compute_demand_mwh_it"])
            hydrogen_demand = float(
                dispatch.hourly["hydrogen_demand_kg"].sum()
            )
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "strategy": candidate.design.strategy,
                    "case_id": case.case_id,
                    "scenario_weight": normalized_by_case[case.case_id],
                    "full_project_capex_cny": common_capex_total + flexible_capex_total,
                    "common_infrastructure_annual_cost_cny": common_annual_total,
                    "flexible_asset_annual_cost_cny": flexible_annual_total,
                    "electrolyzer_stack_replacement_cny": stack_replacement,
                    "annual_operating_margin_cny": annual_operating_margin,
                    "gross_revenue_cny": gross_revenue,
                    "full_project_net_annual_value_cny": full_net,
                    "renewable_generation_mwh": float(
                        dispatch.kpis["renewable_generation_mwh"] * annualization
                    ),
                    "renewable_utilization_rate": dispatch.kpis[
                        "renewable_utilization_rate"
                    ],
                    "curtailment_mwh": float(
                        dispatch.kpis["curtailment_mwh"] * annualization
                    ),
                    "export_land_mwh": float(
                        dispatch.kpis["export_land_mwh"] * annualization
                    ),
                    "spot_compute_service_mwh_it": float(
                        dispatch.kpis["spot_compute_service_mwh_it"] * annualization
                    ),
                    "spot_compute_service_rate": (
                        float(dispatch.kpis["spot_compute_service_mwh_it"])
                        / spot_demand
                        if spot_demand > 0.0
                        else 1.0
                    ),
                    "hydrogen_sales_kg": float(
                        dispatch.kpis["hydrogen_sales_kg"] * annualization
                    ),
                    "hydrogen_demand_service_rate": (
                        float(dispatch.kpis["hydrogen_sales_kg"]) / hydrogen_demand
                        if hydrogen_demand > 0.0
                        else 1.0
                    ),
                    "eens_mwh": float(dispatch.kpis["eens_mwh"] * annualization),
                    "max_offshore_balance_residual_mw": dispatch.kpis[
                        "max_offshore_balance_residual_mw"
                    ],
                    "max_land_balance_residual_mw": dispatch.kpis[
                        "max_land_balance_residual_mw"
                    ],
                    **_capacity_record(candidate.configuration),
                }
            )

    return summarize_scenario_results(
        pd.DataFrame(rows),
        tail_probability=tail_probability,
        risk_aversion=risk_aversion,
    )


def summarize_scenario_results(
    scenario_results: pd.DataFrame,
    *,
    tail_probability: float = 0.20,
    risk_aversion: float = 0.35,
) -> S8RiskScreeningResult:
    """Rebuild risk metrics from checkpointed candidate-scenario rows."""
    if not 0.0 <= risk_aversion <= 1.0:
        raise ValueError("risk_aversion must lie in [0, 1]")
    required = {
        "candidate_id",
        "strategy",
        "case_id",
        "scenario_weight",
        "full_project_capex_cny",
        "full_project_net_annual_value_cny",
        "renewable_utilization_rate",
        "curtailment_mwh",
        "eens_mwh",
        "spot_compute_service_rate",
    }
    # Function annotations do not enumerate the returned mapping keys, so add
    # the explicit capacity contract used in saved S8 ledgers.
    required.update(
        {
            "battery_power_mw",
            "battery_energy_mwh",
            "electrolyzer_power_mw",
            "hydrogen_storage_kg",
            "fuel_cell_power_mw",
            "compute_it_capacity_mw",
            "hydrogen_export_capacity_kg_per_h",
        }
    )
    missing = sorted(required - set(scenario_results.columns))
    if missing:
        raise ValueError(f"scenario results are missing required columns: {missing}")
    scenario_results = scenario_results.copy()
    if scenario_results["strategy"].isna().any() or (
        scenario_results["strategy"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("scenario strategy labels must not be blank")
    duplicates = scenario_results.duplicated(["candidate_id", "case_id"])
    if duplicates.any():
        raise ValueError("scenario results contain duplicate candidate-case rows")
    candidate_case_counts = scenario_results.groupby("candidate_id")["case_id"].nunique()
    if candidate_case_counts.nunique() != 1:
        raise ValueError("every candidate must be evaluated in the same number of cases")
    cases_by_candidate = {
        candidate_id: set(group["case_id"])
        for candidate_id, group in scenario_results.groupby("candidate_id")
    }
    if len({frozenset(values) for values in cases_by_candidate.values()}) != 1:
        raise ValueError("every candidate must be evaluated against the same case identifiers")
    for _, group in scenario_results.groupby("candidate_id"):
        if not np.isclose(group["scenario_weight"].sum(), 1.0, atol=1e-10):
            raise ValueError("scenario weights for each candidate must sum to one")

    best_by_case = scenario_results.groupby("case_id")[
        "full_project_net_annual_value_cny"
    ].transform("max")
    scenario_results["scenario_regret_cny"] = (
        best_by_case - scenario_results["full_project_net_annual_value_cny"]
    )

    summaries: list[dict[str, Any]] = []
    for candidate_id, subset in scenario_results.groupby("candidate_id"):
        subset = subset.copy()
        values = subset["full_project_net_annual_value_cny"].to_numpy(dtype=float)
        weights = subset["scenario_weight"].to_numpy(dtype=float)
        expected = float(np.sum(values * weights))
        lower_tail = weighted_lower_tail_mean(
            values,
            weights,
            tail_probability=tail_probability,
        )
        risk_adjusted = (1.0 - risk_aversion) * expected + risk_aversion * lower_tail
        summaries.append(
            {
                "candidate_id": candidate_id,
                "strategy": str(subset["strategy"].iloc[0]),
                **{
                    key: float(subset[key].iloc[0])
                    for key in (
                        "battery_power_mw",
                        "battery_energy_mwh",
                        "electrolyzer_power_mw",
                        "hydrogen_storage_kg",
                        "fuel_cell_power_mw",
                        "compute_it_capacity_mw",
                        "hydrogen_export_capacity_kg_per_h",
                    )
                },
                "full_project_capex_cny": float(
                    subset["full_project_capex_cny"].iloc[0]
                ),
                "expected_full_net_value_cny_per_year": expected,
                "lower_tail_full_net_value_cny_per_year": lower_tail,
                "worst_case_full_net_value_cny_per_year": float(values.min()),
                "best_case_full_net_value_cny_per_year": float(values.max()),
                "risk_adjusted_score_cny_per_year": risk_adjusted,
                "positive_net_value_probability": float(
                    np.sum(weights[values >= 0.0])
                ),
                "expected_scenario_regret_cny": float(
                    np.sum(subset["scenario_regret_cny"].to_numpy(dtype=float) * weights)
                ),
                "maximum_scenario_regret_cny": float(
                    subset["scenario_regret_cny"].max()
                ),
                "expected_renewable_utilization_rate": float(
                    np.sum(
                        subset["renewable_utilization_rate"].to_numpy(dtype=float)
                        * weights
                    )
                ),
                "expected_curtailment_mwh": float(
                    np.sum(subset["curtailment_mwh"].to_numpy(dtype=float) * weights)
                ),
                "expected_eens_mwh": float(
                    np.sum(subset["eens_mwh"].to_numpy(dtype=float) * weights)
                ),
                "minimum_compute_service_rate": float(
                    subset["spot_compute_service_rate"].min()
                ),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        "risk_adjusted_score_cny_per_year", ascending=False
    ).reset_index(drop=True)
    selected = str(summary.iloc[0]["candidate_id"])
    metadata: dict[str, Any] = {
        "phase": "S8",
        "selection_rule": "max_risk_adjusted_score",
        "risk_adjusted_score": "(1-risk_aversion)*expected + risk_aversion*lower_tail",
        "tail_probability": tail_probability,
        "risk_aversion": risk_aversion,
        "scenario_weights": {
            str(row["case_id"]): float(row["scenario_weight"])
            for _, row in scenario_results.drop_duplicates("case_id").iterrows()
        },
        "operational_foresight": "perfect_within_each_scenario",
        "capacity_decision": "fixed_across_all_scenarios",
        "candidate_scope": "discrete_screening_not_continuous_stochastic_optimization",
    }
    return S8RiskScreeningResult(summary, scenario_results, selected, metadata)


def interpolate_required_capacity(
    capacities: Iterable[float],
    achieved_values: Iterable[float],
    *,
    target: float,
) -> float:
    """Linearly interpolate the minimum monotone capacity reaching ``target``."""
    x = np.asarray(tuple(capacities), dtype=float)
    y = np.asarray(tuple(achieved_values), dtype=float)
    if x.ndim != 1 or len(x) < 2 or len(x) != len(y):
        raise ValueError("capacity and value vectors must have equal length >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(target):
        raise ValueError("capacity, value and target must be finite")
    if (np.diff(x) <= 0.0).any():
        raise ValueError("capacities must be strictly increasing")
    if (np.diff(y) < -1e-10).any():
        raise ValueError("achieved values must be nondecreasing")
    if target <= y[0]:
        return float(x[0])
    if target > y[-1] + 1e-10:
        raise ValueError("target lies above the evaluated capacity range")
    index = int(np.searchsorted(y, target, side="left"))
    if abs(y[index] - target) <= 1e-12 or y[index] == y[index - 1]:
        return float(x[index])
    fraction = (target - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def capacity_mapping(config: SystemConfiguration) -> Mapping[str, float]:
    """Public immutable-style view used by scripts and teaching notebooks."""
    return _capacity_record(config)
