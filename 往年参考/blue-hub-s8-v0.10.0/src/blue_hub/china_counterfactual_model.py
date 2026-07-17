"""Phase 8 / S7 China-calibrated pain-point and counterfactual evaluation.

S7 keeps the auditable hourly S6 capacity-planning core and adds the common
deep-offshore infrastructure that was deliberately outside the S6 incremental
boundary.  It compares complete project designs under an absolute mainland
landing-point limit, so increasing export-cable capacity no longer increases
the receiving grid's capability by assumption.

The module separates three quantities that are often conflated in project
claims: merchant project value, avoided/deferred network expenditure, and
reliability or strategic value.  Only merchant revenue enters the optimizer;
the other quantities are reported as counterfactual comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blue_hub.investment_planning_model import (
    InvestmentCostCase,
    PlanningLimits,
    S6PlanningResult,
    capital_recovery_factor,
    run_s6_investment_planning,
)
from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters


@dataclass(frozen=True)
class ChinaInfrastructureCostCase:
    """Installed common-infrastructure assumptions for one screening case."""

    cost_case_id: str
    discount_rate: float
    offshore_wind_capex_cny_per_kw: float
    offshore_wind_fixed_om_fraction: float
    offshore_wind_lifetime_years: float
    offshore_pv_capex_cny_per_kw: float
    offshore_pv_fixed_om_fraction: float
    offshore_pv_lifetime_years: float
    wave_capex_cny_per_kw: float
    wave_fixed_om_fraction: float
    wave_lifetime_years: float
    hvdc_terminal_capex_cny_per_kw: float
    hvdc_cable_capex_cny_per_mw_km: float
    landing_grid_capex_cny_per_kw: float
    transmission_fixed_om_fraction: float
    transmission_lifetime_years: float
    hub_common_capex_cny: float
    hub_common_fixed_om_fraction: float
    hub_common_lifetime_years: float
    source_basis: str
    notes: str

    def __post_init__(self) -> None:
        if not self.cost_case_id.strip():
            raise ValueError("cost_case_id must not be blank")
        if not 0.0 <= self.discount_rate < 1.0:
            raise ValueError("discount_rate must lie in [0, 1)")
        for name, value in asdict(self).items():
            if name in {"cost_case_id", "source_basis", "notes"}:
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "offshore_wind_lifetime_years",
            "offshore_pv_lifetime_years",
            "wave_lifetime_years",
            "transmission_lifetime_years",
            "hub_common_lifetime_years",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class S7Design:
    """One complete design option used in a common counterfactual comparison."""

    design_id: str
    strategy: str
    wind_capacity_mw: float
    pv_capacity_mw: float
    wave_capacity_mw: float
    tx_capacity_mw: float
    landing_grid_limit_mw: float
    include_hub_common_assets: bool
    enable_flexible_assets: bool

    def __post_init__(self) -> None:
        if not self.design_id.strip() or not self.strategy.strip():
            raise ValueError("design_id and strategy must not be blank")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class S7CounterfactualResult:
    """Hourly planning result plus full-project screening economics."""

    planning: S6PlanningResult
    design: S7Design
    kpis: dict[str, Any]
    common_capex: dict[str, float]
    common_annual_cost: dict[str, float]


def load_china_infrastructure_cost_cases(
    path: str | Path,
) -> tuple[ChinaInfrastructureCostCase, ...]:
    """Load transparent S7 common-infrastructure cases from CSV."""
    frame = pd.read_csv(path, keep_default_na=False)
    cases = tuple(
        ChinaInfrastructureCostCase(**row) for row in frame.to_dict(orient="records")
    )
    identifiers = [case.cost_case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("infrastructure cost case identifiers must be unique")
    return cases


def build_lagged_wave_capacity_factor(
    wind_cf: np.ndarray | pd.Series,
    *,
    target_mean: float = 0.25,
    lag_hours: int = 4,
    smoothing_hours: int = 6,
) -> np.ndarray:
    """Return a bounded wave proxy that lags and smooths the wind resource.

    This is a screening profile, not a site hindcast.  It exists so a claimed
    wind-wave complementarity can be tested rather than assumed.
    """
    values = np.asarray(wind_cf, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("wind_cf must be a finite non-empty vector")
    if not 0.0 <= target_mean <= 1.0:
        raise ValueError("target_mean must lie in [0, 1]")
    if lag_hours < 0 or smoothing_hours <= 0:
        raise ValueError("lag_hours must be nonnegative and smoothing_hours positive")
    lagged = np.roll(values, lag_hours)
    kernel = np.ones(smoothing_hours, dtype=float) / smoothing_hours
    extended = np.concatenate([lagged[-(smoothing_hours - 1) :], lagged])
    smoothed = np.convolve(extended, kernel, mode="valid")[: len(values)]
    mean = float(smoothed.mean())
    if mean <= 0.0:
        return np.full(len(values), target_mean)
    return np.clip(smoothed * target_mean / mean, 0.0, 1.0)


def calibrate_landing_limit_mw(
    frame: pd.DataFrame,
    *,
    wind_capacity_mw: float,
    pv_capacity_mw: float,
    tx_capacity_mw: float,
    target_utilization: float,
) -> float:
    """Calibrate an absolute landing limit to a direct-export utilization target.

    The calculation assumes nonnegative export prices and no flexible assets.
    It includes the island critical load before calculating residual export.
    """
    if not 0.0 < target_utilization <= 1.0:
        raise ValueError("target_utilization must lie in (0, 1]")
    wind = (
        wind_capacity_mw
        * frame["wind_cf"].to_numpy(dtype=float)
        * frame["wind_availability"].to_numpy(dtype=float)
    )
    pv = pv_capacity_mw * frame["pv_cf"].to_numpy(dtype=float)
    wave = (
        frame["wave_generation_mw"].to_numpy(dtype=float)
        if "wave_generation_mw" in frame.columns
        else np.zeros(len(frame))
    )
    renewable = wind + pv + wave
    critical = frame["critical_load"].to_numpy(dtype=float)
    tx_availability = frame["tx_availability"].to_numpy(dtype=float)
    grid_factor = (
        frame["grid_absorption_factor"].to_numpy(dtype=float)
        if "grid_absorption_factor" in frame.columns
        else np.ones(len(frame))
    )
    landing_profile = (
        frame["landing_demand_factor"].to_numpy(dtype=float)
        if "landing_demand_factor" in frame.columns
        else np.ones(len(frame))
    )
    total = float(renewable.sum())
    if total <= 0.0:
        raise ValueError("renewable generation must be positive")

    def utilization(limit: float) -> float:
        export_cap = np.minimum(
            tx_capacity_mw * tx_availability * grid_factor,
            limit * landing_profile,
        )
        used = np.minimum(renewable, critical + export_cap)
        return float(used.sum() / total)

    if utilization(tx_capacity_mw) < target_utilization - 1e-10:
        raise ValueError("target utilization is unattainable with the stated cable capacity")
    low, high = 0.0, tx_capacity_mw
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        if utilization(midpoint) < target_utilization:
            low = midpoint
        else:
            high = midpoint
    return high


def _annual_cost(capex: float, fixed_om_fraction: float, life: float, rate: float) -> float:
    return capex * (capital_recovery_factor(rate, life) + fixed_om_fraction)


def common_infrastructure_costs(
    design: S7Design,
    costs: ChinaInfrastructureCostCase,
    *,
    offshore_distance_km: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return common CAPEX and annualized cost without flexible-asset double counting."""
    capex = {
        "offshore_wind_capex_cny": design.wind_capacity_mw
        * 1_000.0
        * costs.offshore_wind_capex_cny_per_kw,
        "offshore_pv_capex_cny": design.pv_capacity_mw
        * 1_000.0
        * costs.offshore_pv_capex_cny_per_kw,
        "wave_capex_cny": design.wave_capacity_mw * 1_000.0 * costs.wave_capex_cny_per_kw,
        "hvdc_terminal_capex_cny": design.tx_capacity_mw
        * 1_000.0
        * costs.hvdc_terminal_capex_cny_per_kw,
        "hvdc_cable_capex_cny": design.tx_capacity_mw
        * offshore_distance_km
        * costs.hvdc_cable_capex_cny_per_mw_km,
        "landing_grid_capex_cny": design.landing_grid_limit_mw
        * 1_000.0
        * costs.landing_grid_capex_cny_per_kw,
        "hub_common_capex_cny": (
            costs.hub_common_capex_cny if design.include_hub_common_assets else 0.0
        ),
    }
    annual = {
        "offshore_wind_annual_cost_cny": _annual_cost(
            capex["offshore_wind_capex_cny"],
            costs.offshore_wind_fixed_om_fraction,
            costs.offshore_wind_lifetime_years,
            costs.discount_rate,
        ),
        "offshore_pv_annual_cost_cny": _annual_cost(
            capex["offshore_pv_capex_cny"],
            costs.offshore_pv_fixed_om_fraction,
            costs.offshore_pv_lifetime_years,
            costs.discount_rate,
        ),
        "wave_annual_cost_cny": _annual_cost(
            capex["wave_capex_cny"],
            costs.wave_fixed_om_fraction,
            costs.wave_lifetime_years,
            costs.discount_rate,
        ),
        "transmission_annual_cost_cny": _annual_cost(
            capex["hvdc_terminal_capex_cny"]
            + capex["hvdc_cable_capex_cny"]
            + capex["landing_grid_capex_cny"],
            costs.transmission_fixed_om_fraction,
            costs.transmission_lifetime_years,
            costs.discount_rate,
        ),
        "hub_common_annual_cost_cny": _annual_cost(
            capex["hub_common_capex_cny"],
            costs.hub_common_fixed_om_fraction,
            costs.hub_common_lifetime_years,
            costs.discount_rate,
        ),
    }
    return capex, annual


def evaluate_s7_design(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    scenario: ScenarioDefinition,
    flexible_costs: InvestmentCostCase,
    infrastructure_costs: ChinaInfrastructureCostCase,
    limits: PlanningLimits,
    design: S7Design,
) -> S7CounterfactualResult:
    """Optimize flexible capacity and calculate complete-project screening value."""
    frame = timeseries.copy()
    landing_profile = (
        frame["landing_demand_factor"].to_numpy(dtype=float)
        if "landing_demand_factor" in frame.columns
        else np.ones(len(frame))
    )
    frame["grid_absorption_limit_mw"] = design.landing_grid_limit_mw * landing_profile
    if design.wave_capacity_mw > 0.0:
        wave_cf = (
            frame["wave_cf"].to_numpy(dtype=float)
            if "wave_cf" in frame.columns
            else build_lagged_wave_capacity_factor(frame["wind_cf"])
        )
        frame["wave_generation_mw"] = design.wave_capacity_mw * wave_cf
    else:
        frame["wave_generation_mw"] = 0.0
    config = base_config.model_copy(
        update={
            "config_id": design.design_id,
            "wind_capacity_mw": design.wind_capacity_mw,
            "pv_capacity_mw": design.pv_capacity_mw,
            "tx_capacity_mw": design.tx_capacity_mw,
        }
    )
    effective_limits = (
        limits
        if design.enable_flexible_assets
        else PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    planning = run_s6_investment_planning(
        frame, params, config, scenario, flexible_costs, effective_limits
    )
    common_capex, common_annual = common_infrastructure_costs(
        design, infrastructure_costs, offshore_distance_km=scenario.offshore_distance_km
    )
    common_capex_total = float(sum(common_capex.values()))
    common_annual_total = float(sum(common_annual.values()))
    flexible_capex = float(planning.kpis["flexible_asset_capex_cny"])
    flexible_annual = float(planning.kpis["annualized_flexible_fixed_cost_cny"])
    annual_margin = float(planning.kpis["annualized_operating_margin_cny"])
    full_net = annual_margin - flexible_annual - common_annual_total
    renewable = float(planning.kpis["renewable_generation_mwh"])
    export_land = float(planning.kpis["export_land_mwh"])
    gross_revenue = float(
        planning.kpis["annualized_electricity_revenue_cny"]
        + planning.kpis["annualized_hydrogen_revenue_cny"]
        + planning.kpis["annualized_compute_revenue_cny"]
    )
    annual_cash_cost = gross_revenue - full_net
    kpis: dict[str, Any] = {
        "phase": "S7",
        "design_id": design.design_id,
        "strategy": design.strategy,
        "scenario_id": scenario.scenario_id,
        "flexible_cost_case_id": flexible_costs.cost_case_id,
        "infrastructure_cost_case_id": infrastructure_costs.cost_case_id,
        "wind_capacity_mw": design.wind_capacity_mw,
        "pv_capacity_mw": design.pv_capacity_mw,
        "wave_capacity_mw": design.wave_capacity_mw,
        "tx_capacity_mw": design.tx_capacity_mw,
        "landing_grid_limit_mw": design.landing_grid_limit_mw,
        **planning.capacities,
        "renewable_generation_mwh": renewable,
        "renewable_utilization_rate": planning.kpis["renewable_utilization_rate"],
        "curtailment_mwh": planning.kpis["curtailment_mwh"],
        "export_land_mwh": export_land,
        "electrolyzer_energy_mwh": planning.kpis["electrolyzer_energy_mwh"],
        "hydrogen_sales_kg": planning.kpis["hydrogen_sales_kg"],
        "dc_facility_energy_mwh": planning.kpis["dc_facility_energy_mwh"],
        "spot_compute_service_mwh_it": planning.kpis["spot_compute_service_mwh_it"],
        "constrained_grid_hours": planning.kpis["constrained_grid_hours"],
        "electricity_capture_price_cny_per_mwh_land": (
            planning.kpis["annualized_electricity_revenue_cny"] / export_land
            if export_land > 0.0
            else 0.0
        ),
        "common_infrastructure_capex_cny": common_capex_total,
        "flexible_asset_capex_cny": flexible_capex,
        "full_project_capex_cny": common_capex_total + flexible_capex,
        "common_infrastructure_annual_cost_cny": common_annual_total,
        "flexible_asset_annual_cost_cny": flexible_annual,
        "gross_revenue_cny_per_year": gross_revenue,
        "annual_cash_cost_cny_per_year": annual_cash_cost,
        "full_project_net_annual_value_cny": full_net,
        "full_cost_recovery_ratio": (
            gross_revenue / annual_cash_cost if annual_cash_cost > 0.0 else 0.0
        ),
        "additional_revenue_required_cny_per_generated_mwh": (
            max(-full_net, 0.0) / renewable if renewable > 0.0 else 0.0
        ),
        "electricity_export_share_of_generation": (
            float((planning.hourly["export_send_mw"]).sum()) / renewable
            if renewable > 0.0
            else 0.0
        ),
        "compute_power_share_of_generation": (
            planning.kpis["dc_facility_energy_mwh"] / renewable if renewable > 0.0 else 0.0
        ),
        "hydrogen_power_share_of_generation": (
            planning.kpis["electrolyzer_energy_mwh"] / renewable if renewable > 0.0 else 0.0
        ),
        "curtailment_share_of_generation": (
            planning.kpis["curtailment_mwh"] / renewable if renewable > 0.0 else 0.0
        ),
        "max_offshore_balance_residual_mw": planning.kpis[
            "max_offshore_balance_residual_mw"
        ],
        "max_land_balance_residual_mw": planning.kpis["max_land_balance_residual_mw"],
    }
    return S7CounterfactualResult(
        planning=planning,
        design=design,
        kpis=kpis,
        common_capex=common_capex,
        common_annual_cost=common_annual,
    )
