"""Phase 7 / S6 endogenous sizing of flexible offshore-hub assets.

The S6 planner keeps the S5 electricity, battery and hydrogen accounting but
makes six flexible-asset capacities decision variables.  It is an incremental
planning model: wind, PV and the export cable are treated as common existing
assets, while battery, electrolyzer, hydrogen storage, fuel cell and bundled
compute/fibre capacity must recover their own annualized fixed costs.

The representative S6 study uses the optional nationwide spot-compute pool and
sets contracted rigid/flexible arrivals to zero.  An optimal fixed configuration
is subsequently replayed through the full S5 dispatch model for validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from blue_hub import __version__
from blue_hub.provenance import configuration_hash
from blue_hub.scenario_runner import apply_scenario
from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters
from blue_hub.transmission import (
    TransmissionLossSpec,
    build_piecewise_transmission,
    calculate_transmission_flow,
)
from blue_hub.validation import validate_parameters, validate_timeseries


@dataclass(frozen=True)
class InvestmentCostCase:
    """Installed-cost, fixed-O&M and lifetime assumptions for one cost case."""

    cost_case_id: str
    discount_rate: float
    battery_power_capex_cny_per_kw: float
    battery_energy_capex_cny_per_kwh: float
    battery_fixed_om_fraction: float
    battery_lifetime_years: float
    electrolyzer_capex_cny_per_kw: float
    electrolyzer_fixed_om_fraction: float
    electrolyzer_lifetime_years: float
    electrolyzer_replacement_fraction: float
    electrolyzer_replacement_interval_full_load_h: float
    hydrogen_storage_capex_cny_per_kg: float
    hydrogen_storage_fixed_om_fraction: float
    hydrogen_storage_lifetime_years: float
    fuel_cell_capex_cny_per_kw: float
    fuel_cell_fixed_om_fraction: float
    fuel_cell_lifetime_years: float
    compute_fibre_capex_cny_per_kw_it: float
    compute_fibre_fixed_om_fraction: float
    compute_fibre_lifetime_years: float
    source_basis: str
    notes: str
    hydrogen_pipeline_capex_cny_per_kg_h_km: float = 0.0
    hydrogen_pipeline_fixed_om_fraction: float = 0.0
    hydrogen_pipeline_lifetime_years: float = 30.0

    def __post_init__(self) -> None:
        if not self.cost_case_id.strip():
            raise ValueError("cost_case_id must not be blank")
        if not 0.0 <= self.discount_rate < 1.0:
            raise ValueError("discount_rate must lie in [0, 1)")
        numeric = asdict(self)
        for name, value in numeric.items():
            if name in {"cost_case_id", "source_basis", "notes", "discount_rate"}:
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "battery_lifetime_years",
            "electrolyzer_lifetime_years",
            "electrolyzer_replacement_interval_full_load_h",
            "hydrogen_storage_lifetime_years",
            "fuel_cell_lifetime_years",
            "compute_fibre_lifetime_years",
            "hydrogen_pipeline_lifetime_years",
        ):
            if float(numeric[name]) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PlanningLimits:
    """Upper bounds that define the engineering search domain."""

    battery_power_mw: float
    battery_energy_mwh: float
    electrolyzer_power_mw: float
    hydrogen_storage_kg: float
    fuel_cell_power_mw: float
    compute_it_capacity_mw: float
    hydrogen_export_capacity_kg_per_h: float = 1.0e12

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0.0:
                raise ValueError(f"planning limit {name} must be nonnegative")


@dataclass(frozen=True)
class S6PlanningResult:
    """Optimal capacities, annual economics, hourly ledger and provenance."""

    hourly: pd.DataFrame
    capacities: dict[str, float]
    kpis: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Layout:
    periods: int
    segments: int
    charge_start: int
    discharge_start: int
    curtail_start: int
    unmet_start: int
    h2_power_start: int
    h2_sale_start: int
    fuel_cell_start: int
    spot_start: int
    battery_energy_start: int
    h2_inventory_start: int
    battery_power_capacity: int
    battery_energy_capacity: int
    electrolyzer_capacity: int
    h2_storage_capacity: int
    fuel_cell_capacity: int
    compute_capacity: int
    hydrogen_export_capacity: int
    count: int


def _layout(periods: int, segments: int) -> _Layout:
    charge_start = periods * segments
    discharge_start = charge_start + periods
    curtail_start = discharge_start + periods
    unmet_start = curtail_start + periods
    h2_power_start = unmet_start + periods
    h2_sale_start = h2_power_start + periods
    fuel_cell_start = h2_sale_start + periods
    spot_start = fuel_cell_start + periods
    battery_energy_start = spot_start + periods
    h2_inventory_start = battery_energy_start + periods + 1
    battery_power_capacity = h2_inventory_start + periods + 1
    return _Layout(
        periods=periods,
        segments=segments,
        charge_start=charge_start,
        discharge_start=discharge_start,
        curtail_start=curtail_start,
        unmet_start=unmet_start,
        h2_power_start=h2_power_start,
        h2_sale_start=h2_sale_start,
        fuel_cell_start=fuel_cell_start,
        spot_start=spot_start,
        battery_energy_start=battery_energy_start,
        h2_inventory_start=h2_inventory_start,
        battery_power_capacity=battery_power_capacity,
        battery_energy_capacity=battery_power_capacity + 1,
        electrolyzer_capacity=battery_power_capacity + 2,
        h2_storage_capacity=battery_power_capacity + 3,
        fuel_cell_capacity=battery_power_capacity + 4,
        compute_capacity=battery_power_capacity + 5,
        hydrogen_export_capacity=battery_power_capacity + 6,
        count=battery_power_capacity + 7,
    )


def capital_recovery_factor(discount_rate: float, lifetime_years: float) -> float:
    """Return the standard real capital-recovery factor."""
    if lifetime_years <= 0.0:
        raise ValueError("lifetime_years must be positive")
    if discount_rate < 0.0:
        raise ValueError("discount_rate must be nonnegative")
    if discount_rate == 0.0:
        return 1.0 / lifetime_years
    growth = (1.0 + discount_rate) ** lifetime_years
    return discount_rate * growth / (growth - 1.0)


def load_investment_cost_cases(path: str | Path) -> tuple[InvestmentCostCase, ...]:
    """Load transparent S6 cost cases from CSV."""
    frame = pd.read_csv(path, keep_default_na=False)
    cases = tuple(InvestmentCostCase(**row) for row in frame.to_dict(orient="records"))
    ids = [case.cost_case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("investment cost case identifiers must be unique")
    return cases


def _loss_spec(params: TechnologyParameters, technology: str) -> TransmissionLossSpec:
    return TransmissionLossSpec(
        terminal_loss_fraction=params.value(f"tx_terminal_loss_fraction_{technology}"),
        cable_full_load_loss_per_100km=params.value(
            f"tx_cable_full_load_loss_per_100km_{technology}"
        ),
    )


def _optional_series(
    frame: pd.DataFrame,
    name: str,
    default: float,
    *,
    lower: float = 0.0,
    upper: float | None = None,
) -> np.ndarray:
    values = (
        frame[name].to_numpy(dtype=float)
        if name in frame.columns
        else np.full(len(frame), default, dtype=float)
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if (values < lower).any() or (upper is not None and (values > upper).any()):
        raise ValueError(f"{name} lies outside its allowed range")
    return values


def _pue(params: TechnologyParameters, case: str) -> float:
    item = next(item for item in params.items if item.parameter == "data_center_pue")
    values = {
        "optimistic": item.value_low,
        "base": item.value_base,
        "conservative": item.value_high,
    }
    try:
        return values[case.lower()]
    except KeyError as error:
        raise ValueError("pue_case must be optimistic, base or conservative") from error


def _annualized_unit_costs(costs: InvestmentCostCase) -> dict[str, float]:
    def annual(capex: float, fixed_fraction: float, life: float) -> float:
        return capex * (capital_recovery_factor(costs.discount_rate, life) + fixed_fraction)

    return {
        "battery_power_cny_per_kw_year": annual(
            costs.battery_power_capex_cny_per_kw,
            costs.battery_fixed_om_fraction,
            costs.battery_lifetime_years,
        ),
        "battery_energy_cny_per_kwh_year": annual(
            costs.battery_energy_capex_cny_per_kwh,
            costs.battery_fixed_om_fraction,
            costs.battery_lifetime_years,
        ),
        "electrolyzer_cny_per_kw_year": annual(
            costs.electrolyzer_capex_cny_per_kw,
            costs.electrolyzer_fixed_om_fraction,
            costs.electrolyzer_lifetime_years,
        ),
        "hydrogen_storage_cny_per_kg_year": annual(
            costs.hydrogen_storage_capex_cny_per_kg,
            costs.hydrogen_storage_fixed_om_fraction,
            costs.hydrogen_storage_lifetime_years,
        ),
        "fuel_cell_cny_per_kw_year": annual(
            costs.fuel_cell_capex_cny_per_kw,
            costs.fuel_cell_fixed_om_fraction,
            costs.fuel_cell_lifetime_years,
        ),
        "compute_fibre_cny_per_kw_it_year": annual(
            costs.compute_fibre_capex_cny_per_kw_it,
            costs.compute_fibre_fixed_om_fraction,
            costs.compute_fibre_lifetime_years,
        ),
        "hydrogen_pipeline_cny_per_kg_h_km_year": annual(
            costs.hydrogen_pipeline_capex_cny_per_kg_h_km,
            costs.hydrogen_pipeline_fixed_om_fraction,
            costs.hydrogen_pipeline_lifetime_years,
        ),
        "electrolyzer_stack_replacement_cny_per_mwh": (
            costs.electrolyzer_capex_cny_per_kw
            * costs.electrolyzer_replacement_fraction
            * 1_000.0
            / costs.electrolyzer_replacement_interval_full_load_h
        ),
    }


def _safe_max(values: np.ndarray) -> float:
    return float(np.abs(values).max()) if len(values) else 0.0


def run_s6_investment_planning(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    scenario: ScenarioDefinition,
    costs: InvestmentCostCase,
    limits: PlanningLimits,
) -> S6PlanningResult:
    """Jointly choose flexible capacities and their hourly operation."""
    validate_timeseries(timeseries).raise_if_invalid()
    validate_parameters(params).raise_if_invalid()
    if base_config.export_policy != "economic":
        raise ValueError("S6 planning supports economic export only")
    dt = params.value("time_step")
    if dt != 1.0:
        raise ValueError("S6 planning supports a 1 h time step only")

    ts = apply_scenario(timeseries, scenario)
    periods = len(ts)
    if periods <= 0:
        raise ValueError("timeseries must contain at least one period")
    annualization_factor = 8_760.0 / (periods * dt)
    wind = (
        base_config.wind_capacity_mw
        * ts["wind_cf"].to_numpy(dtype=float)
        * ts["wind_availability"].to_numpy(dtype=float)
    )
    pv = base_config.pv_capacity_mw * ts["pv_cf"].to_numpy(dtype=float)
    wave = _optional_series(ts, "wave_generation_mw", 0.0)
    renewable = wind + pv + wave
    critical = ts["critical_load"].to_numpy(dtype=float)
    price = ts["electricity_price"].to_numpy(dtype=float)
    h2_demand = ts["hydrogen_demand"].to_numpy(dtype=float)
    grid_absorption = _optional_series(
        ts, "grid_absorption_factor", 1.0, lower=0.0, upper=1.0
    )
    spot_pool = _optional_series(ts, "national_compute_demand_mw_it", 0.0)
    spot_flexible_fraction = _optional_series(
        ts, "national_compute_flexible_fraction", 1.0, lower=0.0, upper=1.0
    )
    spot_demand = spot_pool * spot_flexible_fraction
    spot_price = (
        _optional_series(
            ts,
            "national_compute_price_cny_per_mwh_it",
            params.value("compute_spot_service_price"),
        )
        * scenario.compute_price_multiplier
    )
    fiber_availability = _optional_series(
        ts, "fiber_availability", 1.0, lower=0.0, upper=1.0
    )
    tx_physical_available = (
        base_config.tx_capacity_mw * ts["tx_availability"].to_numpy(dtype=float)
    )
    proportional_tx_available = tx_physical_available * grid_absorption
    if "grid_absorption_limit_mw" in ts.columns:
        grid_absorption_limit = _optional_series(ts, "grid_absorption_limit_mw", 0.0)
    else:
        grid_absorption_limit = np.full(periods, 1.0e12)
    tx_available = np.minimum(proportional_tx_available, grid_absorption_limit)

    one_way_efficiency = sqrt(params.value("battery_round_trip_efficiency"))
    battery_self_discharge = params.value("battery_self_discharge_per_hour")
    battery_min_fraction = params.value("battery_soc_min")
    battery_max_fraction = params.value("battery_soc_max")
    battery_initial_fraction = base_config.initial_battery_soc_fraction
    if not battery_min_fraction <= battery_initial_fraction <= battery_max_fraction:
        raise ValueError("initial battery SOC fraction must satisfy battery SOC limits")

    h2_sec = params.value("hydrogen_sec_system")
    h2_kg_per_mwh = 1_000.0 / h2_sec
    h2_storage_loss = params.value("hydrogen_storage_loss_per_hour")
    h2_initial_fraction = base_config.initial_hydrogen_inventory_fraction
    h2_sale_price = params.value("hydrogen_sale_price") * scenario.hydrogen_price_multiplier
    h2_transport_cost = params.value("hydrogen_transport_cost")
    h2_variable_cost = params.value("electrolyzer_variable_cost")
    h2_water_cost = params.value("hydrogen_water_consumption") * params.value(
        "desalinated_water_cost"
    )
    fuel_cell_efficiency = params.value("fuel_cell_efficiency_lhv")
    fuel_cell_kg_per_mwh = 1_000.0 / (
        params.value("hydrogen_lhv_kwh_per_kg") * fuel_cell_efficiency
    )
    fuel_cell_variable_cost = params.value("fuel_cell_variable_cost")
    pue = _pue(params, scenario.pue_case)
    compute_variable_cost = params.value("compute_variable_cost")
    compute_ramp_up = params.value("compute_ramp_up")
    compute_ramp_down = params.value("compute_ramp_down")

    loss_spec = _loss_spec(params, base_config.tx_technology)
    segments = int(round(params.value("battery_linearization_segments")))
    if base_config.tx_capacity_mw > 0.0:
        piecewise = build_piecewise_transmission(
            base_config.tx_capacity_mw,
            scenario.offshore_distance_km,
            loss_spec,
            segments,
        )
        widths = piecewise.segment_widths_mw
        slopes = piecewise.land_delivery_slopes
    else:
        segments = 0
        widths = np.empty(0)
        slopes = np.empty(0)

    layout = _layout(periods, segments)
    objective = np.zeros(layout.count)
    tx_cost = params.value("tx_variable_cost")
    for t in range(periods):
        for segment in range(segments):
            objective[t * segments + segment] = (
                (tx_cost - price[t] * slopes[segment]) * dt * annualization_factor
            )
    objective[layout.charge_start : layout.charge_start + periods] = (
        params.value("battery_degradation_cost") * dt * annualization_factor
    )
    objective[layout.discharge_start : layout.discharge_start + periods] = (
        params.value("battery_degradation_cost") * dt * annualization_factor
    )
    objective[layout.curtail_start : layout.curtail_start + periods] = (
        params.value("curtailment_penalty") * dt * annualization_factor
    )
    objective[layout.unmet_start : layout.unmet_start + periods] = (
        params.value("unserved_critical_load_penalty") * dt * annualization_factor
    )
    unit_cost = _annualized_unit_costs(costs)
    objective[layout.h2_power_start : layout.h2_power_start + periods] = (
        (
            (h2_variable_cost + h2_water_cost) * h2_kg_per_mwh
            + unit_cost["electrolyzer_stack_replacement_cny_per_mwh"]
        )
        * dt
        * annualization_factor
    )
    objective[layout.h2_sale_start : layout.h2_sale_start + periods] = -(
        (h2_sale_price - h2_transport_cost) * annualization_factor
    )
    objective[layout.fuel_cell_start : layout.fuel_cell_start + periods] = (
        fuel_cell_variable_cost * dt * annualization_factor
    )
    objective[layout.spot_start : layout.spot_start + periods] = (
        (compute_variable_cost - spot_price) * annualization_factor
    )
    objective[layout.battery_power_capacity] = (
        unit_cost["battery_power_cny_per_kw_year"] * 1_000.0
    )
    objective[layout.battery_energy_capacity] = (
        unit_cost["battery_energy_cny_per_kwh_year"] * 1_000.0
    )
    objective[layout.electrolyzer_capacity] = (
        unit_cost["electrolyzer_cny_per_kw_year"] * 1_000.0
    )
    objective[layout.h2_storage_capacity] = unit_cost[
        "hydrogen_storage_cny_per_kg_year"
    ]
    objective[layout.fuel_cell_capacity] = (
        unit_cost["fuel_cell_cny_per_kw_year"] * 1_000.0
    )
    objective[layout.compute_capacity] = (
        unit_cost["compute_fibre_cny_per_kw_it_year"] * 1_000.0
    )
    objective[layout.hydrogen_export_capacity] = (
        unit_cost["hydrogen_pipeline_cny_per_kg_h_km_year"]
        * scenario.offshore_distance_km
    )

    bounds: list[tuple[float, float | None]] = []
    for _ in range(periods):
        bounds.extend((0.0, float(width)) for width in widths)
    bounds.extend((0.0, None) for _ in range(periods * 2))
    bounds.extend((0.0, float(value)) for value in renewable)
    bounds.extend((0.0, float(value)) for value in critical)
    bounds.extend((0.0, None) for _ in range(periods))
    bounds.extend((0.0, float(value * dt)) for value in h2_demand)
    bounds.extend((0.0, None) for _ in range(periods))
    bounds.extend((0.0, float(value * dt)) for value in spot_demand)
    bounds.extend((0.0, None) for _ in range(2 * (periods + 1)))
    bounds.extend(
        [
            (0.0, limits.battery_power_mw),
            (0.0, limits.battery_energy_mwh),
            (0.0, limits.electrolyzer_power_mw),
            (0.0, limits.hydrogen_storage_kg),
            (0.0, limits.fuel_cell_power_mw),
            (0.0, limits.compute_it_capacity_mw),
            (0.0, limits.hydrogen_export_capacity_kg_per_h),
        ]
    )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    equality_rhs: list[float] = []
    row = 0
    for t in range(periods):
        for segment in range(segments):
            rows.append(row)
            cols.append(t * segments + segment)
            data.append(1.0)
        rows.extend([row] * 7)
        cols.extend(
            [
                layout.charge_start + t,
                layout.curtail_start + t,
                layout.h2_power_start + t,
                layout.spot_start + t,
                layout.discharge_start + t,
                layout.fuel_cell_start + t,
                layout.unmet_start + t,
            ]
        )
        data.extend([1.0, 1.0, 1.0, pue / dt, -1.0, -1.0, -1.0])
        equality_rhs.append(float(renewable[t] - critical[t]))
        row += 1

    for t in range(periods):
        rows.extend([row] * 4)
        cols.extend(
            [
                layout.battery_energy_start + t + 1,
                layout.battery_energy_start + t,
                layout.charge_start + t,
                layout.discharge_start + t,
            ]
        )
        data.extend(
            [
                1.0,
                -(1.0 - battery_self_discharge),
                -one_way_efficiency * dt,
                dt / one_way_efficiency,
            ]
        )
        equality_rhs.append(0.0)
        row += 1

    for t in range(periods):
        rows.extend([row] * 5)
        cols.extend(
            [
                layout.h2_inventory_start + t + 1,
                layout.h2_inventory_start + t,
                layout.h2_power_start + t,
                layout.h2_sale_start + t,
                layout.fuel_cell_start + t,
            ]
        )
        data.extend(
            [
                1.0,
                -(1.0 - h2_storage_loss),
                -h2_kg_per_mwh * dt,
                1.0,
                fuel_cell_kg_per_mwh * dt,
            ]
        )
        equality_rhs.append(0.0)
        row += 1

    for state, capacity, fraction in (
        (layout.battery_energy_start, layout.battery_energy_capacity, battery_initial_fraction),
        (
            layout.battery_energy_start + periods,
            layout.battery_energy_capacity,
            battery_initial_fraction,
        ),
        (layout.h2_inventory_start, layout.h2_storage_capacity, h2_initial_fraction),
        (
            layout.h2_inventory_start + periods,
            layout.h2_storage_capacity,
            h2_initial_fraction,
        ),
    ):
        rows.extend([row, row])
        cols.extend([state, capacity])
        data.extend([1.0, -fraction])
        equality_rhs.append(0.0)
        row += 1
    equality = coo_matrix((data, (rows, cols)), shape=(row, layout.count)).tocsr()

    ir: list[int] = []
    ic: list[int] = []
    idata: list[float] = []
    inequality_rhs: list[float] = []
    export_rows: list[int] = []
    row = 0

    def add_constraint(indices: list[int], coefficients: list[float], rhs: float) -> None:
        nonlocal row
        ir.extend([row] * len(indices))
        ic.extend(indices)
        idata.extend(coefficients)
        inequality_rhs.append(rhs)
        row += 1

    for t in range(periods):
        export_rows.append(row)
        add_constraint(
            [t * segments + segment for segment in range(segments)],
            [1.0] * segments,
            float(tx_available[t]),
        )
    for t in range(periods):
        add_constraint(
            [layout.charge_start + t, layout.battery_power_capacity],
            [1.0, -1.0],
            0.0,
        )
        add_constraint(
            [layout.discharge_start + t, layout.battery_power_capacity],
            [1.0, -1.0],
            0.0,
        )
    for state in range(periods + 1):
        energy_index = layout.battery_energy_start + state
        add_constraint(
            [energy_index, layout.battery_energy_capacity],
            [1.0, -battery_max_fraction],
            0.0,
        )
        add_constraint(
            [energy_index, layout.battery_energy_capacity],
            [-1.0, battery_min_fraction],
            0.0,
        )
        inventory_index = layout.h2_inventory_start + state
        add_constraint(
            [inventory_index, layout.h2_storage_capacity],
            [1.0, -1.0],
            0.0,
        )
    for t in range(periods):
        add_constraint(
            [layout.h2_power_start + t, layout.electrolyzer_capacity],
            [1.0, -1.0],
            0.0,
        )
        add_constraint(
            [layout.fuel_cell_start + t, layout.fuel_cell_capacity],
            [1.0, -1.0],
            0.0,
        )
        add_constraint(
            [layout.h2_sale_start + t, layout.hydrogen_export_capacity],
            [1.0, -dt],
            0.0,
        )
        add_constraint(
            [layout.spot_start + t, layout.compute_capacity],
            [1.0, -dt],
            0.0,
        )
        add_constraint(
            [layout.spot_start + t, layout.compute_capacity],
            [1.0, -fiber_availability[t] * dt],
            0.0,
        )
        for sign, limit in ((1.0, compute_ramp_up), (-1.0, compute_ramp_down)):
            indices = [layout.spot_start + t]
            coefficients = [sign / dt]
            if t > 0:
                indices.append(layout.spot_start + t - 1)
                coefficients.append(-sign / dt)
            add_constraint(indices, coefficients, limit * dt)

    inequality = coo_matrix((idata, (ir, ic)), shape=(row, layout.count)).tocsr()
    solution = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.asarray(inequality_rhs),
        A_eq=equality,
        b_eq=np.asarray(equality_rhs),
        bounds=bounds,
        method="highs",
        options={
            "presolve": True,
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
        },
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(
            f"S6 investment planning failed: status={solution.status}; {solution.message}"
        )

    x = solution.x.copy()
    if not np.isfinite(x).all():
        raise RuntimeError("S6 solver returned non-finite decision variables")
    numerical_zero = 1e-7
    if float(x.min()) < -numerical_zero:
        raise RuntimeError("S6 solver violated a nonnegative variable bound")
    x[(x < 0.0) & (x >= -numerical_zero)] = 0.0
    if segments:
        segment_dispatch = x[: periods * segments].reshape(periods, segments)
        export = segment_dispatch.sum(axis=1)
        piece_land = segment_dispatch @ slopes
    else:
        export = np.zeros(periods)
        piece_land = np.zeros(periods)
    charge = x[layout.charge_start : layout.charge_start + periods]
    discharge = x[layout.discharge_start : layout.discharge_start + periods]
    curtail = x[layout.curtail_start : layout.curtail_start + periods]
    unmet = x[layout.unmet_start : layout.unmet_start + periods]
    h2_power = x[layout.h2_power_start : layout.h2_power_start + periods]
    h2_sale = x[layout.h2_sale_start : layout.h2_sale_start + periods]
    fuel_cell = x[layout.fuel_cell_start : layout.fuel_cell_start + periods]
    spot = x[layout.spot_start : layout.spot_start + periods]
    energy = x[layout.battery_energy_start : layout.battery_energy_start + periods + 1]
    inventory = x[layout.h2_inventory_start : layout.h2_inventory_start + periods + 1]
    capacities = {
        "battery_power_mw": float(x[layout.battery_power_capacity]),
        "battery_energy_mwh": float(x[layout.battery_energy_capacity]),
        "electrolyzer_power_mw": float(x[layout.electrolyzer_capacity]),
        "hydrogen_storage_kg": float(x[layout.h2_storage_capacity]),
        "fuel_cell_power_mw": float(x[layout.fuel_cell_capacity]),
        "compute_it_capacity_mw": float(x[layout.compute_capacity]),
        "subsea_fiber_service_capacity_mw_it": float(x[layout.compute_capacity]),
        "hydrogen_export_capacity_kg_per_h": float(
            x[layout.hydrogen_export_capacity]
        ),
    }

    flow = calculate_transmission_flow(
        export,
        base_config.tx_capacity_mw,
        scenario.offshore_distance_km,
        loss_spec,
    )
    dc_power = pue * spot / dt
    h2_production = h2_power * h2_kg_per_mwh * dt
    h2_fuel_cell_use = fuel_cell * fuel_cell_kg_per_mwh * dt
    battery_residual = (
        energy[1:]
        - (1.0 - battery_self_discharge) * energy[:-1]
        - one_way_efficiency * charge * dt
        + discharge * dt / one_way_efficiency
    )
    h2_residual = (
        inventory[1:]
        - (1.0 - h2_storage_loss) * inventory[:-1]
        - h2_production
        + h2_sale
        + h2_fuel_cell_use
    )
    offshore_residual = (
        renewable
        + discharge
        + fuel_cell
        + unmet
        - critical
        - export
        - charge
        - h2_power
        - dc_power
        - curtail
    )
    land_residual = export - flow.land_mw - flow.total_loss_mw
    linear_error = flow.land_mw - piece_land
    tolerance = params.value("power_balance_tolerance")
    if max(_safe_max(offshore_residual), _safe_max(land_residual)) >= tolerance:
        raise RuntimeError("S6 power-balance audit failed")
    if _safe_max(battery_residual) >= params.value("battery_state_tolerance"):
        raise RuntimeError("S6 battery-state audit failed")
    if _safe_max(h2_residual) >= params.value("hydrogen_state_tolerance"):
        raise RuntimeError("S6 hydrogen-state audit failed")
    if float(linear_error.min()) < -1e-7:
        raise RuntimeError("S6 transmission linearisation overstates land delivery")
    if (np.minimum(charge, discharge) > params.value("battery_simultaneous_tolerance")).any():
        raise RuntimeError("S6 simultaneous battery charge and discharge")

    electricity_revenue = price * flow.land_mw * dt
    tx_variable = tx_cost * export * dt
    curtail_cost = params.value("curtailment_penalty") * curtail * dt
    critical_cost = params.value("unserved_critical_load_penalty") * unmet * dt
    battery_cost = params.value("battery_degradation_cost") * (charge + discharge) * dt
    h2_revenue = h2_sale * h2_sale_price
    h2_transport = h2_sale * h2_transport_cost
    h2_conversion = h2_production * (h2_variable_cost + h2_water_cost)
    stack_replacement = h2_power * dt * unit_cost[
        "electrolyzer_stack_replacement_cny_per_mwh"
    ]
    fuel_cell_cost = fuel_cell * fuel_cell_variable_cost * dt
    compute_revenue = spot * spot_price
    compute_variable = spot * compute_variable_cost
    operating_margin = (
        electricity_revenue
        + h2_revenue
        + compute_revenue
        - tx_variable
        - curtail_cost
        - critical_cost
        - battery_cost
        - h2_transport
        - h2_conversion
        - stack_replacement
        - fuel_cell_cost
        - compute_variable
    )

    capacity_cost_breakdown = {
        "battery_power_annual_cost_cny": capacities["battery_power_mw"]
        * 1_000.0
        * unit_cost["battery_power_cny_per_kw_year"],
        "battery_energy_annual_cost_cny": capacities["battery_energy_mwh"]
        * 1_000.0
        * unit_cost["battery_energy_cny_per_kwh_year"],
        "electrolyzer_annual_cost_cny": capacities["electrolyzer_power_mw"]
        * 1_000.0
        * unit_cost["electrolyzer_cny_per_kw_year"],
        "hydrogen_storage_annual_cost_cny": capacities["hydrogen_storage_kg"]
        * unit_cost["hydrogen_storage_cny_per_kg_year"],
        "fuel_cell_annual_cost_cny": capacities["fuel_cell_power_mw"]
        * 1_000.0
        * unit_cost["fuel_cell_cny_per_kw_year"],
        "compute_fibre_annual_cost_cny": capacities["compute_it_capacity_mw"]
        * 1_000.0
        * unit_cost["compute_fibre_cny_per_kw_it_year"],
        "hydrogen_pipeline_annual_cost_cny": capacities[
            "hydrogen_export_capacity_kg_per_h"
        ]
        * scenario.offshore_distance_km
        * unit_cost["hydrogen_pipeline_cny_per_kg_h_km_year"],
    }
    capex_breakdown = {
        "battery_power_capex_cny": capacities["battery_power_mw"]
        * 1_000.0
        * costs.battery_power_capex_cny_per_kw,
        "battery_energy_capex_cny": capacities["battery_energy_mwh"]
        * 1_000.0
        * costs.battery_energy_capex_cny_per_kwh,
        "electrolyzer_capex_cny": capacities["electrolyzer_power_mw"]
        * 1_000.0
        * costs.electrolyzer_capex_cny_per_kw,
        "hydrogen_storage_capex_cny": capacities["hydrogen_storage_kg"]
        * costs.hydrogen_storage_capex_cny_per_kg,
        "fuel_cell_capex_cny": capacities["fuel_cell_power_mw"]
        * 1_000.0
        * costs.fuel_cell_capex_cny_per_kw,
        "compute_fibre_capex_cny": capacities["compute_it_capacity_mw"]
        * 1_000.0
        * costs.compute_fibre_capex_cny_per_kw_it,
        "hydrogen_pipeline_capex_cny": capacities[
            "hydrogen_export_capacity_kg_per_h"
        ]
        * scenario.offshore_distance_km
        * costs.hydrogen_pipeline_capex_cny_per_kg_h_km,
    }
    fixed_om = (
        (capex_breakdown["battery_power_capex_cny"] + capex_breakdown["battery_energy_capex_cny"])
        * costs.battery_fixed_om_fraction
        + capex_breakdown["electrolyzer_capex_cny"]
        * costs.electrolyzer_fixed_om_fraction
        + capex_breakdown["hydrogen_storage_capex_cny"]
        * costs.hydrogen_storage_fixed_om_fraction
        + capex_breakdown["fuel_cell_capex_cny"] * costs.fuel_cell_fixed_om_fraction
        + capex_breakdown["compute_fibre_capex_cny"]
        * costs.compute_fibre_fixed_om_fraction
        + capex_breakdown["hydrogen_pipeline_capex_cny"]
        * costs.hydrogen_pipeline_fixed_om_fraction
    )
    total_fixed_cost = float(sum(capacity_cost_breakdown.values()))
    total_capex = float(sum(capex_breakdown.values()))
    annual_operating_margin = float(operating_margin.sum() * annualization_factor)
    export_shadow = -np.asarray(solution.ineqlin.marginals)[export_rows]
    renewable_total = float((renewable * dt).sum())
    curtail_total = float((curtail * dt).sum())

    hourly = pd.DataFrame(
        {
            "timestamp": ts["timestamp"].to_numpy(),
            "electricity_price_cny_per_mwh": price,
            "wind_available_mw": wind,
            "pv_available_mw": pv,
            "wave_generation_mw": wave,
            "renewable_available_mw": renewable,
            "critical_load_mw": critical,
            "grid_absorption_factor": grid_absorption,
            "grid_absorption_limit_mw": grid_absorption_limit,
            "tx_available_capacity_mw": tx_available,
            "export_send_mw": export,
            "export_land_mw": flow.land_mw,
            "export_capacity_shadow_cny_per_mw_h_annualized": export_shadow,
            "battery_charge_mw": charge,
            "battery_discharge_mw": discharge,
            "battery_energy_end_mwh": energy[1:],
            "electrolyzer_power_mw": h2_power,
            "hydrogen_production_kg": h2_production,
            "hydrogen_sale_kg": h2_sale,
            "hydrogen_inventory_end_kg": inventory[1:],
            "fuel_cell_power_mw": fuel_cell,
            "national_compute_pool_mwh_it": spot_pool * dt,
            "national_compute_flexible_fraction": spot_flexible_fraction,
            "national_compute_demand_mwh_it": spot_demand * dt,
            "spot_compute_completed_mwh_it": spot,
            "dc_facility_power_mw": dc_power,
            "curtailment_mw": curtail,
            "unmet_critical_load_mw": unmet,
            "operating_margin_cny": operating_margin,
            "offshore_balance_residual_mw": offshore_residual,
            "land_balance_residual_mw": land_residual,
        }
    )
    kpis: dict[str, Any] = {
        "phase": "S6",
        "cost_case_id": costs.cost_case_id,
        "scenario_id": scenario.scenario_id,
        "simulation_hours": periods * dt,
        "annualization_factor": annualization_factor,
        **capacities,
        "renewable_generation_mwh": renewable_total,
        "wind_generation_mwh": float((wind * dt).sum()),
        "pv_generation_mwh": float((pv * dt).sum()),
        "wave_generation_mwh": float((wave * dt).sum()),
        "curtailment_mwh": curtail_total,
        "curtailment_rate": curtail_total / renewable_total if renewable_total else 0.0,
        "renewable_utilization_rate": (
            (renewable_total - curtail_total) / renewable_total if renewable_total else 0.0
        ),
        "export_land_mwh": float((flow.land_mw * dt).sum()),
        "battery_discharge_mwh": float((discharge * dt).sum()),
        "electrolyzer_energy_mwh": float((h2_power * dt).sum()),
        "hydrogen_production_kg": float(h2_production.sum()),
        "hydrogen_sales_kg": float(h2_sale.sum()),
        "fuel_cell_generation_mwh": float((fuel_cell * dt).sum()),
        "spot_compute_service_mwh_it": float(spot.sum()),
        "national_compute_pool_mwh_it": float((spot_pool * dt).sum()),
        "spot_compute_demand_mwh_it": float((spot_demand * dt).sum()),
        "dc_facility_energy_mwh": float((dc_power * dt).sum()),
        "eens_mwh": float((unmet * dt).sum()),
        "constrained_grid_hours": int(
            (tx_available < tx_physical_available - 1e-9).sum()
        ),
        "annualized_electricity_revenue_cny": float(
            electricity_revenue.sum() * annualization_factor
        ),
        "annualized_hydrogen_revenue_cny": float(
            h2_revenue.sum() * annualization_factor
        ),
        "annualized_compute_revenue_cny": float(
            compute_revenue.sum() * annualization_factor
        ),
        "annualized_variable_cost_cny": float(
            (
                tx_variable
                + curtail_cost
                + critical_cost
                + battery_cost
                + h2_transport
                + h2_conversion
                + stack_replacement
                + fuel_cell_cost
                + compute_variable
            ).sum()
            * annualization_factor
        ),
        "annualized_operating_margin_cny": annual_operating_margin,
        **capex_breakdown,
        "flexible_asset_capex_cny": total_capex,
        "annual_fixed_om_cny": float(fixed_om),
        "annualized_capital_recovery_cny": total_fixed_cost - float(fixed_om),
        **capacity_cost_breakdown,
        "annualized_flexible_fixed_cost_cny": total_fixed_cost,
        "net_annual_value_cny": annual_operating_margin - total_fixed_cost,
        "max_offshore_balance_residual_mw": _safe_max(offshore_residual),
        "max_land_balance_residual_mw": _safe_max(land_residual),
        "max_battery_state_residual_mwh": _safe_max(battery_residual),
        "max_hydrogen_state_residual_kg": _safe_max(h2_residual),
        "lp_objective_cny": float(solution.fun),
        "lp_iterations": int(solution.nit),
    }
    payload = {
        "phase": "S6",
        "base_config": base_config.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "costs": asdict(costs),
        "limits": asdict(limits),
        "parameters": [item.model_dump(mode="json") for item in params.items],
    }
    metadata = {
        "phase": "S6",
        "model_version": __version__,
        "configuration_hash": configuration_hash(payload),
        "rows": periods,
        "time_step_hours": dt,
        "solver": "scipy.optimize.linprog.highs",
        "solver_status": int(solution.status),
        "solver_message": solution.message,
        "transmission_segments": segments,
        "planning_boundary": "incremental_flexible_assets_with_fixed_generation_and_export",
        "cost_case": asdict(costs),
        "planning_limits": asdict(limits),
        "annualized_unit_costs": unit_cost,
    }
    return S6PlanningResult(hourly, capacities, kpis, metadata)


def optimal_system_configuration(
    base_config: SystemConfiguration,
    result: S6PlanningResult,
    *,
    config_id: str = "S6_optimal_fixed_replay",
    numerical_zero: float = 1e-6,
) -> SystemConfiguration:
    """Convert an S6 solution into a fixed S5 configuration for replay."""
    values = {
        key: (0.0 if abs(value) < numerical_zero else float(value))
        for key, value in result.capacities.items()
    }
    battery_is_zero = (
        values["battery_power_mw"] == 0.0 or values["battery_energy_mwh"] == 0.0
    )
    if battery_is_zero:
        values["battery_power_mw"] = 0.0
        values["battery_energy_mwh"] = 0.0
    compute_is_zero = values["compute_it_capacity_mw"] == 0.0
    if compute_is_zero:
        values["subsea_fiber_service_capacity_mw_it"] = 0.0
    return base_config.model_copy(
        update={
            "config_id": config_id,
            **values,
            "initial_battery_soc_fraction": (
                base_config.initial_battery_soc_fraction if not battery_is_zero else 0.0
            ),
            "initial_hydrogen_inventory_fraction": (
                base_config.initial_hydrogen_inventory_fraction
                if values["hydrogen_storage_kg"] > 0.0
                else 0.0
            ),
            "battery_reserve_power_mw": 0.0,
            "battery_reserve_duration_h": 0.0,
        }
    )
