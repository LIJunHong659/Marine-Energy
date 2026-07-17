"""Phase 6 / S5 scarcity-aware electricity, storage, hydrogen and compute dispatch.

The S5 formulation preserves one auditable offshore power balance while adding
three mechanisms that were absent from S4: a time-varying mainland absorption
limit, an optional nationwide spot-compute pool, and hydrogen-to-power return.
Every flexible asset can be switched off, which supports fair factorial value
comparisons under identical physical and commercial conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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
class S5DispatchResult:
    """S5 hourly ledger, KPI summary and solver provenance."""

    hourly: pd.DataFrame
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
    rigid_done_start: int
    rigid_unserved_start: int
    flex_done_start: int
    spot_done_start: int
    battery_energy_start: int
    h2_inventory_start: int
    flex_queue_start: int
    count: int


def _layout(periods: int, segments: int) -> _Layout:
    segment_count = periods * segments
    charge_start = segment_count
    discharge_start = charge_start + periods
    curtail_start = discharge_start + periods
    unmet_start = curtail_start + periods
    h2_power_start = unmet_start + periods
    h2_sale_start = h2_power_start + periods
    fuel_cell_start = h2_sale_start + periods
    rigid_done_start = fuel_cell_start + periods
    rigid_unserved_start = rigid_done_start + periods
    flex_done_start = rigid_unserved_start + periods
    spot_done_start = flex_done_start + periods
    battery_energy_start = spot_done_start + periods
    h2_inventory_start = battery_energy_start + periods + 1
    flex_queue_start = h2_inventory_start + periods + 1
    return _Layout(
        periods,
        segments,
        charge_start,
        discharge_start,
        curtail_start,
        unmet_start,
        h2_power_start,
        h2_sale_start,
        fuel_cell_start,
        rigid_done_start,
        rigid_unserved_start,
        flex_done_start,
        spot_done_start,
        battery_energy_start,
        h2_inventory_start,
        flex_queue_start,
        flex_queue_start + periods + 1,
    )


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
    default: float | np.ndarray,
    *,
    lower: float = 0.0,
    upper: float | None = None,
) -> np.ndarray:
    values = (
        frame[name].to_numpy(dtype=float)
        if name in frame.columns
        else np.broadcast_to(default, len(frame)).astype(float, copy=True)
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if (values < lower).any() or (upper is not None and (values > upper).any()):
        interval = f"[{lower}, {upper}]" if upper is not None else f"[{lower}, infinity)"
        raise ValueError(f"{name} must lie in {interval}")
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


def _safe_max(values: np.ndarray) -> float:
    return float(np.abs(values).max()) if len(values) else 0.0


def run_s5_dispatch(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S5DispatchResult:
    """Solve S5 with optional short-, medium- and long-duration flexibility."""
    validate_timeseries(timeseries).raise_if_invalid()
    validate_parameters(params).raise_if_invalid()
    if config.export_policy != "economic":
        raise ValueError("S5 incremental transmission segments require economic export")
    dt = params.value("time_step")
    if dt != 1.0:
        raise ValueError("S5 supports a 1 h time step only")

    ts = apply_scenario(timeseries, scenario)
    periods = len(ts)
    wind = (
        config.wind_capacity_mw
        * ts["wind_cf"].to_numpy(dtype=float)
        * ts["wind_availability"].to_numpy(dtype=float)
    )
    pv = config.pv_capacity_mw * ts["pv_cf"].to_numpy(dtype=float)
    wave = _optional_series(ts, "wave_generation_mw", 0.0)
    renewable = wind + pv + wave
    critical = ts["critical_load"].to_numpy(dtype=float)
    price = ts["electricity_price"].to_numpy(dtype=float)
    h2_demand = ts["hydrogen_demand"].to_numpy(dtype=float)

    compute_enabled = config.compute_it_capacity_mw > 0.0
    if compute_enabled and config.subsea_fiber_service_capacity_mw_it <= 0.0:
        raise ValueError("positive compute capacity requires positive fibre service capacity")
    rigid_arrival = (
        ts["rigid_compute_arrival"].to_numpy(dtype=float) if compute_enabled else np.zeros(periods)
    )
    flex_arrival = (
        ts["flex_compute_arrival"].to_numpy(dtype=float) if compute_enabled else np.zeros(periods)
    )
    grid_absorption = _optional_series(ts, "grid_absorption_factor", 1.0, lower=0.0, upper=1.0)
    national_compute_pool = _optional_series(ts, "national_compute_demand_mw_it", 0.0)
    national_compute_flexible_fraction = _optional_series(
        ts, "national_compute_flexible_fraction", 1.0, lower=0.0, upper=1.0
    )
    national_compute_demand = national_compute_pool * national_compute_flexible_fraction
    spot_price = (
        _optional_series(
            ts,
            "national_compute_price_cny_per_mwh_it",
            params.value("compute_spot_service_price"),
        )
        * scenario.compute_price_multiplier
    )
    tx_physical_available = config.tx_capacity_mw * ts["tx_availability"].to_numpy(dtype=float)
    proportional_tx_available = tx_physical_available * grid_absorption
    if "grid_absorption_limit_mw" in ts.columns:
        grid_absorption_limit = _optional_series(ts, "grid_absorption_limit_mw", 0.0)
    else:
        grid_absorption_limit = np.full(periods, 1.0e12)
    tx_available = np.minimum(proportional_tx_available, grid_absorption_limit)
    fiber_availability = _optional_series(ts, "fiber_availability", 1.0, lower=0.0, upper=1.0)
    fiber_available = config.subsea_fiber_service_capacity_mw_it * fiber_availability

    battery_power = config.battery_power_mw
    battery_energy = config.battery_energy_mwh
    one_way_efficiency = sqrt(params.value("battery_round_trip_efficiency"))
    battery_self_discharge = params.value("battery_self_discharge_per_hour")
    battery_minimum = params.value("battery_soc_min") * battery_energy
    battery_maximum = params.value("battery_soc_max") * battery_energy
    battery_initial = config.initial_battery_soc_fraction * battery_energy
    reserve_energy = (
        config.battery_reserve_power_mw * config.battery_reserve_duration_h / one_way_efficiency
        if battery_power > 0.0
        else 0.0
    )
    battery_scheduled_minimum = battery_minimum + reserve_energy
    if battery_power > 0.0 and not (
        battery_scheduled_minimum <= battery_initial <= battery_maximum
    ):
        raise ValueError("initial battery energy must satisfy SOC and reserve bounds")

    h2_sec = params.value("hydrogen_sec_system")
    h2_kg_per_mwh = 1_000.0 / h2_sec
    h2_storage_loss = params.value("hydrogen_storage_loss_per_hour")
    h2_initial = config.initial_hydrogen_inventory_fraction * config.hydrogen_storage_kg
    h2_sale_price = params.value("hydrogen_sale_price") * scenario.hydrogen_price_multiplier
    h2_transport_cost = params.value("hydrogen_transport_cost")
    h2_variable_cost = params.value("electrolyzer_variable_cost")
    h2_water_cost = params.value("hydrogen_water_consumption") * params.value(
        "desalinated_water_cost"
    )
    hydrogen_lhv_kwh_per_kg = params.value("hydrogen_lhv_kwh_per_kg")
    fuel_cell_efficiency = params.value("fuel_cell_efficiency_lhv")
    fuel_cell_kg_per_mwh = 1_000.0 / (hydrogen_lhv_kwh_per_kg * fuel_cell_efficiency)
    fuel_cell_variable_cost = params.value("fuel_cell_variable_cost")

    pue = _pue(params, scenario.pue_case)
    compute_variable_cost = params.value("compute_variable_cost")
    rigid_price = params.value("compute_rigid_service_price") * scenario.compute_price_multiplier
    flex_price = params.value("compute_flex_service_price") * scenario.compute_price_multiplier
    rigid_penalty = params.value("compute_rigid_sla_penalty")
    flex_queue_capacity = params.value("compute_flex_queue_capacity") if compute_enabled else 0.0
    max_delay_float = params.value("compute_flex_max_delay")
    if not max_delay_float.is_integer():
        raise ValueError("compute_flex_max_delay must be an integer number of hours")
    flex_max_delay = int(max_delay_float)
    compute_ramp_up = params.value("compute_ramp_up")
    compute_ramp_down = params.value("compute_ramp_down")

    loss_spec = _loss_spec(params, config.tx_technology)
    segments = int(round(params.value("battery_linearization_segments")))
    if config.tx_capacity_mw > 0.0:
        piecewise = build_piecewise_transmission(
            config.tx_capacity_mw, scenario.offshore_distance_km, loss_spec, segments
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
            objective[t * segments + segment] = (tx_cost - price[t] * slopes[segment]) * dt
    degradation_cost = params.value("battery_degradation_cost")
    objective[layout.charge_start : layout.charge_start + periods] = degradation_cost * dt
    objective[layout.discharge_start : layout.discharge_start + periods] = degradation_cost * dt
    objective[layout.curtail_start : layout.curtail_start + periods] = (
        params.value("curtailment_penalty") * dt
    )
    objective[layout.unmet_start : layout.unmet_start + periods] = (
        params.value("unserved_critical_load_penalty") * dt
    )
    objective[layout.h2_power_start : layout.h2_power_start + periods] = (
        (h2_variable_cost + h2_water_cost) * h2_kg_per_mwh * dt
    )
    objective[layout.h2_sale_start : layout.h2_sale_start + periods] = -(
        h2_sale_price - h2_transport_cost
    )
    objective[layout.fuel_cell_start : layout.fuel_cell_start + periods] = (
        fuel_cell_variable_cost * dt
    )
    objective[layout.rigid_done_start : layout.rigid_done_start + periods] = (
        compute_variable_cost - rigid_price
    )
    objective[layout.rigid_unserved_start : layout.rigid_unserved_start + periods] = rigid_penalty
    objective[layout.flex_done_start : layout.flex_done_start + periods] = (
        compute_variable_cost - flex_price
    )
    objective[layout.spot_done_start : layout.spot_done_start + periods] = (
        compute_variable_cost - spot_price
    )

    bounds: list[tuple[float, float]] = []
    for _ in range(periods):
        bounds.extend((0.0, float(width)) for width in widths)
    bounds.extend((0.0, battery_power) for _ in range(periods))
    bounds.extend((0.0, battery_power) for _ in range(periods))
    bounds.extend((0.0, float(value)) for value in renewable)
    bounds.extend((0.0, float(value)) for value in critical)
    bounds.extend((0.0, config.electrolyzer_power_mw) for _ in range(periods))
    bounds.extend((0.0, float(value * dt)) for value in h2_demand)
    bounds.extend((0.0, config.fuel_cell_power_mw) for _ in range(periods))
    bounds.extend((0.0, float(value * dt)) for value in rigid_arrival)
    bounds.extend((0.0, float(value * dt)) for value in rigid_arrival)
    bounds.extend((0.0, config.compute_it_capacity_mw * dt) for _ in range(periods))
    bounds.extend((0.0, float(value * dt)) for value in national_compute_demand)
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((battery_initial, battery_initial))
        else:
            bounds.append((battery_scheduled_minimum, battery_maximum))
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((h2_initial, h2_initial))
        else:
            bounds.append((0.0, config.hydrogen_storage_kg))
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((0.0, 0.0))
        else:
            hour = state - 1
            if flex_max_delay == 0:
                due_bound = 0.0
            elif hour >= flex_max_delay:
                due_bound = float(flex_arrival[hour - flex_max_delay + 1 : hour + 1].sum() * dt)
            else:
                due_bound = flex_queue_capacity
            bounds.append((0.0, min(flex_queue_capacity, due_bound)))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(5 * periods)
    for t in range(periods):
        for segment in range(segments):
            rows.append(t)
            cols.append(t * segments + segment)
            data.append(1.0)
        rows.extend([t] * 9)
        cols.extend(
            [
                layout.charge_start + t,
                layout.curtail_start + t,
                layout.h2_power_start + t,
                layout.rigid_done_start + t,
                layout.flex_done_start + t,
                layout.spot_done_start + t,
                layout.discharge_start + t,
                layout.fuel_cell_start + t,
                layout.unmet_start + t,
            ]
        )
        data.extend([1.0, 1.0, 1.0, pue / dt, pue / dt, pue / dt, -1.0, -1.0, -1.0])
        rhs[t] = renewable[t] - critical[t]

        b_row = periods + t
        rows.extend([b_row] * 4)
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

        h_row = 2 * periods + t
        rows.extend([h_row] * 5)
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

        r_row = 3 * periods + t
        rows.extend([r_row] * 2)
        cols.extend([layout.rigid_done_start + t, layout.rigid_unserved_start + t])
        data.extend([1.0, 1.0])
        rhs[r_row] = rigid_arrival[t] * dt

        q_row = 4 * periods + t
        rows.extend([q_row] * 3)
        cols.extend(
            [
                layout.flex_queue_start + t + 1,
                layout.flex_queue_start + t,
                layout.flex_done_start + t,
            ]
        )
        data.extend([1.0, -1.0, 1.0])
        rhs[q_row] = flex_arrival[t] * dt
    equality = coo_matrix((data, (rows, cols)), shape=(5 * periods, layout.count)).tocsr()

    ir: list[int] = []
    ic: list[int] = []
    idata: list[float] = []
    irhs: list[float] = []
    export_constraint_rows: list[int] = []
    row = 0
    for t in range(periods):
        export_constraint_rows.append(row)
        for segment in range(segments):
            ir.append(row)
            ic.append(t * segments + segment)
            idata.append(1.0)
        irhs.append(float(tx_available[t]))
        row += 1

    headroom = max(battery_power - config.battery_reserve_power_mw, 0.0)
    for t in range(periods):
        if config.battery_reserve_power_mw > 0.0:
            ir.extend([row, row])
            ic.extend([layout.discharge_start + t, layout.charge_start + t])
            idata.extend([1.0, -1.0])
            irhs.append(headroom)
            row += 1
        ir.append(row)
        ic.append(layout.h2_sale_start + t)
        idata.append(1.0)
        irhs.append(config.hydrogen_export_capacity_kg_per_h * dt)
        row += 1
        for cap in (config.compute_it_capacity_mw, float(fiber_available[t])):
            ir.extend([row, row, row])
            ic.extend(
                [
                    layout.rigid_done_start + t,
                    layout.flex_done_start + t,
                    layout.spot_done_start + t,
                ]
            )
            idata.extend([1.0, 1.0, 1.0])
            irhs.append(cap * dt)
            row += 1
        for sign, limit in ((1.0, compute_ramp_up), (-1.0, compute_ramp_down)):
            ir.extend([row, row, row])
            ic.extend(
                [
                    layout.rigid_done_start + t,
                    layout.flex_done_start + t,
                    layout.spot_done_start + t,
                ]
            )
            idata.extend([sign / dt, sign / dt, sign / dt])
            if t > 0:
                ir.extend([row, row, row])
                ic.extend(
                    [
                        layout.rigid_done_start + t - 1,
                        layout.flex_done_start + t - 1,
                        layout.spot_done_start + t - 1,
                    ]
                )
                idata.extend([-sign / dt, -sign / dt, -sign / dt])
            irhs.append(limit * dt)
            row += 1
    inequality = coo_matrix((idata, (ir, ic)), shape=(row, layout.count)).tocsr()
    solution = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.asarray(irhs),
        A_eq=equality,
        b_eq=rhs,
        bounds=bounds,
        method="highs",
        options={
            "presolve": True,
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
        },
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"S5 optimisation failed: status={solution.status}; {solution.message}")

    x = solution.x
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
    rigid_done = x[layout.rigid_done_start : layout.rigid_done_start + periods]
    rigid_unserved = x[layout.rigid_unserved_start : layout.rigid_unserved_start + periods]
    flex_done = x[layout.flex_done_start : layout.flex_done_start + periods]
    spot_done = x[layout.spot_done_start : layout.spot_done_start + periods]
    energy = x[layout.battery_energy_start : layout.battery_energy_start + periods + 1]
    inventory = x[layout.h2_inventory_start : layout.h2_inventory_start + periods + 1]
    queue = x[layout.flex_queue_start : layout.flex_queue_start + periods + 1]

    h2_production = h2_power * h2_kg_per_mwh * dt
    h2_fuel_cell_use = fuel_cell * fuel_cell_kg_per_mwh * dt
    it_energy = rigid_done + flex_done + spot_done
    it_power = it_energy / dt
    dc_power = pue * it_power
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
    queue_residual = queue[1:] - queue[:-1] + flex_done - flex_arrival * dt
    rigid_residual = rigid_done + rigid_unserved - rigid_arrival * dt
    flow = calculate_transmission_flow(
        export, config.tx_capacity_mw, scenario.offshore_distance_km, loss_spec
    )
    linear_error = flow.land_mw - piece_land
    critical_served = critical - unmet
    offshore = (
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
    land = export - flow.land_mw - flow.total_loss_mw
    power_tolerance = params.value("power_balance_tolerance")
    if max(_safe_max(offshore), _safe_max(land)) >= power_tolerance:
        raise RuntimeError("S5 power balance residual exceeds tolerance")
    if _safe_max(battery_residual) >= params.value("battery_state_tolerance"):
        raise RuntimeError("S5 battery state audit failed")
    if abs(float(energy[-1] - energy[0])) >= params.value("battery_state_tolerance"):
        raise RuntimeError("S5 battery terminal state audit failed")
    if _safe_max(h2_residual) >= params.value("hydrogen_state_tolerance"):
        raise RuntimeError("S5 hydrogen state audit failed")
    if abs(float(inventory[-1] - inventory[0])) >= params.value("hydrogen_state_tolerance"):
        raise RuntimeError("S5 hydrogen terminal state audit failed")
    if _safe_max(queue_residual) >= params.value("compute_state_tolerance"):
        raise RuntimeError("S5 compute queue audit failed")
    if _safe_max(rigid_residual) >= params.value("compute_state_tolerance"):
        raise RuntimeError("S5 rigid compute audit failed")
    if (np.minimum(charge, discharge) > params.value("battery_simultaneous_tolerance")).any():
        raise RuntimeError("S5 simultaneous battery charge and discharge")
    if (export > tx_available + 1e-7).any():
        raise RuntimeError("S5 mainland absorption constraint is exceeded")
    if (it_power > config.compute_it_capacity_mw + 1e-7).any() or (
        it_power > fiber_available + 1e-7
    ).any():
        raise RuntimeError("S5 compute or fibre capacity is exceeded")
    if float(linear_error.min()) < -1e-7:
        raise RuntimeError("S5 transmission linearisation overstates land delivery")
    completed_cumulative = np.cumsum(flex_done)
    arrivals_cumulative = np.cumsum(flex_arrival * dt)
    deadline_slack = (
        completed_cumulative[flex_max_delay:] - arrivals_cumulative[: periods - flex_max_delay]
        if flex_max_delay < periods
        else np.empty(0)
    )
    if len(deadline_slack) and float(deadline_slack.min()) < -params.value(
        "compute_state_tolerance"
    ):
        raise RuntimeError("S5 flexible compute deadline is violated")

    battery_standing_loss = battery_self_discharge * energy[:-1]
    battery_conversion_loss = (1.0 - one_way_efficiency) * charge * dt + (
        1.0 / one_way_efficiency - 1.0
    ) * discharge * dt
    h2_storage_loss_mass = h2_storage_loss * inventory[:-1]
    electricity_revenue = price * flow.land_mw * dt
    tx_variable = tx_cost * export * dt
    curtail_cost = params.value("curtailment_penalty") * curtail * dt
    critical_cost = params.value("unserved_critical_load_penalty") * unmet * dt
    battery_cost = degradation_cost * (charge + discharge) * dt
    h2_revenue = h2_sale * h2_sale_price
    h2_transport = h2_sale * h2_transport_cost
    h2_conversion = h2_production * h2_variable_cost
    h2_water = h2_production * h2_water_cost
    fuel_cell_cost = fuel_cell * fuel_cell_variable_cost * dt
    compute_revenue = rigid_done * rigid_price + flex_done * flex_price + spot_done * spot_price
    compute_variable = it_energy * compute_variable_cost
    compute_sla = rigid_unserved * rigid_penalty
    margin = (
        electricity_revenue
        + h2_revenue
        + compute_revenue
        - tx_variable
        - curtail_cost
        - critical_cost
        - battery_cost
        - h2_transport
        - h2_conversion
        - h2_water
        - fuel_cell_cost
        - compute_variable
        - compute_sla
    )
    network_limited = (tx_available < tx_physical_available - 1e-7) | (
        export >= tx_available - 1e-7
    )
    network_curtail = np.where(network_limited, curtail, 0.0)
    economic_curtail = curtail - network_curtail
    export_shadow = -np.asarray(solution.ineqlin.marginals)[export_constraint_rows]

    hourly = pd.DataFrame(
        {
            "timestamp": ts["timestamp"].to_numpy(),
            "scenario_id": ts["scenario_id"].to_numpy(),
            "electricity_price_cny_per_mwh": price,
            "wind_available_mw": wind,
            "pv_available_mw": pv,
            "wave_generation_mw": wave,
            "renewable_available_mw": renewable,
            "critical_load_mw": critical,
            "critical_load_served_mw": critical_served,
            "unmet_critical_load_mw": unmet,
            "tx_availability": ts["tx_availability"].to_numpy(dtype=float),
            "grid_absorption_factor": grid_absorption,
            "grid_absorption_limit_mw": grid_absorption_limit,
            "tx_physical_available_capacity_mw": tx_physical_available,
            "tx_available_capacity_mw": tx_available,
            "export_capacity_shadow_cny_per_mw_h": export_shadow,
            "export_send_mw": export,
            "tx_terminal_loss_mw": flow.terminal_loss_mw,
            "tx_cable_loss_mw": flow.cable_loss_mw,
            "tx_total_loss_mw": flow.total_loss_mw,
            "tx_piecewise_land_mw": piece_land,
            "tx_linearization_error_mw": linear_error,
            "export_land_mw": flow.land_mw,
            "battery_charge_mw": charge,
            "battery_discharge_mw": discharge,
            "battery_energy_start_mwh": energy[:-1],
            "battery_energy_end_mwh": energy[1:],
            "battery_total_loss_mwh": battery_standing_loss + battery_conversion_loss,
            "hydrogen_demand_kg": h2_demand * dt,
            "electrolyzer_power_mw": h2_power,
            "hydrogen_production_kg": h2_production,
            "hydrogen_sale_kg": h2_sale,
            "fuel_cell_power_mw": fuel_cell,
            "fuel_cell_hydrogen_use_kg": h2_fuel_cell_use,
            "hydrogen_inventory_start_kg": inventory[:-1],
            "hydrogen_inventory_end_kg": inventory[1:],
            "hydrogen_storage_loss_kg": h2_storage_loss_mass,
            "hydrogen_state_residual_kg": h2_residual,
            "rigid_compute_arrival_mwh_it": rigid_arrival * dt,
            "flex_compute_arrival_mwh_it": flex_arrival * dt,
            "national_compute_pool_mwh_it": national_compute_pool * dt,
            "national_compute_flexible_fraction": national_compute_flexible_fraction,
            "national_compute_demand_mwh_it": national_compute_demand * dt,
            "rigid_compute_completed_mwh_it": rigid_done,
            "rigid_compute_unserved_mwh_it": rigid_unserved,
            "flex_compute_completed_mwh_it": flex_done,
            "spot_compute_completed_mwh_it": spot_done,
            "national_compute_price_cny_per_mwh_it": spot_price,
            "flex_queue_start_mwh_it": queue[:-1],
            "flex_queue_end_mwh_it": queue[1:],
            "flex_queue_state_residual_mwh_it": queue_residual,
            "fiber_availability": fiber_availability,
            "it_power_mw": it_power,
            "dc_facility_power_mw": dc_power,
            "network_curtailment_mw": network_curtail,
            "economic_curtailment_mw": economic_curtail,
            "curtailment_mw": curtail,
            "offshore_balance_residual_mw": offshore,
            "land_balance_residual_mw": land,
            "electricity_revenue_cny": electricity_revenue,
            "tx_variable_cost_cny": tx_variable,
            "curtailment_penalty_cny": curtail_cost,
            "unserved_critical_penalty_cny": critical_cost,
            "battery_degradation_cost_cny": battery_cost,
            "hydrogen_gross_revenue_cny": h2_revenue,
            "hydrogen_transport_cost_cny": h2_transport,
            "electrolyzer_variable_cost_cny": h2_conversion,
            "desalinated_water_cost_cny": h2_water,
            "fuel_cell_variable_cost_cny": fuel_cell_cost,
            "compute_gross_revenue_cny": compute_revenue,
            "compute_variable_cost_cny": compute_variable,
            "compute_sla_penalty_cny": compute_sla,
            "operating_margin_cny": margin,
        }
    )

    renewable_total = float((renewable * dt).sum())
    curtail_total = float((curtail * dt).sum())
    critical_total = float((critical * dt).sum())
    constrained = tx_available < tx_physical_available - 1e-9
    compute_demand_total = float(((rigid_arrival + flex_arrival) * dt).sum())
    contract_compute_service = float((rigid_done + flex_done).sum())
    kpis: dict[str, Any] = {
        "config_id": config.config_id,
        "scenario_id": scenario.scenario_id,
        "simulation_hours": periods * dt,
        "renewable_generation_mwh": renewable_total,
        "wind_generation_mwh": float((wind * dt).sum()),
        "pv_generation_mwh": float((pv * dt).sum()),
        "wave_generation_mwh": float((wave * dt).sum()),
        "renewable_utilization_rate": (
            (renewable_total - curtail_total) / renewable_total if renewable_total else 0.0
        ),
        "curtailment_mwh": curtail_total,
        "curtailment_rate": curtail_total / renewable_total if renewable_total else 0.0,
        "network_curtailment_mwh": float((network_curtail * dt).sum()),
        "economic_curtailment_mwh": float((economic_curtail * dt).sum()),
        "critical_load_mwh": critical_total,
        "critical_load_served_mwh": float((critical_served * dt).sum()),
        "eens_mwh": float((unmet * dt).sum()),
        "lpsp": float((unmet * dt).sum()) / critical_total if critical_total else 0.0,
        "export_send_mwh": float((export * dt).sum()),
        "export_land_mwh": float((flow.land_mw * dt).sum()),
        "transmission_loss_mwh": float((flow.total_loss_mw * dt).sum()),
        "battery_charge_mwh": float((charge * dt).sum()),
        "battery_discharge_mwh": float((discharge * dt).sum()),
        "battery_efc": (
            float(((charge + discharge) * dt).sum() / (2.0 * battery_energy))
            if battery_energy > 0.0
            else 0.0
        ),
        "battery_terminal_state_error_mwh": float(energy[-1] - energy[0]),
        "max_battery_state_residual_mwh": _safe_max(battery_residual),
        "hydrogen_production_kg": float(h2_production.sum()),
        "hydrogen_sales_kg": float(h2_sale.sum()),
        "hydrogen_fuel_cell_use_kg": float(h2_fuel_cell_use.sum()),
        "fuel_cell_generation_mwh": float((fuel_cell * dt).sum()),
        "hydrogen_terminal_state_error_kg": float(inventory[-1] - inventory[0]),
        "max_hydrogen_state_residual_kg": _safe_max(h2_residual),
        "contract_compute_demand_mwh_it": compute_demand_total,
        "contract_compute_service_mwh_it": contract_compute_service,
        "contract_compute_service_rate": (
            contract_compute_service / compute_demand_total if compute_demand_total else 0.0
        ),
        "rigid_compute_unserved_mwh_it": float(rigid_unserved.sum()),
        "spot_compute_demand_mwh_it": float((national_compute_demand * dt).sum()),
        "national_compute_pool_mwh_it": float((national_compute_pool * dt).sum()),
        "spot_compute_service_mwh_it": float(spot_done.sum()),
        "spot_compute_service_constrained_hours_mwh_it": float(spot_done[constrained].sum()),
        "compute_service_mwh_it": float(it_energy.sum()),
        "dc_facility_energy_mwh": float((dc_power * dt).sum()),
        "compute_it_capacity_factor": (
            float(it_energy.sum() / (config.compute_it_capacity_mw * periods * dt))
            if config.compute_it_capacity_mw > 0.0
            else 0.0
        ),
        "data_center_pue": pue,
        "constrained_grid_hours": int(constrained.sum()),
        "constrained_hour_curtailment_mwh": float((curtail[constrained] * dt).sum()),
        "mean_export_capacity_shadow_cny_per_mw_h": float(export_shadow.mean()),
        "max_export_capacity_shadow_cny_per_mw_h": float(export_shadow.max()),
        "sum_export_capacity_shadow_cny_per_mw": float(export_shadow.sum()),
        "electricity_operating_margin_cny": float(
            (electricity_revenue - tx_variable - curtail_cost - critical_cost).sum()
        ),
        "hydrogen_operating_margin_cny": float(
            (h2_revenue - h2_transport - h2_conversion - h2_water - fuel_cell_cost).sum()
        ),
        "compute_operating_margin_cny": float(
            (compute_revenue - compute_variable - compute_sla).sum()
        ),
        "battery_degradation_cost_cny": float(battery_cost.sum()),
        "operating_margin_cny": float(margin.sum()),
        "max_offshore_balance_residual_mw": _safe_max(offshore),
        "max_land_balance_residual_mw": _safe_max(land),
        "lp_objective_cny": float(solution.fun),
        "lp_iterations": int(solution.nit),
    }
    payload = {
        "phase": "S5",
        "config": config.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "parameters": [item.model_dump(mode="json") for item in params.items],
        "optional_input_columns": {
            name: name in timeseries.columns
            for name in (
                "grid_absorption_factor",
                "grid_absorption_limit_mw",
                "national_compute_demand_mw_it",
                "national_compute_flexible_fraction",
                "national_compute_price_cny_per_mwh_it",
                "fiber_availability",
                "wave_generation_mw",
            )
        },
    }
    metadata = {
        "phase": "S5",
        "model_version": __version__,
        "configuration_hash": configuration_hash(payload),
        "rows": periods,
        "time_step_hours": dt,
        "solver": "scipy.optimize.linprog.highs",
        "solver_status": int(solution.status),
        "solver_message": solution.message,
        "transmission_segments": segments,
        "allocation_boundary": "scarcity_aware_single_offshore_power_balance",
    }
    return S5DispatchResult(hourly, kpis, metadata)
