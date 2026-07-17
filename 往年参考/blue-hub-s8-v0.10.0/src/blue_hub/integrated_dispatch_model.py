"""Phase 5 / S4 integrated electricity, battery, hydrogen and compute dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from blue_hub import __version__
from blue_hub.battery import battery_spec_from_parameters, calculate_battery_losses
from blue_hub.compute import compute_spec_from_parameters
from blue_hub.compute_dispatch_model import run_s3_dispatch
from blue_hub.hydrogen import hydrogen_spec_from_parameters
from blue_hub.hydrogen_dispatch_model import run_s2_dispatch
from blue_hub.metrics import calculate_s0_kpis
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
class S4DispatchResult:
    """Joint S4 hourly ledger, KPI summary and solver provenance."""

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
    rigid_done_start: int
    rigid_unserved_start: int
    flex_done_start: int
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
    rigid_done_start = h2_sale_start + periods
    rigid_unserved_start = rigid_done_start + periods
    flex_done_start = rigid_unserved_start + periods
    battery_energy_start = flex_done_start + periods
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
        rigid_done_start,
        rigid_unserved_start,
        flex_done_start,
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


def _validate_config(config: SystemConfiguration) -> None:
    if config.pv_capacity_mw != 0.0 or config.fuel_cell_power_mw != 0.0:
        raise ValueError("S4 currently supports wind, battery, hydrogen and compute only")
    if config.export_policy != "economic":
        raise ValueError("S4 incremental transmission segments require economic export")
    required = {
        "battery_power_mw": config.battery_power_mw,
        "battery_energy_mwh": config.battery_energy_mwh,
        "electrolyzer_power_mw": config.electrolyzer_power_mw,
        "hydrogen_storage_kg": config.hydrogen_storage_kg,
        "compute_it_capacity_mw": config.compute_it_capacity_mw,
        "subsea_fiber_service_capacity_mw_it": config.subsea_fiber_service_capacity_mw_it,
    }
    missing = {name: value for name, value in required.items() if value <= 0.0}
    if missing:
        raise ValueError(f"S4 integrated configuration requires positive capacities: {missing}")


def _augment_s2(
    result: Any, timeseries: pd.DataFrame, scenario: ScenarioDefinition
) -> S4DispatchResult:
    ts = apply_scenario(timeseries, scenario)
    hourly = result.hourly.copy()
    zeros = np.zeros(len(hourly))
    hourly["fiber_availability"] = ts["fiber_availability"].to_numpy(dtype=float)
    for name in (
        "rigid_compute_completed_mwh_it",
        "flex_compute_completed_mwh_it",
        "flex_queue_start_mwh_it",
        "flex_queue_end_mwh_it",
        "flex_queue_state_residual_mwh_it",
        "it_power_mw",
        "dc_facility_power_mw",
        "compute_gross_revenue_cny",
        "compute_variable_cost_cny",
        "compute_sla_penalty_cny",
    ):
        hourly[name] = zeros
    hourly["rigid_compute_arrival_mwh_it"] = ts["rigid_compute_arrival"].to_numpy()
    hourly["flex_compute_arrival_mwh_it"] = ts["flex_compute_arrival"].to_numpy()
    hourly["rigid_compute_unserved_mwh_it"] = hourly["rigid_compute_arrival_mwh_it"]
    kpis = {
        **result.kpis,
        "compute_service_mwh_it": 0.0,
        "compute_service_rate": 0.0,
        "rigid_compute_unserved_mwh_it": float(hourly["rigid_compute_arrival_mwh_it"].sum()),
        "flex_queue_terminal_error_mwh_it": 0.0,
        "max_flex_queue_state_residual_mwh_it": 0.0,
        "compute_operating_margin_cny": 0.0,
    }
    return S4DispatchResult(hourly, kpis, {**result.metadata, "phase": "S4_ablation_S2"})


def _augment_s3(
    result: Any, timeseries: pd.DataFrame, scenario: ScenarioDefinition
) -> S4DispatchResult:
    ts = apply_scenario(timeseries, scenario)
    hourly = result.hourly.copy()
    zeros = np.zeros(len(hourly))
    for name in (
        "battery_charge_mw",
        "battery_discharge_mw",
        "battery_energy_start_mwh",
        "battery_energy_end_mwh",
        "electrolyzer_power_mw",
        "hydrogen_production_kg",
        "hydrogen_sale_kg",
        "hydrogen_inventory_start_kg",
        "hydrogen_inventory_end_kg",
        "hydrogen_state_residual_kg",
        "hydrogen_gross_revenue_cny",
        "hydrogen_transport_cost_cny",
        "electrolyzer_variable_cost_cny",
        "desalinated_water_cost_cny",
    ):
        hourly[name] = zeros
    hourly["hydrogen_demand_kg"] = ts["hydrogen_demand"].to_numpy()
    kpis = {
        **result.kpis,
        "battery_efc": 0.0,
        "hydrogen_production_kg": 0.0,
        "hydrogen_sales_kg": 0.0,
        "hydrogen_service_rate": 0.0,
        "hydrogen_terminal_state_error_kg": 0.0,
        "max_hydrogen_state_residual_kg": 0.0,
        "hydrogen_operating_margin_cny": 0.0,
    }
    return S4DispatchResult(hourly, kpis, {**result.metadata, "phase": "S4_ablation_S3"})


def run_s4_dispatch(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S4DispatchResult:
    """Solve the fully integrated S4 allocation problem with one power balance."""
    validate_timeseries(timeseries).raise_if_invalid()
    validate_parameters(params).raise_if_invalid()
    if config.compute_it_capacity_mw == 0.0:
        return _augment_s2(
            run_s2_dispatch(timeseries, params, config, scenario), timeseries, scenario
        )
    if config.electrolyzer_power_mw == 0.0 and config.battery_power_mw == 0.0:
        return _augment_s3(
            run_s3_dispatch(timeseries, params, config, scenario), timeseries, scenario
        )
    _validate_config(config)
    if "fiber_availability" not in timeseries.columns:
        raise ValueError("S4 requires a fiber_availability time-series column")
    dt = params.value("time_step")
    if dt != 1.0:
        raise ValueError("S4 supports a 1 h time step only")
    battery = battery_spec_from_parameters(config, params)
    hydrogen = hydrogen_spec_from_parameters(config, params, scenario)
    compute = compute_spec_from_parameters(config, params, scenario)
    ts = apply_scenario(timeseries, scenario)
    periods = len(ts)
    wind = config.wind_capacity_mw * ts["wind_cf"].to_numpy() * ts["wind_availability"].to_numpy()
    critical = ts["critical_load"].to_numpy()
    price = ts["electricity_price"].to_numpy()
    h2_demand = ts["hydrogen_demand"].to_numpy()
    rigid_arrival = ts["rigid_compute_arrival"].to_numpy()
    flex_arrival = ts["flex_compute_arrival"].to_numpy()
    tx_available = config.tx_capacity_mw * ts["tx_availability"].to_numpy()
    fiber_available = (
        compute.subsea_fiber_service_capacity_mw_it * ts["fiber_availability"].to_numpy()
    )
    loss_spec = _loss_spec(params, config.tx_technology)
    segments = int(round(params.value("battery_linearization_segments")))
    if config.tx_capacity_mw > 0.0:
        piecewise = build_piecewise_transmission(
            config.tx_capacity_mw, scenario.offshore_distance_km, loss_spec, segments
        )
        widths, slopes = piecewise.segment_widths_mw, piecewise.land_delivery_slopes
    else:
        segments, widths, slopes = 0, np.empty(0), np.empty(0)
    layout = _layout(periods, segments)
    objective = np.zeros(layout.count)
    tx_cost = params.value("tx_variable_cost")
    for t in range(periods):
        for segment in range(segments):
            objective[t * segments + segment] = (tx_cost - price[t] * slopes[segment]) * dt
    objective[layout.charge_start : layout.charge_start + periods] = (
        battery.degradation_cost_cny_per_mwh_throughput * dt
    )
    objective[layout.discharge_start : layout.discharge_start + periods] = (
        battery.degradation_cost_cny_per_mwh_throughput * dt
    )
    objective[layout.curtail_start : layout.curtail_start + periods] = (
        params.value("curtailment_penalty") * dt
    )
    objective[layout.unmet_start : layout.unmet_start + periods] = (
        params.value("unserved_critical_load_penalty") * dt
    )
    objective[layout.h2_power_start : layout.h2_power_start + periods] = (
        hydrogen.conversion_and_water_cost_cny_per_kg * hydrogen.kg_per_mwh * dt
    )
    objective[
        layout.h2_sale_start : layout.h2_sale_start + periods
    ] = -hydrogen.net_sale_value_cny_per_kg
    objective[layout.rigid_done_start : layout.rigid_done_start + periods] = (
        compute.variable_cost_cny_per_mwh_it - compute.rigid_service_price_cny_per_mwh_it
    )
    objective[layout.rigid_unserved_start : layout.rigid_unserved_start + periods] = (
        compute.rigid_sla_penalty_cny_per_mwh_it
    )
    objective[layout.flex_done_start : layout.flex_done_start + periods] = (
        compute.variable_cost_cny_per_mwh_it - compute.flex_service_price_cny_per_mwh_it
    )

    bounds: list[tuple[float, float]] = []
    for _ in range(periods):
        bounds.extend((0.0, float(width)) for width in widths)
    bounds.extend((0.0, battery.power_mw) for _ in range(periods))
    bounds.extend((0.0, battery.power_mw) for _ in range(periods))
    bounds.extend((0.0, float(value)) for value in wind)
    bounds.extend((0.0, float(value)) for value in critical)
    bounds.extend((0.0, hydrogen.power_mw) for _ in range(periods))
    bounds.extend((0.0, float(value * dt)) for value in h2_demand)
    bounds.extend((0.0, float(value * dt)) for value in rigid_arrival)
    bounds.extend((0.0, float(value * dt)) for value in rigid_arrival)
    bounds.extend((0.0, compute.it_capacity_mw * dt) for _ in range(periods))
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((battery.initial_energy_mwh, battery.initial_energy_mwh))
        else:
            bounds.append((battery.scheduled_minimum_energy_mwh, battery.maximum_energy_mwh))
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((hydrogen.initial_inventory_kg, hydrogen.initial_inventory_kg))
        else:
            bounds.append((0.0, hydrogen.storage_capacity_kg))
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((0.0, 0.0))
        else:
            hour = state - 1
            if compute.flex_max_delay_h == 0:
                due_bound = 0.0
            elif hour >= compute.flex_max_delay_h:
                due_bound = float(
                    flex_arrival[hour - compute.flex_max_delay_h + 1 : hour + 1].sum() * dt
                )
            else:
                due_bound = compute.flex_queue_capacity_mwh_it
            bounds.append((0.0, min(compute.flex_queue_capacity_mwh_it, due_bound)))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(5 * periods)
    for t in range(periods):
        for segment in range(segments):
            rows.append(t)
            cols.append(t * segments + segment)
            data.append(1.0)
        rows.extend([t] * 7)
        cols.extend(
            [
                layout.charge_start + t,
                layout.curtail_start + t,
                layout.h2_power_start + t,
                layout.rigid_done_start + t,
                layout.flex_done_start + t,
                layout.discharge_start + t,
                layout.unmet_start + t,
            ]
        )
        data.extend([1.0, 1.0, 1.0, compute.pue / dt, compute.pue / dt, -1.0, -1.0])
        rhs[t] = wind[t] - critical[t]
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
                -(1.0 - battery.self_discharge_per_hour),
                -battery.charge_efficiency * dt,
                dt / battery.discharge_efficiency,
            ]
        )
        h_row = 2 * periods + t
        rows.extend([h_row] * 4)
        cols.extend(
            [
                layout.h2_inventory_start + t + 1,
                layout.h2_inventory_start + t,
                layout.h2_power_start + t,
                layout.h2_sale_start + t,
            ]
        )
        data.extend([1.0, -(1.0 - hydrogen.storage_loss_per_hour), -hydrogen.kg_per_mwh * dt, 1.0])
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
    row = 0
    for t in range(periods):
        for segment in range(segments):
            ir.append(row)
            ic.append(t * segments + segment)
            idata.append(1.0)
        irhs.append(float(tx_available[t]))
        row += 1
    headroom = battery.power_mw - battery.reserve_power_mw
    for t in range(periods):
        if battery.reserve_power_mw > 0.0:
            ir.extend([row, row])
            ic.extend([layout.discharge_start + t, layout.charge_start + t])
            idata.extend([1.0, -1.0])
            irhs.append(headroom)
            row += 1
        for cap in (compute.it_capacity_mw, float(fiber_available[t])):
            ir.extend([row, row])
            ic.extend([layout.rigid_done_start + t, layout.flex_done_start + t])
            idata.extend([1.0, 1.0])
            irhs.append(cap * dt)
            row += 1
        for sign, limit in (
            (1.0, compute.ramp_up_mw_it_per_h),
            (-1.0, compute.ramp_down_mw_it_per_h),
        ):
            ir.extend([row, row])
            ic.extend([layout.rigid_done_start + t, layout.flex_done_start + t])
            idata.extend([sign / dt, sign / dt])
            if t > 0:
                ir.extend([row, row])
                ic.extend([layout.rigid_done_start + t - 1, layout.flex_done_start + t - 1])
                idata.extend([-sign / dt, -sign / dt])
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
        raise RuntimeError(f"S4 optimisation failed: status={solution.status}; {solution.message}")
    x = solution.x
    if segments:
        segment_dispatch = x[: periods * segments].reshape(periods, segments)
        export = segment_dispatch.sum(axis=1)
        piece_land = segment_dispatch @ slopes
    else:
        segment_dispatch = np.empty((periods, 0))
        export = np.zeros(periods)
        piece_land = np.zeros(periods)
    charge = x[layout.charge_start : layout.charge_start + periods]
    discharge = x[layout.discharge_start : layout.discharge_start + periods]
    curtail = x[layout.curtail_start : layout.curtail_start + periods]
    unmet = x[layout.unmet_start : layout.unmet_start + periods]
    h2_power = x[layout.h2_power_start : layout.h2_power_start + periods]
    h2_sale = x[layout.h2_sale_start : layout.h2_sale_start + periods]
    rigid_done = x[layout.rigid_done_start : layout.rigid_done_start + periods]
    rigid_unserved = x[layout.rigid_unserved_start : layout.rigid_unserved_start + periods]
    flex_done = x[layout.flex_done_start : layout.flex_done_start + periods]
    energy = x[layout.battery_energy_start : layout.battery_energy_start + periods + 1]
    inventory = x[layout.h2_inventory_start : layout.h2_inventory_start + periods + 1]
    queue = x[layout.flex_queue_start : layout.flex_queue_start + periods + 1]
    h2_production = h2_power * hydrogen.kg_per_mwh * dt
    it_energy = rigid_done + flex_done
    it_power = it_energy / dt
    dc_power = compute.pue * it_power
    h2_residual = (
        inventory[1:]
        - (1.0 - hydrogen.storage_loss_per_hour) * inventory[:-1]
        - h2_production
        + h2_sale
    )
    battery_residual = (
        energy[1:]
        - (1.0 - battery.self_discharge_per_hour) * energy[:-1]
        - battery.charge_efficiency * charge * dt
        + discharge * dt / battery.discharge_efficiency
    )
    queue_residual = queue[1:] - queue[:-1] + flex_done - flex_arrival * dt
    rigid_residual = rigid_done + rigid_unserved - rigid_arrival * dt
    flow = calculate_transmission_flow(
        export, config.tx_capacity_mw, scenario.offshore_distance_km, loss_spec
    )
    linear_error = flow.land_mw - piece_land
    critical_served = critical - unmet
    offshore = wind + discharge + unmet - critical - export - charge - h2_power - dc_power - curtail
    land = export - flow.land_mw - flow.total_loss_mw
    tol = params.value("power_balance_tolerance")
    if max(float(np.abs(offshore).max()), float(np.abs(land).max())) >= tol:
        raise RuntimeError("S4 power balance residual exceeds tolerance")
    if float(np.abs(battery_residual).max()) >= params.value("battery_state_tolerance") or abs(
        float(energy[-1] - energy[0])
    ) >= params.value("battery_state_tolerance"):
        raise RuntimeError("S4 battery state audit failed")
    if float(np.abs(h2_residual).max()) >= params.value("hydrogen_state_tolerance") or abs(
        float(inventory[-1] - inventory[0])
    ) >= params.value("hydrogen_state_tolerance"):
        raise RuntimeError("S4 hydrogen state audit failed")
    if float(np.abs(queue_residual).max()) >= params.value("compute_state_tolerance") or abs(
        float(queue[-1] - queue[0])
    ) >= params.value("compute_state_tolerance"):
        raise RuntimeError("S4 compute queue audit failed")
    if float(np.abs(rigid_residual).max()) >= params.value("compute_state_tolerance"):
        raise RuntimeError("S4 rigid compute audit failed")
    if (np.minimum(charge, discharge) > params.value("battery_simultaneous_tolerance")).any():
        raise RuntimeError("S4 simultaneous battery charge and discharge")
    completed_cumulative = np.cumsum(flex_done)
    arrivals_cumulative = np.cumsum(flex_arrival * dt)
    delay = compute.flex_max_delay_h
    deadline_slack = (
        completed_cumulative[delay:] - arrivals_cumulative[: periods - delay]
        if delay < periods
        else np.empty(0)
    )
    if len(deadline_slack) and float(deadline_slack.min()) < -params.value(
        "compute_state_tolerance"
    ):
        raise RuntimeError("S4 flexible compute deadline is violated")
    if (it_power > compute.it_capacity_mw + 1e-7).any() or (
        it_power > fiber_available + 1e-7
    ).any():
        raise RuntimeError("S4 compute capacity is exceeded")
    if float(linear_error.min()) < -1e-7:
        raise RuntimeError("S4 transmission linearisation overstates land delivery")

    h2_loss = hydrogen.storage_loss_per_hour * inventory[:-1]
    losses = calculate_battery_losses(energy[:-1], charge, discharge, battery, dt)
    electricity_revenue = price * flow.land_mw * dt
    tx_variable = tx_cost * export * dt
    curtail_cost = params.value("curtailment_penalty") * curtail * dt
    critical_cost = params.value("unserved_critical_load_penalty") * unmet * dt
    battery_cost = battery.degradation_cost_cny_per_mwh_throughput * (charge + discharge) * dt
    h2_revenue = h2_sale * hydrogen.sale_price_cny_per_kg
    h2_transport = h2_sale * hydrogen.transport_cost_cny_per_kg
    h2_variable = h2_production * hydrogen.variable_cost_cny_per_kg
    h2_water = h2_production * hydrogen.water_consumption_m3_per_kg
    h2_water_cost = h2_water * hydrogen.water_cost_cny_per_m3
    compute_revenue = (
        rigid_done * compute.rigid_service_price_cny_per_mwh_it
        + flex_done * compute.flex_service_price_cny_per_mwh_it
    )
    compute_variable = it_energy * compute.variable_cost_cny_per_mwh_it
    compute_sla = rigid_unserved * compute.rigid_sla_penalty_cny_per_mwh_it
    margin = (
        electricity_revenue
        + h2_revenue
        + compute_revenue
        - tx_variable
        - curtail_cost
        - critical_cost
        - battery_cost
        - h2_transport
        - h2_variable
        - h2_water_cost
        - compute_variable
        - compute_sla
    )
    feasible_export = np.minimum(
        np.maximum(wind + discharge - critical_served - charge - h2_power - dc_power, 0.0),
        tx_available,
    )
    marginal_export = (
        price * (1.0 - loss_spec.terminal_loss_fraction)
        - tx_cost
        + params.value("curtailment_penalty")
    )
    economic_curtail = np.where(
        (marginal_export <= 0.0) & (tx_available > 0.0) & (export <= 1e-7), curtail, 0.0
    )
    hourly = pd.DataFrame(
        {
            "timestamp": ts["timestamp"].to_numpy(),
            "scenario_id": ts["scenario_id"].to_numpy(),
            "electricity_price_cny_per_mwh": price,
            "wind_available_mw": wind,
            "critical_load_mw": critical,
            "critical_load_served_mw": critical_served,
            "unmet_critical_load_mw": unmet,
            "renewable_surplus_mw": np.maximum(wind - critical_served, 0.0),
            "tx_availability": ts["tx_availability"].to_numpy(),
            "tx_available_capacity_mw": tx_available,
            "fiber_availability": ts["fiber_availability"].to_numpy(),
            "export_feasible_limit_mw": feasible_export,
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
            "battery_total_loss_mwh": losses.total_loss_mwh,
            "hydrogen_demand_kg": h2_demand * dt,
            "electrolyzer_power_mw": h2_power,
            "hydrogen_production_kg": h2_production,
            "hydrogen_sale_kg": h2_sale,
            "hydrogen_inventory_start_kg": inventory[:-1],
            "hydrogen_inventory_end_kg": inventory[1:],
            "hydrogen_storage_loss_kg": h2_loss,
            "hydrogen_state_residual_kg": h2_residual,
            "rigid_compute_arrival_mwh_it": rigid_arrival * dt,
            "flex_compute_arrival_mwh_it": flex_arrival * dt,
            "rigid_compute_completed_mwh_it": rigid_done,
            "rigid_compute_unserved_mwh_it": rigid_unserved,
            "flex_compute_completed_mwh_it": flex_done,
            "flex_queue_start_mwh_it": queue[:-1],
            "flex_queue_end_mwh_it": queue[1:],
            "flex_queue_state_residual_mwh_it": queue_residual,
            "it_power_mw": it_power,
            "dc_facility_power_mw": dc_power,
            "network_curtailment_mw": curtail - economic_curtail,
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
            "electrolyzer_variable_cost_cny": h2_variable,
            "desalinated_water_cost_cny": h2_water_cost,
            "compute_gross_revenue_cny": compute_revenue,
            "compute_variable_cost_cny": compute_variable,
            "compute_sla_penalty_cny": compute_sla,
            "operating_margin_cny": margin,
        }
    )
    kpis = calculate_s0_kpis(hourly, config, time_step_hours=dt)
    h2_demand_total = float((h2_demand * dt).sum())
    compute_demand = float(((rigid_arrival + flex_arrival) * dt).sum())
    compute_service = float(it_energy.sum())
    kpis.update(
        {
            "battery_charge_mwh": float((charge * dt).sum()),
            "battery_discharge_mwh": float((discharge * dt).sum()),
            "battery_efc": float(((charge + discharge) * dt).sum() / (2 * battery.energy_mwh)),
            "battery_final_soc": float(energy[-1] / battery.energy_mwh),
            "max_battery_state_residual_mwh": float(np.abs(battery_residual).max()),
            "hydrogen_production_kg": float(h2_production.sum()),
            "hydrogen_sales_kg": float(h2_sale.sum()),
            "hydrogen_demand_kg": h2_demand_total,
            "hydrogen_service_rate": float(h2_sale.sum() / h2_demand_total)
            if h2_demand_total
            else 0.0,
            "hydrogen_terminal_state_error_kg": float(inventory[-1] - inventory[0]),
            "max_hydrogen_state_residual_kg": float(np.abs(h2_residual).max()),
            "electrolyzer_capacity_factor": float(
                (h2_power * dt).sum() / (hydrogen.power_mw * periods * dt)
            ),
            "compute_service_mwh_it": compute_service,
            "compute_demand_mwh_it": compute_demand,
            "compute_service_rate": compute_service / compute_demand if compute_demand else 0.0,
            "rigid_compute_unserved_mwh_it": float(rigid_unserved.sum()),
            "flex_queue_max_mwh_it": float(queue.max()),
            "flex_queue_terminal_error_mwh_it": float(queue[-1] - queue[0]),
            "max_flex_queue_state_residual_mwh_it": float(np.abs(queue_residual).max()),
            "minimum_flex_deadline_slack_mwh_it": float(deadline_slack.min())
            if len(deadline_slack)
            else 0.0,
            "compute_it_capacity_factor": compute_service / (compute.it_capacity_mw * periods * dt),
            "data_center_pue": compute.pue,
            "electricity_operating_margin_cny": float(
                (electricity_revenue - tx_variable - curtail_cost - critical_cost).sum()
            ),
            "hydrogen_operating_margin_cny": float(
                (h2_revenue - h2_transport - h2_variable - h2_water_cost).sum()
            ),
            "compute_operating_margin_cny": float(
                (compute_revenue - compute_variable - compute_sla).sum()
            ),
            "battery_degradation_cost_cny": float(battery_cost.sum()),
            "lp_objective_cny": float(solution.fun),
            "lp_iterations": int(solution.nit),
        }
    )
    payload = {
        "phase": "S4",
        "config": config.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "parameters": [item.model_dump(mode="json") for item in params.items],
    }
    metadata = {
        "phase": "S4",
        "model_version": __version__,
        "configuration_hash": configuration_hash(payload),
        "rows": periods,
        "time_step_hours": dt,
        "solver": "scipy.optimize.linprog.highs",
        "solver_status": int(solution.status),
        "solver_message": solution.message,
        "transmission_segments": segments,
        "allocation_boundary": "single_offshore_power_balance",
    }
    return S4DispatchResult(hourly, kpis, metadata)
