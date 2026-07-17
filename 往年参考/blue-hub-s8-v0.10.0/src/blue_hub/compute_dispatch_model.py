"""Phase 4 / S3 hourly optimisation of export and green-compute service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from blue_hub import __version__
from blue_hub.compute import compute_spec_from_parameters
from blue_hub.dispatch_model import run_s0_dispatch
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
class S3DispatchResult:
    """Hourly S3 ledger, KPIs and deterministic solver provenance."""

    hourly: pd.DataFrame
    kpis: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _VariableLayout:
    periods: int
    segments: int
    curtailment_start: int
    unmet_critical_start: int
    rigid_completed_start: int
    rigid_unserved_start: int
    flex_completed_start: int
    flex_queue_start: int
    variable_count: int


def _layout(periods: int, segments: int) -> _VariableLayout:
    segment_count = periods * segments
    curtailment_start = segment_count
    unmet_critical_start = curtailment_start + periods
    rigid_completed_start = unmet_critical_start + periods
    rigid_unserved_start = rigid_completed_start + periods
    flex_completed_start = rigid_unserved_start + periods
    flex_queue_start = flex_completed_start + periods
    return _VariableLayout(
        periods=periods,
        segments=segments,
        curtailment_start=curtailment_start,
        unmet_critical_start=unmet_critical_start,
        rigid_completed_start=rigid_completed_start,
        rigid_unserved_start=rigid_unserved_start,
        flex_completed_start=flex_completed_start,
        flex_queue_start=flex_queue_start,
        variable_count=flex_queue_start + periods + 1,
    )


def _loss_spec(params: TechnologyParameters, technology: str) -> TransmissionLossSpec:
    return TransmissionLossSpec(
        terminal_loss_fraction=params.value(f"tx_terminal_loss_fraction_{technology}"),
        cable_full_load_loss_per_100km=params.value(
            f"tx_cable_full_load_loss_per_100km_{technology}"
        ),
    )


def _validate_s3_configuration(config: SystemConfiguration) -> None:
    unsupported = {
        "pv_capacity_mw": config.pv_capacity_mw,
        "battery_power_mw": config.battery_power_mw,
        "battery_energy_mwh": config.battery_energy_mwh,
        "electrolyzer_power_mw": config.electrolyzer_power_mw,
        "hydrogen_storage_kg": config.hydrogen_storage_kg,
        "fuel_cell_power_mw": config.fuel_cell_power_mw,
    }
    nonzero = {name: value for name, value in unsupported.items() if value != 0.0}
    if nonzero:
        raise ValueError(f"S3 configuration contains future-phase capacities: {nonzero}")
    if config.export_policy != "economic":
        raise ValueError("S3 incremental transmission segments currently require economic export")
    if config.compute_it_capacity_mw == 0.0 and config.subsea_fiber_service_capacity_mw_it != 0.0:
        raise ValueError("subsea-fibre service capacity requires positive IT capacity in S3")


def _s0_ablation(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S3DispatchResult:
    """Preserve exact S0 results when no compute capacity is installed."""
    s0 = run_s0_dispatch(timeseries, params, config, scenario)
    ts = apply_scenario(timeseries, scenario)
    hourly = s0.hourly.copy()
    zeros = np.zeros(len(hourly))
    hourly["fiber_availability"] = ts["fiber_availability"].to_numpy(dtype=float)
    hourly["rigid_compute_arrival_mwh_it"] = ts["rigid_compute_arrival"].to_numpy(dtype=float)
    hourly["flex_compute_arrival_mwh_it"] = ts["flex_compute_arrival"].to_numpy(dtype=float)
    hourly["rigid_compute_completed_mwh_it"] = zeros
    hourly["rigid_compute_unserved_mwh_it"] = hourly["rigid_compute_arrival_mwh_it"]
    hourly["flex_compute_completed_mwh_it"] = zeros
    hourly["flex_queue_start_mwh_it"] = zeros
    hourly["flex_queue_end_mwh_it"] = zeros
    hourly["flex_queue_state_residual_mwh_it"] = zeros
    hourly["it_power_mw"] = zeros
    hourly["dc_facility_power_mw"] = zeros
    hourly["compute_gross_revenue_cny"] = zeros
    hourly["compute_variable_cost_cny"] = zeros
    hourly["compute_sla_penalty_cny"] = zeros
    kpis = {
        **s0.kpis,
        "compute_service_mwh_it": 0.0,
        "compute_demand_mwh_it": float(
            hourly["rigid_compute_arrival_mwh_it"].sum()
            + hourly["flex_compute_arrival_mwh_it"].sum()
        ),
        "compute_service_rate": 0.0,
        "rigid_compute_service_rate": 0.0,
        "flex_compute_service_rate": 0.0,
        "rigid_compute_unserved_mwh_it": float(hourly["rigid_compute_arrival_mwh_it"].sum()),
        "flex_queue_min_mwh_it": 0.0,
        "flex_queue_max_mwh_it": 0.0,
        "flex_queue_terminal_error_mwh_it": 0.0,
        "max_flex_queue_state_residual_mwh_it": 0.0,
        "compute_it_capacity_factor": 0.0,
        "subsea_fiber_service_utilization": 0.0,
        "data_center_pue": 0.0,
        "compute_gross_revenue_cny": 0.0,
        "compute_variable_cost_cny": 0.0,
        "compute_sla_penalty_cny": 0.0,
        "compute_operating_margin_cny": 0.0,
    }
    return S3DispatchResult(
        hourly=hourly,
        kpis=kpis,
        metadata={**s0.metadata, "phase": "S3_ablation", "solver": "S0_exact_fallback"},
    )


def run_s3_dispatch(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S3DispatchResult:
    """Optimise direct electricity export and fibre-delivered green-compute service.

    Work is measured in MWh-IT.  PUE converts completed IT work to facility
    electricity at the offshore bus.  The fibre link is a service-egress proxy
    in MWh-IT/h, not an optical-bandwidth or electrical-loss model.
    """
    validate_timeseries(timeseries).raise_if_invalid()
    validate_parameters(params).raise_if_invalid()
    _validate_s3_configuration(config)
    if config.compute_it_capacity_mw == 0.0:
        return _s0_ablation(timeseries, params, config, scenario)
    if "fiber_availability" not in timeseries.columns:
        raise ValueError("S3 requires a fiber_availability time-series column")

    dt = params.value("time_step")
    if dt != 1.0:
        raise ValueError("S3 currently supports a 1 h time step only")
    compute = compute_spec_from_parameters(config, params, scenario)
    ts = apply_scenario(timeseries, scenario)
    periods = len(ts)
    wind = (
        config.wind_capacity_mw
        * ts["wind_cf"].to_numpy(dtype=float)
        * ts["wind_availability"].to_numpy(dtype=float)
    )
    critical = ts["critical_load"].to_numpy(dtype=float)
    price = ts["electricity_price"].to_numpy(dtype=float)
    rigid_arrival = ts["rigid_compute_arrival"].to_numpy(dtype=float)
    flex_arrival = ts["flex_compute_arrival"].to_numpy(dtype=float)
    tx_availability = ts["tx_availability"].to_numpy(dtype=float)
    fiber_availability = ts["fiber_availability"].to_numpy(dtype=float)
    available_tx_capacity = config.tx_capacity_mw * tx_availability
    available_fiber_service_capacity = (
        compute.subsea_fiber_service_capacity_mw_it * fiber_availability
    )
    loss_spec = _loss_spec(params, config.tx_technology)

    segment_count = int(round(params.value("battery_linearization_segments")))
    if segment_count < 1:
        raise ValueError("battery_linearization_segments must be at least one")
    if config.tx_capacity_mw > 0.0:
        piecewise = build_piecewise_transmission(
            config.tx_capacity_mw,
            scenario.offshore_distance_km,
            loss_spec,
            segment_count,
        )
        widths = piecewise.segment_widths_mw
        slopes = piecewise.land_delivery_slopes
    else:
        segment_count = 0
        widths = np.empty(0)
        slopes = np.empty(0)

    layout = _layout(periods, segment_count)
    objective = np.zeros(layout.variable_count)
    tx_variable_cost = params.value("tx_variable_cost")
    for t in range(periods):
        for segment in range(segment_count):
            objective[t * segment_count + segment] = (
                tx_variable_cost - price[t] * slopes[segment]
            ) * dt
    objective[layout.curtailment_start : layout.curtailment_start + periods] = (
        params.value("curtailment_penalty") * dt
    )
    objective[layout.unmet_critical_start : layout.unmet_critical_start + periods] = (
        params.value("unserved_critical_load_penalty") * dt
    )
    objective[layout.rigid_completed_start : layout.rigid_completed_start + periods] = (
        compute.variable_cost_cny_per_mwh_it - compute.rigid_service_price_cny_per_mwh_it
    )
    objective[layout.rigid_unserved_start : layout.rigid_unserved_start + periods] = (
        compute.rigid_sla_penalty_cny_per_mwh_it
    )
    objective[layout.flex_completed_start : layout.flex_completed_start + periods] = (
        compute.variable_cost_cny_per_mwh_it - compute.flex_service_price_cny_per_mwh_it
    )

    bounds: list[tuple[float, float]] = []
    for _ in range(periods):
        bounds.extend((0.0, float(width)) for width in widths)
    bounds.extend((0.0, float(value)) for value in wind)
    bounds.extend((0.0, float(value)) for value in critical)
    bounds.extend((0.0, float(value * dt)) for value in rigid_arrival)
    bounds.extend((0.0, float(value * dt)) for value in rigid_arrival)
    bounds.extend((0.0, compute.it_capacity_mw * dt) for _ in range(periods))
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((0.0, 0.0))
        else:
            completed_hour = state - 1
            if compute.flex_max_delay_h == 0:
                delay_limited_capacity = 0.0
            elif completed_hour >= compute.flex_max_delay_h:
                delay_limited_capacity = float(
                    flex_arrival[
                        completed_hour - compute.flex_max_delay_h + 1 : completed_hour + 1
                    ].sum()
                    * dt
                )
            else:
                delay_limited_capacity = compute.flex_queue_capacity_mwh_it
            bounds.append(
                (0.0, min(compute.flex_queue_capacity_mwh_it, delay_limited_capacity))
            )

    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_data: list[float] = []
    equality_rhs = np.zeros(3 * periods)
    facility_factor = compute.pue / dt
    for t in range(periods):
        for segment in range(segment_count):
            equality_rows.append(t)
            equality_columns.append(t * segment_count + segment)
            equality_data.append(1.0)
        equality_rows.extend([t, t, t, t, t])
        equality_columns.extend(
            [
                layout.curtailment_start + t,
                layout.rigid_completed_start + t,
                layout.flex_completed_start + t,
                layout.unmet_critical_start + t,
                layout.unmet_critical_start + t,
            ]
        )
        equality_data.extend([1.0, facility_factor, facility_factor, -1.0, 0.0])
        equality_rhs[t] = wind[t] - critical[t]

        rigid_row = periods + t
        equality_rows.extend([rigid_row, rigid_row])
        equality_columns.extend([layout.rigid_completed_start + t, layout.rigid_unserved_start + t])
        equality_data.extend([1.0, 1.0])
        equality_rhs[rigid_row] = rigid_arrival[t] * dt

        queue_row = 2 * periods + t
        equality_rows.extend([queue_row, queue_row, queue_row])
        equality_columns.extend(
            [
                layout.flex_queue_start + t + 1,
                layout.flex_queue_start + t,
                layout.flex_completed_start + t,
            ]
        )
        equality_data.extend([1.0, -1.0, 1.0])
        equality_rhs[queue_row] = flex_arrival[t] * dt
    equality_matrix = coo_matrix(
        (equality_data, (equality_rows, equality_columns)),
        shape=(3 * periods, layout.variable_count),
    ).tocsr()

    inequality_rows: list[int] = []
    inequality_columns: list[int] = []
    inequality_data: list[float] = []
    inequality_rhs_values: list[float] = []
    row = 0
    if segment_count:
        for t in range(periods):
            for segment in range(segment_count):
                inequality_rows.append(row)
                inequality_columns.append(t * segment_count + segment)
                inequality_data.append(1.0)
            inequality_rhs_values.append(float(available_tx_capacity[t]))
            row += 1
    for t in range(periods):
        inequality_rows.extend([row, row])
        inequality_columns.extend(
            [layout.rigid_completed_start + t, layout.flex_completed_start + t]
        )
        inequality_data.extend([1.0, 1.0])
        inequality_rhs_values.append(compute.it_capacity_mw * dt)
        row += 1
        inequality_rows.extend([row, row])
        inequality_columns.extend(
            [layout.rigid_completed_start + t, layout.flex_completed_start + t]
        )
        inequality_data.extend([1.0, 1.0])
        inequality_rhs_values.append(float(available_fiber_service_capacity[t] * dt))
        row += 1
    for t in range(periods):
        inequality_rows.append(row)
        inequality_columns.append(layout.rigid_completed_start + t)
        inequality_data.append(1.0 / dt)
        inequality_rows.append(row)
        inequality_columns.append(layout.flex_completed_start + t)
        inequality_data.append(1.0 / dt)
        if t > 0:
            inequality_rows.append(row)
            inequality_columns.append(layout.rigid_completed_start + t - 1)
            inequality_data.append(-1.0 / dt)
            inequality_rows.append(row)
            inequality_columns.append(layout.flex_completed_start + t - 1)
            inequality_data.append(-1.0 / dt)
        inequality_rhs_values.append(compute.ramp_up_mw_it_per_h * dt)
        row += 1
        inequality_rows.append(row)
        inequality_columns.append(layout.rigid_completed_start + t)
        inequality_data.append(-1.0 / dt)
        inequality_rows.append(row)
        inequality_columns.append(layout.flex_completed_start + t)
        inequality_data.append(-1.0 / dt)
        if t > 0:
            inequality_rows.append(row)
            inequality_columns.append(layout.rigid_completed_start + t - 1)
            inequality_data.append(1.0 / dt)
            inequality_rows.append(row)
            inequality_columns.append(layout.flex_completed_start + t - 1)
            inequality_data.append(1.0 / dt)
        inequality_rhs_values.append(compute.ramp_down_mw_it_per_h * dt)
        row += 1
    inequality_matrix = coo_matrix(
        (inequality_data, (inequality_rows, inequality_columns)),
        shape=(row, layout.variable_count),
    ).tocsr()
    inequality_rhs = np.asarray(inequality_rhs_values)

    solution = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs,
        A_eq=equality_matrix,
        b_eq=equality_rhs,
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
            f"S3 linear optimization failed: status={solution.status}; {solution.message}"
        )
    values = solution.x
    if segment_count:
        segment_dispatch = values[: periods * segment_count].reshape(periods, segment_count)
        export_send = segment_dispatch.sum(axis=1)
        piecewise_land = segment_dispatch @ slopes
    else:
        segment_dispatch = np.empty((periods, 0))
        export_send = np.zeros(periods)
        piecewise_land = np.zeros(periods)
    curtailment = values[layout.curtailment_start : layout.curtailment_start + periods]
    unmet_critical = values[
        layout.unmet_critical_start : layout.unmet_critical_start + periods
    ]
    rigid_completed = values[
        layout.rigid_completed_start : layout.rigid_completed_start + periods
    ]
    rigid_unserved = values[
        layout.rigid_unserved_start : layout.rigid_unserved_start + periods
    ]
    flex_completed = values[
        layout.flex_completed_start : layout.flex_completed_start + periods
    ]
    flex_queue = values[layout.flex_queue_start : layout.flex_queue_start + periods + 1]
    it_energy = rigid_completed + flex_completed
    it_power = it_energy / dt
    facility_power = compute.pue * it_power
    queue_residual = flex_queue[1:] - flex_queue[:-1] + flex_completed - flex_arrival * dt
    rigid_residual = rigid_completed + rigid_unserved - rigid_arrival * dt

    segment_order_violation = 0.0
    for segment in range(1, segment_count):
        previous_not_full = segment_dispatch[:, segment - 1] < widths[segment - 1] - 1e-7
        if previous_not_full.any():
            segment_order_violation = max(
                segment_order_violation,
                float(segment_dispatch[previous_not_full, segment].max(initial=0.0)),
            )
    if segment_order_violation > 1e-6:
        raise RuntimeError("incremental transmission segment order is invalid")

    flow = calculate_transmission_flow(
        export_send,
        config.tx_capacity_mw,
        scenario.offshore_distance_km,
        loss_spec,
    )
    linearization_error = flow.land_mw - piecewise_land
    if float(linearization_error.min(initial=0.0)) < -1e-7:
        raise RuntimeError("piecewise model overstates concave transmission delivery")
    critical_served = critical - unmet_critical
    offshore_residual = (
        wind
        + unmet_critical
        - critical
        - export_send
        - facility_power
        - curtailment
    )
    land_residual = export_send - flow.land_mw - flow.total_loss_mw
    power_tolerance = params.value("power_balance_tolerance")
    state_tolerance = params.value("compute_state_tolerance")
    if float(np.abs(offshore_residual).max(initial=0.0)) >= power_tolerance:
        raise RuntimeError("S3 offshore power balance residual exceeds tolerance")
    if float(np.abs(land_residual).max(initial=0.0)) >= power_tolerance:
        raise RuntimeError("S3 land-side balance residual exceeds tolerance")
    if float(np.abs(queue_residual).max(initial=0.0)) >= state_tolerance:
        raise RuntimeError("S3 flexible compute queue residual exceeds tolerance")
    if float(np.abs(rigid_residual).max(initial=0.0)) >= state_tolerance:
        raise RuntimeError("S3 rigid compute service residual exceeds tolerance")
    if abs(float(flex_queue[-1] - flex_queue[0])) >= state_tolerance:
        raise RuntimeError("S3 flexible compute queue terminal state does not close")
    if (it_power > compute.it_capacity_mw + 1e-7).any():
        raise RuntimeError("S3 IT capacity is exceeded")
    if (it_power > available_fiber_service_capacity + 1e-7).any():
        raise RuntimeError("S3 subsea-fibre service capacity is exceeded")
    deadline_slack = np.empty(0)
    if compute.flex_max_delay_h < periods:
        completed_cumulative = np.cumsum(flex_completed)
        arrivals_cumulative = np.cumsum(flex_arrival * dt)
        delay = compute.flex_max_delay_h
        deadline_slack = completed_cumulative[delay:] - arrivals_cumulative[: periods - delay]
        if float(deadline_slack.min()) < -state_tolerance:
            raise RuntimeError("S3 flexible compute maximum delay is violated")

    marginal_export_value = (
        price * (1.0 - loss_spec.terminal_loss_fraction)
        - tx_variable_cost
        + params.value("curtailment_penalty")
    )
    economic_mask = (
        (marginal_export_value <= 0.0) & (available_tx_capacity > 0.0) & (export_send <= 1e-7)
    )
    economic_curtailment = np.where(economic_mask, curtailment, 0.0)
    network_curtailment = curtailment - economic_curtailment
    feasible_export_limit = np.minimum(
        np.maximum(wind - critical_served - facility_power, 0.0), available_tx_capacity
    )

    electricity_revenue = price * flow.land_mw * dt
    transmission_variable_cost = tx_variable_cost * export_send * dt
    curtailment_cost = params.value("curtailment_penalty") * curtailment * dt
    unserved_critical_cost = params.value("unserved_critical_load_penalty") * unmet_critical * dt
    compute_gross_revenue = (
        rigid_completed * compute.rigid_service_price_cny_per_mwh_it
        + flex_completed * compute.flex_service_price_cny_per_mwh_it
    )
    compute_variable_cost = it_energy * compute.variable_cost_cny_per_mwh_it
    compute_sla_penalty = rigid_unserved * compute.rigid_sla_penalty_cny_per_mwh_it
    operating_margin = (
        electricity_revenue
        + compute_gross_revenue
        - transmission_variable_cost
        - curtailment_cost
        - unserved_critical_cost
        - compute_variable_cost
        - compute_sla_penalty
    )

    hourly = pd.DataFrame(
        {
            "timestamp": ts["timestamp"].to_numpy(),
            "scenario_id": ts["scenario_id"].to_numpy(),
            "electricity_price_cny_per_mwh": price,
            "wind_available_mw": wind,
            "critical_load_mw": critical,
            "critical_load_served_mw": critical_served,
            "unmet_critical_load_mw": unmet_critical,
            "renewable_surplus_mw": np.maximum(wind - critical_served, 0.0),
            "tx_availability": tx_availability,
            "tx_available_capacity_mw": available_tx_capacity,
            "fiber_availability": fiber_availability,
            "fiber_available_service_capacity_mw_it": available_fiber_service_capacity,
            "export_feasible_limit_mw": feasible_export_limit,
            "export_send_mw": export_send,
            "tx_terminal_loss_mw": flow.terminal_loss_mw,
            "tx_cable_loss_mw": flow.cable_loss_mw,
            "tx_total_loss_mw": flow.total_loss_mw,
            "tx_piecewise_land_mw": piecewise_land,
            "tx_linearization_error_mw": linearization_error,
            "export_land_mw": flow.land_mw,
            "rigid_compute_arrival_mwh_it": rigid_arrival * dt,
            "flex_compute_arrival_mwh_it": flex_arrival * dt,
            "rigid_compute_completed_mwh_it": rigid_completed,
            "rigid_compute_unserved_mwh_it": rigid_unserved,
            "flex_compute_completed_mwh_it": flex_completed,
            "flex_queue_start_mwh_it": flex_queue[:-1],
            "flex_queue_end_mwh_it": flex_queue[1:],
            "flex_queue_state_residual_mwh_it": queue_residual,
            "it_power_mw": it_power,
            "dc_facility_power_mw": facility_power,
            "network_curtailment_mw": network_curtailment,
            "economic_curtailment_mw": economic_curtailment,
            "curtailment_mw": curtailment,
            "offshore_balance_residual_mw": offshore_residual,
            "land_balance_residual_mw": land_residual,
            "electricity_revenue_cny": electricity_revenue,
            "tx_variable_cost_cny": transmission_variable_cost,
            "curtailment_penalty_cny": curtailment_cost,
            "unserved_critical_penalty_cny": unserved_critical_cost,
            "compute_gross_revenue_cny": compute_gross_revenue,
            "compute_variable_cost_cny": compute_variable_cost,
            "compute_sla_penalty_cny": compute_sla_penalty,
            "operating_margin_cny": operating_margin,
        }
    )
    kpis = calculate_s0_kpis(hourly, config, time_step_hours=dt)
    rigid_demand = float((rigid_arrival * dt).sum())
    flex_demand = float((flex_arrival * dt).sum())
    compute_demand = rigid_demand + flex_demand
    compute_service = float(it_energy.sum())
    fiber_available_energy = float((available_fiber_service_capacity * dt).sum())
    kpis.update(
        {
            "compute_service_mwh_it": compute_service,
            "compute_demand_mwh_it": compute_demand,
            "compute_service_rate": compute_service / compute_demand if compute_demand else 0.0,
            "rigid_compute_demand_mwh_it": rigid_demand,
            "rigid_compute_completed_mwh_it": float(rigid_completed.sum()),
            "rigid_compute_unserved_mwh_it": float(rigid_unserved.sum()),
            "rigid_compute_service_rate": (
                float(rigid_completed.sum() / rigid_demand) if rigid_demand else 0.0
            ),
            "flex_compute_demand_mwh_it": flex_demand,
            "flex_compute_completed_mwh_it": float(flex_completed.sum()),
            "flex_compute_service_rate": (
                float(flex_completed.sum() / flex_demand) if flex_demand else 0.0
            ),
            "flex_queue_min_mwh_it": float(flex_queue.min()),
            "flex_queue_max_mwh_it": float(flex_queue.max()),
            "flex_queue_terminal_error_mwh_it": float(flex_queue[-1] - flex_queue[0]),
            "max_flex_queue_state_residual_mwh_it": float(
                np.abs(queue_residual).max(initial=0.0)
            ),
            "minimum_flex_deadline_slack_mwh_it": float(deadline_slack.min())
            if len(deadline_slack)
            else 0.0,
            "compute_it_capacity_factor": float(
                compute_service / (compute.it_capacity_mw * periods * dt)
            ),
            "subsea_fiber_service_utilization": (
                compute_service / fiber_available_energy if fiber_available_energy else 0.0
            ),
            "data_center_pue": compute.pue,
            "compute_gross_revenue_cny": float(compute_gross_revenue.sum()),
            "compute_variable_cost_cny": float(compute_variable_cost.sum()),
            "compute_sla_penalty_cny": float(compute_sla_penalty.sum()),
            "compute_operating_margin_cny": float(
                compute_gross_revenue.sum()
                - compute_variable_cost.sum()
                - compute_sla_penalty.sum()
            ),
            "tx_linearization_error_mwh": float((linearization_error * dt).sum()),
            "max_tx_linearization_error_mw": float(linearization_error.max(initial=0.0)),
            "lp_objective_cny": float(solution.fun),
            "lp_iterations": int(solution.nit),
        }
    )
    fingerprint_payload = {
        "phase": "S3",
        "config": config.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "parameters": [item.model_dump(mode="json") for item in params.items],
    }
    metadata = {
        "phase": "S3",
        "model_version": __version__,
        "configuration_hash": configuration_hash(fingerprint_payload),
        "rows": periods,
        "time_step_hours": dt,
        "solver": "scipy.optimize.linprog.highs",
        "solver_status": int(solution.status),
        "solver_message": solution.message,
        "transmission_segments": segment_count,
        "compute_unit": "MWh-IT",
        "subsea_fiber_boundary": "service_egress_proxy_mwh_it_per_h",
    }
    return S3DispatchResult(hourly=hourly, kpis=kpis, metadata=metadata)
