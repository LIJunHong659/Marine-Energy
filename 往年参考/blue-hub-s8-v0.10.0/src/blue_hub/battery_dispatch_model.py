"""Phase 2 / S1 battery-coupled hourly optimization model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from blue_hub import __version__
from blue_hub.battery import battery_spec_from_parameters, calculate_battery_losses
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
class S1DispatchResult:
    """Hourly S1 ledger, aggregate KPIs and solver provenance."""

    hourly: pd.DataFrame
    kpis: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _VariableLayout:
    periods: int
    segments: int
    charge_start: int
    discharge_start: int
    curtailment_start: int
    unmet_start: int
    energy_start: int
    variable_count: int


def _layout(periods: int, segments: int) -> _VariableLayout:
    segment_count = periods * segments
    charge_start = segment_count
    discharge_start = charge_start + periods
    curtailment_start = discharge_start + periods
    unmet_start = curtailment_start + periods
    energy_start = unmet_start + periods
    return _VariableLayout(
        periods=periods,
        segments=segments,
        charge_start=charge_start,
        discharge_start=discharge_start,
        curtailment_start=curtailment_start,
        unmet_start=unmet_start,
        energy_start=energy_start,
        variable_count=energy_start + periods + 1,
    )


def _validate_s1_configuration(config: SystemConfiguration) -> None:
    unsupported = {
        "pv_capacity_mw": config.pv_capacity_mw,
        "electrolyzer_power_mw": config.electrolyzer_power_mw,
        "hydrogen_storage_kg": config.hydrogen_storage_kg,
        "compute_it_capacity_mw": config.compute_it_capacity_mw,
        "subsea_fiber_service_capacity_mw_it": config.subsea_fiber_service_capacity_mw_it,
        "fuel_cell_power_mw": config.fuel_cell_power_mw,
    }
    nonzero = {name: value for name, value in unsupported.items() if value != 0.0}
    if nonzero:
        raise ValueError(f"S1 configuration contains future-phase capacities: {nonzero}")
    if config.export_policy != "economic":
        raise ValueError("S1 incremental transmission segments currently require economic export")


def _loss_spec(params: TechnologyParameters, technology: str) -> TransmissionLossSpec:
    return TransmissionLossSpec(
        terminal_loss_fraction=params.value(f"tx_terminal_loss_fraction_{technology}"),
        cable_full_load_loss_per_100km=params.value(
            f"tx_cable_full_load_loss_per_100km_{technology}"
        ),
    )


def _s0_ablation(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S1DispatchResult:
    s0 = run_s0_dispatch(timeseries, params, config, scenario)
    hourly = s0.hourly.copy()
    zeros = np.zeros(len(hourly))
    hourly["tx_piecewise_land_mw"] = hourly["export_land_mw"]
    hourly["tx_linearization_error_mw"] = zeros
    hourly["battery_charge_mw"] = zeros
    hourly["battery_discharge_mw"] = zeros
    hourly["battery_energy_start_mwh"] = zeros
    hourly["battery_energy_end_mwh"] = zeros
    hourly["battery_soc_start"] = zeros
    hourly["battery_soc_end"] = zeros
    hourly["battery_standing_loss_mwh"] = zeros
    hourly["battery_charge_loss_mwh"] = zeros
    hourly["battery_discharge_loss_mwh"] = zeros
    hourly["battery_total_loss_mwh"] = zeros
    hourly["battery_state_residual_mwh"] = zeros
    hourly["battery_degradation_cost_cny"] = zeros
    kpis = {
        **s0.kpis,
        "battery_charge_mwh": 0.0,
        "battery_discharge_mwh": 0.0,
        "battery_total_loss_mwh": 0.0,
        "battery_efc": 0.0,
        "battery_min_soc": 0.0,
        "battery_max_soc": 0.0,
        "battery_final_soc": 0.0,
        "battery_max_simultaneous_charge_discharge_mw": 0.0,
        "battery_terminal_state_error_mwh": 0.0,
        "max_battery_state_residual_mwh": 0.0,
        "tx_linearization_error_mwh": 0.0,
    }
    return S1DispatchResult(
        hourly=hourly,
        kpis=kpis,
        metadata={**s0.metadata, "phase": "S1_ablation", "solver": "S0_exact_fallback"},
    )


def run_s1_dispatch(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S1DispatchResult:
    """Optimize battery charge, discharge, export, curtailment and critical-load service."""
    validate_timeseries(timeseries).raise_if_invalid()
    validate_parameters(params).raise_if_invalid()
    _validate_s1_configuration(config)
    if config.battery_power_mw == 0.0 and config.battery_energy_mwh == 0.0:
        return _s0_ablation(timeseries, params, config, scenario)

    specification = battery_spec_from_parameters(config, params)
    dt = params.value("time_step")
    if dt != 1.0:
        raise ValueError("S1 currently supports a 1 h time step only")
    ts = apply_scenario(timeseries, scenario)
    periods = len(ts)
    wind = (
        config.wind_capacity_mw
        * ts["wind_cf"].to_numpy(dtype=float)
        * ts["wind_availability"].to_numpy(dtype=float)
    )
    critical = ts["critical_load"].to_numpy(dtype=float)
    price = ts["electricity_price"].to_numpy(dtype=float)
    tx_availability = ts["tx_availability"].to_numpy(dtype=float)
    available_tx_capacity = config.tx_capacity_mw * tx_availability
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
            column = t * segment_count + segment
            objective[column] = (tx_variable_cost - price[t] * slopes[segment]) * dt
    degradation_cost = specification.degradation_cost_cny_per_mwh_throughput
    objective[layout.charge_start : layout.charge_start + periods] = degradation_cost * dt
    objective[layout.discharge_start : layout.discharge_start + periods] = degradation_cost * dt
    objective[layout.curtailment_start : layout.curtailment_start + periods] = (
        params.value("curtailment_penalty") * dt
    )
    objective[layout.unmet_start : layout.unmet_start + periods] = (
        params.value("unserved_critical_load_penalty") * dt
    )

    bounds: list[tuple[float, float]] = []
    for _ in range(periods):
        bounds.extend((0.0, float(width)) for width in widths)
    bounds.extend((0.0, specification.power_mw) for _ in range(periods))
    bounds.extend((0.0, specification.power_mw) for _ in range(periods))
    bounds.extend((0.0, float(value)) for value in wind)
    bounds.extend((0.0, float(value)) for value in critical)
    for state in range(periods + 1):
        if state in (0, periods):
            initial = specification.initial_energy_mwh
            bounds.append((initial, initial))
        else:
            bounds.append(
                (
                    specification.scheduled_minimum_energy_mwh,
                    specification.maximum_energy_mwh,
                )
            )

    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_data: list[float] = []
    equality_rhs = np.zeros(2 * periods)
    retention = 1.0 - specification.self_discharge_per_hour
    for t in range(periods):
        for segment in range(segment_count):
            equality_rows.append(t)
            equality_columns.append(t * segment_count + segment)
            equality_data.append(1.0)
        equality_rows.extend([t, t, t, t])
        equality_columns.extend(
            [
                layout.charge_start + t,
                layout.curtailment_start + t,
                layout.discharge_start + t,
                layout.unmet_start + t,
            ]
        )
        equality_data.extend([1.0, 1.0, -1.0, -1.0])
        equality_rhs[t] = wind[t] - critical[t]

        state_row = periods + t
        equality_rows.extend([state_row, state_row, state_row, state_row])
        equality_columns.extend(
            [
                layout.energy_start + t + 1,
                layout.energy_start + t,
                layout.charge_start + t,
                layout.discharge_start + t,
            ]
        )
        equality_data.extend(
            [
                1.0,
                -retention,
                -specification.charge_efficiency * dt,
                dt / specification.discharge_efficiency,
            ]
        )
    equality_matrix = coo_matrix(
        (equality_data, (equality_rows, equality_columns)),
        shape=(2 * periods, layout.variable_count),
    ).tocsr()

    inequality_row_count = (periods if segment_count else 0) + (
        periods if specification.reserve_power_mw > 0.0 else 0
    )
    if inequality_row_count:
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
        if specification.reserve_power_mw > 0.0:
            headroom = specification.power_mw - specification.reserve_power_mw
            for t in range(periods):
                inequality_rows.extend([row, row])
                inequality_columns.extend([layout.discharge_start + t, layout.charge_start + t])
                inequality_data.extend([1.0, -1.0])
                inequality_rhs_values.append(headroom)
                row += 1
        inequality_matrix = coo_matrix(
            (inequality_data, (inequality_rows, inequality_columns)),
            shape=(inequality_row_count, layout.variable_count),
        ).tocsr()
        inequality_rhs = np.asarray(inequality_rhs_values)
    else:
        inequality_matrix = None
        inequality_rhs = None

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
            f"S1 linear optimization failed: status={solution.status}; {solution.message}"
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
    charge = values[layout.charge_start : layout.charge_start + periods]
    discharge = values[layout.discharge_start : layout.discharge_start + periods]
    curtailment = values[layout.curtailment_start : layout.curtailment_start + periods]
    unmet = values[layout.unmet_start : layout.unmet_start + periods]
    energy = values[layout.energy_start : layout.energy_start + periods + 1]

    simultaneous = np.minimum(charge, discharge)
    simultaneous_tolerance = params.value("battery_simultaneous_tolerance")
    if float(simultaneous.max(initial=0.0)) > simultaneous_tolerance:
        raise RuntimeError("S1 solution contains simultaneous battery charge and discharge")
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

    critical_served = critical - unmet
    offshore_residual = wind + discharge + unmet - critical - export_send - charge - curtailment
    land_residual = export_send - flow.land_mw - flow.total_loss_mw
    state_residual = (
        energy[1:]
        - retention * energy[:-1]
        - specification.charge_efficiency * charge * dt
        + discharge * dt / specification.discharge_efficiency
    )
    state_tolerance = params.value("battery_state_tolerance")
    terminal_error = energy[-1] - energy[0]
    if float(np.abs(state_residual).max(initial=0.0)) >= state_tolerance:
        raise RuntimeError("battery state equation residual exceeds tolerance")
    if abs(float(terminal_error)) >= state_tolerance:
        raise RuntimeError("battery terminal state does not match its initial state")

    battery_losses = calculate_battery_losses(energy[:-1], charge, discharge, specification, dt)
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
        np.maximum(wind + discharge - critical_served - charge, 0.0),
        available_tx_capacity,
    )

    electricity_revenue = price * flow.land_mw * dt
    transmission_variable_cost = tx_variable_cost * export_send * dt
    curtailment_cost = params.value("curtailment_penalty") * curtailment * dt
    unserved_cost = params.value("unserved_critical_load_penalty") * unmet * dt
    battery_degradation_cost = degradation_cost * (charge + discharge) * dt
    operating_margin = (
        electricity_revenue
        - transmission_variable_cost
        - curtailment_cost
        - unserved_cost
        - battery_degradation_cost
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
            "tx_availability": tx_availability,
            "tx_available_capacity_mw": available_tx_capacity,
            "export_feasible_limit_mw": feasible_export_limit,
            "export_send_mw": export_send,
            "tx_terminal_loss_mw": flow.terminal_loss_mw,
            "tx_cable_loss_mw": flow.cable_loss_mw,
            "tx_total_loss_mw": flow.total_loss_mw,
            "tx_piecewise_land_mw": piecewise_land,
            "tx_linearization_error_mw": linearization_error,
            "export_land_mw": flow.land_mw,
            "battery_charge_mw": charge,
            "battery_discharge_mw": discharge,
            "battery_energy_start_mwh": energy[:-1],
            "battery_energy_end_mwh": energy[1:],
            "battery_soc_start": energy[:-1] / specification.energy_mwh,
            "battery_soc_end": energy[1:] / specification.energy_mwh,
            "battery_standing_loss_mwh": battery_losses.standing_loss_mwh,
            "battery_charge_loss_mwh": battery_losses.charge_conversion_loss_mwh,
            "battery_discharge_loss_mwh": battery_losses.discharge_conversion_loss_mwh,
            "battery_total_loss_mwh": battery_losses.total_loss_mwh,
            "battery_state_residual_mwh": state_residual,
            "network_curtailment_mw": network_curtailment,
            "economic_curtailment_mw": economic_curtailment,
            "curtailment_mw": curtailment,
            "offshore_balance_residual_mw": offshore_residual,
            "land_balance_residual_mw": land_residual,
            "electricity_revenue_cny": electricity_revenue,
            "tx_variable_cost_cny": transmission_variable_cost,
            "curtailment_penalty_cny": curtailment_cost,
            "unserved_critical_penalty_cny": unserved_cost,
            "battery_degradation_cost_cny": battery_degradation_cost,
            "operating_margin_cny": operating_margin,
        }
    )
    power_tolerance = params.value("power_balance_tolerance")
    if float(np.abs(offshore_residual).max(initial=0.0)) >= power_tolerance:
        raise RuntimeError("S1 offshore power balance residual exceeds tolerance")
    if float(np.abs(land_residual).max(initial=0.0)) >= power_tolerance:
        raise RuntimeError("S1 land-side balance residual exceeds tolerance")

    kpis = calculate_s0_kpis(hourly, config, time_step_hours=dt)
    throughput = float(((charge + discharge) * dt).sum())
    kpis.update(
        {
            "battery_charge_mwh": float((charge * dt).sum()),
            "battery_discharge_mwh": float((discharge * dt).sum()),
            "battery_total_loss_mwh": float(battery_losses.total_loss_mwh.sum()),
            "battery_throughput_mwh": throughput,
            "battery_efc": throughput / (2.0 * specification.energy_mwh),
            "battery_min_soc": float((energy / specification.energy_mwh).min()),
            "battery_max_soc": float((energy / specification.energy_mwh).max()),
            "battery_final_soc": float(energy[-1] / specification.energy_mwh),
            "battery_reserve_power_mw": specification.reserve_power_mw,
            "battery_reserve_duration_h": specification.reserve_duration_h,
            "battery_reserve_energy_mwh": specification.reserve_energy_above_minimum_mwh,
            "battery_scheduled_minimum_soc": (
                specification.scheduled_minimum_energy_mwh / specification.energy_mwh
            ),
            "battery_max_simultaneous_charge_discharge_mw": float(simultaneous.max(initial=0.0)),
            "battery_terminal_state_error_mwh": float(terminal_error),
            "max_battery_state_residual_mwh": float(np.abs(state_residual).max(initial=0.0)),
            "battery_degradation_cost_cny": float(battery_degradation_cost.sum()),
            "tx_linearization_error_mwh": float((linearization_error * dt).sum()),
            "max_tx_linearization_error_mw": float(linearization_error.max(initial=0.0)),
            "lp_objective_cny": float(solution.fun),
            "lp_iterations": int(solution.nit),
        }
    )
    fingerprint_payload = {
        "phase": "S1",
        "config": config.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "parameters": [item.model_dump(mode="json") for item in params.items],
    }
    metadata = {
        "phase": "S1",
        "model_version": __version__,
        "configuration_hash": configuration_hash(fingerprint_payload),
        "rows": periods,
        "time_step_hours": dt,
        "solver": "scipy.optimize.linprog.highs",
        "solver_status": int(solution.status),
        "solver_message": solution.message,
        "transmission_segments": segment_count,
    }
    return S1DispatchResult(hourly=hourly, kpis=kpis, metadata=metadata)
