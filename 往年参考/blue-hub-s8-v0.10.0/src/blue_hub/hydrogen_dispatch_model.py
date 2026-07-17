"""Phase 3 / S2 co-optimization of export, battery, hydrogen and inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from blue_hub import __version__
from blue_hub.battery import BatterySpec, battery_spec_from_parameters, calculate_battery_losses
from blue_hub.battery_dispatch_model import run_s1_dispatch
from blue_hub.hydrogen import hydrogen_spec_from_parameters
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
class S2DispatchResult:
    """Hourly S2 ledger, KPIs and deterministic solver provenance."""

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
    hydrogen_power_start: int
    hydrogen_sale_start: int
    battery_energy_start: int
    hydrogen_inventory_start: int
    variable_count: int


def _layout(periods: int, segments: int) -> _VariableLayout:
    segment_count = periods * segments
    charge_start = segment_count
    discharge_start = charge_start + periods
    curtailment_start = discharge_start + periods
    unmet_start = curtailment_start + periods
    hydrogen_power_start = unmet_start + periods
    hydrogen_sale_start = hydrogen_power_start + periods
    battery_energy_start = hydrogen_sale_start + periods
    hydrogen_inventory_start = battery_energy_start + periods + 1
    return _VariableLayout(
        periods=periods,
        segments=segments,
        charge_start=charge_start,
        discharge_start=discharge_start,
        curtailment_start=curtailment_start,
        unmet_start=unmet_start,
        hydrogen_power_start=hydrogen_power_start,
        hydrogen_sale_start=hydrogen_sale_start,
        battery_energy_start=battery_energy_start,
        hydrogen_inventory_start=hydrogen_inventory_start,
        variable_count=hydrogen_inventory_start + periods + 1,
    )


def _loss_spec(params: TechnologyParameters, technology: str) -> TransmissionLossSpec:
    return TransmissionLossSpec(
        terminal_loss_fraction=params.value(f"tx_terminal_loss_fraction_{technology}"),
        cable_full_load_loss_per_100km=params.value(
            f"tx_cable_full_load_loss_per_100km_{technology}"
        ),
    )


def _validate_s2_configuration(config: SystemConfiguration) -> None:
    unsupported = {
        "pv_capacity_mw": config.pv_capacity_mw,
        "compute_it_capacity_mw": config.compute_it_capacity_mw,
        "subsea_fiber_service_capacity_mw_it": config.subsea_fiber_service_capacity_mw_it,
        "fuel_cell_power_mw": config.fuel_cell_power_mw,
    }
    nonzero = {name: value for name, value in unsupported.items() if value != 0.0}
    if nonzero:
        raise ValueError(f"S2 configuration contains future-phase capacities: {nonzero}")
    if config.export_policy != "economic":
        raise ValueError("S2 incremental transmission segments currently require economic export")
    if config.electrolyzer_power_mw == 0.0 and config.hydrogen_storage_kg != 0.0:
        raise ValueError("hydrogen storage requires positive electrolyzer power in S2")


def _s1_ablation(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S2DispatchResult:
    """Retain exact S1 results when no hydrogen capacity is installed."""
    s1 = run_s1_dispatch(timeseries, params, config, scenario)
    ts = apply_scenario(timeseries, scenario)
    hourly = s1.hourly.copy()
    zeros = np.zeros(len(hourly))
    hourly["hydrogen_demand_kg"] = ts["hydrogen_demand"].to_numpy(dtype=float)
    hourly["electrolyzer_power_mw"] = zeros
    hourly["hydrogen_production_kg"] = zeros
    hourly["hydrogen_sale_kg"] = zeros
    hourly["hydrogen_unsatisfied_demand_kg"] = hourly["hydrogen_demand_kg"]
    hourly["hydrogen_inventory_start_kg"] = zeros
    hourly["hydrogen_inventory_end_kg"] = zeros
    hourly["hydrogen_storage_loss_kg"] = zeros
    hourly["hydrogen_state_residual_kg"] = zeros
    hourly["hydrogen_water_m3"] = zeros
    hourly["hydrogen_gross_revenue_cny"] = zeros
    hourly["hydrogen_transport_cost_cny"] = zeros
    hourly["electrolyzer_variable_cost_cny"] = zeros
    hourly["desalinated_water_cost_cny"] = zeros
    kpis = {
        **s1.kpis,
        "hydrogen_production_kg": 0.0,
        "hydrogen_sales_kg": 0.0,
        "hydrogen_demand_kg": float(hourly["hydrogen_demand_kg"].sum()),
        "hydrogen_unsatisfied_demand_kg": float(hourly["hydrogen_demand_kg"].sum()),
        "hydrogen_service_rate": 0.0,
        "hydrogen_storage_loss_kg": 0.0,
        "hydrogen_water_m3": 0.0,
        "hydrogen_inventory_min_kg": 0.0,
        "hydrogen_inventory_max_kg": 0.0,
        "hydrogen_initial_inventory_kg": 0.0,
        "hydrogen_final_inventory_kg": 0.0,
        "hydrogen_terminal_state_error_kg": 0.0,
        "max_hydrogen_state_residual_kg": 0.0,
        "electrolyzer_capacity_factor": 0.0,
        "hydrogen_sec_system_kwh_per_kg": params.value("hydrogen_sec_system"),
        "hydrogen_gross_revenue_cny": 0.0,
        "hydrogen_transport_cost_cny": 0.0,
        "electrolyzer_variable_cost_cny": 0.0,
        "desalinated_water_cost_cny": 0.0,
        "hydrogen_operating_margin_cny": 0.0,
    }
    return S2DispatchResult(
        hourly=hourly,
        kpis=kpis,
        metadata={**s1.metadata, "phase": "S2_ablation", "solver": "S1_exact_fallback"},
    )


def _battery_arrays(
    values: np.ndarray,
    layout: _VariableLayout,
    specification: BatterySpec | None,
    time_step_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float, float]:
    periods = layout.periods
    charge = values[layout.charge_start : layout.charge_start + periods]
    discharge = values[layout.discharge_start : layout.discharge_start + periods]
    energy = values[layout.battery_energy_start : layout.battery_energy_start + periods + 1]
    if specification is None:
        zeros = np.zeros(periods)
        return charge, discharge, energy, {
            "standing": zeros,
            "charge": zeros,
            "discharge": zeros,
            "total": zeros,
        }, 0.0, 0.0
    losses = calculate_battery_losses(
        energy[:-1], charge, discharge, specification, time_step_hours
    )
    state_residual = (
        energy[1:]
        - (1.0 - specification.self_discharge_per_hour) * energy[:-1]
        - specification.charge_efficiency * charge * time_step_hours
        + discharge * time_step_hours / specification.discharge_efficiency
    )
    return charge, discharge, energy, {
        "standing": losses.standing_loss_mwh,
        "charge": losses.charge_conversion_loss_mwh,
        "discharge": losses.discharge_conversion_loss_mwh,
        "total": losses.total_loss_mwh,
    }, float(np.abs(state_residual).max(initial=0.0)), float(energy[-1] - energy[0])


def run_s2_dispatch(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S2DispatchResult:
    """Jointly optimize power balance, export, battery and hydrogen inventory.

    Hydrogen sales are limited by hourly demand.  Initial and terminal inventory
    are fixed to the same declared level, so a final inventory build cannot
    masquerade as annual revenue or shift energy across the simulation boundary.
    """
    validate_timeseries(timeseries).raise_if_invalid()
    validate_parameters(params).raise_if_invalid()
    _validate_s2_configuration(config)
    if config.electrolyzer_power_mw == 0.0:
        return _s1_ablation(timeseries, params, config, scenario)

    dt = params.value("time_step")
    if dt != 1.0:
        raise ValueError("S2 currently supports a 1 h time step only")
    hydrogen = hydrogen_spec_from_parameters(config, params, scenario)
    battery: BatterySpec | None = None
    if config.battery_power_mw > 0.0:
        battery = battery_spec_from_parameters(config, params)

    ts = apply_scenario(timeseries, scenario)
    periods = len(ts)
    wind = (
        config.wind_capacity_mw
        * ts["wind_cf"].to_numpy(dtype=float)
        * ts["wind_availability"].to_numpy(dtype=float)
    )
    critical = ts["critical_load"].to_numpy(dtype=float)
    price = ts["electricity_price"].to_numpy(dtype=float)
    demand = ts["hydrogen_demand"].to_numpy(dtype=float)
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
            objective[t * segment_count + segment] = (
                tx_variable_cost - price[t] * slopes[segment]
            ) * dt
    if battery is not None:
        objective[layout.charge_start : layout.charge_start + periods] = (
            battery.degradation_cost_cny_per_mwh_throughput * dt
        )
        objective[layout.discharge_start : layout.discharge_start + periods] = (
            battery.degradation_cost_cny_per_mwh_throughput * dt
        )
    objective[layout.curtailment_start : layout.curtailment_start + periods] = (
        params.value("curtailment_penalty") * dt
    )
    objective[layout.unmet_start : layout.unmet_start + periods] = (
        params.value("unserved_critical_load_penalty") * dt
    )
    objective[layout.hydrogen_power_start : layout.hydrogen_power_start + periods] = (
        hydrogen.conversion_and_water_cost_cny_per_kg * hydrogen.kg_per_mwh * dt
    )
    objective[layout.hydrogen_sale_start : layout.hydrogen_sale_start + periods] = (
        -hydrogen.net_sale_value_cny_per_kg
    )

    bounds: list[tuple[float, float]] = []
    for _ in range(periods):
        bounds.extend((0.0, float(width)) for width in widths)
    battery_power = battery.power_mw if battery is not None else 0.0
    bounds.extend((0.0, battery_power) for _ in range(periods))
    bounds.extend((0.0, battery_power) for _ in range(periods))
    bounds.extend((0.0, float(value)) for value in wind)
    bounds.extend((0.0, float(value)) for value in critical)
    bounds.extend((0.0, hydrogen.power_mw) for _ in range(periods))
    bounds.extend((0.0, float(value * dt)) for value in demand)
    if battery is None:
        bounds.extend((0.0, 0.0) for _ in range(periods + 1))
    else:
        for state in range(periods + 1):
            if state in (0, periods):
                bounds.append((battery.initial_energy_mwh, battery.initial_energy_mwh))
            else:
                bounds.append(
                    (battery.scheduled_minimum_energy_mwh, battery.maximum_energy_mwh)
                )
    for state in range(periods + 1):
        if state in (0, periods):
            bounds.append((hydrogen.initial_inventory_kg, hydrogen.initial_inventory_kg))
        else:
            bounds.append((0.0, hydrogen.storage_capacity_kg))

    battery_rows = periods if battery is not None else 0
    total_rows = periods + battery_rows + periods
    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_data: list[float] = []
    equality_rhs = np.zeros(total_rows)
    for t in range(periods):
        for segment in range(segment_count):
            equality_rows.append(t)
            equality_columns.append(t * segment_count + segment)
            equality_data.append(1.0)
        equality_rows.extend([t, t, t, t, t])
        equality_columns.extend(
            [
                layout.charge_start + t,
                layout.curtailment_start + t,
                layout.hydrogen_power_start + t,
                layout.discharge_start + t,
                layout.unmet_start + t,
            ]
        )
        equality_data.extend([1.0, 1.0, 1.0, -1.0, -1.0])
        equality_rhs[t] = wind[t] - critical[t]

        if battery is not None:
            state_row = periods + t
            retention = 1.0 - battery.self_discharge_per_hour
            equality_rows.extend([state_row, state_row, state_row, state_row])
            equality_columns.extend(
                [
                    layout.battery_energy_start + t + 1,
                    layout.battery_energy_start + t,
                    layout.charge_start + t,
                    layout.discharge_start + t,
                ]
            )
            equality_data.extend(
                [
                    1.0,
                    -retention,
                    -battery.charge_efficiency * dt,
                    dt / battery.discharge_efficiency,
                ]
            )

        hydrogen_row = periods + battery_rows + t
        retention_hydrogen = 1.0 - hydrogen.storage_loss_per_hour
        equality_rows.extend([hydrogen_row, hydrogen_row, hydrogen_row, hydrogen_row])
        equality_columns.extend(
            [
                layout.hydrogen_inventory_start + t + 1,
                layout.hydrogen_inventory_start + t,
                layout.hydrogen_power_start + t,
                layout.hydrogen_sale_start + t,
            ]
        )
        equality_data.extend([1.0, -retention_hydrogen, -hydrogen.kg_per_mwh * dt, 1.0])
    equality_matrix = coo_matrix(
        (equality_data, (equality_rows, equality_columns)),
        shape=(total_rows, layout.variable_count),
    ).tocsr()

    inequality_row_count = (periods if segment_count else 0) + (
        periods if battery is not None and battery.reserve_power_mw > 0.0 else 0
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
        if battery is not None and battery.reserve_power_mw > 0.0:
            headroom = battery.power_mw - battery.reserve_power_mw
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
            f"S2 linear optimization failed: status={solution.status}; {solution.message}"
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
    charge, discharge, energy, battery_losses, max_battery_residual, battery_terminal_error = (
        _battery_arrays(values, layout, battery, dt)
    )
    curtailment = values[layout.curtailment_start : layout.curtailment_start + periods]
    unmet = values[layout.unmet_start : layout.unmet_start + periods]
    hydrogen_power = values[
        layout.hydrogen_power_start : layout.hydrogen_power_start + periods
    ]
    hydrogen_sales = values[
        layout.hydrogen_sale_start : layout.hydrogen_sale_start + periods
    ]
    inventory = values[
        layout.hydrogen_inventory_start : layout.hydrogen_inventory_start + periods + 1
    ]
    hydrogen_production = hydrogen_power * hydrogen.kg_per_mwh * dt
    hydrogen_storage_loss = hydrogen.storage_loss_per_hour * inventory[:-1]
    hydrogen_state_residual = (
        inventory[1:]
        - (1.0 - hydrogen.storage_loss_per_hour) * inventory[:-1]
        - hydrogen_production
        + hydrogen_sales
    )

    simultaneous = np.minimum(charge, discharge)
    simultaneous_tolerance = params.value("battery_simultaneous_tolerance")
    if float(simultaneous.max(initial=0.0)) > simultaneous_tolerance:
        raise RuntimeError("S2 solution contains simultaneous battery charge and discharge")
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
    offshore_residual = (
        wind
        + discharge
        + unmet
        - critical
        - export_send
        - charge
        - hydrogen_power
        - curtailment
    )
    land_residual = export_send - flow.land_mw - flow.total_loss_mw
    power_tolerance = params.value("power_balance_tolerance")
    hydrogen_tolerance = params.value("hydrogen_state_tolerance")
    if float(np.abs(offshore_residual).max(initial=0.0)) >= power_tolerance:
        raise RuntimeError("S2 offshore power balance residual exceeds tolerance")
    if float(np.abs(land_residual).max(initial=0.0)) >= power_tolerance:
        raise RuntimeError("S2 land-side balance residual exceeds tolerance")
    if max_battery_residual >= params.value("battery_state_tolerance"):
        raise RuntimeError("S2 battery state equation residual exceeds tolerance")
    if abs(battery_terminal_error) >= params.value("battery_state_tolerance"):
        raise RuntimeError("S2 battery terminal state does not match its initial state")
    if float(np.abs(hydrogen_state_residual).max(initial=0.0)) >= hydrogen_tolerance:
        raise RuntimeError("S2 hydrogen inventory equation residual exceeds tolerance")
    hydrogen_terminal_error = float(inventory[-1] - inventory[0])
    if abs(hydrogen_terminal_error) >= hydrogen_tolerance:
        raise RuntimeError("S2 hydrogen terminal inventory does not match its initial inventory")

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
        np.maximum(wind + discharge - critical_served - charge - hydrogen_power, 0.0),
        available_tx_capacity,
    )

    electricity_revenue = price * flow.land_mw * dt
    transmission_variable_cost = tx_variable_cost * export_send * dt
    curtailment_cost = params.value("curtailment_penalty") * curtailment * dt
    unserved_cost = params.value("unserved_critical_load_penalty") * unmet * dt
    battery_degradation_cost = (
        (battery.degradation_cost_cny_per_mwh_throughput if battery is not None else 0.0)
        * (charge + discharge)
        * dt
    )
    hydrogen_gross_revenue = hydrogen_sales * hydrogen.sale_price_cny_per_kg
    hydrogen_transport_cost = hydrogen_sales * hydrogen.transport_cost_cny_per_kg
    electrolyzer_variable_cost = hydrogen_production * hydrogen.variable_cost_cny_per_kg
    hydrogen_water = hydrogen_production * hydrogen.water_consumption_m3_per_kg
    desalinated_water_cost = hydrogen_water * hydrogen.water_cost_cny_per_m3
    operating_margin = (
        electricity_revenue
        + hydrogen_gross_revenue
        - transmission_variable_cost
        - curtailment_cost
        - unserved_cost
        - battery_degradation_cost
        - hydrogen_transport_cost
        - electrolyzer_variable_cost
        - desalinated_water_cost
    )

    battery_energy_capacity = battery.energy_mwh if battery is not None else 0.0
    if battery is not None:
        battery_soc_start = energy[:-1] / battery_energy_capacity
        battery_soc_end = energy[1:] / battery_energy_capacity
    else:
        battery_soc_start = np.zeros(periods)
        battery_soc_end = np.zeros(periods)
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
            "battery_soc_start": battery_soc_start,
            "battery_soc_end": battery_soc_end,
            "battery_standing_loss_mwh": battery_losses["standing"],
            "battery_charge_loss_mwh": battery_losses["charge"],
            "battery_discharge_loss_mwh": battery_losses["discharge"],
            "battery_total_loss_mwh": battery_losses["total"],
            "hydrogen_demand_kg": demand * dt,
            "electrolyzer_power_mw": hydrogen_power,
            "hydrogen_production_kg": hydrogen_production,
            "hydrogen_sale_kg": hydrogen_sales,
            "hydrogen_unsatisfied_demand_kg": demand * dt - hydrogen_sales,
            "hydrogen_inventory_start_kg": inventory[:-1],
            "hydrogen_inventory_end_kg": inventory[1:],
            "hydrogen_storage_loss_kg": hydrogen_storage_loss,
            "hydrogen_state_residual_kg": hydrogen_state_residual,
            "hydrogen_water_m3": hydrogen_water,
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
            "hydrogen_gross_revenue_cny": hydrogen_gross_revenue,
            "hydrogen_transport_cost_cny": hydrogen_transport_cost,
            "electrolyzer_variable_cost_cny": electrolyzer_variable_cost,
            "desalinated_water_cost_cny": desalinated_water_cost,
            "operating_margin_cny": operating_margin,
        }
    )

    kpis = calculate_s0_kpis(hourly, config, time_step_hours=dt)
    throughput = float(((charge + discharge) * dt).sum())
    demand_total = float((demand * dt).sum())
    hydrogen_operating_margin = float(
        hydrogen_gross_revenue.sum()
        - hydrogen_transport_cost.sum()
        - electrolyzer_variable_cost.sum()
        - desalinated_water_cost.sum()
    )
    kpis.update(
        {
            "battery_charge_mwh": float((charge * dt).sum()),
            "battery_discharge_mwh": float((discharge * dt).sum()),
            "battery_total_loss_mwh": float(battery_losses["total"].sum()),
            "battery_throughput_mwh": throughput,
            "battery_efc": (
                throughput / (2.0 * battery.energy_mwh) if battery is not None else 0.0
            ),
            "battery_min_soc": float(np.min(np.concatenate([battery_soc_start, battery_soc_end]))),
            "battery_max_soc": float(np.max(np.concatenate([battery_soc_start, battery_soc_end]))),
            "battery_final_soc": float(battery_soc_end[-1]) if battery is not None else 0.0,
            "battery_reserve_power_mw": battery.reserve_power_mw if battery is not None else 0.0,
            "battery_reserve_duration_h": (
                battery.reserve_duration_h if battery is not None else 0.0
            ),
            "battery_reserve_energy_mwh": (
                battery.reserve_energy_above_minimum_mwh if battery is not None else 0.0
            ),
            "battery_scheduled_minimum_soc": (
                battery.scheduled_minimum_energy_mwh / battery.energy_mwh
                if battery is not None
                else 0.0
            ),
            "battery_max_simultaneous_charge_discharge_mw": float(simultaneous.max(initial=0.0)),
            "battery_terminal_state_error_mwh": battery_terminal_error,
            "max_battery_state_residual_mwh": max_battery_residual,
            "battery_degradation_cost_cny": float(battery_degradation_cost.sum()),
            "tx_linearization_error_mwh": float((linearization_error * dt).sum()),
            "max_tx_linearization_error_mw": float(linearization_error.max(initial=0.0)),
            "hydrogen_production_kg": float(hydrogen_production.sum()),
            "hydrogen_sales_kg": float(hydrogen_sales.sum()),
            "hydrogen_demand_kg": demand_total,
            "hydrogen_unsatisfied_demand_kg": float(demand_total - hydrogen_sales.sum()),
            "hydrogen_service_rate": float(hydrogen_sales.sum() / demand_total)
            if demand_total > 0.0
            else 0.0,
            "hydrogen_storage_loss_kg": float(hydrogen_storage_loss.sum()),
            "hydrogen_water_m3": float(hydrogen_water.sum()),
            "hydrogen_inventory_min_kg": float(inventory.min()),
            "hydrogen_inventory_max_kg": float(inventory.max()),
            "hydrogen_initial_inventory_kg": float(inventory[0]),
            "hydrogen_final_inventory_kg": float(inventory[-1]),
            "hydrogen_terminal_state_error_kg": hydrogen_terminal_error,
            "max_hydrogen_state_residual_kg": float(
                np.abs(hydrogen_state_residual).max(initial=0.0)
            ),
            "electrolyzer_capacity_factor": float(
                (hydrogen_power * dt).sum() / (hydrogen.power_mw * periods * dt)
            ),
            "hydrogen_sec_system_kwh_per_kg": hydrogen.system_sec_kwh_per_kg,
            "hydrogen_gross_revenue_cny": float(hydrogen_gross_revenue.sum()),
            "hydrogen_transport_cost_cny": float(hydrogen_transport_cost.sum()),
            "electrolyzer_variable_cost_cny": float(electrolyzer_variable_cost.sum()),
            "desalinated_water_cost_cny": float(desalinated_water_cost.sum()),
            "hydrogen_operating_margin_cny": hydrogen_operating_margin,
            "lp_objective_cny": float(solution.fun),
            "lp_iterations": int(solution.nit),
        }
    )
    fingerprint_payload = {
        "phase": "S2",
        "config": config.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "parameters": [item.model_dump(mode="json") for item in params.items],
    }
    metadata = {
        "phase": "S2",
        "model_version": __version__,
        "configuration_hash": configuration_hash(fingerprint_payload),
        "rows": periods,
        "time_step_hours": dt,
        "solver": "scipy.optimize.linprog.highs",
        "solver_status": int(solution.status),
        "solver_message": solution.message,
        "transmission_segments": segment_count,
        "hydrogen_sec_boundary": "integrated_electrolyzer_system_only_once",
    }
    return S2DispatchResult(hourly=hourly, kpis=kpis, metadata=metadata)
