"""Analytical Phase 1 / S0 dispatch baseline and executable optimization oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from blue_hub import __version__
from blue_hub.metrics import calculate_s0_kpis
from blue_hub.provenance import configuration_hash
from blue_hub.scenario_runner import apply_scenario
from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters
from blue_hub.transmission import (
    TransmissionLossSpec,
    calculate_transmission_flow,
    optimal_export_send,
)
from blue_hub.validation import validate_parameters, validate_timeseries


@dataclass(frozen=True)
class S0DispatchResult:
    """Hourly S0 ledger, aggregated KPIs and deterministic run metadata."""

    hourly: pd.DataFrame
    kpis: dict[str, Any]
    metadata: dict[str, Any]


def _validate_s0_configuration(config: SystemConfiguration) -> None:
    unsupported = {
        "pv_capacity_mw": config.pv_capacity_mw,
        "battery_power_mw": config.battery_power_mw,
        "battery_energy_mwh": config.battery_energy_mwh,
        "electrolyzer_power_mw": config.electrolyzer_power_mw,
        "hydrogen_storage_kg": config.hydrogen_storage_kg,
        "compute_it_capacity_mw": config.compute_it_capacity_mw,
        "subsea_fiber_service_capacity_mw_it": config.subsea_fiber_service_capacity_mw_it,
        "fuel_cell_power_mw": config.fuel_cell_power_mw,
    }
    nonzero = {name: value for name, value in unsupported.items() if value != 0.0}
    if nonzero:
        raise ValueError(f"S0 configuration contains unsupported nonzero capacities: {nonzero}")


def _loss_spec(params: TechnologyParameters, technology: str) -> TransmissionLossSpec:
    return TransmissionLossSpec(
        terminal_loss_fraction=params.value(f"tx_terminal_loss_fraction_{technology}"),
        cable_full_load_loss_per_100km=params.value(
            f"tx_cable_full_load_loss_per_100km_{technology}"
        ),
    )


def run_s0_dispatch(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> S0DispatchResult:
    """Run the wind-load-export-curtailment baseline with two explicit balances."""
    validate_timeseries(timeseries).raise_if_invalid()
    validate_parameters(params).raise_if_invalid()
    _validate_s0_configuration(config)
    time_step_hours = params.value("time_step")
    if time_step_hours != 1.0:
        raise ValueError("S0 currently supports a 1 h time step only")

    ts = apply_scenario(timeseries, scenario)
    wind_available = (
        config.wind_capacity_mw
        * ts["wind_cf"].to_numpy(dtype=float)
        * ts["wind_availability"].to_numpy(dtype=float)
    )
    critical_load = ts["critical_load"].to_numpy(dtype=float)
    critical_served = np.minimum(wind_available, critical_load)
    unmet_critical = critical_load - critical_served
    surplus = wind_available - critical_served

    tx_availability = ts["tx_availability"].to_numpy(dtype=float)
    available_tx_capacity = config.tx_capacity_mw * tx_availability
    feasible_export_limit = np.minimum(surplus, available_tx_capacity)
    price = ts["electricity_price"].to_numpy(dtype=float)
    spec = _loss_spec(params, config.tx_technology)
    export_send = optimal_export_send(
        surplus_mw=surplus,
        available_capacity_mw=available_tx_capacity,
        electricity_price_cny_per_mwh=price,
        installed_capacity_mw=config.tx_capacity_mw,
        distance_km=scenario.offshore_distance_km,
        loss_spec=spec,
        variable_cost_cny_per_mwh_send=params.value("tx_variable_cost"),
        curtailment_penalty_cny_per_mwh=params.value("curtailment_penalty"),
        policy=config.export_policy,
    )
    flow = calculate_transmission_flow(
        export_send,
        capacity_mw=config.tx_capacity_mw,
        distance_km=scenario.offshore_distance_km,
        loss_spec=spec,
    )

    network_curtailment = np.maximum(surplus - feasible_export_limit, 0.0)
    economic_curtailment = np.maximum(feasible_export_limit - export_send, 0.0)
    curtailment = network_curtailment + economic_curtailment
    offshore_residual = wind_available - critical_served - export_send - curtailment
    land_residual = export_send - flow.land_mw - flow.total_loss_mw

    dt = time_step_hours
    electricity_revenue = price * flow.land_mw * dt
    tx_variable_cost = params.value("tx_variable_cost") * export_send * dt
    curtailment_cost = params.value("curtailment_penalty") * curtailment * dt
    unserved_cost = params.value("unserved_critical_load_penalty") * unmet_critical * dt
    operating_margin = electricity_revenue - tx_variable_cost - curtailment_cost - unserved_cost

    hourly = pd.DataFrame(
        {
            "timestamp": ts["timestamp"].to_numpy(),
            "scenario_id": ts["scenario_id"].to_numpy(),
            "electricity_price_cny_per_mwh": price,
            "wind_available_mw": wind_available,
            "critical_load_mw": critical_load,
            "critical_load_served_mw": critical_served,
            "unmet_critical_load_mw": unmet_critical,
            "renewable_surplus_mw": surplus,
            "tx_availability": tx_availability,
            "tx_available_capacity_mw": available_tx_capacity,
            "export_feasible_limit_mw": feasible_export_limit,
            "export_send_mw": export_send,
            "tx_terminal_loss_mw": flow.terminal_loss_mw,
            "tx_cable_loss_mw": flow.cable_loss_mw,
            "tx_total_loss_mw": flow.total_loss_mw,
            "export_land_mw": flow.land_mw,
            "network_curtailment_mw": network_curtailment,
            "economic_curtailment_mw": economic_curtailment,
            "curtailment_mw": curtailment,
            "offshore_balance_residual_mw": offshore_residual,
            "land_balance_residual_mw": land_residual,
            "electricity_revenue_cny": electricity_revenue,
            "tx_variable_cost_cny": tx_variable_cost,
            "curtailment_penalty_cny": curtailment_cost,
            "unserved_critical_penalty_cny": unserved_cost,
            "operating_margin_cny": operating_margin,
        }
    )
    tolerance = params.value("power_balance_tolerance")
    max_residual = max(
        float(np.max(np.abs(offshore_residual))),
        float(np.max(np.abs(land_residual))),
    )
    if max_residual >= tolerance:
        raise RuntimeError(
            f"S0 power balance residual {max_residual:.3e} MW exceeds {tolerance:.3e} MW"
        )

    kpis = calculate_s0_kpis(hourly, config, time_step_hours=time_step_hours)
    fingerprint_payload = {
        "phase": "S0",
        "config": config.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "parameters": [item.model_dump(mode="json") for item in params.items],
    }
    metadata = {
        "phase": "S0",
        "model_version": __version__,
        "configuration_hash": configuration_hash(fingerprint_payload),
        "rows": len(hourly),
        "time_step_hours": time_step_hours,
        "dispatch_method": "analytical_hourly_profit_maximization",
    }
    return S0DispatchResult(hourly=hourly, kpis=kpis, metadata=metadata)
