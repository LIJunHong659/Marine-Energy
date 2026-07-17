"""KPI calculations for the S0 offshore wind export baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from blue_hub.schemas import SystemConfiguration


def calculate_s0_kpis(
    hourly: pd.DataFrame,
    config: SystemConfiguration,
    time_step_hours: float = 1.0,
) -> dict[str, Any]:
    """Aggregate energy, reliability, transmission and operating-value metrics."""
    if time_step_hours <= 0.0:
        raise ValueError("time_step_hours must be positive")
    hours = len(hourly) * time_step_hours

    def energy(column: str) -> float:
        return float(hourly[column].sum() * time_step_hours)

    wind = energy("wind_available_mw")
    critical = energy("critical_load_mw")
    send = energy("export_send_mw")
    loss = energy("tx_total_loss_mw")
    curtailment = energy("curtailment_mw")
    critical_served = energy("critical_load_served_mw")
    unmet = energy("unmet_critical_load_mw")

    return {
        "config_id": config.config_id,
        "scenario_id": str(hourly["scenario_id"].iloc[0]),
        "simulation_hours": hours,
        "renewable_generation_mwh": wind,
        "renewable_utilization_rate": (wind - curtailment) / wind if wind else 0.0,
        "curtailment_rate": curtailment / wind if wind else 0.0,
        "curtailment_mwh": curtailment,
        "network_curtailment_mwh": energy("network_curtailment_mw"),
        "economic_curtailment_mwh": energy("economic_curtailment_mw"),
        "critical_load_mwh": critical,
        "critical_load_served_mwh": critical_served,
        "eens_mwh": unmet,
        "lpsp": unmet / critical if critical else 0.0,
        "export_send_mwh": send,
        "export_land_mwh": energy("export_land_mw"),
        "transmission_loss_mwh": loss,
        "transmission_loss_rate_send_side": loss / send if send else 0.0,
        "transmission_capacity_factor": (
            send / (config.tx_capacity_mw * hours)
            if config.tx_capacity_mw > 0.0 and hours > 0.0
            else 0.0
        ),
        "negative_price_export_mwh": float(
            hourly.loc[hourly["electricity_price_cny_per_mwh"] < 0.0, "export_send_mw"].sum()
            * time_step_hours
        ),
        "electricity_revenue_cny": float(hourly["electricity_revenue_cny"].sum()),
        "tx_variable_cost_cny": float(hourly["tx_variable_cost_cny"].sum()),
        "curtailment_penalty_cny": float(hourly["curtailment_penalty_cny"].sum()),
        "unserved_critical_penalty_cny": float(hourly["unserved_critical_penalty_cny"].sum()),
        "operating_margin_cny": float(hourly["operating_margin_cny"].sum()),
        "max_offshore_balance_residual_mw": float(
            hourly["offshore_balance_residual_mw"].abs().max()
        ),
        "max_land_balance_residual_mw": float(hourly["land_balance_residual_mw"].abs().max()),
    }
