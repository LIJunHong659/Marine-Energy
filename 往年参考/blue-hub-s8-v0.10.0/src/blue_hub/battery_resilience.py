"""Event-only battery ride-through audit without perfect foresight or precharging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from blue_hub.battery import BatterySpec


@dataclass(frozen=True)
class BatteryResilienceResult:
    """Greedy critical-load support trajectory for an islanded event."""

    hourly: pd.DataFrame
    kpis: dict[str, float]


def simulate_islanded_critical_load(
    critical_load_mw: npt.ArrayLike,
    specification: BatterySpec,
    initial_soc: float,
    time_step_hours: float = 1.0,
) -> BatteryResilienceResult:
    """Discharge only as needed to serve critical load after an unforeseen outage."""
    specification.validate()
    if not specification.soc_min <= initial_soc <= specification.soc_max:
        raise ValueError("event initial SOC must lie within operational bounds")
    if time_step_hours <= 0.0:
        raise ValueError("time_step_hours must be positive")
    load = np.asarray(critical_load_mw, dtype=float)
    if load.ndim != 1 or not np.isfinite(load).all() or (load < 0.0).any():
        raise ValueError("critical load must be a finite non-negative vector")

    energy_start = np.zeros(len(load))
    energy_end = np.zeros(len(load))
    discharge = np.zeros(len(load))
    unmet = np.zeros(len(load))
    standing_loss = np.zeros(len(load))
    energy = initial_soc * specification.energy_mwh
    for t, demand in enumerate(load):
        energy_start[t] = energy
        standing_loss[t] = specification.self_discharge_per_hour * energy
        energy_after_standing = energy - standing_loss[t]
        deliverable_from_energy = max(
            0.0,
            (energy_after_standing - specification.minimum_energy_mwh)
            * specification.discharge_efficiency
            / time_step_hours,
        )
        discharge[t] = min(demand, specification.power_mw, deliverable_from_energy)
        unmet[t] = demand - discharge[t]
        energy = (
            energy_after_standing
            - discharge[t] * time_step_hours / specification.discharge_efficiency
        )
        energy_end[t] = energy

    shortfall_indices = np.flatnonzero(unmet > 1e-9)
    hours_before_first_shortfall = (
        float(shortfall_indices[0]) * time_step_hours
        if shortfall_indices.size
        else float(len(load)) * time_step_hours
    )
    hourly = pd.DataFrame(
        {
            "event_hour": np.arange(len(load)),
            "critical_load_mw": load,
            "battery_discharge_mw": discharge,
            "unmet_critical_load_mw": unmet,
            "battery_energy_start_mwh": energy_start,
            "battery_energy_end_mwh": energy_end,
            "battery_soc_start": energy_start / specification.energy_mwh,
            "battery_soc_end": energy_end / specification.energy_mwh,
            "battery_standing_loss_mwh": standing_loss,
        }
    )
    kpis = {
        "initial_soc": initial_soc,
        "hours_before_first_shortfall": hours_before_first_shortfall,
        "critical_load_mwh": float(load.sum() * time_step_hours),
        "served_energy_mwh": float(discharge.sum() * time_step_hours),
        "eens_mwh": float(unmet.sum() * time_step_hours),
        "final_soc": float(energy / specification.energy_mwh),
    }
    return BatteryResilienceResult(hourly=hourly, kpis=kpis)
