"""Deterministic synthetic inputs for contract and regression testing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_timeseries(
    hours: int = 24,
    start: str = "2026-01-01 00:00:00+08:00",
    scenario_id: str = "base",
) -> pd.DataFrame:
    """Generate repeatable hourly profiles without pretending they are site observations."""
    if hours <= 0:
        raise ValueError("hours must be positive")
    if not scenario_id.strip():
        raise ValueError("scenario_id must not be blank")

    timestamp = pd.date_range(start=start, periods=hours, freq="h")
    t = np.arange(hours, dtype=float)
    hour_of_day = timestamp.hour.to_numpy(dtype=float)

    wind_cf = np.clip(
        0.47
        + 0.16 * np.sin(2.0 * np.pi * (t + 3.0) / 24.0)
        + 0.08 * np.sin(2.0 * np.pi * t / 168.0),
        0.08,
        0.88,
    )
    daylight_shape = np.sin(np.pi * (hour_of_day - 6.0) / 12.0)
    pv_cf = 0.72 * np.clip(daylight_shape, 0.0, None)

    electricity_price = np.full(hours, 360.0)
    electricity_price[(hour_of_day >= 0) & (hour_of_day < 7)] = 240.0
    electricity_price[(hour_of_day >= 17) & (hour_of_day < 22)] = 620.0

    critical_load = 10.0 + 0.8 * np.sin(2.0 * np.pi * (t - 4.0) / 24.0)
    rigid_arrival = np.full(hours, 20.0)
    flex_arrival = 8.0 + 4.0 * (wind_cf > 0.55)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "wind_cf": wind_cf,
            "pv_cf": pv_cf,
            "electricity_price": electricity_price,
            "grid_carbon_intensity": np.full(hours, 0.55),
            "critical_load": critical_load,
            "rigid_compute_arrival": rigid_arrival,
            "flex_compute_arrival": flex_arrival,
            "hydrogen_demand": np.full(hours, 1_000.0),
            "tx_availability": np.ones(hours),
            "fiber_availability": np.ones(hours),
            "wind_availability": np.full(hours, 0.97),
            "scenario_id": np.full(hours, scenario_id),
        }
    )
