"""Small scenario helpers for 24h examples."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .balance import DispatchRequest, IntegratedResult, evaluate_integrated_hour
from .parameters import ModelParameters, default_parameters


@dataclass(frozen=True)
class HourlyScenario:
    hour: int
    available_power_mw: float


def generate_typical_day(base_mw: float = 220.0, amplitude_mw: float = 90.0) -> list[HourlyScenario]:
    """Generate a deterministic 24h renewable-power profile for testing."""

    rows: list[HourlyScenario] = []
    for hour in range(24):
        wind_shape = 0.6 + 0.4 * math.sin((hour - 4) * math.pi / 12.0) ** 2
        solar_shape = max(0.0, math.sin((hour - 6) * math.pi / 12.0))
        available = base_mw * wind_shape + amplitude_mw * solar_shape
        rows.append(HourlyScenario(hour=hour, available_power_mw=round(available, 6)))
    return rows


def simple_greedy_dispatch(
    params: ModelParameters | None = None,
    scenario: list[HourlyScenario] | None = None,
) -> list[IntegratedResult]:
    """Run a feasible greedy dispatch for a 24h smoke test.

    Priority: marine load first, then split remaining power across grid export,
    compute and hydrogen. The rule is intentionally simple; it is not an
    economic optimizer.
    """

    params = default_parameters() if params is None else params
    scenario = generate_typical_day() if scenario is None else scenario
    storage_kg = 0.0
    results: list[IntegratedResult] = []

    for row in scenario:
        remaining = row.available_power_mw
        marine_mw = min(params.marine.total_load_mw, remaining)
        remaining -= marine_mw

        post_marine = remaining
        grid_mw = min(
            post_marine * 0.55,
            params.power_export.cable_capacity_mw,
            params.power_export.grid_accept_max_mw,
        )
        remaining -= grid_mw

        compute_target_mw = post_marine * 0.30
        compute_mw = min(remaining, compute_target_mw, params.compute.compute_power_max_mw)
        if 0.0 < compute_mw < params.compute.compute_power_min_mw <= remaining:
            compute_mw = params.compute.compute_power_min_mw
        elif 0.0 < compute_mw < params.compute.compute_power_min_mw:
            compute_mw = 0.0
        remaining -= compute_mw

        h2_mw = min(max(remaining, 0.0), params.hydrogen.electrolyzer_power_max_mw)
        remaining -= h2_mw

        produced_kg = 1000.0 * h2_mw * params.time_step_h / params.hydrogen.sec_kwh_per_kg
        available_h2 = storage_kg + produced_kg
        pipe_kg = min(
            available_h2,
            params.hydrogen.pipe_capacity_kg_per_h * params.time_step_h,
        )
        request = DispatchRequest(
            grid_power_mw=grid_mw,
            compute_power_mw=compute_mw,
            h2_power_mw=h2_mw,
            marine_power_mw=marine_mw,
            h2_pipe_output_kg=pipe_kg,
            h2_ship_output_kg=0.0,
        )
        result = evaluate_integrated_hour(
            row.hour,
            row.available_power_mw,
            storage_kg,
            request,
            params,
        )
        storage_kg = result.storage_end_kg
        results.append(result)

    return results
