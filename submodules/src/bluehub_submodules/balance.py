"""Integrated hourly balance connecting all Day1-Day7 submodels."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hydrogen_output import HydrogenResult
from .marine_load import MarineLoadResult, evaluate_marine_load
from .objectives import ObjectiveBreakdown, calculate_objective_breakdown
from .parameters import ModelParameters
from .power_export import PowerExportResult, evaluate_power_export
from .compute_load import ComputeLoadResult, evaluate_compute_load
from .hydrogen_output import evaluate_hydrogen_output


@dataclass(frozen=True)
class DispatchRequest:
    """Requested dispatch variables for one time step."""

    grid_power_mw: float
    compute_power_mw: float
    h2_power_mw: float
    marine_power_mw: float
    h2_pipe_output_kg: float = 0.0
    h2_ship_output_kg: float = 0.0
    storage_charge_mw: float = 0.0
    storage_discharge_mw: float = 0.0
    curtailment_mw: float | None = None


@dataclass(frozen=True)
class IntegratedResult:
    hour: int
    available_power_mw: float
    storage_start_kg: float
    storage_end_kg: float
    request: DispatchRequest
    power_export: PowerExportResult
    compute: ComputeLoadResult
    hydrogen: HydrogenResult
    marine: MarineLoadResult
    curtailment_mw: float
    offshore_balance_residual_mw: float
    objective: ObjectiveBreakdown
    violations: tuple[str, ...] = field(default_factory=tuple)


def evaluate_integrated_hour(
    hour: int,
    available_power_mw: float,
    storage_start_kg: float,
    request: DispatchRequest,
    params: ModelParameters,
) -> IntegratedResult:
    """Evaluate one integrated dispatch hour.

    This is a deterministic evaluator, not an optimizer. It is intended as the
    auditable foundation that later optimization code can call or reproduce.
    """

    params.validate()
    dt = params.time_step_h
    violations: list[str] = []
    if available_power_mw < 0:
        violations.append("P_available,t must be non-negative.")

    marine_available_mw = max(available_power_mw + request.storage_discharge_mw, 0.0)
    marine = evaluate_marine_load(
        marine_available_mw,
        params.marine,
        time_step_h=dt,
        requested_power_mw=request.marine_power_mw,
    )
    power_export = evaluate_power_export(
        request.grid_power_mw,
        params.power_export,
        time_step_h=dt,
    )
    compute = evaluate_compute_load(
        request.compute_power_mw,
        params.compute,
        time_step_h=dt,
    )
    hydrogen = evaluate_hydrogen_output(
        request.h2_power_mw,
        request.h2_pipe_output_kg,
        request.h2_ship_output_kg,
        storage_start_kg,
        params.hydrogen,
        time_step_h=dt,
    )

    used_power_mw = (
        marine.served_power_mw
        + power_export.exported_power_mw
        + compute.facility_power_mw
        + hydrogen.electrolyzer_power_mw
        + max(request.storage_charge_mw, 0.0)
    )
    supplied_power_mw = max(available_power_mw, 0.0) + max(request.storage_discharge_mw, 0.0)
    if request.curtailment_mw is None:
        curtailment_mw = max(supplied_power_mw - used_power_mw, 0.0)
    else:
        curtailment_mw = max(request.curtailment_mw, 0.0)
        if request.curtailment_mw < 0:
            violations.append("P_curt,t must be non-negative.")

    residual = supplied_power_mw - used_power_mw - curtailment_mw
    if abs(residual) > params.power_balance_tolerance_mw:
        violations.append("offshore power balance residual exceeds tolerance.")

    objective = calculate_objective_breakdown(
        power_revenue_cny=power_export.revenue_cny,
        compute_revenue_cny=compute.revenue_cny,
        hydrogen_revenue_cny=hydrogen.revenue_cny,
        marine_value_cny=marine.revenue_cny,
        power_variable_cost_cny=power_export.variable_cost_cny,
        compute_variable_cost_cny=compute.variable_cost_cny,
        hydrogen_variable_cost_cny=hydrogen.variable_cost_cny,
        marine_unmet_penalty_cny=marine.unmet_penalty_cny,
    )

    violations.extend(marine.violations)
    violations.extend(power_export.violations)
    violations.extend(compute.violations)
    violations.extend(hydrogen.violations)

    return IntegratedResult(
        hour=hour,
        available_power_mw=available_power_mw,
        storage_start_kg=storage_start_kg,
        storage_end_kg=hydrogen.storage_end_kg,
        request=request,
        power_export=power_export,
        compute=compute,
        hydrogen=hydrogen,
        marine=marine,
        curtailment_mw=curtailment_mw,
        offshore_balance_residual_mw=residual,
        objective=objective,
        violations=tuple(violations),
    )


def summarize_results(results: list[IntegratedResult]) -> dict[str, float]:
    """Summarize hourly integrated results into report-friendly KPIs."""

    total = ObjectiveBreakdown()
    for result in results:
        total = total + result.objective
    return {
        "hours": float(len(results)),
        "export_sent_mwh": sum(r.power_export.exported_power_mw for r in results),
        "export_delivered_mwh": sum(r.power_export.delivered_power_mw for r in results),
        "compute_service_mwh_it": sum(r.compute.service_mwh_it for r in results),
        "hydrogen_produced_kg": sum(r.hydrogen.produced_kg for r in results),
        "hydrogen_delivered_kg": sum(r.hydrogen.delivered_kg for r in results),
        "marine_served_mwh": sum(r.marine.served_power_mw for r in results),
        "curtailment_mwh": sum(r.curtailment_mw for r in results),
        "operating_margin_cny": total.operating_margin_cny,
        "total_revenue_cny": total.total_revenue_cny,
        "total_cost_cny": total.total_cost_cny,
        "max_abs_balance_residual_mw": max(
            (abs(r.offshore_balance_residual_mw) for r in results),
            default=0.0,
        ),
        "violation_count": float(sum(len(r.violations) for r in results)),
    }

