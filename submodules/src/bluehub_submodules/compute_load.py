"""Green compute and subsea-fiber service model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ComputeLoadParams:
    """Parameters for the compute-load and fiber-output chain."""

    compute_power_max_mw: float = 120.0
    compute_power_min_mw: float = 10.0
    pue: float = 1.15
    fiber_service_capacity_mw_it: float = 120.0
    fiber_bandwidth_max_gbps: float = 800.0
    fiber_latency_ms: float = 20.0
    price_compute_cny_per_mwh_it: float = 1500.0
    variable_cost_cny_per_mwh_it: float = 20.0

    def validate(self) -> None:
        if self.compute_power_max_mw < 0 or self.compute_power_min_mw < 0:
            raise ValueError("compute power bounds must be non-negative.")
        if self.compute_power_min_mw > self.compute_power_max_mw:
            raise ValueError("compute_power_min_mw cannot exceed compute_power_max_mw.")
        if self.pue < 1.0:
            raise ValueError("PUE must be at least 1.0.")
        if self.fiber_service_capacity_mw_it < 0:
            raise ValueError("fiber_service_capacity_mw_it must be non-negative.")
        if self.fiber_bandwidth_max_gbps < 0 or self.fiber_latency_ms < 0:
            raise ValueError("fiber bandwidth and latency must be non-negative.")
        if self.price_compute_cny_per_mwh_it < 0:
            raise ValueError("price_compute_cny_per_mwh_it must be non-negative.")
        if self.variable_cost_cny_per_mwh_it < 0:
            raise ValueError("variable_cost_cny_per_mwh_it must be non-negative.")


@dataclass(frozen=True)
class ComputeLoadResult:
    requested_facility_power_mw: float
    facility_power_mw: float
    it_power_mw: float
    service_mwh_it: float
    revenue_cny: float
    variable_cost_cny: float
    violations: tuple[str, ...] = field(default_factory=tuple)


def evaluate_compute_load(
    requested_facility_power_mw: float,
    params: ComputeLoadParams,
    time_step_h: float = 1.0,
) -> ComputeLoadResult:
    """Evaluate one time step of data-center power and fiber service.

    `requested_facility_power_mw` is total data-center facility power. IT power
    is obtained through `P_it = P_compute / PUE`. Fiber constrains delivered
    service in MWh-IT/h and does not represent electric power transfer.
    """

    params.validate()
    if time_step_h <= 0:
        raise ValueError("time_step_h must be positive.")

    violations: list[str] = []
    if requested_facility_power_mw < 0:
        violations.append("P_compute,t must be non-negative.")

    facility_power_mw = min(
        max(requested_facility_power_mw, 0.0),
        params.compute_power_max_mw,
    )
    if requested_facility_power_mw > params.compute_power_max_mw:
        violations.append("P_compute,t exceeds P_compute_max.")
    if 0.0 < requested_facility_power_mw < params.compute_power_min_mw:
        violations.append("P_compute,t is below P_compute_min while online.")

    it_power_mw = facility_power_mw / params.pue
    fiber_limited_it_mw = min(it_power_mw, params.fiber_service_capacity_mw_it)
    if it_power_mw > params.fiber_service_capacity_mw_it:
        violations.append("IT service exceeds fiber service capacity.")
        it_power_mw = fiber_limited_it_mw
        facility_power_mw = it_power_mw * params.pue

    service_mwh_it = it_power_mw * time_step_h
    revenue_cny = service_mwh_it * params.price_compute_cny_per_mwh_it
    variable_cost_cny = service_mwh_it * params.variable_cost_cny_per_mwh_it

    return ComputeLoadResult(
        requested_facility_power_mw=requested_facility_power_mw,
        facility_power_mw=facility_power_mw,
        it_power_mw=it_power_mw,
        service_mwh_it=service_mwh_it,
        revenue_cny=revenue_cny,
        variable_cost_cny=variable_cost_cny,
        violations=tuple(violations),
    )

