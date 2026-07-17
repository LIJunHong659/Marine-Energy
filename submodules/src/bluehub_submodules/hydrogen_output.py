"""Hydrogen production, storage, pipeline and shipping output model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HydrogenParams:
    """Parameters for hydrogen production and output."""

    electrolyzer_power_max_mw: float = 150.0
    sec_kwh_per_kg: float = 57.5
    pipe_capacity_kg_per_h: float = 1800.0
    ship_capacity_kg_per_h: float = 1200.0
    storage_max_kg: float = 30_000.0
    pipe_loss_fraction: float = 0.01
    ship_loss_fraction: float = 0.02
    price_h2_cny_per_kg: float = 30.0
    pipe_transport_cost_cny_per_kg: float = 1.5
    ship_transport_cost_cny_per_kg: float = 3.0
    electrolyzer_variable_cost_cny_per_kg: float = 0.5

    def validate(self) -> None:
        if self.electrolyzer_power_max_mw < 0:
            raise ValueError("electrolyzer_power_max_mw must be non-negative.")
        if self.sec_kwh_per_kg <= 0:
            raise ValueError("sec_kwh_per_kg must be positive.")
        if self.pipe_capacity_kg_per_h < 0 or self.ship_capacity_kg_per_h < 0:
            raise ValueError("hydrogen output capacities must be non-negative.")
        if self.storage_max_kg < 0:
            raise ValueError("storage_max_kg must be non-negative.")
        if not 0 <= self.pipe_loss_fraction < 1:
            raise ValueError("pipe_loss_fraction must be in [0, 1).")
        if not 0 <= self.ship_loss_fraction < 1:
            raise ValueError("ship_loss_fraction must be in [0, 1).")
        if self.price_h2_cny_per_kg < 0:
            raise ValueError("price_h2_cny_per_kg must be non-negative.")
        if self.pipe_transport_cost_cny_per_kg < 0:
            raise ValueError("pipe transport cost must be non-negative.")
        if self.ship_transport_cost_cny_per_kg < 0:
            raise ValueError("ship transport cost must be non-negative.")
        if self.electrolyzer_variable_cost_cny_per_kg < 0:
            raise ValueError("electrolyzer variable cost must be non-negative.")


@dataclass(frozen=True)
class HydrogenResult:
    requested_electrolyzer_power_mw: float
    electrolyzer_power_mw: float
    produced_kg: float
    pipe_output_kg: float
    ship_output_kg: float
    delivered_kg: float
    storage_start_kg: float
    storage_end_kg: float
    revenue_cny: float
    variable_cost_cny: float
    violations: tuple[str, ...] = field(default_factory=tuple)


def hydrogen_production_kg(power_mw: float, sec_kwh_per_kg: float, time_step_h: float) -> float:
    """Convert electrolysis power to hydrogen mass using system SEC."""

    if sec_kwh_per_kg <= 0:
        raise ValueError("sec_kwh_per_kg must be positive.")
    if time_step_h <= 0:
        raise ValueError("time_step_h must be positive.")
    if power_mw < 0:
        raise ValueError("power_mw must be non-negative.")
    return 1000.0 * power_mw * time_step_h / sec_kwh_per_kg


def evaluate_hydrogen_output(
    requested_electrolyzer_power_mw: float,
    requested_pipe_output_kg: float,
    requested_ship_output_kg: float,
    storage_start_kg: float,
    params: HydrogenParams,
    time_step_h: float = 1.0,
) -> HydrogenResult:
    """Evaluate one hydrogen production and output step.

    Production is added to beginning inventory. Pipeline and shipping output are
    clipped by capacity and available storage. Revenue is based on delivered H2
    after route-specific losses.
    """

    params.validate()
    if time_step_h <= 0:
        raise ValueError("time_step_h must be positive.")

    violations: list[str] = []
    if requested_electrolyzer_power_mw < 0:
        violations.append("P_h2_el,t must be non-negative.")
    if requested_pipe_output_kg < 0 or requested_ship_output_kg < 0:
        violations.append("H_pipe,t and H_ship,t must be non-negative.")
    if storage_start_kg < 0:
        violations.append("H_storage,t must be non-negative.")
    if storage_start_kg > params.storage_max_kg:
        violations.append("H_storage,t exceeds H_storage_max.")

    electrolyzer_power_mw = min(
        max(requested_electrolyzer_power_mw, 0.0),
        params.electrolyzer_power_max_mw,
    )
    if requested_electrolyzer_power_mw > params.electrolyzer_power_max_mw:
        violations.append("P_h2_el,t exceeds electrolyzer capacity.")

    produced_kg = hydrogen_production_kg(
        electrolyzer_power_mw,
        params.sec_kwh_per_kg,
        time_step_h,
    )
    available_kg = max(storage_start_kg, 0.0) + produced_kg

    pipe_cap_kg = params.pipe_capacity_kg_per_h * time_step_h
    ship_cap_kg = params.ship_capacity_kg_per_h * time_step_h
    pipe_output_kg = min(max(requested_pipe_output_kg, 0.0), pipe_cap_kg, available_kg)
    if requested_pipe_output_kg > pipe_cap_kg:
        violations.append("H_pipe,t exceeds pipeline output capacity.")

    available_after_pipe = available_kg - pipe_output_kg
    ship_output_kg = min(max(requested_ship_output_kg, 0.0), ship_cap_kg, available_after_pipe)
    if requested_ship_output_kg > ship_cap_kg:
        violations.append("H_ship,t exceeds equivalent shipping capacity.")
    if requested_pipe_output_kg + requested_ship_output_kg > available_kg:
        violations.append("Hydrogen output exceeds produced plus stored hydrogen.")

    storage_end_kg = available_after_pipe - ship_output_kg
    if storage_end_kg > params.storage_max_kg:
        violations.append("H_storage,t exceeds H_storage_max after production.")
        storage_end_kg = params.storage_max_kg

    delivered_pipe_kg = pipe_output_kg * (1.0 - params.pipe_loss_fraction)
    delivered_ship_kg = ship_output_kg * (1.0 - params.ship_loss_fraction)
    delivered_kg = delivered_pipe_kg + delivered_ship_kg
    revenue_cny = delivered_kg * params.price_h2_cny_per_kg
    variable_cost_cny = produced_kg * params.electrolyzer_variable_cost_cny_per_kg
    variable_cost_cny += pipe_output_kg * params.pipe_transport_cost_cny_per_kg
    variable_cost_cny += ship_output_kg * params.ship_transport_cost_cny_per_kg

    return HydrogenResult(
        requested_electrolyzer_power_mw=requested_electrolyzer_power_mw,
        electrolyzer_power_mw=electrolyzer_power_mw,
        produced_kg=produced_kg,
        pipe_output_kg=pipe_output_kg,
        ship_output_kg=ship_output_kg,
        delivered_kg=delivered_kg,
        storage_start_kg=storage_start_kg,
        storage_end_kg=storage_end_kg,
        revenue_cny=revenue_cny,
        variable_cost_cny=variable_cost_cny,
        violations=tuple(violations),
    )

