"""Power export model for the electricity-subsea-cable chain."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PowerExportParams:
    """Parameters for electricity export through a subsea cable."""

    cable_capacity_mw: float = 700.0
    grid_accept_max_mw: float = 450.0
    cable_loss_fraction: float = 0.08
    price_power_cny_per_kwh: float = 0.35
    variable_cost_cny_per_mwh_send: float = 5.0
    cable_capex_cny_per_km: float = 10_000_000.0
    cable_distance_km: float = 200.0
    converter_cost_cny: float = 0.0
    om_cable_fraction_per_year: float = 0.02

    def validate(self) -> None:
        if self.cable_capacity_mw < 0:
            raise ValueError("cable_capacity_mw must be non-negative.")
        if self.grid_accept_max_mw < 0:
            raise ValueError("grid_accept_max_mw must be non-negative.")
        if not 0 <= self.cable_loss_fraction < 1:
            raise ValueError("cable_loss_fraction must be in [0, 1).")
        if self.price_power_cny_per_kwh < 0:
            raise ValueError("price_power_cny_per_kwh must be non-negative.")
        if self.variable_cost_cny_per_mwh_send < 0:
            raise ValueError("variable_cost_cny_per_mwh_send must be non-negative.")
        if self.cable_capex_cny_per_km < 0 or self.cable_distance_km < 0:
            raise ValueError("cable cost and distance must be non-negative.")
        if self.converter_cost_cny < 0 or self.om_cable_fraction_per_year < 0:
            raise ValueError("converter cost and O&M fraction must be non-negative.")


@dataclass(frozen=True)
class PowerExportResult:
    requested_power_mw: float
    exported_power_mw: float
    delivered_power_mw: float
    lost_power_mw: float
    revenue_cny: float
    variable_cost_cny: float
    violations: tuple[str, ...] = field(default_factory=tuple)


def evaluate_power_export(
    requested_power_mw: float,
    params: PowerExportParams,
    time_step_h: float = 1.0,
) -> PowerExportResult:
    """Evaluate one time step of power export.

    The dispatched export is clipped by cable capacity and onshore acceptance.
    Any clipping is reported as a constraint violation against the requested
    dispatch. Revenue is calculated on delivered electricity.
    """

    params.validate()
    if time_step_h <= 0:
        raise ValueError("time_step_h must be positive.")

    violations: list[str] = []
    if requested_power_mw < 0:
        violations.append("P_grid,t must be non-negative.")

    feasible_max = min(params.cable_capacity_mw, params.grid_accept_max_mw)
    exported_power_mw = min(max(requested_power_mw, 0.0), feasible_max)
    if requested_power_mw > feasible_max:
        violations.append(
            "P_grid,t exceeds min(P_grid_max, P_grid_accept_max)."
        )

    delivered_power_mw = exported_power_mw * (1.0 - params.cable_loss_fraction)
    lost_power_mw = exported_power_mw - delivered_power_mw
    delivered_mwh = delivered_power_mw * time_step_h
    sent_mwh = exported_power_mw * time_step_h
    revenue_cny = delivered_mwh * 1000.0 * params.price_power_cny_per_kwh
    variable_cost_cny = sent_mwh * params.variable_cost_cny_per_mwh_send

    return PowerExportResult(
        requested_power_mw=requested_power_mw,
        exported_power_mw=exported_power_mw,
        delivered_power_mw=delivered_power_mw,
        lost_power_mw=lost_power_mw,
        revenue_cny=revenue_cny,
        variable_cost_cny=variable_cost_cny,
        violations=tuple(violations),
    )


def annualized_cable_cost_cny(params: PowerExportParams, capital_recovery_factor: float) -> float:
    """Return a simple annualized cable-system cost."""

    params.validate()
    if capital_recovery_factor < 0:
        raise ValueError("capital_recovery_factor must be non-negative.")
    capex = params.cable_capex_cny_per_km * params.cable_distance_km
    capex += params.converter_cost_cny
    return capex * (capital_recovery_factor + params.om_cable_fraction_per_year)

