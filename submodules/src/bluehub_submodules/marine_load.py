"""Marine auxiliary-load model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarineLoadParams:
    """Parameters for island auxiliary and marine energy use."""

    aux_load_mw: float = 8.0
    desal_load_mw: float = 4.0
    equipment_load_mw: float = 6.0
    flexible_fraction: float = 0.30
    value_cny_per_mwh_served: float = 0.0
    unmet_penalty_cny_per_mwh: float = 10_000.0

    def validate(self) -> None:
        if self.aux_load_mw < 0 or self.desal_load_mw < 0 or self.equipment_load_mw < 0:
            raise ValueError("marine load components must be non-negative.")
        if not 0 <= self.flexible_fraction <= 1:
            raise ValueError("flexible_fraction must be in [0, 1].")
        if self.value_cny_per_mwh_served < 0 or self.unmet_penalty_cny_per_mwh < 0:
            raise ValueError("marine value and penalty must be non-negative.")

    @property
    def total_load_mw(self) -> float:
        return self.aux_load_mw + self.desal_load_mw + self.equipment_load_mw

    @property
    def rigid_load_mw(self) -> float:
        return self.total_load_mw * (1.0 - self.flexible_fraction)


@dataclass(frozen=True)
class MarineLoadResult:
    requested_power_mw: float
    served_power_mw: float
    unmet_power_mw: float
    revenue_cny: float
    unmet_penalty_cny: float
    violations: tuple[str, ...] = field(default_factory=tuple)


def evaluate_marine_load(
    available_power_mw: float,
    params: MarineLoadParams,
    time_step_h: float = 1.0,
    requested_power_mw: float | None = None,
) -> MarineLoadResult:
    """Evaluate island auxiliary and marine energy service.

    If requested power is omitted, the full modeled marine load is requested.
    Service is clipped by available power. Unserved rigid and flexible load are
    both reported; the penalty can be interpreted as a placeholder for EENS.
    """

    params.validate()
    if time_step_h <= 0:
        raise ValueError("time_step_h must be positive.")

    violations: list[str] = []
    if available_power_mw < 0:
        violations.append("available_power_mw must be non-negative.")
    requested = params.total_load_mw if requested_power_mw is None else requested_power_mw
    if requested < 0:
        violations.append("P_marine,t must be non-negative.")

    served_power_mw = min(max(available_power_mw, 0.0), max(requested, 0.0))
    unmet_power_mw = max(requested, 0.0) - served_power_mw
    if served_power_mw < params.rigid_load_mw:
        violations.append("served marine power is below rigid marine load.")

    served_mwh = served_power_mw * time_step_h
    unmet_mwh = unmet_power_mw * time_step_h
    return MarineLoadResult(
        requested_power_mw=requested,
        served_power_mw=served_power_mw,
        unmet_power_mw=unmet_power_mw,
        revenue_cny=served_mwh * params.value_cny_per_mwh_served,
        unmet_penalty_cny=unmet_mwh * params.unmet_penalty_cny_per_mwh,
        violations=tuple(violations),
    )

