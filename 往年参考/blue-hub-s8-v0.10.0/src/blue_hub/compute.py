"""Green-compute service contracts for Phase 4 / S3."""

from __future__ import annotations

from dataclasses import dataclass

from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters


@dataclass(frozen=True)
class ComputeSpec:
    """IT energy, facility PUE, fibre service and workload-SLA assumptions."""

    it_capacity_mw: float
    subsea_fiber_service_capacity_mw_it: float
    pue: float
    rigid_service_price_cny_per_mwh_it: float
    flex_service_price_cny_per_mwh_it: float
    variable_cost_cny_per_mwh_it: float
    rigid_sla_penalty_cny_per_mwh_it: float
    flex_queue_capacity_mwh_it: float
    flex_max_delay_h: int
    ramp_up_mw_it_per_h: float
    ramp_down_mw_it_per_h: float

    def validate(self) -> None:
        if self.it_capacity_mw <= 0.0:
            raise ValueError("S3 IT capacity must be positive")
        if self.subsea_fiber_service_capacity_mw_it <= 0.0:
            raise ValueError("S3 subsea-fibre service capacity must be positive")
        if self.pue < 1.0:
            raise ValueError("PUE must be at least one")
        if min(
            self.rigid_service_price_cny_per_mwh_it,
            self.flex_service_price_cny_per_mwh_it,
            self.variable_cost_cny_per_mwh_it,
        ) < 0.0:
            raise ValueError("compute service prices and variable cost must be non-negative")
        if self.rigid_sla_penalty_cny_per_mwh_it <= 0.0:
            raise ValueError("rigid compute SLA penalty must be positive")
        if self.flex_queue_capacity_mwh_it <= 0.0:
            raise ValueError("flexible compute queue capacity must be positive")
        if self.flex_max_delay_h < 0:
            raise ValueError("flexible compute maximum delay must be non-negative")
        if self.ramp_up_mw_it_per_h <= 0.0 or self.ramp_down_mw_it_per_h <= 0.0:
            raise ValueError("compute ramp limits must be positive")


def _pue_for_case(params: TechnologyParameters, pue_case: str) -> float:
    for item in params.items:
        if item.parameter != "data_center_pue":
            continue
        values = {
            "optimistic": item.value_low,
            "base": item.value_base,
            "conservative": item.value_high,
        }
        try:
            return values[pue_case.lower()]
        except KeyError as error:
            raise ValueError(
                "pue_case must be one of optimistic, base or conservative"
            ) from error
    raise KeyError("unknown technology parameter: data_center_pue")


def compute_spec_from_parameters(
    config: SystemConfiguration,
    params: TechnologyParameters,
    scenario: ScenarioDefinition,
) -> ComputeSpec:
    """Build the S3 service contract with PUE selected by scenario case."""
    max_delay = params.value("compute_flex_max_delay")
    if not max_delay.is_integer():
        raise ValueError("compute_flex_max_delay must be an integer number of hours")
    specification = ComputeSpec(
        it_capacity_mw=config.compute_it_capacity_mw,
        subsea_fiber_service_capacity_mw_it=config.subsea_fiber_service_capacity_mw_it,
        pue=_pue_for_case(params, scenario.pue_case),
        rigid_service_price_cny_per_mwh_it=(
            params.value("compute_rigid_service_price") * scenario.compute_price_multiplier
        ),
        flex_service_price_cny_per_mwh_it=(
            params.value("compute_flex_service_price") * scenario.compute_price_multiplier
        ),
        variable_cost_cny_per_mwh_it=params.value("compute_variable_cost"),
        rigid_sla_penalty_cny_per_mwh_it=params.value("compute_rigid_sla_penalty"),
        flex_queue_capacity_mwh_it=params.value("compute_flex_queue_capacity"),
        flex_max_delay_h=int(max_delay),
        ramp_up_mw_it_per_h=params.value("compute_ramp_up"),
        ramp_down_mw_it_per_h=params.value("compute_ramp_down"),
    )
    specification.validate()
    return specification
