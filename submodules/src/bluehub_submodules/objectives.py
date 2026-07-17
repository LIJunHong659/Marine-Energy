"""Objective accounting for the integrated submodels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Revenue, cost and net-value accounting for one or many time steps."""

    power_revenue_cny: float = 0.0
    compute_revenue_cny: float = 0.0
    hydrogen_revenue_cny: float = 0.0
    marine_value_cny: float = 0.0
    power_variable_cost_cny: float = 0.0
    compute_variable_cost_cny: float = 0.0
    hydrogen_variable_cost_cny: float = 0.0
    marine_unmet_penalty_cny: float = 0.0

    @property
    def total_revenue_cny(self) -> float:
        return (
            self.power_revenue_cny
            + self.compute_revenue_cny
            + self.hydrogen_revenue_cny
            + self.marine_value_cny
        )

    @property
    def total_cost_cny(self) -> float:
        return (
            self.power_variable_cost_cny
            + self.compute_variable_cost_cny
            + self.hydrogen_variable_cost_cny
            + self.marine_unmet_penalty_cny
        )

    @property
    def operating_margin_cny(self) -> float:
        return self.total_revenue_cny - self.total_cost_cny

    def __add__(self, other: "ObjectiveBreakdown") -> "ObjectiveBreakdown":
        return ObjectiveBreakdown(
            power_revenue_cny=self.power_revenue_cny + other.power_revenue_cny,
            compute_revenue_cny=self.compute_revenue_cny + other.compute_revenue_cny,
            hydrogen_revenue_cny=self.hydrogen_revenue_cny + other.hydrogen_revenue_cny,
            marine_value_cny=self.marine_value_cny + other.marine_value_cny,
            power_variable_cost_cny=self.power_variable_cost_cny
            + other.power_variable_cost_cny,
            compute_variable_cost_cny=self.compute_variable_cost_cny
            + other.compute_variable_cost_cny,
            hydrogen_variable_cost_cny=self.hydrogen_variable_cost_cny
            + other.hydrogen_variable_cost_cny,
            marine_unmet_penalty_cny=self.marine_unmet_penalty_cny
            + other.marine_unmet_penalty_cny,
        )


def calculate_objective_breakdown(
    *,
    power_revenue_cny: float = 0.0,
    compute_revenue_cny: float = 0.0,
    hydrogen_revenue_cny: float = 0.0,
    marine_value_cny: float = 0.0,
    power_variable_cost_cny: float = 0.0,
    compute_variable_cost_cny: float = 0.0,
    hydrogen_variable_cost_cny: float = 0.0,
    marine_unmet_penalty_cny: float = 0.0,
) -> ObjectiveBreakdown:
    """Create an objective breakdown from separate value streams."""

    return ObjectiveBreakdown(
        power_revenue_cny=power_revenue_cny,
        compute_revenue_cny=compute_revenue_cny,
        hydrogen_revenue_cny=hydrogen_revenue_cny,
        marine_value_cny=marine_value_cny,
        power_variable_cost_cny=power_variable_cost_cny,
        compute_variable_cost_cny=compute_variable_cost_cny,
        hydrogen_variable_cost_cny=hydrogen_variable_cost_cny,
        marine_unmet_penalty_cny=marine_unmet_penalty_cny,
    )

