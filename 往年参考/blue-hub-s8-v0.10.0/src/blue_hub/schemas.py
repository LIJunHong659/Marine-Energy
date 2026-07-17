"""Typed configuration schemas used at every model boundary."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Forbid silent fields and mutation in research configuration objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceGrade(StrEnum):
    """Evidence grade defined by the project specification."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class TimeSeriesRecord(StrictModel):
    """One hourly input record with explicit physical domains."""

    timestamp: datetime
    wind_cf: float = Field(ge=0.0, le=1.0)
    pv_cf: float = Field(ge=0.0, le=1.0)
    electricity_price: float
    grid_carbon_intensity: float = Field(ge=0.0)
    critical_load: float = Field(ge=0.0)
    rigid_compute_arrival: float = Field(ge=0.0)
    flex_compute_arrival: float = Field(ge=0.0)
    hydrogen_demand: float = Field(ge=0.0)
    tx_availability: float = Field(ge=0.0, le=1.0)
    fiber_availability: float = Field(default=1.0, ge=0.0, le=1.0)
    grid_absorption_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    grid_absorption_limit_mw: float = Field(default=1.0e12, ge=0.0)
    landing_demand_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    national_compute_demand_mw_it: float = Field(default=0.0, ge=0.0)
    national_compute_flexible_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    national_compute_price_cny_per_mwh_it: float = Field(default=0.0, ge=0.0)
    wave_generation_mw: float = Field(default=0.0, ge=0.0)
    wind_availability: float = Field(ge=0.0, le=1.0)
    scenario_id: str = Field(min_length=1)


class TechnologyParameter(StrictModel):
    """One traceable scalar parameter with uncertainty bounds."""

    parameter: str = Field(min_length=1)
    value_base: float
    value_low: float
    value_high: float
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_grade: SourceGrade
    notes: str = ""

    @model_validator(mode="after")
    def check_bounds(self) -> TechnologyParameter:
        if not self.value_low <= self.value_base <= self.value_high:
            raise ValueError("parameter bounds must satisfy low <= base <= high")
        return self


class TechnologyParameters(StrictModel):
    """A duplicate-free collection of technology parameters."""

    items: tuple[TechnologyParameter, ...]

    @model_validator(mode="after")
    def check_unique_names(self) -> TechnologyParameters:
        names = [item.parameter for item in self.items]
        if len(names) != len(set(names)):
            raise ValueError("technology parameter names must be unique")
        return self

    def value(self, name: str) -> float:
        """Return the base value for *name* or raise a clear error."""
        for item in self.items:
            if item.parameter == name:
                return item.value_base
        raise KeyError(f"unknown technology parameter: {name}")


class SystemConfiguration(StrictModel):
    """Capacity decision vector and initial state fractions."""

    config_id: str = Field(min_length=1)
    wind_capacity_mw: float = Field(gt=0.0)
    pv_capacity_mw: float = Field(ge=0.0)
    tx_capacity_mw: float = Field(ge=0.0)
    battery_power_mw: float = Field(ge=0.0)
    battery_energy_mwh: float = Field(ge=0.0)
    electrolyzer_power_mw: float = Field(ge=0.0)
    hydrogen_storage_kg: float = Field(ge=0.0)
    hydrogen_export_capacity_kg_per_h: float = Field(default=1.0e12, ge=0.0)
    compute_it_capacity_mw: float = Field(ge=0.0)
    subsea_fiber_service_capacity_mw_it: float = Field(default=0.0, ge=0.0)
    fuel_cell_power_mw: float = Field(ge=0.0)
    initial_battery_soc_fraction: float = Field(ge=0.0, le=1.0)
    initial_hydrogen_inventory_fraction: float = Field(ge=0.0, le=1.0)
    tx_technology: Literal["hvdc", "hvac"] = "hvdc"
    export_policy: Literal["economic", "must_take"] = "economic"
    battery_reserve_power_mw: float = Field(default=0.0, ge=0.0)
    battery_reserve_duration_h: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def check_storage_pairing(self) -> SystemConfiguration:
        if (self.battery_power_mw == 0.0) != (self.battery_energy_mwh == 0.0):
            raise ValueError("battery power and energy must both be zero or both be positive")
        if (self.battery_reserve_power_mw == 0.0) != (self.battery_reserve_duration_h == 0.0):
            raise ValueError("battery reserve power and duration must both be zero or positive")
        if self.battery_reserve_power_mw > self.battery_power_mw:
            raise ValueError("battery reserve power cannot exceed installed battery power")
        return self


class ScenarioDefinition(StrictModel):
    """Scenario multipliers and exogenous disruption settings."""

    scenario_id: str = Field(min_length=1)
    offshore_distance_km: float = Field(gt=0.0)
    electricity_price_multiplier: float = Field(gt=0.0)
    hydrogen_price_multiplier: float = Field(gt=0.0)
    hydrogen_demand_multiplier: float = Field(default=1.0, gt=0.0)
    compute_price_multiplier: float = Field(gt=0.0)
    compute_demand_multiplier: float = Field(default=1.0, gt=0.0)
    wind_year: str = Field(min_length=1)
    tx_outage_hours: int = Field(ge=0, le=8760)
    tx_outage_start_hour: int = Field(default=0, ge=0, le=8759)
    fiber_outage_hours: int = Field(default=0, ge=0, le=8760)
    fiber_outage_start_hour: int = Field(default=0, ge=0, le=8759)
    price_event_hours: int = Field(default=0, ge=0, le=8760)
    price_event_start_hour: int = Field(default=0, ge=0, le=8759)
    price_event_cny_per_mwh: float = 0.0
    wind_event_hours: int = Field(default=0, ge=0, le=8760)
    wind_event_start_hour: int = Field(default=0, ge=0, le=8759)
    wind_event_availability: float = Field(default=1.0, ge=0.0, le=1.0)
    typhoon_case: str = Field(min_length=1)
    battery_cost_multiplier: float = Field(gt=0.0)
    electrolyzer_cost_multiplier: float = Field(gt=0.0)
    pue_case: str = Field(min_length=1)


class ConfigurationCandidate(StrictModel):
    """One row in the outer capacity search grid."""

    config_id: str = Field(min_length=1)
    tx_capacity_mw: float = Field(ge=0.0)
    battery_power_mw: float = Field(ge=0.0)
    battery_energy_mwh: float = Field(ge=0.0)
    electrolyzer_power_mw: float = Field(ge=0.0)
    hydrogen_storage_kg: float = Field(ge=0.0)
    compute_it_capacity_mw: float = Field(ge=0.0)
    subsea_fiber_service_capacity_mw_it: float = Field(default=0.0, ge=0.0)
    fuel_cell_power_mw: float = Field(ge=0.0)
