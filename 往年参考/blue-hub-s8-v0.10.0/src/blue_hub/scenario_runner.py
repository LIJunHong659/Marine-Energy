"""Deterministic transformations from a base time series to one S0 scenario."""

from __future__ import annotations

import pandas as pd

from blue_hub.schemas import ScenarioDefinition
from blue_hub.validation import validate_timeseries


def apply_scenario(df: pd.DataFrame, scenario: ScenarioDefinition) -> pd.DataFrame:
    """Apply price scaling and an explicitly located contiguous cable outage."""
    validate_timeseries(df).raise_if_invalid()
    transformed = df.copy(deep=True)
    transformed["electricity_price"] = (
        transformed["electricity_price"] * scenario.electricity_price_multiplier
    )
    transformed["hydrogen_demand"] = (
        transformed["hydrogen_demand"] * scenario.hydrogen_demand_multiplier
    )
    transformed["rigid_compute_arrival"] = (
        transformed["rigid_compute_arrival"] * scenario.compute_demand_multiplier
    )
    transformed["flex_compute_arrival"] = (
        transformed["flex_compute_arrival"] * scenario.compute_demand_multiplier
    )
    transformed["scenario_id"] = scenario.scenario_id

    if scenario.typhoon_case.lower() != "none":
        raise NotImplementedError(
            "S0 does not synthesize typhoon availability; provide it in wind_availability"
        )
    if scenario.tx_outage_hours:
        start = scenario.tx_outage_start_hour
        stop = start + scenario.tx_outage_hours
        if stop > len(transformed):
            raise ValueError("transmission outage window exceeds the time-series horizon")
        tx_column = transformed.columns.get_loc("tx_availability")
        transformed.iloc[start:stop, tx_column] = 0.0
    if scenario.fiber_outage_hours:
        if "fiber_availability" not in transformed.columns:
            raise ValueError("fiber outage requires a fiber_availability time-series column")
        start = scenario.fiber_outage_start_hour
        stop = start + scenario.fiber_outage_hours
        if stop > len(transformed):
            raise ValueError("fiber outage window exceeds the time-series horizon")
        fiber_column = transformed.columns.get_loc("fiber_availability")
        transformed.iloc[start:stop, fiber_column] = 0.0
    if scenario.price_event_hours:
        start = scenario.price_event_start_hour
        stop = start + scenario.price_event_hours
        if stop > len(transformed):
            raise ValueError("price event window exceeds the time-series horizon")
        price_column = transformed.columns.get_loc("electricity_price")
        transformed.iloc[start:stop, price_column] = scenario.price_event_cny_per_mwh
    if scenario.wind_event_hours:
        start = scenario.wind_event_start_hour
        stop = start + scenario.wind_event_hours
        if stop > len(transformed):
            raise ValueError("wind event window exceeds the time-series horizon")
        wind_column = transformed.columns.get_loc("wind_availability")
        transformed.iloc[start:stop, wind_column] = scenario.wind_event_availability

    validate_timeseries(transformed).raise_if_invalid()
    return transformed
