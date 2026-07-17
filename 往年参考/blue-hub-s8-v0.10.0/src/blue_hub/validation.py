"""Strict, aggregated validation for tabular research inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from blue_hub.schemas import TechnologyParameters

TIMESERIES_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "wind_cf",
    "pv_cf",
    "electricity_price",
    "grid_carbon_intensity",
    "critical_load",
    "rigid_compute_arrival",
    "flex_compute_arrival",
    "hydrogen_demand",
    "tx_availability",
    "wind_availability",
    "scenario_id",
)

BOUNDED_FRACTION_COLUMNS: tuple[str, ...] = (
    "wind_cf",
    "pv_cf",
    "tx_availability",
    "wind_availability",
)

NONNEGATIVE_COLUMNS: tuple[str, ...] = (
    "grid_carbon_intensity",
    "critical_load",
    "rigid_compute_arrival",
    "flex_compute_arrival",
    "hydrogen_demand",
)

OPTIONAL_BOUNDED_FRACTION_COLUMNS: tuple[str, ...] = (
    "fiber_availability",
    "grid_absorption_factor",
    "landing_demand_factor",
    "national_compute_flexible_fraction",
)

OPTIONAL_NONNEGATIVE_COLUMNS: tuple[str, ...] = (
    "national_compute_demand_mw_it",
    "national_compute_price_cny_per_mwh_it",
    "grid_absorption_limit_mw",
    "wave_generation_mw",
)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: Severity = Severity.ERROR


class ValidationError(ValueError):
    """Raised when a validation report contains one or more errors."""


@dataclass(frozen=True)
class ValidationReport:
    """Collect all input problems so users can fix them in one pass."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def raise_if_invalid(self) -> None:
        if self.is_valid:
            return
        details = "; ".join(f"{item.code}: {item.message}" for item in self.issues)
        raise ValidationError(details)


def validate_timeseries(df: pd.DataFrame, expected_step_hours: int = 1) -> ValidationReport:
    """Validate the complete hourly input contract without silent coercion."""
    issues: list[ValidationIssue] = []
    missing = [column for column in TIMESERIES_COLUMNS if column not in df.columns]
    if missing:
        issues.append(ValidationIssue("missing_columns", f"missing required columns: {missing}"))
        return ValidationReport(tuple(issues))
    if df.empty:
        return ValidationReport((ValidationIssue("empty", "time series has no rows"),))

    required = df.loc[:, TIMESERIES_COLUMNS]
    null_counts = required.isna().sum()
    if int(null_counts.sum()) > 0:
        bad = {name: int(count) for name, count in null_counts.items() if count > 0}
        issues.append(ValidationIssue("missing_values", f"null values found: {bad}"))

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        issues.append(
            ValidationIssue("invalid_timestamp", "one or more timestamps cannot be parsed")
        )
    else:
        if timestamps.duplicated().any():
            issues.append(ValidationIssue("duplicate_timestamp", "timestamps must be unique"))
        if not timestamps.is_monotonic_increasing:
            issues.append(ValidationIssue("timestamp_order", "timestamps must be increasing"))
        if len(timestamps) > 1:
            expected = pd.Timedelta(hours=expected_step_hours)
            bad_steps = timestamps.diff().iloc[1:] != expected
            if bad_steps.any():
                issues.append(
                    ValidationIssue(
                        "timestamp_gap",
                        f"all time steps must equal {expected_step_hours} hour(s)",
                    )
                )

    numeric_columns = BOUNDED_FRACTION_COLUMNS + NONNEGATIVE_COLUMNS + ("electricity_price",)
    numeric: dict[str, pd.Series] = {}
    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        numeric[column] = converted
        invalid = converted.isna() & df[column].notna()
        if invalid.any():
            issues.append(ValidationIssue("non_numeric", f"{column} contains non-numeric values"))
        finite = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            issues.append(ValidationIssue("non_finite", f"{column} contains infinite values"))

    for column in BOUNDED_FRACTION_COLUMNS:
        values = numeric[column]
        if ((values < 0.0) | (values > 1.0)).fillna(False).any():
            issues.append(ValidationIssue("fraction_bounds", f"{column} must be within [0, 1]"))

    for column in NONNEGATIVE_COLUMNS:
        if (numeric[column] < 0.0).fillna(False).any():
            issues.append(ValidationIssue("negative_value", f"{column} must be non-negative"))

    optional_numeric: dict[str, pd.Series] = {}
    for column in OPTIONAL_BOUNDED_FRACTION_COLUMNS + OPTIONAL_NONNEGATIVE_COLUMNS:
        if column not in df.columns:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        optional_numeric[column] = converted
        if converted.isna().any():
            issues.append(
                ValidationIssue("invalid_optional_value", f"{column} must be numeric and non-null")
            )
        finite = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            issues.append(ValidationIssue("non_finite", f"{column} contains infinite values"))
    for column in OPTIONAL_BOUNDED_FRACTION_COLUMNS:
        if (
            column in optional_numeric
            and ((optional_numeric[column] < 0.0) | (optional_numeric[column] > 1.0))
            .fillna(False)
            .any()
        ):
            issues.append(ValidationIssue("fraction_bounds", f"{column} must be within [0, 1]"))
    for column in OPTIONAL_NONNEGATIVE_COLUMNS:
        if column in optional_numeric and (optional_numeric[column] < 0.0).fillna(False).any():
            issues.append(ValidationIssue("negative_value", f"{column} must be non-negative"))

    if df["scenario_id"].astype("string").str.strip().eq("").fillna(False).any():
        issues.append(ValidationIssue("empty_scenario", "scenario_id must not be blank"))

    return ValidationReport(tuple(issues))


def validate_parameters(params: TechnologyParameters) -> ValidationReport:
    """Apply cross-parameter checks that are not intrinsic to row schemas."""
    issues: list[ValidationIssue] = []
    units = {item.parameter: item.unit for item in params.items}
    expected_units = {
        "wind_capacity": "MW",
        "hydrogen_sec_system": "kWh/kg",
        "data_center_pue": "ratio",
        "time_step": "h",
        "power_balance_tolerance": "MW",
        "tx_terminal_loss_fraction_hvdc": "fraction",
        "tx_cable_full_load_loss_per_100km_hvdc": "fraction/100km",
        "tx_variable_cost": "CNY/MWh-send",
        "curtailment_penalty": "CNY/MWh-curtailed",
        "unserved_critical_load_penalty": "CNY/MWh-unserved",
        "battery_self_discharge_per_hour": "fraction/h",
        "battery_soc_min": "fraction",
        "battery_soc_max": "fraction",
        "battery_degradation_cost": "CNY/MWh-throughput",
        "battery_linearization_segments": "count",
        "hydrogen_sale_price": "CNY/kg",
        "hydrogen_transport_cost": "CNY/kg",
        "electrolyzer_variable_cost": "CNY/kg",
        "hydrogen_storage_loss_per_hour": "fraction/h",
        "hydrogen_water_consumption": "m3/kg",
        "desalinated_water_cost": "CNY/m3",
        "hydrogen_state_tolerance": "kg",
        "compute_rigid_service_price": "CNY/MWh-IT",
        "compute_flex_service_price": "CNY/MWh-IT",
        "compute_variable_cost": "CNY/MWh-IT",
        "compute_rigid_sla_penalty": "CNY/MWh-IT-unserved",
        "compute_flex_queue_capacity": "MWh-IT",
        "compute_flex_max_delay": "h",
        "compute_ramp_up": "MW-IT/h",
        "compute_ramp_down": "MW-IT/h",
        "compute_state_tolerance": "MWh-IT",
        "compute_spot_service_price": "CNY/MWh-IT",
        "hydrogen_lhv_kwh_per_kg": "kWh/kg",
        "fuel_cell_efficiency_lhv": "fraction",
        "fuel_cell_variable_cost": "CNY/MWh-electric",
    }
    for name, expected in expected_units.items():
        if name not in units:
            issues.append(
                ValidationIssue("missing_parameter", f"required parameter {name} is absent")
            )
        elif units[name] != expected:
            issues.append(
                ValidationIssue(
                    "unit_mismatch", f"{name} must use {expected}, received {units[name]}"
                )
            )
    if "time_step" in units and params.value("time_step") != 1.0:
        issues.append(ValidationIssue("time_step", "the core model currently requires a 1 h step"))
    return ValidationReport(tuple(issues))
