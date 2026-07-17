import numpy as np

from blue_hub.synthetic import generate_synthetic_timeseries
from blue_hub.validation import ValidationError, validate_timeseries


def issue_codes(frame) -> set[str]:
    return {item.code for item in validate_timeseries(frame).issues}


def test_missing_column_is_rejected() -> None:
    frame = generate_synthetic_timeseries().drop(columns="wind_cf")
    assert "missing_columns" in issue_codes(frame)


def test_duplicate_and_gap_are_rejected() -> None:
    duplicate = generate_synthetic_timeseries()
    duplicate.loc[1, "timestamp"] = duplicate.loc[0, "timestamp"]
    assert {"duplicate_timestamp", "timestamp_gap"} <= issue_codes(duplicate)

    gap = generate_synthetic_timeseries().drop(index=5).reset_index(drop=True)
    assert "timestamp_gap" in issue_codes(gap)


def test_nan_is_not_silently_filled() -> None:
    frame = generate_synthetic_timeseries()
    frame.loc[4, "electricity_price"] = np.nan
    report = validate_timeseries(frame)
    assert "missing_values" in {item.code for item in report.issues}
    try:
        report.raise_if_invalid()
    except ValidationError as exc:
        assert "missing_values" in str(exc)
    else:
        raise AssertionError("invalid report did not raise")


def test_fraction_and_nonnegative_domains() -> None:
    frame = generate_synthetic_timeseries()
    frame.loc[2, "wind_cf"] = 1.01
    frame.loc[3, "critical_load"] = -1.0
    assert {"fraction_bounds", "negative_value"} <= issue_codes(frame)


def test_negative_electricity_price_is_allowed() -> None:
    frame = generate_synthetic_timeseries()
    frame.loc[0, "electricity_price"] = -50.0
    assert validate_timeseries(frame).is_valid


def test_s5_optional_columns_are_validated_when_present() -> None:
    frame = generate_synthetic_timeseries()
    frame["grid_absorption_factor"] = 1.01
    frame["national_compute_demand_mw_it"] = -1.0
    assert {"fraction_bounds", "negative_value"} <= issue_codes(frame)
