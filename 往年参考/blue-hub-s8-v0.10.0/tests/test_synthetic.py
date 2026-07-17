from pathlib import Path

import pandas as pd

from blue_hub.loaders import (
    load_configurations,
    load_parameters,
    load_scenarios,
    load_system_configuration,
)
from blue_hub.synthetic import generate_synthetic_timeseries
from blue_hub.validation import validate_timeseries

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_data_are_deterministic_and_valid() -> None:
    first = generate_synthetic_timeseries(168)
    second = generate_synthetic_timeseries(168)
    pd.testing.assert_frame_equal(first, second)
    assert validate_timeseries(first).is_valid


def test_full_year_contract_is_valid() -> None:
    frame = generate_synthetic_timeseries(8760)
    assert len(frame) == 8760
    assert validate_timeseries(frame).is_valid


def test_all_configuration_files_load() -> None:
    params = load_parameters(ROOT / "configs/technology_parameters.csv")
    system = load_system_configuration(ROOT / "configs/base_case.yaml")
    scenarios = load_scenarios(ROOT / "configs/scenario_matrix.csv")
    candidates = load_configurations(ROOT / "configs/configuration_grid.csv")
    assert params.value("hydrogen_sec_system") == 57.5
    assert system.config_id == "S0_base"
    assert scenarios[0].offshore_distance_km == 100.0
    assert candidates[0].tx_capacity_mw == 700.0
