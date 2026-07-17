"""File loaders that preserve schema errors instead of silently repairing data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from blue_hub.schemas import (
    ConfigurationCandidate,
    ScenarioDefinition,
    SystemConfiguration,
    TechnologyParameter,
    TechnologyParameters,
)
from blue_hub.validation import validate_parameters, validate_timeseries


def load_timeseries(path: str | Path) -> pd.DataFrame:
    """Read and validate a model time series."""
    frame = pd.read_csv(path)
    validate_timeseries(frame).raise_if_invalid()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def load_parameters(path: str | Path) -> TechnologyParameters:
    """Read traceable technology parameters and validate units."""
    frame = pd.read_csv(path, keep_default_na=False)
    items = tuple(TechnologyParameter(**record) for record in frame.to_dict(orient="records"))
    parameters = TechnologyParameters(items=items)
    validate_parameters(parameters).raise_if_invalid()
    return parameters


def load_system_configuration(path: str | Path) -> SystemConfiguration:
    """Load the base system configuration from YAML."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return SystemConfiguration(**payload)


def load_scenarios(path: str | Path) -> tuple[ScenarioDefinition, ...]:
    """Load all scenario rows with strict typing."""
    frame = pd.read_csv(path, keep_default_na=False)
    return tuple(ScenarioDefinition(**row) for row in frame.to_dict(orient="records"))


def load_configurations(path: str | Path) -> tuple[ConfigurationCandidate, ...]:
    """Load all capacity candidates with strict typing."""
    frame = pd.read_csv(path, keep_default_na=False)
    return tuple(ConfigurationCandidate(**row) for row in frame.to_dict(orient="records"))
