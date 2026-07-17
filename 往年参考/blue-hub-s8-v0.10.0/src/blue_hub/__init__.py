"""Core contracts and validation utilities for the Blue Hub model."""

from blue_hub.synthetic import generate_synthetic_timeseries
from blue_hub.validation import ValidationError, ValidationReport, validate_timeseries

__all__ = [
    "ValidationError",
    "ValidationReport",
    "generate_synthetic_timeseries",
    "validate_timeseries",
]

__version__ = "0.10.0"
