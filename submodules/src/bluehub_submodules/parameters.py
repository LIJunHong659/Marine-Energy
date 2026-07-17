"""Parameter containers for the Day1-Day7 submodels."""

from __future__ import annotations

from dataclasses import dataclass, field

from .compute_load import ComputeLoadParams
from .hydrogen_output import HydrogenParams
from .marine_load import MarineLoadParams
from .power_export import PowerExportParams


@dataclass(frozen=True)
class ModelParameters:
    """Grouped parameter set used by the integrated model."""

    power_export: PowerExportParams = field(default_factory=PowerExportParams)
    compute: ComputeLoadParams = field(default_factory=ComputeLoadParams)
    hydrogen: HydrogenParams = field(default_factory=HydrogenParams)
    marine: MarineLoadParams = field(default_factory=MarineLoadParams)
    time_step_h: float = 1.0
    power_balance_tolerance_mw: float = 1e-6

    def validate(self) -> None:
        """Raise ValueError if basic parameter units or ranges are invalid."""

        if self.time_step_h <= 0:
            raise ValueError("time_step_h must be positive.")
        if self.power_balance_tolerance_mw <= 0:
            raise ValueError("power_balance_tolerance_mw must be positive.")
        self.power_export.validate()
        self.compute.validate()
        self.hydrogen.validate()
        self.marine.validate()


def default_parameters() -> ModelParameters:
    """Return a transparent default parameter set for 24h examples.

    Values are screening assumptions, not project-calibrated engineering data.
    Replace them with the parameter evidence table before using model results in
    the report.
    """

    params = ModelParameters()
    params.validate()
    return params

