"""Parameter containers for the Day1-Day7 submodels."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from .compute_load import ComputeLoadParams
from .hydrogen_output import HydrogenParams
from .marine_load import MarineLoadParams
from .power_export import PowerExportParams

T = TypeVar("T")


@dataclass(frozen=True)
class CaseMetadata:
    """Frozen metadata shared by all integrated runs."""

    case_id: str = "common_case_v1"
    scenario_type: str = "interface_smoke"
    timezone: str = "Asia/Shanghai"
    site_id: str = "SITE_REF_V1"
    source_capacity_mw: float = 40.0
    evaluation_year: int = 2026
    operation_year_label: str = "Y0"
    evaluation_life_years: int = 25
    random_seed: int = 42

    def validate(self) -> None:
        if self.case_id != "common_case_v1":
            raise ValueError("case_id must be common_case_v1.")
        if self.scenario_type not in {"interface_smoke", "engineering_base"}:
            raise ValueError("scenario_type must be interface_smoke or engineering_base.")
        if not self.timezone:
            raise ValueError("timezone must be non-empty.")
        if not self.site_id:
            raise ValueError("site_id must be non-empty.")
        if self.source_capacity_mw <= 0:
            raise ValueError("source_capacity_mw must be positive.")
        if self.evaluation_year <= 0:
            raise ValueError("evaluation_year must be positive.")
        if not self.operation_year_label:
            raise ValueError("operation_year_label must be non-empty.")
        if self.evaluation_life_years <= 0:
            raise ValueError("evaluation_life_years must be positive.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")


@dataclass(frozen=True)
class BatteryParams:
    """BESS capacity metadata reserved for 4.5 integration."""

    bess_power_mw: float = 0.0
    bess_energy_mwh: float = 0.0
    roundtrip_efficiency: float = 0.90

    def validate(self) -> None:
        if self.bess_power_mw < 0 or self.bess_energy_mwh < 0:
            raise ValueError("BESS power and energy capacity must be non-negative.")
        if not 0 < self.roundtrip_efficiency <= 1:
            raise ValueError("roundtrip_efficiency must be in (0, 1].")


@dataclass(frozen=True)
class ModelParameters:
    """Grouped parameter set used by the integrated model."""

    case: CaseMetadata = field(default_factory=CaseMetadata)
    battery: BatteryParams = field(default_factory=BatteryParams)
    power_export: PowerExportParams = field(default_factory=PowerExportParams)
    compute: ComputeLoadParams = field(default_factory=ComputeLoadParams)
    hydrogen: HydrogenParams = field(default_factory=HydrogenParams)
    marine: MarineLoadParams = field(default_factory=MarineLoadParams)
    time_step_h: float = 1.0
    power_balance_tolerance_mw: float = 1e-6

    def validate(self) -> None:
        """Raise ValueError if basic parameter units or ranges are invalid."""

        self.case.validate()
        self.battery.validate()
        if self.time_step_h <= 0:
            raise ValueError("time_step_h must be positive.")
        if self.power_balance_tolerance_mw <= 0:
            raise ValueError("power_balance_tolerance_mw must be positive.")
        self.power_export.validate()
        self.compute.validate()
        self.hydrogen.validate()
        self.marine.validate()


def _load_dataclass_section(section_name: str, raw: Any, cls: type[T]) -> T:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{section_name} section must be a table.")

    allowed_keys = {field.name for field in fields(cls)}
    unexpected_keys = sorted(set(raw) - allowed_keys)
    if unexpected_keys:
        joined = ", ".join(unexpected_keys)
        raise ValueError(f"{section_name} section has unknown keys: {joined}.")

    return cls(**dict(raw))


def parameters_from_mapping(data: Mapping[str, Any]) -> ModelParameters:
    """Build model parameters from a nested mapping.

    The expected shape is a nested mapping with these top-level sections:

    case
    battery
    power_export
    compute
    hydrogen
    marine

    Optional top-level keys:
    - time_step_h
    - power_balance_tolerance_mw
    """

    allowed_top_level_keys = {
        "case",
        "battery",
        "power_export",
        "compute",
        "hydrogen",
        "marine",
        "time_step_h",
        "power_balance_tolerance_mw",
    }
    unexpected_keys = sorted(set(data) - allowed_top_level_keys)
    if unexpected_keys:
        joined = ", ".join(unexpected_keys)
        raise ValueError(f"parameter mapping has unknown keys: {joined}.")

    params = ModelParameters(
        case=_load_dataclass_section("case", data.get("case"), CaseMetadata),
        battery=_load_dataclass_section("battery", data.get("battery"), BatteryParams),
        power_export=_load_dataclass_section(
            "power_export", data.get("power_export"), PowerExportParams
        ),
        compute=_load_dataclass_section("compute", data.get("compute"), ComputeLoadParams),
        hydrogen=_load_dataclass_section("hydrogen", data.get("hydrogen"), HydrogenParams),
        marine=_load_dataclass_section("marine", data.get("marine"), MarineLoadParams),
        time_step_h=float(data.get("time_step_h", 1.0)),
        power_balance_tolerance_mw=float(data.get("power_balance_tolerance_mw", 1e-6)),
    )
    params.validate()
    return params


def load_parameters_from_yaml(path: str | Path) -> ModelParameters:
    """Load model parameters from a YAML file."""

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env setup
        raise ModuleNotFoundError(
            "PyYAML is required to load YAML parameter files."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError("YAML root must be a mapping.")
    return parameters_from_mapping(data)


def load_parameters_from_file(path: str | Path) -> ModelParameters:
    """Load model parameters from a YAML file based on file suffix."""

    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return load_parameters_from_yaml(path)
    raise ValueError("Unsupported parameter file suffix. Use .yaml or .yml.")


def default_parameters() -> ModelParameters:
    """Return a transparent default parameter set for 24h examples.

    Values are screening assumptions, not project-calibrated engineering data.
    Replace them with the parameter evidence table before using model results in
    the report.
    """

    params = ModelParameters()
    params.validate()
    return params
