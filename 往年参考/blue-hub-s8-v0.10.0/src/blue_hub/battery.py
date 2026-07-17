"""Battery parameterization and state-ledger diagnostics for Phase 2 / S1."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import numpy.typing as npt

from blue_hub.schemas import SystemConfiguration, TechnologyParameters

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BatterySpec:
    """Operational battery parameters used by the hourly energy model."""

    power_mw: float
    energy_mwh: float
    charge_efficiency: float
    discharge_efficiency: float
    self_discharge_per_hour: float
    soc_min: float
    soc_max: float
    initial_soc: float
    degradation_cost_cny_per_mwh_throughput: float
    reserve_power_mw: float
    reserve_duration_h: float

    @property
    def minimum_energy_mwh(self) -> float:
        return self.soc_min * self.energy_mwh

    @property
    def maximum_energy_mwh(self) -> float:
        return self.soc_max * self.energy_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc * self.energy_mwh

    @property
    def reserve_energy_above_minimum_mwh(self) -> float:
        return self.reserve_power_mw * self.reserve_duration_h / self.discharge_efficiency

    @property
    def scheduled_minimum_energy_mwh(self) -> float:
        return self.minimum_energy_mwh + self.reserve_energy_above_minimum_mwh

    def validate(self) -> None:
        if self.power_mw <= 0.0 or self.energy_mwh <= 0.0:
            raise ValueError("S1 battery power and energy capacities must be positive")
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError("charge efficiency must lie in (0, 1]")
        if not 0.0 < self.discharge_efficiency <= 1.0:
            raise ValueError("discharge efficiency must lie in (0, 1]")
        if not 0.0 <= self.self_discharge_per_hour < 1.0:
            raise ValueError("self-discharge must lie in [0, 1)")
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("SOC bounds must satisfy 0 <= min < max <= 1")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial SOC must lie within operational bounds")
        if self.degradation_cost_cny_per_mwh_throughput <= 0.0:
            raise ValueError("battery degradation cost must be positive")
        if self.reserve_power_mw < 0.0 or self.reserve_duration_h < 0.0:
            raise ValueError("battery reserve settings must be non-negative")
        if self.reserve_power_mw > self.power_mw:
            raise ValueError("battery reserve power exceeds installed power")
        if self.scheduled_minimum_energy_mwh > self.maximum_energy_mwh:
            raise ValueError("battery reserve energy exceeds the usable SOC window")
        if self.initial_energy_mwh < self.scheduled_minimum_energy_mwh:
            raise ValueError("initial battery energy is below the scheduled reserve floor")


def battery_spec_from_parameters(
    config: SystemConfiguration,
    params: TechnologyParameters,
) -> BatterySpec:
    """Build a symmetric-efficiency battery from the round-trip efficiency contract."""
    one_way_efficiency = sqrt(params.value("battery_round_trip_efficiency"))
    specification = BatterySpec(
        power_mw=config.battery_power_mw,
        energy_mwh=config.battery_energy_mwh,
        charge_efficiency=one_way_efficiency,
        discharge_efficiency=one_way_efficiency,
        self_discharge_per_hour=params.value("battery_self_discharge_per_hour"),
        soc_min=params.value("battery_soc_min"),
        soc_max=params.value("battery_soc_max"),
        initial_soc=config.initial_battery_soc_fraction,
        degradation_cost_cny_per_mwh_throughput=params.value("battery_degradation_cost"),
        reserve_power_mw=config.battery_reserve_power_mw,
        reserve_duration_h=config.battery_reserve_duration_h,
    )
    specification.validate()
    return specification


@dataclass(frozen=True)
class BatteryLossLedger:
    """Per-step standing and conversion losses in MWh."""

    standing_loss_mwh: FloatArray
    charge_conversion_loss_mwh: FloatArray
    discharge_conversion_loss_mwh: FloatArray
    total_loss_mwh: FloatArray


def calculate_battery_losses(
    energy_start_mwh: npt.ArrayLike,
    charge_mw: npt.ArrayLike,
    discharge_mw: npt.ArrayLike,
    specification: BatterySpec,
    time_step_hours: float,
) -> BatteryLossLedger:
    """Return a transparent loss ledger consistent with the state equation."""
    energy = np.asarray(energy_start_mwh, dtype=float)
    charge = np.asarray(charge_mw, dtype=float)
    discharge = np.asarray(discharge_mw, dtype=float)
    standing = specification.self_discharge_per_hour * energy
    charge_loss = (1.0 - specification.charge_efficiency) * charge * time_step_hours
    discharge_loss = (1.0 / specification.discharge_efficiency - 1.0) * discharge * time_step_hours
    total = standing + charge_loss + discharge_loss
    return BatteryLossLedger(standing, charge_loss, discharge_loss, total)
