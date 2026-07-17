"""Hydrogen conversion and inventory contracts for Phase 3 / S2."""

from __future__ import annotations

from dataclasses import dataclass

from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters


@dataclass(frozen=True)
class HydrogenSpec:
    """Physical and commercial assumptions for the electrolyzer value chain.

    ``system_sec_kwh_per_kg`` is a full electrolyzer-system boundary.  Water is
    therefore costed as a material input only; this module deliberately does
    not add separate desalination or compression electricity to the power
    balance.
    """

    power_mw: float
    storage_capacity_kg: float
    system_sec_kwh_per_kg: float
    storage_loss_per_hour: float
    initial_inventory_fraction: float
    sale_price_cny_per_kg: float
    transport_cost_cny_per_kg: float
    variable_cost_cny_per_kg: float
    water_consumption_m3_per_kg: float
    water_cost_cny_per_m3: float

    @property
    def kg_per_mwh(self) -> float:
        """Hydrogen output implied by one MWh of system electricity."""
        return 1_000.0 / self.system_sec_kwh_per_kg

    @property
    def initial_inventory_kg(self) -> float:
        return self.initial_inventory_fraction * self.storage_capacity_kg

    @property
    def net_sale_value_cny_per_kg(self) -> float:
        return self.sale_price_cny_per_kg - self.transport_cost_cny_per_kg

    @property
    def conversion_and_water_cost_cny_per_kg(self) -> float:
        return self.variable_cost_cny_per_kg + (
            self.water_consumption_m3_per_kg * self.water_cost_cny_per_m3
        )

    def validate(self) -> None:
        if self.power_mw <= 0.0:
            raise ValueError("S2 electrolyzer power must be positive")
        if self.storage_capacity_kg < 0.0:
            raise ValueError("hydrogen storage capacity must be non-negative")
        if self.system_sec_kwh_per_kg <= 0.0:
            raise ValueError("hydrogen system SEC must be positive")
        if not 0.0 <= self.storage_loss_per_hour < 1.0:
            raise ValueError("hydrogen storage loss must lie in [0, 1)")
        if not 0.0 <= self.initial_inventory_fraction <= 1.0:
            raise ValueError("initial hydrogen inventory fraction must lie in [0, 1]")
        if min(
            self.sale_price_cny_per_kg,
            self.transport_cost_cny_per_kg,
            self.variable_cost_cny_per_kg,
            self.water_consumption_m3_per_kg,
            self.water_cost_cny_per_m3,
        ) < 0.0:
            raise ValueError("hydrogen prices, costs and water use must be non-negative")


def hydrogen_spec_from_parameters(
    config: SystemConfiguration,
    params: TechnologyParameters,
    scenario: ScenarioDefinition,
) -> HydrogenSpec:
    """Build a unit-consistent S2 specification from auditable inputs."""
    specification = HydrogenSpec(
        power_mw=config.electrolyzer_power_mw,
        storage_capacity_kg=config.hydrogen_storage_kg,
        system_sec_kwh_per_kg=params.value("hydrogen_sec_system"),
        storage_loss_per_hour=params.value("hydrogen_storage_loss_per_hour"),
        initial_inventory_fraction=config.initial_hydrogen_inventory_fraction,
        sale_price_cny_per_kg=(
            params.value("hydrogen_sale_price") * scenario.hydrogen_price_multiplier
        ),
        transport_cost_cny_per_kg=params.value("hydrogen_transport_cost"),
        variable_cost_cny_per_kg=params.value("electrolyzer_variable_cost"),
        water_consumption_m3_per_kg=params.value("hydrogen_water_consumption"),
        water_cost_cny_per_m3=params.value("desalinated_water_cost"),
    )
    specification.validate()
    return specification
