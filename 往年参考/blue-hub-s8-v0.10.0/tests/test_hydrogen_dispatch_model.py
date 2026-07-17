from pathlib import Path

import numpy as np
import pytest

from blue_hub.battery_dispatch_model import run_s1_dispatch
from blue_hub.hydrogen import hydrogen_spec_from_parameters
from blue_hub.hydrogen_dispatch_model import run_s2_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.schemas import TechnologyParameter, TechnologyParameters
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    scenarios = {
        item.scenario_id: item for item in load_scenarios(ROOT / "configs/scenario_matrix.csv")
    }
    s1_config = load_system_configuration(ROOT / "configs/s1_battery_100mw_400mwh.yaml")
    hydrogen_config = load_system_configuration(ROOT / "configs/s2_hydrogen_100mw_72000kg.yaml")
    battery_hydrogen_config = load_system_configuration(
        ROOT / "configs/s2_battery_hydrogen_100mw_400mwh_100mw_72000kg.yaml"
    )
    return parameters, scenarios, s1_config, hydrogen_config, battery_hydrogen_config


def replace_parameter(
    parameters: TechnologyParameters, name: str, value: float
) -> TechnologyParameters:
    items = tuple(
        TechnologyParameter(
            **{
                **item.model_dump(),
                "value_base": value,
                "value_low": min(item.value_low, value),
                "value_high": max(item.value_high, value),
            }
        )
        if item.parameter == name
        else item
        for item in parameters.items
    )
    return TechnologyParameters(items=items)


def test_integrated_sec_conversion_is_applied_once() -> None:
    parameters, scenarios, _, hydrogen_config, _ = inputs()
    specification = hydrogen_spec_from_parameters(hydrogen_config, parameters, scenarios["base"])
    assert specification.kg_per_mwh == pytest.approx(1_000.0 / 57.5)
    assert 100.0 * specification.kg_per_mwh == pytest.approx(1_739.1304347826087)
    assert specification.conversion_and_water_cost_cny_per_kg == pytest.approx(0.59)


def test_zero_hydrogen_capacity_is_exact_s1_ablation() -> None:
    parameters, scenarios, s1_config, _, _ = inputs()
    frame = generate_synthetic_timeseries(168)
    s1 = run_s1_dispatch(frame, parameters, s1_config, scenarios["base"])
    s2 = run_s2_dispatch(frame, parameters, s1_config, scenarios["base"])
    assert s2.metadata["phase"] == "S2_ablation"
    assert s2.kpis["operating_margin_cny"] == pytest.approx(s1.kpis["operating_margin_cny"])
    assert s2.kpis["export_land_mwh"] == pytest.approx(s1.kpis["export_land_mwh"])
    assert s2.kpis["hydrogen_production_kg"] == 0.0


def test_hydrogen_inventory_is_cyclic_and_mass_conserving() -> None:
    parameters, scenarios, _, _, battery_hydrogen_config = inputs()
    result = run_s2_dispatch(
        generate_synthetic_timeseries(168), parameters, battery_hydrogen_config, scenarios["base"]
    )
    hourly = result.hourly
    assert result.kpis["hydrogen_initial_inventory_kg"] == pytest.approx(
        result.kpis["hydrogen_final_inventory_kg"]
    )
    assert result.kpis["max_hydrogen_state_residual_kg"] < parameters.value(
        "hydrogen_state_tolerance"
    )
    assert hourly["hydrogen_inventory_start_kg"].min() >= -1e-8
    assert hourly["hydrogen_inventory_end_kg"].max() <= (
        battery_hydrogen_config.hydrogen_storage_kg + 1e-8
    )
    assert hourly["hydrogen_water_m3"].sum() == pytest.approx(
        hourly["hydrogen_production_kg"].sum()
        * parameters.value("hydrogen_water_consumption")
    )


def test_zero_demand_prevents_unpriced_inventory_build() -> None:
    parameters, scenarios, _, hydrogen_config, _ = inputs()
    frame = generate_synthetic_timeseries(168)
    frame["hydrogen_demand"] = 0.0
    result = run_s2_dispatch(frame, parameters, hydrogen_config, scenarios["base"])
    assert result.kpis["hydrogen_sales_kg"] == pytest.approx(0.0, abs=1e-8)
    assert result.kpis["hydrogen_production_kg"] == pytest.approx(0.0, abs=1e-8)
    assert result.kpis["hydrogen_gross_revenue_cny"] == pytest.approx(0.0, abs=1e-8)


def test_higher_sec_reduces_hydrogen_output_when_energy_is_constrained() -> None:
    parameters, scenarios, _, hydrogen_config, _ = inputs()
    frame = generate_synthetic_timeseries(72)
    frame["hydrogen_demand"] = 1_000_000.0
    no_storage_no_export = hydrogen_config.model_copy(
        update={"tx_capacity_mw": 0.0, "hydrogen_storage_kg": 0.0}
    )
    base = run_s2_dispatch(frame, parameters, no_storage_no_export, scenarios["base"])
    high_sec = run_s2_dispatch(
        frame,
        replace_parameter(parameters, "hydrogen_sec_system", 65.0),
        no_storage_no_export,
        scenarios["base"],
    )
    assert base.kpis["hydrogen_production_kg"] > high_sec.kpis["hydrogen_production_kg"]
    assert np.allclose(
        base.hourly["hydrogen_sale_kg"], base.hourly["hydrogen_production_kg"], atol=1e-8
    )
