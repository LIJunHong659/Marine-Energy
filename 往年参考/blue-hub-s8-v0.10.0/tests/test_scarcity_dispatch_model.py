from pathlib import Path

import numpy as np
import pytest

from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.scarcity_dispatch_model import run_s5_dispatch
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    parameters = load_parameters(ROOT / "configs/technology_parameters.csv")
    scenario = {
        item.scenario_id: item for item in load_scenarios(ROOT / "configs/scenario_matrix.csv")
    }["base"].model_copy(update={"hydrogen_price_multiplier": 0.75})
    config = load_system_configuration(
        ROOT / "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )
    return parameters, scenario, config


def without_assets(config):
    return config.model_copy(
        update={
            "config_id": "S5_none_test",
            "battery_power_mw": 0.0,
            "battery_energy_mwh": 0.0,
            "electrolyzer_power_mw": 0.0,
            "hydrogen_storage_kg": 0.0,
            "fuel_cell_power_mw": 0.0,
            "compute_it_capacity_mw": 0.0,
            "subsea_fiber_service_capacity_mw_it": 0.0,
            "initial_battery_soc_fraction": 0.0,
            "initial_hydrogen_inventory_fraction": 0.0,
        }
    )


def test_mainland_absorption_is_separate_from_physical_cable_capacity() -> None:
    parameters, scenario, config = inputs()
    frame = generate_synthetic_timeseries(48)
    frame["grid_absorption_factor"] = 0.25
    result = run_s5_dispatch(frame, parameters, without_assets(config), scenario)
    hourly = result.hourly
    assert np.allclose(
        hourly["tx_available_capacity_mw"],
        hourly["tx_physical_available_capacity_mw"] * 0.25,
    )
    assert (hourly["export_send_mw"] <= hourly["tx_available_capacity_mw"] + 1e-8).all()
    assert result.kpis["curtailment_mwh"] > 0.0


def test_nationwide_spot_compute_absorbs_constrained_surplus() -> None:
    parameters, scenario, config = inputs()
    frame = generate_synthetic_timeseries(72)
    frame["grid_absorption_factor"] = 0.25
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    frame["national_compute_demand_mw_it"] = 200.0
    frame["national_compute_price_cny_per_mwh_it"] = 360.0
    direct = run_s5_dispatch(frame, parameters, without_assets(config), scenario)
    compute = run_s5_dispatch(
        frame,
        parameters,
        without_assets(config).model_copy(
            update={
                "config_id": "S5_compute_test",
                "compute_it_capacity_mw": 200.0,
                "subsea_fiber_service_capacity_mw_it": 200.0,
            }
        ),
        scenario,
    )
    assert compute.kpis["spot_compute_service_mwh_it"] > 0.0
    assert compute.kpis["curtailment_mwh"] < direct.kpis["curtailment_mwh"]
    assert (
        compute.hourly["spot_compute_completed_mwh_it"]
        <= compute.hourly["national_compute_demand_mwh_it"] + 1e-8
    ).all()
    assert compute.kpis["max_offshore_balance_residual_mw"] < parameters.value(
        "power_balance_tolerance"
    )


def test_stored_hydrogen_returns_power_and_reduces_lull_eens() -> None:
    parameters, scenario, config = inputs()
    frame = generate_synthetic_timeseries(168)
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    frame["hydrogen_demand"] = 0.0
    frame["grid_absorption_factor"] = 1.0
    frame.loc[48:119, "wind_availability"] = 0.0
    direct = run_s5_dispatch(frame, parameters, without_assets(config), scenario)
    hydrogen = run_s5_dispatch(
        frame,
        parameters,
        without_assets(config).model_copy(
            update={
                "config_id": "S5_hydrogen_resilience_test",
                "electrolyzer_power_mw": 200.0,
                "hydrogen_storage_kg": 500_000.0,
                "fuel_cell_power_mw": 20.0,
                "initial_hydrogen_inventory_fraction": 0.5,
            }
        ),
        scenario,
    )
    assert direct.kpis["eens_mwh"] > 0.0
    assert hydrogen.kpis["fuel_cell_generation_mwh"] > 0.0
    assert hydrogen.kpis["eens_mwh"] < direct.kpis["eens_mwh"]
    assert hydrogen.kpis["hydrogen_terminal_state_error_kg"] == pytest.approx(0.0, abs=1e-7)
    assert hydrogen.kpis["max_hydrogen_state_residual_kg"] < parameters.value(
        "hydrogen_state_tolerance"
    )


def test_s5_supports_offshore_pv_in_the_shared_power_balance() -> None:
    parameters, scenario, config = inputs()
    frame = generate_synthetic_timeseries(24)
    pv_config = without_assets(config).model_copy(update={"pv_capacity_mw": 100.0})
    result = run_s5_dispatch(frame, parameters, pv_config, scenario)
    assert result.kpis["pv_generation_mwh"] > 0.0
    assert result.kpis["renewable_generation_mwh"] == pytest.approx(
        result.kpis["wind_generation_mwh"] + result.kpis["pv_generation_mwh"]
    )
