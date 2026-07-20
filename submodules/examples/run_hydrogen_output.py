from pathlib import Path

from bluehub_submodules import load_parameters_from_file
from bluehub_submodules.hydrogen_output import (
    evaluate_hydrogen_output,
    hydrogen_production_kg,
)


CASE_ID = "common_case_v1"
SCENARIO_TYPE = "interface_smoke"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "parameters.yaml"


def main() -> None:
    params = load_parameters_from_file(CONFIG_PATH)
    h2_params = params.hydrogen

    requested_electrolyzer_power_mw = min(
        50.0,
        h2_params.electrolyzer_power_max_mw,
    )
    produced_kg = hydrogen_production_kg(
        requested_electrolyzer_power_mw,
        h2_params.sec_kwh_per_kg,
        params.time_step_h,
    )
    requested_pipe_output_kg = min(
        produced_kg,
        h2_params.pipe_capacity_kg_per_h * params.time_step_h,
        500.0,
    )

    result = evaluate_hydrogen_output(
        requested_electrolyzer_power_mw=requested_electrolyzer_power_mw,
        requested_pipe_output_kg=requested_pipe_output_kg,
        requested_ship_output_kg=0.0,
        storage_start_kg=0.0,
        params=h2_params,
        time_step_h=params.time_step_h,
    )

    print(f"caseId: {CASE_ID}")
    print(f"scenarioType: {SCENARIO_TYPE}")
    print("model: hydrogen_output")
    print(f"requested_electrolyzer_power_mw_at_mp_electrolyzer: {result.requested_electrolyzer_power_mw}")
    print(f"electrolyzer_power_mw_at_mp_electrolyzer: {result.electrolyzer_power_mw}")
    print(f"produced_kg: {result.produced_kg}")
    print(f"pipe_output_kg: {result.pipe_output_kg}")
    print(f"ship_output_kg: {result.ship_output_kg}")
    print(f"delivered_kg_at_mp_h2_delivery: {result.delivered_kg}")
    print(f"storage_start_kg: {result.storage_start_kg}")
    print(f"storage_end_kg: {result.storage_end_kg}")
    print(f"revenue_cny: {result.revenue_cny}")
    print(f"variable_cost_cny: {result.variable_cost_cny}")
    print(f"violations: {list(result.violations)}")


if __name__ == "__main__":
    main()
