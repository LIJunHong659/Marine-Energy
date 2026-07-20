from pathlib import Path

from bluehub_submodules import DispatchRequest, evaluate_integrated_hour, load_parameters_from_file


CASE_ID = "common_case_v1"
SCENARIO_TYPE = "interface_smoke"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "parameters.yaml"


def main() -> None:
    params = load_parameters_from_file(CONFIG_PATH)
    available_power_mw = max(220.0, params.marine.total_load_mw)
    remaining_power_mw = available_power_mw

    marine_power_mw = min(params.marine.total_load_mw, remaining_power_mw)
    remaining_power_mw -= marine_power_mw

    grid_power_mw = min(
        remaining_power_mw * 0.5,
        params.power_export.cable_capacity_mw,
        params.power_export.grid_accept_max_mw,
    )
    remaining_power_mw -= grid_power_mw

    if params.compute.compute_power_max_mw <= 0.0:
        compute_power_mw = 0.0
    else:
        compute_target_mw = min(
            60.0,
            params.compute.compute_power_max_mw,
        )
        compute_target_mw = max(compute_target_mw, params.compute.compute_power_min_mw)
        compute_power_mw = min(remaining_power_mw, compute_target_mw)
        if 0.0 < compute_power_mw < params.compute.compute_power_min_mw:
            compute_power_mw = 0.0
    remaining_power_mw -= compute_power_mw

    h2_power_mw = min(
        max(remaining_power_mw, 0.0),
        params.hydrogen.electrolyzer_power_max_mw,
    )
    produced_kg = 1000.0 * h2_power_mw * params.time_step_h / params.hydrogen.sec_kwh_per_kg
    h2_pipe_output_kg = min(
        produced_kg,
        params.hydrogen.pipe_capacity_kg_per_h * params.time_step_h,
    )

    request = DispatchRequest(
        grid_power_mw=grid_power_mw,
        compute_power_mw=compute_power_mw,
        h2_power_mw=h2_power_mw,
        marine_power_mw=marine_power_mw,
        h2_pipe_output_kg=h2_pipe_output_kg,
        h2_ship_output_kg=0.0,
    )
    result = evaluate_integrated_hour(
        hour=0,
        available_power_mw=available_power_mw,
        storage_start_kg=0.0,
        request=request,
        params=params,
    )

    print(f"caseId: {CASE_ID}")
    print(f"scenarioType: {SCENARIO_TYPE}")
    print("model: integrated_balance")
    print(f"hour: {result.hour}")
    print(f"available_power_mw_at_mp_03_source_poi: {result.available_power_mw}")
    print(f"marine_power_mw: {result.marine.served_power_mw}")
    print(f"exported_power_mw_at_mp_cable_send: {result.power_export.exported_power_mw}")
    print(f"compute_facility_power_mw_at_mp_dc_facility: {result.compute.facility_power_mw}")
    print(f"electrolyzer_power_mw_at_mp_electrolyzer: {result.hydrogen.electrolyzer_power_mw}")
    print(f"curtailment_mw: {result.curtailment_mw}")
    print(f"offshore_balance_residual_mw: {result.offshore_balance_residual_mw}")
    print(f"operating_margin_cny: {result.objective.operating_margin_cny}")
    print(f"violations: {list(result.violations)}")


if __name__ == "__main__":
    main()
