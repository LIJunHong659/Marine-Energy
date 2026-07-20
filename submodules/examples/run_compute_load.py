from pathlib import Path

from bluehub_submodules import load_parameters_from_file
from bluehub_submodules.compute_load import evaluate_compute_load


CASE_ID = "common_case_v1"
SCENARIO_TYPE = "interface_smoke"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "parameters.yaml"


def main() -> None:
    params = load_parameters_from_file(CONFIG_PATH)
    compute_params = params.compute

    if compute_params.compute_power_max_mw <= 0.0:
        requested_facility_power_mw = 0.0
    else:
        requested_facility_power_mw = min(
            compute_params.compute_power_max_mw,
            max(compute_params.compute_power_min_mw, 80.0),
        )

    result = evaluate_compute_load(
        requested_facility_power_mw=requested_facility_power_mw,
        params=compute_params,
        time_step_h=params.time_step_h,
    )

    print(f"caseId: {CASE_ID}")
    print(f"scenarioType: {SCENARIO_TYPE}")
    print("model: compute_load")
    print(f"requested_facility_power_mw_at_mp_dc_facility: {result.requested_facility_power_mw}")
    print(f"facility_power_mw_at_mp_dc_facility: {result.facility_power_mw}")
    print(f"it_power_mw: {result.it_power_mw}")
    print(f"service_mwh_it: {result.service_mwh_it}")
    print(f"revenue_cny: {result.revenue_cny}")
    print(f"variable_cost_cny: {result.variable_cost_cny}")
    print(f"violations: {list(result.violations)}")


if __name__ == "__main__":
    main()
