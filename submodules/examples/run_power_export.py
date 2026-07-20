from pathlib import Path

from bluehub_submodules import load_parameters_from_file
from bluehub_submodules.power_export import evaluate_power_export


CASE_ID = "common_case_v1"
SCENARIO_TYPE = "interface_smoke"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "parameters.yaml"


def main() -> None:
    params = load_parameters_from_file(CONFIG_PATH)
    export_params = params.power_export

    feasible_max_mw = min(
        export_params.cable_capacity_mw,
        export_params.grid_accept_max_mw,
    )
    requested_power_mw = min(100.0, feasible_max_mw)

    result = evaluate_power_export(
        requested_power_mw=requested_power_mw,
        params=export_params,
        time_step_h=params.time_step_h,
    )

    print(f"caseId: {CASE_ID}")
    print(f"scenarioType: {SCENARIO_TYPE}")
    print("model: power_export")
    print(f"requested_power_mw: {result.requested_power_mw}")
    print(f"exported_power_mw_at_mp_cable_send: {result.exported_power_mw}")
    print(f"delivered_power_mw_at_mp_cable_receive: {result.delivered_power_mw}")
    print(f"lost_power_mw: {result.lost_power_mw}")
    print(f"revenue_cny: {result.revenue_cny}")
    print(f"variable_cost_cny: {result.variable_cost_cny}")
    print(f"violations: {list(result.violations)}")


if __name__ == "__main__":
    main()
