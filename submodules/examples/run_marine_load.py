from pathlib import Path

from bluehub_submodules import load_parameters_from_file
from bluehub_submodules.marine_load import evaluate_marine_load


CASE_ID = "common_case_v1"
SCENARIO_TYPE = "interface_smoke"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "parameters.yaml"


def main() -> None:
    params = load_parameters_from_file(CONFIG_PATH)
    marine_params = params.marine

    requested_power_mw = marine_params.total_load_mw
    available_power_mw = requested_power_mw

    result = evaluate_marine_load(
        available_power_mw=available_power_mw,
        params=marine_params,
        time_step_h=params.time_step_h,
        requested_power_mw=requested_power_mw,
    )

    print(f"caseId: {CASE_ID}")
    print(f"scenarioType: {SCENARIO_TYPE}")
    print("model: marine_load")
    print(f"available_power_mw: {available_power_mw}")
    print(f"requested_power_mw: {result.requested_power_mw}")
    print(f"served_power_mw: {result.served_power_mw}")
    print(f"unmet_power_mw: {result.unmet_power_mw}")
    print(f"revenue_cny: {result.revenue_cny}")
    print(f"unmet_penalty_cny: {result.unmet_penalty_cny}")
    print(f"violations: {list(result.violations)}")


if __name__ == "__main__":
    main()
