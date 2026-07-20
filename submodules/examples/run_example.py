from pathlib import Path

from bluehub_submodules import (
    default_parameters,
    ModelParameters,
    load_parameters_from_file,
    simple_greedy_dispatch,
    summarize_results,
)


SUMMARY_LABELS = {
    "hours": "Hours / 小时",
    "export_sent_mwh": "Export sent (MWh) / 外送电量（MWh）",
    "export_delivered_mwh": "Export delivered (MWh) / 实际送出电量（MWh）",
    "compute_service_mwh_it": "Compute service (MWh IT) / 算力服务量（MWh IT）",
    "hydrogen_produced_kg": "Hydrogen produced (kg) / 制氢产量（kg）",
    "hydrogen_delivered_kg": "Hydrogen delivered (kg) / 交付氢气量（kg）",
    "marine_served_mwh": "Marine served (MWh) / 海洋负载服务量（MWh）",
    "curtailment_mwh": "Curtailment (MWh) / 弃电量（MWh）",
    "operating_margin_cny": "Operating margin (CNY) / 经营毛利（元）",
    "total_revenue_cny": "Total revenue (CNY) / 总收入（元）",
    "total_cost_cny": "Total cost (CNY) / 总成本（元）",
    "max_abs_balance_residual_mw": "Max abs balance residual (MW) / 最大平衡残差（MW）",
    "violation_count": "Violation count / 违规次数",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_CASE_PATH = (
    PROJECT_ROOT / "4.1边界与口径" / "common_case_v1" / "common_case_v1.yaml"
)
LEGACY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "parameters.yaml"


def load_example_parameters() -> ModelParameters:
    if COMMON_CASE_PATH.exists():
        return load_parameters_from_file(COMMON_CASE_PATH)
    if LEGACY_CONFIG_PATH.exists():
        return load_parameters_from_file(LEGACY_CONFIG_PATH)
    return default_parameters()


def main() -> None:
    params = load_example_parameters()
    results = simple_greedy_dispatch(params)
    summary = summarize_results(results)
    print(f"Case ID / 参数集: {params.case.case_id}")
    print(f"Scenario / 场景: {params.case.scenario_type}")
    print(f"Site / 场址: {params.case.site_id}")
    print(f"Timezone / 时区: {params.case.timezone}")
    for key, value in summary.items():
        print(f"{SUMMARY_LABELS.get(key, key)}: {value}")


if __name__ == "__main__":
    main()
