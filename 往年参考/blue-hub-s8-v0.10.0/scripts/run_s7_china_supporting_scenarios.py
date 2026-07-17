"""Resume the S7 supporting cases after the primary strategy counterfactual.

The script is deliberately separate from ``run_s7_china_value_analysis.py``:
the primary 8760-hour strategy table is saved first, and this runner adds the
price-pair, portability, cost, multi-resource and resilience checks without
discarding completed results if a long batch is interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_s7_china_value_analysis import (
    _china_reference_timeseries,
    _counterfactual_record,
    _design,
    _load_limits,
    _plot_allocation,
    _plot_frontier,
    _record,
    _scenario,
    _with_markets,
    _without_assets,
    _write,
)

from blue_hub import __version__
from blue_hub.china_counterfactual_model import (
    evaluate_s7_design,
    load_china_infrastructure_cost_cases,
)
from blue_hub.investment_planning_model import (
    PlanningLimits,
    load_investment_cost_cases,
    optimal_system_configuration,
)
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.scarcity_dispatch_model import run_s5_dispatch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, output: Path, hours: int) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters(root / "configs/technology_parameters.csv")
    base_scenario = load_scenarios(root / "configs/scenario_matrix.csv")[0]
    base_config = load_system_configuration(
        root / "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )
    flexible_cases = {
        item.cost_case_id: item
        for item in load_investment_cost_cases(
            root / "configs/s7_china_flexible_cost_cases.csv"
        )
    }
    infrastructure_cases = {
        item.cost_case_id: item
        for item in load_china_infrastructure_cost_cases(
            root / "configs/s7_china_infrastructure_cost_cases.csv"
        )
    }
    reference_flexible = flexible_cases["china_reference"]
    reference_common = infrastructure_cases["china_reference"]
    limits = _load_limits(root / "configs/s7_china_planning_limits.yaml")
    zero = PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    frame = _china_reference_timeseries(hours)
    cluster_landing = 450.0
    direct_design = _design(
        "s7_resume_direct", "direct_existing", tx=700.0, landing=cluster_landing,
        flexible=False, hub=False,
    )
    direct = evaluate_s7_design(
        frame, parameters, base_config,
        _scenario(base_scenario, "s7_resume_direct", hydrogen_price=25.0),
        reference_flexible, reference_common, zero, direct_design,
    )
    hub_design = _design(
        "s7_resume_hub", "integrated_hub", tx=700.0, landing=cluster_landing,
        flexible=True, hub=True,
    )

    frontier_rows: list[dict[str, object]] = []
    frontier_results = {}
    for compute_price in (1_000.0, 2_000.0):
        for hydrogen_price in (20.0, 30.0):
            result = evaluate_s7_design(
                _with_markets(frame, compute_price=compute_price),
                parameters, base_config,
                _scenario(
                    base_scenario,
                    f"s7_resume_c{compute_price:g}_h{hydrogen_price:g}",
                    hydrogen_price=hydrogen_price,
                ),
                reference_flexible, reference_common, limits, hub_design,
            )
            frontier_results[(compute_price, hydrogen_price)] = result
            row = _counterfactual_record(result, direct)
            row["compute_price_cny_per_mwh_it"] = compute_price
            row["hydrogen_price_cny_per_kg"] = hydrogen_price
            frontier_rows.append(row)
    frontier = pd.DataFrame(frontier_rows)
    _write(frontier, output / "s7_contract_value_frontier.csv")
    _plot_frontier(frontier, figures / "s7_contract_value_frontier.png")
    contract_result = frontier_results[(2_000.0, 30.0)]
    contract_result.planning.hourly.to_csv(
        output / "s7_representative_contract_hourly.csv", index=False, float_format="%.8g"
    )
    _plot_allocation(contract_result, figures / "s7_representative_energy_allocation.png")

    portability_rows: list[dict[str, object]] = []
    for fraction in (0.15, 0.35, 0.60):
        result = (
            contract_result
            if fraction == 0.35
            else evaluate_s7_design(
                _with_markets(
                    frame, compute_price=2_000.0, compute_flexible_fraction=fraction
                ),
                parameters, base_config,
                _scenario(
                    base_scenario, f"s7_portability_{fraction:.2f}", hydrogen_price=30.0
                ),
                reference_flexible, reference_common, limits, hub_design,
            )
        )
        row = _counterfactual_record(result, direct)
        row["national_compute_flexible_fraction"] = fraction
        portability_rows.append(row)
    portability = pd.DataFrame(portability_rows)
    _write(portability, output / "s7_compute_portability_sensitivity.csv")

    cost_rows: list[dict[str, object]] = []
    contract_frame = _with_markets(frame, compute_price=2_000.0)
    for suffix in ("low", "reference", "high"):
        if suffix == "reference":
            result, baseline = contract_result, direct
        else:
            flexible = flexible_cases[f"china_{suffix}"]
            common = infrastructure_cases[f"china_{suffix}"]
            baseline = evaluate_s7_design(
                contract_frame, parameters, base_config,
                _scenario(base_scenario, f"s7_cost_{suffix}_direct", hydrogen_price=30.0),
                flexible, common, zero,
                _design(
                    f"s7_cost_{suffix}_direct", "direct_existing", tx=700.0,
                    landing=cluster_landing, flexible=False, hub=False,
                ),
            )
            result = evaluate_s7_design(
                contract_frame, parameters, base_config,
                _scenario(base_scenario, f"s7_cost_{suffix}_hub", hydrogen_price=30.0),
                flexible, common, limits, hub_design,
            )
        row = _counterfactual_record(result, baseline)
        row["cost_case"] = suffix
        cost_rows.append(row)
    costs = pd.DataFrame(cost_rows)
    _write(costs, output / "s7_cost_uncertainty.csv")

    resource_rows: list[dict[str, object]] = []
    for resource_case, pv, wave in (("wind_only", 0.0, 0.0), ("wind_pv_wave", 200.0, 50.0)):
        result = (
            contract_result
            if resource_case == "wind_only"
            else evaluate_s7_design(
                contract_frame, parameters, base_config,
                _scenario(base_scenario, "s7_wind_pv_wave", hydrogen_price=30.0),
                reference_flexible, reference_common, limits,
                _design(
                    "s7_wind_pv_wave", "integrated_hub", tx=700.0,
                    landing=cluster_landing, flexible=True, hub=True, pv=pv, wave=wave,
                ),
            )
        )
        row = _record(result)
        row["resource_case"] = resource_case
        row["renewable_output_cv"] = float(
            result.planning.hourly["renewable_available_mw"].std()
            / result.planning.hourly["renewable_available_mw"].mean()
        )
        row["incremental_full_net_vs_wind_only_cny_per_year"] = (
            result.kpis["full_project_net_annual_value_cny"]
            - contract_result.kpis["full_project_net_annual_value_cny"]
        )
        resource_rows.append(row)
    resources = pd.DataFrame(resource_rows)
    _write(resources, output / "s7_resource_addon_screening.csv")

    stress_rows: list[dict[str, object]] = []
    outage_frame = contract_frame.copy()
    outage_frame["grid_absorption_limit_mw"] = (
        cluster_landing * outage_frame["landing_demand_factor"]
    )
    outage_scenario = _scenario(
        base_scenario, "s7_fixed_hub_cable_outage_72h", hydrogen_price=30.0
    ).model_copy(update={"tx_outage_hours": 72, "tx_outage_start_hour": 1_000})
    fixed_hub = run_s5_dispatch(
        outage_frame, parameters,
        optimal_system_configuration(
            base_config, contract_result.planning, config_id="s7_fixed_contract_hub"
        ),
        outage_scenario,
    )
    fixed_direct = run_s5_dispatch(
        outage_frame, parameters,
        _without_assets(base_config, config_id="s7_fixed_direct", tx_capacity_mw=700.0),
        outage_scenario,
    )
    for strategy, result in (("direct", fixed_direct), ("integrated_hub", fixed_hub)):
        event = result.hourly.iloc[1_000:1_072]
        stress_rows.append(
            {
                "stress_case": "fixed_capacity_cable_outage_72h",
                "strategy": strategy,
                "event_curtailment_mwh": float(event["curtailment_mw"].sum()),
                "event_compute_service_mwh_it": float(
                    event["spot_compute_completed_mwh_it"].sum()
                ),
                "event_hydrogen_production_kg": float(event["hydrogen_production_kg"].sum()),
                "event_eens_mwh": float(event["unmet_critical_load_mw"].sum()),
                "max_offshore_balance_residual_mw": result.kpis[
                    "max_offshore_balance_residual_mw"
                ],
            }
        )

    reliability_frame = frame.copy()
    reliability_frame["national_compute_demand_mw_it"] = 0.0
    reliability_frame["hydrogen_demand"] = 0.0
    reliability_frame["critical_load"] = 50.0
    lull_start = 4_000
    lull_end = 4_720
    reliability_frame.loc[lull_start:lull_end - 1, "wind_availability"] = 0.03
    reliability_scenario = _scenario(
        base_scenario, "s7_30day_low_wind_reliability", hydrogen_price=20.0
    )
    reliability_direct = evaluate_s7_design(
        reliability_frame, parameters, base_config, reliability_scenario,
        reference_flexible, reference_common, zero,
        _design("s7_reliability_direct", "direct_existing", tx=700.0,
                landing=cluster_landing, flexible=False, hub=False),
    )
    reliability_hub = evaluate_s7_design(
        reliability_frame, parameters, base_config, reliability_scenario,
        reference_flexible, reference_common, limits,
        _design("s7_reliability_hub", "integrated_hub_reliability", tx=700.0,
                landing=cluster_landing, flexible=True, hub=True),
    )
    for strategy, result in (("direct", reliability_direct), ("integrated_hub", reliability_hub)):
        lull = result.planning.hourly.iloc[lull_start:lull_end]
        stress_rows.append(
            {
                "stress_case": "endogenous_30day_low_wind",
                "strategy": strategy,
                "event_curtailment_mwh": float(lull["curtailment_mw"].sum()),
                "event_compute_service_mwh_it": 0.0,
                "event_hydrogen_production_kg": float(lull["hydrogen_production_kg"].sum()),
                "event_eens_mwh": float(lull["unmet_critical_load_mw"].sum()),
                "battery_power_mw": result.kpis["battery_power_mw"],
                "battery_energy_mwh": result.kpis["battery_energy_mwh"],
                "electrolyzer_power_mw": result.kpis["electrolyzer_power_mw"],
                "hydrogen_storage_kg": result.kpis["hydrogen_storage_kg"],
                "fuel_cell_power_mw": result.kpis["fuel_cell_power_mw"],
                "incremental_full_net_vs_direct_cny_per_year": (
                    result.kpis["full_project_net_annual_value_cny"]
                    - reliability_direct.kpis["full_project_net_annual_value_cny"]
                ),
                "max_offshore_balance_residual_mw": result.kpis[
                    "max_offshore_balance_residual_mw"
                ],
            }
        )
    stress = pd.DataFrame(stress_rows)
    _write(stress, output / "s7_fixed_and_reliability_stress_tests.csv")

    strategy_table = pd.read_csv(output / "s7_strategy_counterfactual.csv")
    cable_only = strategy_table.loc[strategy_table["strategy"] == "cable_only"].iloc[0]
    reinforced = strategy_table.loc[strategy_table["strategy"] == "grid_reinforced"].iloc[0]
    hub_reference = strategy_table.loc[strategy_table["strategy"] == "integrated_hub"].iloc[0]
    claims = pd.DataFrame(
        [
            {
                "claim_id": "P1_national_severe_curtailment",
                "assessment": "not_supported_as_a_current_national_generalization",
                "model_metric": 0.041,
                "unit": "fraction",
                "interpretation": "The national reference is calibrated to 4.1% non-utilization; severe curtailment must be shown at a project landing point.",
            },
            {
                "claim_id": "P2_local_landing_bottleneck",
                "assessment": "supported_conditionally",
                "model_metric": 1.0 - float(direct.kpis["renewable_utilization_rate"]),
                "unit": "fraction",
                "interpretation": "A 450 MW landing limit creates local curtailment under the same resource year.",
            },
            {
                "claim_id": "P3_cable_expansion_alone",
                "assessment": "not_sufficient_when_landing_grid_is_fixed",
                "model_metric": float(cable_only["utilization_gain_vs_direct_percentage_points"]),
                "unit": "percentage_points",
                "interpretation": "A 700 to 1000 MW cable increase does not relax an unchanged 450 MW mainland landing limit.",
            },
            {
                "claim_id": "P4_high_cost_low_price_conflict",
                "assessment": "supported_in_reference_screening_case",
                "model_metric": float(direct.kpis["full_cost_recovery_ratio"]),
                "unit": "ratio",
                "interpretation": "Direct export does not recover screened deep-offshore common costs at the reference price profile.",
            },
            {
                "claim_id": "V1_compute_value",
                "assessment": "contract_and_portability_conditional",
                "model_metric": float(contract_result.kpis["compute_it_capacity_mw"]),
                "unit": "MW_IT",
                "interpretation": "Compute enters at 2000 CNY/MWh-IT and 35% movable demand; the price and task-structure frontier remains explicit.",
            },
            {
                "claim_id": "V2_hydrogen_value",
                "assessment": "price_and_pipeline_conditional",
                "model_metric": float(contract_result.kpis["electrolyzer_power_mw"]),
                "unit": "MW",
                "interpretation": "Hydrogen capacity is selected only after a distance-dependent pipeline is paid for and an offtake price is present.",
            },
            {
                "claim_id": "V3_grid_or_hub_choice",
                "assessment": "scenario_dependent",
                "model_metric": float(
                    hub_reference["full_project_net_annual_value_cny"]
                    - reinforced["full_project_net_annual_value_cny"]
                ),
                "unit": "CNY/year",
                "interpretation": "Grid reinforcement and a hub must be compared by common resource, landing capacity and contractual value, not by a universal claim.",
            },
        ]
    )
    _write(claims, output / "s7_claim_assessment.csv")

    output_files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "phase": "S7",
        "model_version": __version__,
        "simulation_hours": hours,
        "calibration": {
            "wind_full_load_hours_target": 3300.0,
            "national_utilization_reference": 0.959,
            "local_cluster_landing_limit_mw": cluster_landing,
            "offshore_distance_km": 200.0,
            "reference_compute_portable_fraction": 0.35,
        },
        "representative_contract": {
            "compute_service_value_cny_per_mwh_it": 2000.0,
            "hydrogen_price_cny_per_kg": 30.0,
            "capacities": contract_result.planning.capacities,
            "kpis": contract_result.kpis,
        },
        "source_files": {
            str(path.relative_to(root)): _sha256(path)
            for path in (
                root / "configs/s7_china_evidence_register.csv",
                root / "configs/s7_china_flexible_cost_cases.csv",
                root / "configs/s7_china_infrastructure_cost_cases.csv",
                root / "configs/s7_china_planning_limits.yaml",
            )
        },
        "output_files": {str(path.relative_to(output)): _sha256(path) for path in output_files},
        "limitations": [
            "The annual profiles are deterministic synthetic screening inputs, not site measurements.",
            "D-grade cost ranges are not supplier quotations or a bankable estimate.",
            "The hourly LP assumes perfect foresight and omits unit commitment and seconds-scale stability.",
            "Wave output is a lagged screening proxy, not a measured hindcast.",
            "Merchant, network and reliability values are kept separate to avoid double counting.",
        ],
        "cost_cases": {
            "flexible_reference": asdict(reference_flexible),
            "common_reference": asdict(reference_common),
            "planning_limits": asdict(limits),
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--output", type=Path, default=Path("outputs/s7_china_value_analysis"))
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.output, args.hours)


if __name__ == "__main__":
    main()
