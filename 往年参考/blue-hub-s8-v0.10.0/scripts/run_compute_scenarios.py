"""Run Phase 4 / S3 green-compute service and fibre-link analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blue_hub import __version__
from blue_hub.compute_dispatch_model import S3DispatchResult, run_s3_dispatch
from blue_hub.dispatch_model import run_s0_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_dispatch_results
from blue_hub.schemas import ScenarioDefinition
from blue_hub.synthetic import generate_synthetic_timeseries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot_scenario_value(comparison: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = comparison["scenario_id"].str.replace("_", " ")
    values = comparison["compute_incremental_operating_value_cny"] / 1e6
    axis.bar(labels, values, color="#009E73")
    axis.set_ylabel("Incremental operating value (million CNY)")
    axis.set_title("S3 compute value relative to the direct-electricity baseline")
    axis.tick_params(axis="x", rotation=25)
    axis.axhline(0.0, color="black", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s3_scenario_compute_value.png", dpi=180)
    plt.close(figure)


def _plot_capacity_value(sensitivity: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        sensitivity["compute_it_capacity_mw"],
        sensitivity["incremental_operating_value_million_cny"],
        marker="o",
        color="#009E73",
    )
    axis.set_xlabel("IT capacity (MW-IT)")
    axis.set_ylabel("Incremental operating value (million CNY)")
    axis.set_title("Compute capacity value at 500 MW transmission")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s3_compute_capacity_value.png", dpi=180)
    plt.close(figure)


def _plot_fiber_outage(result: S3DispatchResult, output: Path, start_hour: int) -> None:
    event = result.hourly.iloc[start_hour : start_hour + 48].copy()
    x = np.arange(len(event))
    figure, (upper, lower) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    upper.plot(x, event["export_send_mw"], label="Electricity export", color="#0072B2")
    upper.plot(x, event["it_power_mw"], label="IT power", color="#009E73")
    upper.plot(x, event["dc_facility_power_mw"], label="Facility power", color="#D55E00")
    upper.set_ylabel("Power (MW)")
    upper.set_title("24-hour subsea-fibre outage: power allocation and workload recovery")
    upper.legend(ncol=3)
    lower.plot(x, event["flex_queue_end_mwh_it"], label="Flexible-work queue", color="#CC79A7")
    lower.bar(
        x,
        event["rigid_compute_unserved_mwh_it"],
        label="Rigid task unserved",
        color="#999999",
        alpha=0.6,
    )
    lower.set_ylabel("MWh-IT")
    lower.set_xlabel("Hours from fibre outage start")
    lower.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s3_fiber_outage_response.png", dpi=180)
    plt.close(figure)


def _scenario_with_compute_price(
    base: ScenarioDefinition, price_cny_per_mwh_it: float, base_price_cny_per_mwh_it: float
) -> ScenarioDefinition:
    return base.model_copy(
        update={
            "scenario_id": f"compute_price_{price_cny_per_mwh_it:g}",
            "compute_price_multiplier": price_cny_per_mwh_it / base_price_cny_per_mwh_it,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--output", type=Path, default=Path("outputs/s3_compute_analysis"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    parameters = load_parameters("configs/technology_parameters.csv")
    scenarios = {item.scenario_id: item for item in load_scenarios("configs/scenario_matrix.csv")}
    s0_config = load_system_configuration("configs/base_case.yaml")
    compute_config = load_system_configuration("configs/s3_compute_100mw_fiber_100mw_it.yaml")
    timeseries = generate_synthetic_timeseries(args.hours)

    scenario_ids = (
        "base",
        "compute_low_price",
        "compute_high_price",
        "compute_high_demand",
        "compute_pue_optimistic",
        "compute_pue_conservative",
        "negative_price_24h",
        "cable_outage_24h",
        "fiber_outage_24h",
    )
    comparison_records: list[dict[str, object]] = []
    compute_results: dict[str, S3DispatchResult] = {}
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        s0 = run_s0_dispatch(timeseries, parameters, s0_config, scenario)
        s3 = run_s3_dispatch(timeseries, parameters, compute_config, scenario)
        compute_results[scenario_id] = s3
        comparison_records.append(
            {
                "scenario_id": scenario_id,
                "s0_operating_margin_cny": s0.kpis["operating_margin_cny"],
                "s3_operating_margin_cny": s3.kpis["operating_margin_cny"],
                "compute_incremental_operating_value_cny": (
                    s3.kpis["operating_margin_cny"] - s0.kpis["operating_margin_cny"]
                ),
                "compute_incremental_operating_value_million_cny": (
                    s3.kpis["operating_margin_cny"] - s0.kpis["operating_margin_cny"]
                )
                / 1e6,
                "compute_service_mwh_it": s3.kpis["compute_service_mwh_it"],
                "compute_service_rate": s3.kpis["compute_service_rate"],
                "rigid_compute_unserved_mwh_it": s3.kpis["rigid_compute_unserved_mwh_it"],
                "flex_queue_max_mwh_it": s3.kpis["flex_queue_max_mwh_it"],
                "data_center_pue": s3.kpis["data_center_pue"],
                "curtailment_reduction_mwh": s0.kpis["curtailment_mwh"]
                - s3.kpis["curtailment_mwh"],
                "configuration_hash": s3.metadata["configuration_hash"],
            }
        )
        export_dispatch_results(s3, args.output / "scenario_hourly" / scenario_id)
    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(args.output / "s3_scenario_comparison.csv", index=False)

    congested_s0_config = s0_config.model_copy(update={"tx_capacity_mw": 500.0})
    congested_s0 = run_s0_dispatch(timeseries, parameters, congested_s0_config, scenarios["base"])
    capacity_records: list[dict[str, object]] = []
    for capacity in (50.0, 100.0, 150.0):
        config = compute_config.model_copy(
            update={
                "config_id": f"S3_tx500_C{capacity:g}_F{capacity:g}",
                "tx_capacity_mw": 500.0,
                "compute_it_capacity_mw": capacity,
                "subsea_fiber_service_capacity_mw_it": capacity,
            }
        )
        result = run_s3_dispatch(timeseries, parameters, config, scenarios["base"])
        incremental_value = result.kpis["operating_margin_cny"] - congested_s0.kpis[
            "operating_margin_cny"
        ]
        capacity_records.append(
            {
                "compute_it_capacity_mw": capacity,
                "subsea_fiber_service_capacity_mw_it": capacity,
                "incremental_operating_value_cny": incremental_value,
                "incremental_operating_value_million_cny": incremental_value / 1e6,
                "compute_service_rate": result.kpis["compute_service_rate"],
                "compute_it_capacity_factor": result.kpis["compute_it_capacity_factor"],
                "curtailment_reduction_mwh": (
                    congested_s0.kpis["curtailment_mwh"] - result.kpis["curtailment_mwh"]
                ),
                "configuration_hash": result.metadata["configuration_hash"],
            }
        )
    capacity_sensitivity = pd.DataFrame(capacity_records)
    capacity_sensitivity.to_csv(args.output / "s3_compute_capacity_sensitivity.csv", index=False)

    price_records: list[dict[str, object]] = []
    base_price = parameters.value("compute_rigid_service_price")
    for price in (300.0, 500.0, 700.0, 850.0, 1000.0, 1200.0):
        scenario = _scenario_with_compute_price(scenarios["base"], price, base_price)
        result = run_s3_dispatch(timeseries, parameters, compute_config, scenario)
        price_records.append(
            {
                "rigid_compute_service_price_cny_per_mwh_it": price,
                "operating_margin_cny": result.kpis["operating_margin_cny"],
                "compute_service_mwh_it": result.kpis["compute_service_mwh_it"],
                "compute_service_rate": result.kpis["compute_service_rate"],
                "rigid_compute_unserved_mwh_it": result.kpis["rigid_compute_unserved_mwh_it"],
                "compute_operating_margin_cny": result.kpis["compute_operating_margin_cny"],
                "configuration_hash": result.metadata["configuration_hash"],
            }
        )
    price_sensitivity = pd.DataFrame(price_records)
    price_sensitivity.to_csv(args.output / "s3_compute_price_sensitivity.csv", index=False)

    _plot_scenario_value(comparison, args.output)
    _plot_capacity_value(capacity_sensitivity, args.output)
    _plot_fiber_outage(
        compute_results["fiber_outage_24h"],
        args.output,
        scenarios["fiber_outage_24h"].fiber_outage_start_hour,
    )
    artifacts = {
        str(path.relative_to(args.output)): _sha256(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "model_version": __version__,
                "phase": "S3_compute_analysis",
                "hours": args.hours,
                "scenario_count": len(comparison),
                "capacity_case_count": len(capacity_sensitivity),
                "price_case_count": len(price_sensitivity),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(comparison)} scenario comparisons, {len(capacity_sensitivity)} capacity "
        f"cases and {len(price_sensitivity)} price cases to {args.output}"
    )


if __name__ == "__main__":
    main()
