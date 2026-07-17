"""Run Phase 3 / S2 hydrogen, storage and event analyses reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blue_hub import __version__
from blue_hub.battery_dispatch_model import run_s1_dispatch
from blue_hub.hydrogen_dispatch_model import S2DispatchResult, run_s2_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_dispatch_results
from blue_hub.schemas import ScenarioDefinition, TechnologyParameter, TechnologyParameters
from blue_hub.synthetic import generate_synthetic_timeseries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_parameter(
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


def _plot_scenario_value(comparison: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = comparison["scenario_id"].str.replace("_", " ")
    values = comparison["hydrogen_incremental_operating_value_cny"] / 1e6
    axis.bar(labels, values, color="#0072B2")
    axis.set_ylabel("Incremental operating value (million CNY)")
    axis.set_title("S2 hydrogen value relative to the S1 battery baseline")
    axis.tick_params(axis="x", rotation=25)
    axis.axhline(0.0, color="black", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s2_scenario_hydrogen_value.png", dpi=180)
    plt.close(figure)


def _plot_capacity_heatmap(sensitivity: pd.DataFrame, output: Path) -> None:
    pivot = sensitivity.pivot(
        index="electrolyzer_power_mw",
        columns="storage_duration_h",
        values="incremental_operating_value_million_cny",
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    axis.set_xlabel("Hydrogen storage duration at 1,000 kg/h demand (h)")
    axis.set_ylabel("Electrolyzer power (MW)")
    axis.set_title("Hydrogen capacity value at 500 MW transmission")
    for row, power in enumerate(pivot.index):
        for column, duration in enumerate(pivot.columns):
            value = pivot.loc[power, duration]
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="Million CNY/year")
    figure.tight_layout()
    figure.savefig(output / "figures" / "s2_hydrogen_capacity_value.png", dpi=180)
    plt.close(figure)


def _plot_outage_dispatch(result: S2DispatchResult, output: Path, start_hour: int) -> None:
    event = result.hourly.iloc[start_hour : start_hour + 72].copy()
    figure, (upper, lower) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(event))
    upper.plot(x, event["export_send_mw"], label="Export send", color="#0072B2")
    upper.plot(x, event["electrolyzer_power_mw"], label="Electrolyzer", color="#D55E00")
    upper.plot(x, event["battery_charge_mw"], label="Battery charge", color="#009E73")
    upper.plot(x, event["battery_discharge_mw"], label="Battery discharge", color="#CC79A7")
    upper.set_ylabel("Power (MW)")
    upper.set_title("72-hour cable outage: coordinated battery, export and hydrogen response")
    upper.legend(ncol=2)
    lower.plot(x, event["hydrogen_inventory_end_kg"], label="Hydrogen inventory", color="#D55E00")
    lower.set_ylabel("Inventory (kg)")
    lower.set_xlabel("Hours from outage start")
    lower.legend()
    figure.tight_layout()
    figure.savefig(output / "figures" / "s2_extended_outage_dispatch.png", dpi=180)
    plt.close(figure)


def _scenario_with_hydrogen_price(
    base: ScenarioDefinition, price_cny_per_kg: float, parameter_price_cny_per_kg: float
) -> ScenarioDefinition:
    return base.model_copy(
        update={
            "scenario_id": f"hydrogen_price_{price_cny_per_kg:g}",
            "hydrogen_price_multiplier": price_cny_per_kg / parameter_price_cny_per_kg,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--output", type=Path, default=Path("outputs/s2_hydrogen_analysis"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    parameters = load_parameters("configs/technology_parameters.csv")
    scenarios = {item.scenario_id: item for item in load_scenarios("configs/scenario_matrix.csv")}
    s1_config = load_system_configuration("configs/s1_battery_100mw_400mwh.yaml")
    hydrogen_config = load_system_configuration("configs/s2_hydrogen_100mw_72000kg.yaml")
    battery_hydrogen_config = load_system_configuration(
        "configs/s2_battery_hydrogen_100mw_400mwh_100mw_72000kg.yaml"
    )
    timeseries = generate_synthetic_timeseries(args.hours)

    scenario_ids = (
        "base",
        "hydrogen_low_price",
        "hydrogen_high_price",
        "hydrogen_high_demand",
        "negative_price_24h",
        "cable_outage_24h",
        "extended_cable_outage_72h",
    )
    comparison_records: list[dict[str, object]] = []
    s2_results: dict[str, S2DispatchResult] = {}
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        s1 = run_s1_dispatch(timeseries, parameters, s1_config, scenario)
        s2 = run_s2_dispatch(timeseries, parameters, battery_hydrogen_config, scenario)
        s2_results[scenario_id] = s2
        comparison_records.append(
            {
                "scenario_id": scenario_id,
                "s1_operating_margin_cny": s1.kpis["operating_margin_cny"],
                "s2_operating_margin_cny": s2.kpis["operating_margin_cny"],
                "hydrogen_incremental_operating_value_cny": (
                    s2.kpis["operating_margin_cny"] - s1.kpis["operating_margin_cny"]
                ),
                "hydrogen_incremental_operating_value_million_cny": (
                    s2.kpis["operating_margin_cny"] - s1.kpis["operating_margin_cny"]
                )
                / 1e6,
                "hydrogen_sales_kg": s2.kpis["hydrogen_sales_kg"],
                "hydrogen_service_rate": s2.kpis["hydrogen_service_rate"],
                "hydrogen_storage_loss_kg": s2.kpis["hydrogen_storage_loss_kg"],
                "electrolyzer_capacity_factor": s2.kpis["electrolyzer_capacity_factor"],
                "curtailment_reduction_mwh": (
                    s1.kpis["curtailment_mwh"] - s2.kpis["curtailment_mwh"]
                ),
                "configuration_hash": s2.metadata["configuration_hash"],
            }
        )
        export_dispatch_results(s2, args.output / "scenario_hourly" / scenario_id)
    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(args.output / "s2_scenario_comparison.csv", index=False)

    congested_s1_config = s1_config.model_copy(update={"tx_capacity_mw": 500.0})
    congested_s1 = run_s1_dispatch(timeseries, parameters, congested_s1_config, scenarios["base"])
    capacity_records: list[dict[str, object]] = []
    for power in (50.0, 100.0, 150.0):
        for duration in (24.0, 72.0, 168.0):
            storage = duration * 1_000.0
            config = hydrogen_config.model_copy(
                update={
                    "config_id": f"S2_tx500_H{power:g}_S{storage:g}",
                    "tx_capacity_mw": 500.0,
                    "electrolyzer_power_mw": power,
                    "hydrogen_storage_kg": storage,
                }
            )
            result = run_s2_dispatch(timeseries, parameters, config, scenarios["base"])
            incremental_value = result.kpis["operating_margin_cny"] - congested_s1.kpis[
                "operating_margin_cny"
            ]
            capacity_records.append(
                {
                    "electrolyzer_power_mw": power,
                    "storage_duration_h": duration,
                    "hydrogen_storage_kg": storage,
                    "incremental_operating_value_cny": incremental_value,
                    "incremental_operating_value_million_cny": incremental_value / 1e6,
                    "hydrogen_service_rate": result.kpis["hydrogen_service_rate"],
                    "electrolyzer_capacity_factor": result.kpis["electrolyzer_capacity_factor"],
                    "curtailment_reduction_mwh": (
                        congested_s1.kpis["curtailment_mwh"] - result.kpis["curtailment_mwh"]
                    ),
                    "configuration_hash": result.metadata["configuration_hash"],
                }
            )
    capacity = pd.DataFrame(capacity_records).sort_values(
        ["electrolyzer_power_mw", "storage_duration_h"]
    )
    capacity.to_csv(args.output / "s2_hydrogen_capacity_sensitivity.csv", index=False)

    price_records: list[dict[str, object]] = []
    parameter_price = parameters.value("hydrogen_sale_price")
    for price_cny_per_kg in (15.0, 20.0, 25.0, 30.0, 35.0, 40.0):
        scenario = _scenario_with_hydrogen_price(
            scenarios["base"], price_cny_per_kg, parameter_price
        )
        result = run_s2_dispatch(timeseries, parameters, battery_hydrogen_config, scenario)
        price_records.append(
            {
                "hydrogen_sale_price_cny_per_kg": price_cny_per_kg,
                "operating_margin_cny": result.kpis["operating_margin_cny"],
                "hydrogen_sales_kg": result.kpis["hydrogen_sales_kg"],
                "hydrogen_service_rate": result.kpis["hydrogen_service_rate"],
                "electrolyzer_capacity_factor": result.kpis["electrolyzer_capacity_factor"],
                "hydrogen_operating_margin_cny": result.kpis["hydrogen_operating_margin_cny"],
                "configuration_hash": result.metadata["configuration_hash"],
            }
        )
    price_sensitivity = pd.DataFrame(price_records)
    price_sensitivity.to_csv(args.output / "s2_hydrogen_price_sensitivity.csv", index=False)

    _plot_scenario_value(comparison, args.output)
    _plot_capacity_heatmap(capacity, args.output)
    _plot_outage_dispatch(
        s2_results["extended_cable_outage_72h"],
        args.output,
        scenarios["extended_cable_outage_72h"].tx_outage_start_hour,
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
                "phase": "S2_hydrogen_analysis",
                "hours": args.hours,
                "scenario_count": len(comparison),
                "capacity_case_count": len(capacity),
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
        f"wrote {len(comparison)} scenario comparisons, {len(capacity)} capacity cases and "
        f"{len(price_sensitivity)} price cases to {args.output}"
    )


if __name__ == "__main__":
    main()
