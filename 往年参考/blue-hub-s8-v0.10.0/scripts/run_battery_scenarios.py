"""Run S1 scenario comparisons and battery power-duration sensitivities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blue_hub import __version__
from blue_hub.battery import battery_spec_from_parameters
from blue_hub.battery_dispatch_model import S1DispatchResult, run_s1_dispatch
from blue_hub.battery_resilience import simulate_islanded_critical_load
from blue_hub.dispatch_model import run_s0_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_dispatch_results
from blue_hub.schemas import TechnologyParameters
from blue_hub.synthetic import generate_synthetic_timeseries


def _float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_parameter(
    parameters: TechnologyParameters, name: str, value: float
) -> TechnologyParameters:
    items = tuple(
        item.model_copy(update={"value_base": value}) if item.parameter == name else item
        for item in parameters.items
    )
    if not any(item.parameter == name for item in parameters.items):
        raise KeyError(f"unknown parameter: {name}")
    return TechnologyParameters(items=items)


def _plot_scenario_comparison(comparison: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = comparison["scenario_id"].str.replace("_24h", "", regex=False)
    x = np.arange(len(comparison))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    axes[0].bar(x, comparison["battery_incremental_operating_value_cny"] / 1e6)
    axes[0].set_ylabel("Incremental operating value (million CNY/year)")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].bar(x - 0.18, comparison["s0_eens_mwh"], width=0.36, label="Without battery")
    axes[1].bar(x + 0.18, comparison["s1_eens_mwh"], width=0.36, label="With battery")
    axes[1].set_ylabel("EENS (MWh/year)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "battery_scenario_value.png", dpi=180)
    plt.close(fig)


def _plot_capacity_heatmap(sensitivity: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    pivot = sensitivity.pivot(
        index="battery_power_mw",
        columns="battery_duration_h",
        values="incremental_operating_value_million_cny",
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    axis.set(xlabel="Battery duration (h)", ylabel="Battery power (MW)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", color="white")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Incremental operating value (million CNY/year)")
    fig.tight_layout()
    fig.savefig(figures / "battery_power_duration_value.png", dpi=180)
    plt.close(fig)


def _plot_compound_event(result: S1DispatchResult, output: Path) -> None:
    figures = output / "figures"
    event = result.hourly.iloc[60:108]
    hour = np.arange(60, 108)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    axes[0].plot(hour, event["critical_load_mw"], label="Critical load")
    axes[0].plot(hour, event["battery_discharge_mw"], label="Battery discharge")
    axes[0].plot(hour, event["unmet_critical_load_mw"], label="Unmet load")
    axes[0].axvspan(72, 96, color="black", alpha=0.08, label="Wind + cable outage")
    axes[0].set_ylabel("Power (MW)")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].plot(hour, 100.0 * event["battery_soc_start"], color="tab:green")
    axes[1].axvspan(72, 96, color="black", alpha=0.08)
    axes[1].set(xlabel="Hour of synthetic year", ylabel="Battery SOC (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "compound_outage_battery_response.png", dpi=180)
    plt.close(fig)


def _plot_initial_soc_resilience(resilience: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    soc_percent = 100.0 * resilience["initial_soc"]
    axes[0].plot(soc_percent, resilience["hours_before_first_shortfall"], marker="o")
    axes[0].set(xlabel="Initial SOC (%)", ylabel="Hours before first shortfall")
    axes[1].plot(soc_percent, resilience["eens_mwh"], marker="o", color="tab:red")
    axes[1].set(xlabel="Initial SOC (%)", ylabel="24-hour EENS (MWh)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "initial_soc_resilience.png", dpi=180)
    plt.close(fig)


def _plot_parameter_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    pivot = sensitivity.pivot(
        index="battery_round_trip_efficiency",
        columns="battery_degradation_cost_cny_per_mwh_throughput",
        values="incremental_operating_value_million_cny",
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{value:.0%}" for value in pivot.index])
    axis.set(
        xlabel="Degradation cost (CNY/MWh throughput)",
        ylabel="Round-trip efficiency",
    )
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(
                column,
                row,
                f"{pivot.iloc[row, column]:.1f}",
                ha="center",
                va="center",
                color="white",
            )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Incremental operating value (million CNY/year)")
    fig.tight_layout()
    fig.savefig(figures / "battery_parameter_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--powers", default="50,100,150")
    parser.add_argument("--durations", default="2,4,6")
    parser.add_argument("--sensitivity-tx-capacity", type=float, default=500.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/s1_battery_analysis"))
    args = parser.parse_args()

    parameters = load_parameters("configs/technology_parameters.csv")
    s0_config = load_system_configuration("configs/base_case.yaml")
    battery_config = load_system_configuration("configs/s1_battery_100mw_400mwh.yaml")
    reserve_config = load_system_configuration("configs/s1_battery_100mw_400mwh_reserve.yaml")
    scenarios = {item.scenario_id: item for item in load_scenarios("configs/scenario_matrix.csv")}
    timeseries = generate_synthetic_timeseries(args.hours)
    args.output.mkdir(parents=True, exist_ok=True)

    scenario_ids = [
        "base",
        "negative_price_24h",
        "cable_outage_24h",
        "wind_lull_24h",
        "compound_outage_24h",
    ]
    comparison_records: list[dict[str, object]] = []
    battery_results: dict[str, S1DispatchResult] = {}
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        s0 = run_s0_dispatch(timeseries, parameters, s0_config, scenario)
        s1 = run_s1_dispatch(timeseries, parameters, battery_config, scenario)
        battery_results[scenario_id] = s1
        comparison_records.append(
            {
                "scenario_id": scenario_id,
                "s0_operating_margin_cny": s0.kpis["operating_margin_cny"],
                "s1_operating_margin_cny": s1.kpis["operating_margin_cny"],
                "battery_incremental_operating_value_cny": (
                    s1.kpis["operating_margin_cny"] - s0.kpis["operating_margin_cny"]
                ),
                "s0_curtailment_mwh": s0.kpis["curtailment_mwh"],
                "s1_curtailment_mwh": s1.kpis["curtailment_mwh"],
                "s0_eens_mwh": s0.kpis["eens_mwh"],
                "s1_eens_mwh": s1.kpis["eens_mwh"],
                "battery_charge_mwh": s1.kpis["battery_charge_mwh"],
                "battery_discharge_mwh": s1.kpis["battery_discharge_mwh"],
                "battery_efc": s1.kpis["battery_efc"],
                "battery_degradation_cost_cny": s1.kpis["battery_degradation_cost_cny"],
                "configuration_hash": s1.metadata["configuration_hash"],
            }
        )
        export_dispatch_results(s1, args.output / "scenario_hourly" / scenario_id)
    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(args.output / "s1_scenario_comparison.csv", index=False)

    base_battery = battery_results["base"]
    reserve_result = run_s1_dispatch(timeseries, parameters, reserve_config, scenarios["base"])
    reserve_tradeoff = pd.DataFrame(
        [
            {
                "reserve_power_mw": reserve_config.battery_reserve_power_mw,
                "reserve_duration_h": reserve_config.battery_reserve_duration_h,
                "reserve_energy_above_minimum_mwh": reserve_result.kpis[
                    "battery_reserve_energy_mwh"
                ],
                "scheduled_minimum_soc": reserve_result.kpis["battery_scheduled_minimum_soc"],
                "operating_margin_without_reserve_cny": base_battery.kpis["operating_margin_cny"],
                "operating_margin_with_reserve_cny": reserve_result.kpis["operating_margin_cny"],
                "reserve_opportunity_cost_cny": (
                    base_battery.kpis["operating_margin_cny"]
                    - reserve_result.kpis["operating_margin_cny"]
                ),
                "battery_efc_without_reserve": base_battery.kpis["battery_efc"],
                "battery_efc_with_reserve": reserve_result.kpis["battery_efc"],
                "configuration_hash": reserve_result.metadata["configuration_hash"],
            }
        ]
    )
    reserve_tradeoff.to_csv(args.output / "reserve_tradeoff.csv", index=False)

    congested_s0_config = s0_config.model_copy(
        update={"tx_capacity_mw": args.sensitivity_tx_capacity}
    )
    congested_s0 = run_s0_dispatch(timeseries, parameters, congested_s0_config, scenarios["base"])
    sensitivity_records: list[dict[str, object]] = []
    for power in _float_list(args.powers):
        for duration in _float_list(args.durations):
            energy = power * duration
            config = battery_config.model_copy(
                update={
                    "config_id": f"S1_B{power:g}_E{energy:g}",
                    "tx_capacity_mw": args.sensitivity_tx_capacity,
                    "battery_power_mw": power,
                    "battery_energy_mwh": energy,
                }
            )
            result = run_s1_dispatch(timeseries, parameters, config, scenarios["base"])
            incremental_value = (
                result.kpis["operating_margin_cny"] - congested_s0.kpis["operating_margin_cny"]
            )
            sensitivity_records.append(
                {
                    "battery_power_mw": power,
                    "battery_duration_h": duration,
                    "battery_energy_mwh": energy,
                    "incremental_operating_value_cny": incremental_value,
                    "incremental_operating_value_million_cny": incremental_value / 1e6,
                    "break_even_annual_cost_cny_per_kw_year": incremental_value / power / 1000.0,
                    "break_even_annual_cost_cny_per_kwh_year": incremental_value / energy / 1000.0,
                    "curtailment_reduction_mwh": (
                        congested_s0.kpis["curtailment_mwh"] - result.kpis["curtailment_mwh"]
                    ),
                    "battery_efc": result.kpis["battery_efc"],
                    "battery_loss_mwh": result.kpis["battery_total_loss_mwh"],
                    "configuration_hash": result.metadata["configuration_hash"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_records).sort_values(
        ["battery_power_mw", "battery_duration_h"]
    )
    sensitivity.to_csv(args.output / "s1_battery_capacity_sensitivity.csv", index=False)

    parameter_records: list[dict[str, object]] = []
    for round_trip_efficiency in (0.80, 0.85, 0.90):
        for degradation_cost in (40.0, 80.0, 120.0, 145.0, 150.0, 160.0):
            varied = _replace_parameter(
                parameters,
                "battery_round_trip_efficiency",
                round_trip_efficiency,
            )
            varied = _replace_parameter(
                varied,
                "battery_degradation_cost",
                degradation_cost,
            )
            result = run_s1_dispatch(timeseries, varied, battery_config, scenarios["base"])
            incremental_value = (
                result.kpis["operating_margin_cny"]
                - comparison.loc[
                    comparison["scenario_id"] == "base", "s0_operating_margin_cny"
                ].iloc[0]
            )
            parameter_records.append(
                {
                    "battery_round_trip_efficiency": round_trip_efficiency,
                    "battery_degradation_cost_cny_per_mwh_throughput": degradation_cost,
                    "incremental_operating_value_cny": incremental_value,
                    "incremental_operating_value_million_cny": incremental_value / 1e6,
                    "battery_efc": result.kpis["battery_efc"],
                    "battery_total_loss_mwh": result.kpis["battery_total_loss_mwh"],
                    "configuration_hash": result.metadata["configuration_hash"],
                }
            )
    parameter_sensitivity = pd.DataFrame(parameter_records)
    parameter_sensitivity.to_csv(args.output / "battery_parameter_sensitivity.csv", index=False)

    compound_timeseries = battery_results["compound_outage_24h"].hourly
    critical_event = compound_timeseries.iloc[72:96]["critical_load_mw"].to_numpy()
    battery_specification = battery_spec_from_parameters(battery_config, parameters)
    resilience_records = []
    for initial_soc in (0.3, 0.5, 0.7, 0.9):
        audit = simulate_islanded_critical_load(
            critical_event,
            battery_specification,
            initial_soc,
        )
        resilience_records.append(audit.kpis)
    resilience = pd.DataFrame(resilience_records)
    resilience.to_csv(args.output / "event_initial_soc_sensitivity.csv", index=False)

    _plot_scenario_comparison(comparison, args.output)
    _plot_capacity_heatmap(sensitivity, args.output)
    _plot_compound_event(battery_results["compound_outage_24h"], args.output)
    _plot_initial_soc_resilience(resilience, args.output)
    _plot_parameter_sensitivity(parameter_sensitivity, args.output)

    artifacts = {
        str(path.relative_to(args.output)): _sha256(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "model_version": __version__,
                "phase": "S1_battery_analysis",
                "hours": args.hours,
                "scenario_count": len(comparison),
                "capacity_case_count": len(sensitivity),
                "parameter_case_count": len(parameter_sensitivity),
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
        f"wrote {len(comparison)} scenario comparisons and "
        f"{len(sensitivity)} battery capacity cases to {args.output}"
    )


if __name__ == "__main__":
    main()
