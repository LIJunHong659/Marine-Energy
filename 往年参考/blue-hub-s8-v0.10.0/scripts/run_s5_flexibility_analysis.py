"""Run Phase 6 / S5 scarcity, factorial value and seasonal-resilience cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-blue-hub")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blue_hub import __version__
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_dispatch_results
from blue_hub.scarcity_dispatch_model import S5DispatchResult, run_s5_dispatch
from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters
from blue_hub.synthetic import generate_synthetic_timeseries

ASSET_LABELS = {
    "none": "仅直接输电",
    "B": "电池",
    "H": "氢",
    "C": "算力",
    "BH": "电池+氢",
    "BC": "电池+算力",
    "HC": "氢+算力",
    "BHC": "电池+氢+算力",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(
    base: ScenarioDefinition, scenario_id: str, h2_price: float = 0.75
) -> ScenarioDefinition:
    return base.model_copy(
        update={
            "scenario_id": scenario_id,
            "hydrogen_price_multiplier": h2_price,
        }
    )


def _asset_config(
    base: SystemConfiguration, assets: str, *, config_id: str | None = None
) -> SystemConfiguration:
    has_battery = "B" in assets
    has_hydrogen = "H" in assets
    has_compute = "C" in assets
    return base.model_copy(
        update={
            "config_id": config_id or f"S5_{assets or 'none'}",
            "battery_power_mw": base.battery_power_mw if has_battery else 0.0,
            "battery_energy_mwh": base.battery_energy_mwh if has_battery else 0.0,
            "electrolyzer_power_mw": base.electrolyzer_power_mw if has_hydrogen else 0.0,
            "hydrogen_storage_kg": base.hydrogen_storage_kg if has_hydrogen else 0.0,
            "fuel_cell_power_mw": base.fuel_cell_power_mw if has_hydrogen else 0.0,
            "compute_it_capacity_mw": base.compute_it_capacity_mw if has_compute else 0.0,
            "subsea_fiber_service_capacity_mw_it": (
                base.subsea_fiber_service_capacity_mw_it if has_compute else 0.0
            ),
            "initial_battery_soc_fraction": (
                base.initial_battery_soc_fraction if has_battery else 0.0
            ),
            "initial_hydrogen_inventory_fraction": (
                base.initial_hydrogen_inventory_fraction if has_hydrogen else 0.0
            ),
            "battery_reserve_power_mw": 0.0,
            "battery_reserve_duration_h": 0.0,
        }
    )


def _analysis_timeseries(hours: int, mode: str) -> pd.DataFrame:
    frame = generate_synthetic_timeseries(hours, scenario_id=mode)
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    frame["national_compute_demand_mw_it"] = 200.0
    frame["national_compute_price_cny_per_mwh_it"] = 360.0
    if mode == "loose_export":
        frame["grid_absorption_factor"] = 1.0
    elif mode in {"demand_mismatch", "cable_outage_72h"}:
        price = frame["electricity_price"].to_numpy(dtype=float)
        wind = frame["wind_cf"].to_numpy(dtype=float)
        factor = np.ones(hours)
        low_demand = price <= 360.0
        factor[low_demand] = 0.55
        factor[low_demand & (wind >= 0.52)] = 0.32
        frame["grid_absorption_factor"] = factor
        if mode == "cable_outage_72h":
            start = min(4_000, max(hours // 2 - 36, 0))
            stop = min(start + 72, hours)
            frame.loc[start : stop - 1, "tx_availability"] = 0.0
    elif mode == "seasonal_lull":
        frame["grid_absorption_factor"] = 1.0
        frame["national_compute_demand_mw_it"] = 0.0
        frame["hydrogen_demand"] = 0.0
        start = min(3_000, max(hours // 3, 0))
        stop = min(start + 2_160, hours)
        frame.loc[start : stop - 1, "wind_availability"] = 0.01
    else:
        raise ValueError(f"unknown analysis mode: {mode}")
    return frame


def _record(result: S5DispatchResult, case: str, mode: str) -> dict[str, object]:
    kpi = result.kpis
    return {
        "mode": mode,
        "asset_case": case,
        "asset_label_zh": ASSET_LABELS[case],
        "operating_margin_cny": kpi["operating_margin_cny"],
        "curtailment_mwh": kpi["curtailment_mwh"],
        "renewable_utilization_rate": kpi["renewable_utilization_rate"],
        "export_land_mwh": kpi["export_land_mwh"],
        "battery_discharge_mwh": kpi["battery_discharge_mwh"],
        "hydrogen_production_kg": kpi["hydrogen_production_kg"],
        "fuel_cell_generation_mwh": kpi["fuel_cell_generation_mwh"],
        "spot_compute_service_mwh_it": kpi["spot_compute_service_mwh_it"],
        "spot_compute_service_constrained_hours_mwh_it": kpi[
            "spot_compute_service_constrained_hours_mwh_it"
        ],
        "eens_mwh": kpi["eens_mwh"],
        "sum_export_capacity_shadow_cny_per_mw": kpi["sum_export_capacity_shadow_cny_per_mw"],
        "configuration_hash": result.metadata["configuration_hash"],
    }


def _factorial(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    scenario: ScenarioDefinition,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, S5DispatchResult]]:
    cases = ("none", "B", "H", "C", "BH", "BC", "HC", "BHC")
    results: dict[str, S5DispatchResult] = {}
    records: list[dict[str, object]] = []
    for case in cases:
        assets = "" if case == "none" else case
        result = run_s5_dispatch(
            timeseries,
            params,
            _asset_config(base_config, assets),
            scenario,
        )
        results[case] = result
        records.append(_record(result, case, mode))
    frame = pd.DataFrame(records)
    baseline = float(frame.loc[frame["asset_case"] == "none", "operating_margin_cny"].iloc[0])
    frame["incremental_operating_value_cny"] = frame["operating_margin_cny"] - baseline
    frame["avoided_curtailment_mwh"] = (
        float(frame.loc[frame["asset_case"] == "none", "curtailment_mwh"].iloc[0])
        - frame["curtailment_mwh"]
    )
    return frame, results


def _interactions(frame: pd.DataFrame, mode: str) -> dict[str, object]:
    value = frame.set_index("asset_case")["operating_margin_cny"].to_dict()
    incremental = {case: value[case] - value["none"] for case in value}
    i_bh = value["BH"] - value["B"] - value["H"] + value["none"]
    i_bc = value["BC"] - value["B"] - value["C"] + value["none"]
    i_hc = value["HC"] - value["H"] - value["C"] + value["none"]
    i_bhc = (
        value["BHC"]
        - value["BH"]
        - value["BC"]
        - value["HC"]
        + value["B"]
        + value["H"]
        + value["C"]
        - value["none"]
    )
    shapley_b = (
        incremental["B"] / 3.0
        + ((value["BH"] - value["H"]) + (value["BC"] - value["C"])) / 6.0
        + (value["BHC"] - value["HC"]) / 3.0
    )
    shapley_h = (
        incremental["H"] / 3.0
        + ((value["BH"] - value["B"]) + (value["HC"] - value["C"])) / 6.0
        + (value["BHC"] - value["BC"]) / 3.0
    )
    shapley_c = (
        incremental["C"] / 3.0
        + ((value["BC"] - value["B"]) + (value["HC"] - value["H"])) / 6.0
        + (value["BHC"] - value["BH"]) / 3.0
    )
    return {
        "mode": mode,
        "battery_standalone_value_cny": incremental["B"],
        "hydrogen_standalone_value_cny": incremental["H"],
        "compute_standalone_value_cny": incremental["C"],
        "full_hub_incremental_value_cny": incremental["BHC"],
        "pair_interaction_battery_hydrogen_cny": i_bh,
        "pair_interaction_battery_compute_cny": i_bc,
        "pair_interaction_hydrogen_compute_cny": i_hc,
        "triple_interaction_cny": i_bhc,
        "shapley_battery_cny": shapley_b,
        "shapley_hydrogen_cny": shapley_h,
        "shapley_compute_cny": shapley_c,
        "shapley_sum_check_cny": shapley_b + shapley_h + shapley_c,
    }


def _break_even_metrics(
    interactions: pd.DataFrame, base_config: SystemConfiguration
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in interactions.to_dict(orient="records"):
        mode = str(row["mode"])
        for resource, value_key, capacity, unit in (
            (
                "battery",
                "shapley_battery_cny",
                base_config.battery_energy_mwh * 1_000.0,
                "CNY/kWh-year",
            ),
            (
                "hydrogen_chain",
                "shapley_hydrogen_cny",
                base_config.electrolyzer_power_mw * 1_000.0,
                "CNY/kW-electrolyzer-year",
            ),
            (
                "compute",
                "shapley_compute_cny",
                base_config.compute_it_capacity_mw * 1_000.0,
                "CNY/kW-IT-year",
            ),
        ):
            value = float(row[value_key])
            records.append(
                {
                    "mode": mode,
                    "resource": resource,
                    "shapley_operating_value_cny_per_year": value,
                    "capacity_denominator": capacity,
                    "annualized_cost_ceiling": value / capacity if capacity else 0.0,
                    "ceiling_unit": unit,
                    "interpretation": (
                        "Maximum annualized fixed cost before financing and risk adjustments"
                    ),
                }
            )
    return pd.DataFrame(records)


def _equivalent_cable(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    scenario: ScenarioDefinition,
    target_curtailment_mwh: float,
) -> tuple[pd.DataFrame, S5DispatchResult]:
    direct = _asset_config(base_config, "")
    low = base_config.tx_capacity_mw
    high = 3_000.0
    records: list[dict[str, float]] = []
    best: S5DispatchResult | None = None
    for iteration in range(11):
        capacity = high if iteration == 0 else (low + high) / 2.0
        config = direct.model_copy(
            update={"config_id": f"S5_equivalent_cable_{capacity:.3f}", "tx_capacity_mw": capacity}
        )
        result = run_s5_dispatch(timeseries, params, config, scenario)
        curtailment = float(result.kpis["curtailment_mwh"])
        records.append(
            {
                "iteration": iteration,
                "tx_capacity_mw": capacity,
                "curtailment_mwh": curtailment,
                "target_curtailment_mwh": target_curtailment_mwh,
            }
        )
        if curtailment <= target_curtailment_mwh + 1.0:
            high = capacity
            best = result
        else:
            low = capacity
    if best is None:
        raise RuntimeError("equivalent-cable search upper bound is insufficient")
    table = pd.DataFrame(records)
    table["equivalent_tx_capacity_mw"] = high
    table["equivalent_added_cable_capacity_mw"] = high - base_config.tx_capacity_mw
    return table, best


def _cable_marginal_value(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for case, assets in (("direct", ""), ("full_hub", "BHC")):
        margins: dict[float, float] = {}
        for capacity in (650.0, 700.0, 750.0):
            config = _asset_config(base_config, assets).model_copy(
                update={"config_id": f"S5_{case}_{capacity:.0f}", "tx_capacity_mw": capacity}
            )
            result = run_s5_dispatch(timeseries, params, config, scenario)
            margins[capacity] = float(result.kpis["operating_margin_cny"])
            records.append(
                {
                    "case": case,
                    "tx_capacity_mw": capacity,
                    "operating_margin_cny": margins[capacity],
                    "central_marginal_value_cny_per_mw_year": np.nan,
                }
            )
        derivative = (margins[750.0] - margins[650.0]) / 100.0
        for record in records:
            if record["case"] == case:
                record["central_marginal_value_cny_per_mw_year"] = derivative
    return pd.DataFrame(records)


def _seasonal_resilience(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    scenario: ScenarioDefinition,
) -> tuple[pd.DataFrame, dict[float, S5DispatchResult]]:
    results: dict[float, S5DispatchResult] = {}
    records: list[dict[str, object]] = []
    for storage in (72_000.0, 500_000.0, 1_500_000.0):
        config = _asset_config(base_config, "H").model_copy(
            update={
                "config_id": f"S5_seasonal_H2_{storage:.0f}",
                "hydrogen_storage_kg": storage,
                "fuel_cell_power_mw": 20.0,
                "initial_hydrogen_inventory_fraction": 0.5,
            }
        )
        result = run_s5_dispatch(timeseries, params, config, scenario)
        results[storage] = result
        records.append(
            {
                "hydrogen_storage_kg": storage,
                "electrolyzer_power_mw": config.electrolyzer_power_mw,
                "fuel_cell_power_mw": config.fuel_cell_power_mw,
                "eens_mwh": result.kpis["eens_mwh"],
                "critical_load_served_mwh": result.kpis["critical_load_served_mwh"],
                "fuel_cell_generation_mwh": result.kpis["fuel_cell_generation_mwh"],
                "hydrogen_fuel_cell_use_kg": result.kpis["hydrogen_fuel_cell_use_kg"],
                "minimum_hydrogen_inventory_kg": float(
                    result.hourly["hydrogen_inventory_end_kg"].min()
                ),
                "operating_margin_cny": result.kpis["operating_margin_cny"],
            }
        )
    no_hydrogen = run_s5_dispatch(timeseries, params, _asset_config(base_config, ""), scenario)
    base_eens = float(no_hydrogen.kpis["eens_mwh"])
    frame = pd.DataFrame(records)
    frame["avoided_eens_mwh"] = base_eens - frame["eens_mwh"]
    frame["avoided_eens_rate"] = frame["avoided_eens_mwh"] / base_eens if base_eens else 0.0
    frame["direct_case_eens_mwh"] = base_eens
    results[0.0] = no_hydrogen
    return frame, results


def _price_sensitivity(
    timeseries: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    base_scenario: ScenarioDefinition,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for h2_multiplier, compute_price in product((0.60, 0.75, 0.90), (300.0, 360.0, 420.0)):
        frame = timeseries.copy()
        frame["national_compute_price_cny_per_mwh_it"] = compute_price
        scenario = _scenario(
            base_scenario,
            f"price_h{h2_multiplier:.2f}_c{compute_price:.0f}",
            h2_price=h2_multiplier,
        )
        result = run_s5_dispatch(frame, params, _asset_config(base_config, "BHC"), scenario)
        records.append(
            {
                "hydrogen_price_multiplier": h2_multiplier,
                "hydrogen_sale_price_cny_per_kg": (
                    params.value("hydrogen_sale_price") * h2_multiplier
                ),
                "spot_compute_price_cny_per_mwh_it": compute_price,
                "operating_margin_cny": result.kpis["operating_margin_cny"],
                "curtailment_mwh": result.kpis["curtailment_mwh"],
                "hydrogen_production_kg": result.kpis["hydrogen_production_kg"],
                "spot_compute_service_mwh_it": result.kpis["spot_compute_service_mwh_it"],
            }
        )
    return pd.DataFrame(records)


def _plot_factorial(factorial: pd.DataFrame, output: Path) -> None:
    pivot = factorial.pivot(
        index="asset_case", columns="mode", values="incremental_operating_value_cny"
    )
    order = ["B", "H", "C", "BH", "BC", "HC", "BHC"]
    pivot = pivot.reindex(order)
    figure, axis = plt.subplots(figsize=(11, 6))
    pivot.div(1e6).plot.bar(ax=axis, color=["#8CB9D9", "#D55E00"])
    axis.set_xlabel("")
    axis.set_ylabel("Annual operating value above direct export (million CNY)")
    axis.set_title("Flexibility value under loose and constrained mainland absorption")
    if list(pivot.columns) == ["demand_mismatch", "loose_export"]:
        axis.legend(["Demand mismatch", "Loose export"])
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s5_factorial_incremental_value.png", dpi=200)
    plt.close(figure)


def _plot_mismatch_dispatch(result: S5DispatchResult, output: Path) -> None:
    hourly = result.hourly
    stress = hourly.loc[hourly["grid_absorption_factor"] < 0.4]
    center = int(stress.index[0]) if not stress.empty else 0
    window = hourly.iloc[max(center - 24, 0) : min(center + 144, len(hourly))]
    x = np.arange(len(window))
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.stackplot(
        x,
        window["export_send_mw"],
        window["dc_facility_power_mw"],
        window["electrolyzer_power_mw"],
        window["battery_charge_mw"],
        labels=["Direct export", "Compute facility", "Electrolyzer", "Battery charge"],
        colors=["#2878B5", "#35A16B", "#E17C05", "#8C6BB1"],
        alpha=0.85,
    )
    axis.plot(
        x, window["renewable_available_mw"], color="#222222", lw=1.3, label="Renewable available"
    )
    axis.plot(
        x,
        window["tx_available_capacity_mw"],
        color="#D62728",
        lw=1.2,
        ls="--",
        label="Mainland absorption limit",
    )
    axis.set_xlabel("Representative hours")
    axis.set_ylabel("Power (MW)")
    axis.set_title("Multi-outlet dispatch during constrained mainland absorption")
    axis.legend(ncol=3, loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s5_mismatch_dispatch.png", dpi=200)
    plt.close(figure)


def _plot_seasonal(results: dict[float, S5DispatchResult], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for storage, color in ((72_000.0, "#E17C05"), (500_000.0, "#35A16B"), (1_500_000.0, "#2878B5")):
        hourly = results[storage].hourly
        axes[0].plot(
            np.arange(len(hourly)),
            hourly["hydrogen_inventory_end_kg"] / 1e3,
            label=f"{storage / 1e3:.0f} t H2 storage",
            color=color,
            lw=1.1,
        )
        axes[1].plot(
            np.arange(len(hourly)),
            hourly["unmet_critical_load_mw"],
            label=f"{storage / 1e3:.0f} t H2 storage",
            color=color,
            lw=1.0,
        )
    direct = results[0.0].hourly
    axes[1].plot(
        np.arange(len(direct)),
        direct["unmet_critical_load_mw"],
        color="#333333",
        lw=1.0,
        alpha=0.7,
        label="Direct export only",
    )
    axes[0].set_ylabel("Hydrogen inventory (t)")
    axes[0].set_title("Seasonal hydrogen dispatch during a continuous 90-day wind lull")
    axes[1].set_ylabel("Unserved critical load (MW)")
    axes[1].set_xlabel("Hour of year")
    for axis in axes:
        axis.legend(ncol=4)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s5_seasonal_hydrogen_resilience.png", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8_760)
    parser.add_argument("--output", type=Path, default=Path("outputs/s5_flexibility_value"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)

    params = load_parameters("configs/technology_parameters.csv")
    scenarios = {item.scenario_id: item for item in load_scenarios("configs/scenario_matrix.csv")}
    base_scenario = scenarios["base"]
    base_config = load_system_configuration(
        "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )

    factorial_tables: list[pd.DataFrame] = []
    interaction_records: list[dict[str, object]] = []
    factorial_results: dict[str, dict[str, S5DispatchResult]] = {}
    for mode in ("loose_export", "demand_mismatch"):
        timeseries = _analysis_timeseries(args.hours, mode)
        scenario = _scenario(base_scenario, mode)
        table, results = _factorial(timeseries, params, base_config, scenario, mode)
        factorial_tables.append(table)
        factorial_results[mode] = results
        interaction_records.append(_interactions(table, mode))
    factorial = pd.concat(factorial_tables, ignore_index=True)
    interactions = pd.DataFrame(interaction_records)
    break_even = _break_even_metrics(interactions, base_config)
    factorial.to_csv(args.output / "s5_factorial_value.csv", index=False)
    interactions.to_csv(args.output / "s5_synergy_and_shapley.csv", index=False)
    break_even.to_csv(args.output / "s5_annualized_cost_ceiling.csv", index=False)

    mismatch_timeseries = _analysis_timeseries(args.hours, "demand_mismatch")
    mismatch_scenario = _scenario(base_scenario, "demand_mismatch")
    full_mismatch = factorial_results["demand_mismatch"]["BHC"]
    export_dispatch_results(full_mismatch, args.output / "scenario_hourly" / "demand_mismatch_full")
    export_dispatch_results(
        factorial_results["demand_mismatch"]["none"],
        args.output / "scenario_hourly" / "demand_mismatch_direct",
    )

    outage_timeseries = _analysis_timeseries(args.hours, "cable_outage_72h")
    outage_scenario = _scenario(base_scenario, "cable_outage_72h")
    outage_records: list[dict[str, object]] = []
    for case, assets in (("none", ""), ("C", "C"), ("H", "H"), ("BHC", "BHC")):
        result = run_s5_dispatch(
            outage_timeseries, params, _asset_config(base_config, assets), outage_scenario
        )
        outage_records.append(_record(result, case, "cable_outage_72h"))
        if case in {"none", "BHC"}:
            export_dispatch_results(
                result, args.output / "scenario_hourly" / f"cable_outage_72h_{case}"
            )
    outage_table = pd.DataFrame(outage_records)
    outage_table["avoided_curtailment_mwh"] = (
        float(outage_table.loc[outage_table["asset_case"] == "none", "curtailment_mwh"].iloc[0])
        - outage_table["curtailment_mwh"]
    )
    outage_table.to_csv(args.output / "s5_cable_outage_72h.csv", index=False)

    equivalent, equivalent_result = _equivalent_cable(
        mismatch_timeseries,
        params,
        base_config,
        mismatch_scenario,
        float(full_mismatch.kpis["curtailment_mwh"]),
    )
    equivalent.to_csv(args.output / "s5_equivalent_cable_capacity.csv", index=False)
    export_dispatch_results(
        equivalent_result, args.output / "scenario_hourly" / "equivalent_cable_direct"
    )
    cable_marginal = _cable_marginal_value(
        mismatch_timeseries, params, base_config, mismatch_scenario
    )
    cable_marginal.to_csv(args.output / "s5_cable_marginal_value.csv", index=False)

    seasonal_timeseries = _analysis_timeseries(args.hours, "seasonal_lull")
    seasonal_scenario = _scenario(base_scenario, "seasonal_lull")
    seasonal_table, seasonal_results = _seasonal_resilience(
        seasonal_timeseries, params, base_config, seasonal_scenario
    )
    seasonal_table.to_csv(args.output / "s5_seasonal_hydrogen_resilience.csv", index=False)
    export_dispatch_results(
        seasonal_results[1_500_000.0],
        args.output / "scenario_hourly" / "seasonal_lull_h2_1500t",
    )
    export_dispatch_results(
        seasonal_results[0.0], args.output / "scenario_hourly" / "seasonal_lull_direct"
    )

    price_table = _price_sensitivity(mismatch_timeseries, params, base_config, base_scenario)
    price_table.to_csv(args.output / "s5_price_sensitivity.csv", index=False)

    _plot_factorial(factorial, args.output)
    _plot_mismatch_dispatch(full_mismatch, args.output)
    _plot_seasonal(seasonal_results, args.output)

    artifacts = {
        str(path.relative_to(args.output)): _sha256(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    manifest = {
        "model_version": __version__,
        "phase": "S5_flexibility_value",
        "hours": args.hours,
        "factorial_case_count": len(factorial),
        "price_sensitivity_case_count": len(price_table),
        "seasonal_storage_case_count": len(seasonal_table),
        "synthetic_data_warning": (
            "All results are mechanism-identification calculations on deterministic synthetic "
            "hourly profiles, not project investment forecasts."
        ),
        "artifacts": artifacts,
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(factorial)} factorial, {len(price_table)} price and "
        f"{len(seasonal_table)} seasonal cases to {args.output}"
    )


if __name__ == "__main__":
    main()
