"""Run Phase 7 / S6 investment sizing, threshold and replay analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-blue-hub")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from blue_hub import __version__
from blue_hub.investment_planning_model import (
    PlanningLimits,
    S6PlanningResult,
    load_investment_cost_cases,
    optimal_system_configuration,
    run_s6_investment_planning,
)
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_dispatch_results
from blue_hub.scarcity_dispatch_model import run_s5_dispatch
from blue_hub.schemas import ScenarioDefinition, SystemConfiguration, TechnologyParameters
from blue_hub.synthetic import generate_synthetic_timeseries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_limits(path: Path) -> PlanningLimits:
    with path.open(encoding="utf-8") as stream:
        return PlanningLimits(**yaml.safe_load(stream))


def _scenario(
    base: ScenarioDefinition,
    scenario_id: str,
    hydrogen_price_cny_per_kg: float,
) -> ScenarioDefinition:
    return base.model_copy(
        update={
            "scenario_id": scenario_id,
            "hydrogen_price_multiplier": hydrogen_price_cny_per_kg / 30.0,
        }
    )


def _timeseries(
    hours: int,
    scarcity: str,
    *,
    compute_price_cny_per_mwh_it: float = 360.0,
    compute_demand_mw_it: float = 300.0,
) -> pd.DataFrame:
    frame = generate_synthetic_timeseries(hours, scenario_id=f"s6_{scarcity}")
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    frame["national_compute_demand_mw_it"] = compute_demand_mw_it
    frame["national_compute_price_cny_per_mwh_it"] = compute_price_cny_per_mwh_it
    price = frame["electricity_price"].to_numpy(dtype=float)
    wind = frame["wind_cf"].to_numpy(dtype=float)
    factor = np.ones(hours)
    if scarcity == "loose":
        pass
    elif scarcity == "moderate":
        factor[price <= 360.0] = 0.55
        factor[(price <= 360.0) & (wind >= 0.52)] = 0.32
    elif scarcity == "severe":
        factor[price <= 360.0] = 0.40
        factor[(price <= 360.0) & (wind >= 0.52)] = 0.20
    else:
        raise ValueError(f"unknown scarcity case: {scarcity}")
    frame["grid_absorption_factor"] = factor
    return frame


def _zero_limits() -> PlanningLimits:
    return PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _selected_assets(capacities: dict[str, float]) -> str:
    selected: list[str] = []
    for key, label in (
        ("battery_power_mw", "B"),
        ("electrolyzer_power_mw", "H2"),
        ("fuel_cell_power_mw", "FC"),
        ("compute_it_capacity_mw", "C"),
    ):
        if capacities[key] > 1e-4:
            selected.append(label)
    return "+".join(selected) if selected else "none"


def _record(
    result: S6PlanningResult,
    direct: S6PlanningResult,
    *,
    scarcity: str,
    compute_price: float,
    hydrogen_price: float,
) -> dict[str, object]:
    kpi = result.kpis
    baseline = direct.kpis
    return {
        "scarcity_case": scarcity,
        "cost_case_id": kpi["cost_case_id"],
        "compute_price_cny_per_mwh_it": compute_price,
        "hydrogen_price_cny_per_kg": hydrogen_price,
        "selected_assets": _selected_assets(result.capacities),
        **result.capacities,
        "annualized_operating_margin_cny": kpi["annualized_operating_margin_cny"],
        "flexible_asset_capex_cny": kpi["flexible_asset_capex_cny"],
        "annual_fixed_om_cny": kpi["annual_fixed_om_cny"],
        "annualized_capital_recovery_cny": kpi["annualized_capital_recovery_cny"],
        "annualized_flexible_fixed_cost_cny": kpi["annualized_flexible_fixed_cost_cny"],
        "incremental_operating_margin_cny": (
            kpi["annualized_operating_margin_cny"]
            - baseline["annualized_operating_margin_cny"]
        ),
        "incremental_net_value_cny": (
            kpi["net_annual_value_cny"] - baseline["net_annual_value_cny"]
        ),
        "curtailment_mwh": kpi["curtailment_mwh"],
        "avoided_curtailment_mwh": baseline["curtailment_mwh"] - kpi["curtailment_mwh"],
        "hydrogen_sales_kg": kpi["hydrogen_sales_kg"],
        "spot_compute_service_mwh_it": kpi["spot_compute_service_mwh_it"],
        "eens_mwh": kpi["eens_mwh"],
        "configuration_hash": result.metadata["configuration_hash"],
    }


def _solve(
    frame: pd.DataFrame,
    params: TechnologyParameters,
    config: SystemConfiguration,
    scenario: ScenarioDefinition,
    cost_case,
    limits: PlanningLimits,
) -> S6PlanningResult:
    return run_s6_investment_planning(frame, params, config, scenario, cost_case, limits)


def _ar1(rng: np.random.Generator, periods: int, phi: float) -> np.ndarray:
    innovations = rng.normal(0.0, 1.0, periods)
    values = np.zeros(periods)
    for t in range(1, periods):
        values[t] = phi * values[t - 1] + np.sqrt(1.0 - phi**2) * innovations[t]
    return values


def _perturbed_year(base: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create a deterministic synthetic out-of-sample year with correlated shocks."""
    rng = np.random.default_rng(seed)
    frame = base.copy()
    periods = len(frame)
    wind_noise = _ar1(rng, periods, 0.94)
    grid_noise = _ar1(rng, periods, 0.85)
    price_noise = _ar1(rng, periods, 0.75)
    compute_noise = _ar1(rng, periods, 0.80)
    wind_scale = rng.uniform(0.88, 1.10)
    demand_scale = rng.uniform(0.85, 1.15)
    frame["wind_cf"] = np.clip(
        frame["wind_cf"].to_numpy(dtype=float) * wind_scale + 0.035 * wind_noise,
        0.02,
        0.95,
    )
    frame["grid_absorption_factor"] = np.clip(
        frame["grid_absorption_factor"].to_numpy(dtype=float)
        + rng.uniform(-0.08, 0.08)
        + 0.035 * grid_noise,
        0.10,
        1.00,
    )
    frame["electricity_price"] = np.clip(
        frame["electricity_price"].to_numpy(dtype=float)
        * np.exp(0.10 * price_noise),
        80.0,
        1_000.0,
    )
    frame["national_compute_demand_mw_it"] = np.clip(
        frame["national_compute_demand_mw_it"].to_numpy(dtype=float)
        * demand_scale
        * (1.0 + 0.06 * compute_noise),
        0.0,
        None,
    )
    frame["national_compute_price_cny_per_mwh_it"] = np.clip(
        frame["national_compute_price_cny_per_mwh_it"].to_numpy(dtype=float)
        * np.exp(0.08 * compute_noise),
        100.0,
        None,
    )
    frame["hydrogen_demand"] *= rng.uniform(0.85, 1.15)
    if seed % 3 == 0:
        start = int(rng.integers(0, max(periods - 48, 1)))
        frame.loc[start : min(start + 47, periods - 1), "tx_availability"] = 0.0
    if seed % 4 == 0:
        start = int(rng.integers(0, max(periods - 48, 1)))
        frame.loc[start : min(start + 47, periods - 1), "fiber_availability"] = 0.0
    if seed % 5 == 0:
        start = int(rng.integers(0, max(periods - 168, 1)))
        frame.loc[start : min(start + 167, periods - 1), "wind_availability"] *= 0.15
    frame["scenario_id"] = f"s6_oos_seed_{seed}"
    return frame


def _without_assets(config: SystemConfiguration, config_id: str) -> SystemConfiguration:
    return config.model_copy(
        update={
            "config_id": config_id,
            "battery_power_mw": 0.0,
            "battery_energy_mwh": 0.0,
            "electrolyzer_power_mw": 0.0,
            "hydrogen_storage_kg": 0.0,
            "fuel_cell_power_mw": 0.0,
            "compute_it_capacity_mw": 0.0,
            "subsea_fiber_service_capacity_mw_it": 0.0,
            "initial_battery_soc_fraction": 0.0,
            "initial_hydrogen_inventory_fraction": 0.0,
            "battery_reserve_power_mw": 0.0,
            "battery_reserve_duration_h": 0.0,
        }
    )


def _out_of_sample_replay(
    base_frame: pd.DataFrame,
    params: TechnologyParameters,
    base_config: SystemConfiguration,
    scenario: ScenarioDefinition,
    design: S6PlanningResult,
    seeds: int,
) -> pd.DataFrame:
    optimal_config = optimal_system_configuration(base_config, design)
    direct_config = _without_assets(base_config, "S6_oos_direct")
    fixed_cost = float(design.kpis["annualized_flexible_fixed_cost_cny"])
    stack_cost_rate = float(
        design.metadata["annualized_unit_costs"][
            "electrolyzer_stack_replacement_cny_per_mwh"
        ]
    )
    records: list[dict[str, object]] = []
    for seed in range(1, seeds + 1):
        frame = _perturbed_year(base_frame, seed)
        annualization_factor = 8_760.0 / len(frame)
        direct = run_s5_dispatch(frame, params, direct_config, scenario)
        flexible = run_s5_dispatch(frame, params, optimal_config, scenario)
        stack_replacement = (
            flexible.kpis["hydrogen_production_kg"]
            * params.value("hydrogen_sec_system")
            / 1_000.0
            * stack_cost_rate
        )
        incremental_operating = (
            flexible.kpis["operating_margin_cny"]
            - stack_replacement
            - direct.kpis["operating_margin_cny"]
        ) * annualization_factor
        records.append(
            {
                "seed": seed,
                "wind_generation_mwh": flexible.kpis["wind_generation_mwh"],
                "direct_curtailment_mwh": direct.kpis["curtailment_mwh"],
                "flexible_curtailment_mwh": flexible.kpis["curtailment_mwh"],
                "avoided_curtailment_mwh": (
                    direct.kpis["curtailment_mwh"] - flexible.kpis["curtailment_mwh"]
                ),
                "incremental_operating_margin_cny": incremental_operating,
                "annualized_flexible_fixed_cost_cny": fixed_cost,
                "incremental_net_value_cny": incremental_operating - fixed_cost,
                "hydrogen_sales_kg": flexible.kpis["hydrogen_sales_kg"],
                "spot_compute_service_mwh_it": flexible.kpis[
                    "spot_compute_service_mwh_it"
                ],
                "eens_mwh": flexible.kpis["eens_mwh"],
                "tx_outage_hours": int((frame["tx_availability"] == 0.0).sum()),
                "fiber_outage_hours": int((frame["fiber_availability"] == 0.0).sum()),
                "configuration_hash": flexible.metadata["configuration_hash"],
            }
        )
    return pd.DataFrame(records)


def _plot_frontiers(compute: pd.DataFrame, hydrogen: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        compute["compute_price_cny_per_mwh_it"],
        compute["compute_it_capacity_mw"],
        marker="o",
        color="#1d4ed8",
        label="Compute",
    )
    axes[0].set_xlabel("Compute service price (CNY/MWh-IT)")
    axes[0].set_ylabel("Optimal compute capacity (MW-IT)")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Compute contract entry range")
    axes[1].plot(
        hydrogen["hydrogen_price_cny_per_kg"],
        hydrogen["electrolyzer_power_mw"],
        marker="o",
        color="#0f766e",
        label="Electrolyzer (MW)",
    )
    axes[1].plot(
        hydrogen["hydrogen_price_cny_per_kg"],
        hydrogen["hydrogen_storage_kg"] / 1_000.0,
        marker="s",
        color="#d97706",
        label="Hydrogen storage (tonne)",
    )
    axes[1].set_xlabel("Delivered hydrogen price (CNY/kg)")
    axes[1].set_ylabel("Optimal capacity (MW or tonne)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_title("Hydrogen contract entry range")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cost_scarcity(matrix: pd.DataFrame, path: Path) -> None:
    order_cost = ["cost_low", "cost_reference", "cost_high"]
    order_scarcity = ["loose", "moderate", "severe"]
    pivot = (
        matrix.pivot(
            index="cost_case_id", columns="scarcity_case", values="incremental_net_value_cny"
        )
        .reindex(index=order_cost, columns=order_scarcity)
        / 1e6
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(order_scarcity)), ["Loose", "Moderate", "Severe"])
    ax.set_yticks(range(len(order_cost)), ["Low", "Reference", "High"])
    for i in range(len(order_cost)):
        for j in range(len(order_scarcity)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center")
    ax.set_title("Incremental net annual value of flexible assets")
    fig.colorbar(image, ax=ax, label="million CNY/year")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_oos(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("incremental_net_value_cny")
    colors = np.where(ordered["incremental_net_value_cny"] >= 0.0, "#0f766e", "#b91c1c")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(ordered["seed"].astype(str), ordered["incremental_net_value_cny"] / 1e6, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Synthetic out-of-sample year")
    ax.set_ylabel("Incremental net annual value (million CNY/year)")
    ax.set_title("Fixed design under out-of-sample supply-demand shocks")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8_760)
    parser.add_argument("--oos-years", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/s6_investment_value")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    params = load_parameters(root / "configs/technology_parameters.csv")
    base_config = load_system_configuration(
        root / "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )
    scenarios = {
        item.scenario_id: item
        for item in load_scenarios(root / "configs/scenario_matrix.csv")
    }
    base_scenario = scenarios["base"]
    costs = {
        item.cost_case_id: item
        for item in load_investment_cost_cases(root / "configs/s6_investment_cost_cases.csv")
    }
    limits = _load_limits(root / "configs/s6_planning_limits.yaml")

    matrix_records: list[dict[str, object]] = []
    direct_by_scarcity: dict[str, S6PlanningResult] = {}
    for scarcity in ("loose", "moderate", "severe"):
        frame = _timeseries(args.hours, scarcity)
        scenario = _scenario(base_scenario, f"s6_{scarcity}", 22.5)
        direct = _solve(
            frame,
            params,
            base_config,
            scenario,
            costs["cost_reference"],
            _zero_limits(),
        )
        direct_by_scarcity[scarcity] = direct
        for cost_case in costs.values():
            result = _solve(frame, params, base_config, scenario, cost_case, limits)
            matrix_records.append(
                _record(
                    result,
                    direct,
                    scarcity=scarcity,
                    compute_price=360.0,
                    hydrogen_price=22.5,
                )
            )
    matrix = pd.DataFrame(matrix_records)
    matrix_path = output / "s6_cost_scarcity_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    moderate_direct = direct_by_scarcity["moderate"]
    compute_records: list[dict[str, object]] = []
    for compute_price in (360.0, 900.0, 1_200.0, 1_800.0, 2_400.0, 3_200.0):
        frame = _timeseries(args.hours, "moderate", compute_price_cny_per_mwh_it=compute_price)
        scenario = _scenario(base_scenario, f"s6_compute_{compute_price:.0f}", 22.5)
        result = _solve(
            frame,
            params,
            base_config,
            scenario,
            costs["cost_reference"],
            limits,
        )
        compute_records.append(
            _record(
                result,
                moderate_direct,
                scarcity="moderate",
                compute_price=compute_price,
                hydrogen_price=22.5,
            )
        )
    compute_frontier = pd.DataFrame(compute_records)
    compute_path = output / "s6_compute_price_frontier.csv"
    compute_frontier.to_csv(compute_path, index=False)

    hydrogen_records: list[dict[str, object]] = []
    for hydrogen_price in (22.5, 27.5, 30.0, 35.0, 40.0, 50.0):
        frame = _timeseries(args.hours, "moderate")
        scenario = _scenario(base_scenario, f"s6_hydrogen_{hydrogen_price:.1f}", hydrogen_price)
        result = _solve(
            frame,
            params,
            base_config,
            scenario,
            costs["cost_reference"],
            limits,
        )
        hydrogen_records.append(
            _record(
                result,
                moderate_direct,
                scarcity="moderate",
                compute_price=360.0,
                hydrogen_price=hydrogen_price,
            )
        )
    hydrogen_frontier = pd.DataFrame(hydrogen_records)
    hydrogen_path = output / "s6_hydrogen_price_frontier.csv"
    hydrogen_frontier.to_csv(hydrogen_path, index=False)

    strategic_frame = _timeseries(
        args.hours, "severe", compute_price_cny_per_mwh_it=1_800.0
    )
    strategic_scenario = _scenario(base_scenario, "s6_strategic_contract", 35.0)
    strategic = _solve(
        strategic_frame,
        params,
        base_config,
        strategic_scenario,
        costs["cost_reference"],
        limits,
    )
    strategic_record = _record(
        strategic,
        direct_by_scarcity["severe"],
        scarcity="severe",
        compute_price=1_800.0,
        hydrogen_price=35.0,
    )
    strategic_path = output / "s6_strategic_case_summary.csv"
    pd.DataFrame([strategic_record]).to_csv(strategic_path, index=False)
    export_dispatch_results(strategic, output / "strategic_design_hourly")

    oos = _out_of_sample_replay(
        strategic_frame,
        params,
        base_config,
        strategic_scenario,
        strategic,
        args.oos_years,
    )
    oos_path = output / "s6_out_of_sample_replay.csv"
    oos.to_csv(oos_path, index=False)
    net = oos["incremental_net_value_cny"]
    oos_summary = pd.DataFrame(
        [
            {
                "oos_years": len(oos),
                "mean_incremental_net_value_cny": net.mean(),
                "median_incremental_net_value_cny": net.median(),
                "p10_incremental_net_value_cny": net.quantile(0.10),
                "worst_incremental_net_value_cny": net.min(),
                "best_incremental_net_value_cny": net.max(),
                "positive_net_value_share": float((net > 0.0).mean()),
                "mean_avoided_curtailment_mwh": oos["avoided_curtailment_mwh"].mean(),
                "design_selected_assets": strategic_record["selected_assets"],
                **strategic.capacities,
            }
        ]
    )
    oos_summary_path = output / "s6_out_of_sample_summary.csv"
    oos_summary.to_csv(oos_summary_path, index=False)

    frontier_figure = figures / "s6_contract_price_frontiers.png"
    matrix_figure = figures / "s6_cost_scarcity_net_value.png"
    oos_figure = figures / "s6_out_of_sample_net_value.png"
    _plot_frontiers(compute_frontier, hydrogen_frontier, frontier_figure)
    _plot_cost_scarcity(matrix, matrix_figure)
    _plot_oos(oos, oos_figure)

    artifacts = [
        matrix_path,
        compute_path,
        hydrogen_path,
        strategic_path,
        oos_path,
        oos_summary_path,
        frontier_figure,
        matrix_figure,
        oos_figure,
    ]
    manifest = {
        "phase": "S6",
        "model_version": __version__,
        "hours": args.hours,
        "oos_years": args.oos_years,
        "cost_cases": [asdict(case) for case in costs.values()],
        "planning_limits": asdict(limits),
        "strategic_configuration_hash": strategic.metadata["configuration_hash"],
        "artifacts": {str(path.relative_to(output)): _sha256(path) for path in artifacts},
        "limitations": [
            "synthetic time series rather than site and landing-point observations",
            "perfect-foresight dispatch in both design and out-of-sample replay",
            "compute and fibre installed cost remains an engineering screening range",
            "wave energy and full wind/export-project finance remain outside S6",
        ],
    }
    manifest_path = output / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(matrix.to_string(index=False))
    print(compute_frontier.to_string(index=False))
    print(hydrogen_frontier.to_string(index=False))
    print(pd.DataFrame([strategic_record]).to_string(index=False))
    print(oos_summary.to_string(index=False))
    print(f"Wrote S6 analysis to {output}")


if __name__ == "__main__":
    main()
