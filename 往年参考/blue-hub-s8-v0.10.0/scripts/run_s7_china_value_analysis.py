"""Run Phase 8 / S7 China-calibrated pain-point and value analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-blue-hub-s7")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from blue_hub import __version__
from blue_hub.china_counterfactual_model import (
    S7CounterfactualResult,
    S7Design,
    calibrate_landing_limit_mw,
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
from blue_hub.synthetic import generate_synthetic_timeseries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_limits(path: Path) -> PlanningLimits:
    with path.open(encoding="utf-8") as stream:
        return PlanningLimits(**yaml.safe_load(stream))


def _rescale_clipped(values: np.ndarray, target_mean: float) -> np.ndarray:
    low, high = 0.0, 5.0
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        mean = float(np.clip(values * midpoint, 0.02, 0.95).mean())
        if mean < target_mean:
            low = midpoint
        else:
            high = midpoint
    return np.clip(values * high, 0.02, 0.95)


def _china_reference_timeseries(hours: int) -> pd.DataFrame:
    """Build a deterministic China-context screening year, not site observations."""
    frame = generate_synthetic_timeseries(hours, scenario_id="s7_china_reference")
    timestamp = pd.to_datetime(frame["timestamp"])
    t = np.arange(hours, dtype=float)
    hour = timestamp.dt.hour.to_numpy()
    day = t / 24.0

    seasonal = 1.0 + 0.16 * np.cos(2.0 * np.pi * (day - 20.0) / 365.0)
    synoptic = 1.0 + 0.09 * np.sin(2.0 * np.pi * day / 11.0)
    raw_wind = frame["wind_cf"].to_numpy(dtype=float) * seasonal * synoptic
    target_gross_cf = (3_300.0 / 8_760.0) / 0.97
    frame["wind_cf"] = _rescale_clipped(raw_wind, target_gross_cf)

    price = np.full(hours, 350.0)
    price[(hour >= 0) & (hour < 7)] = 210.0
    price[(hour >= 10) & (hour < 16)] = 300.0
    price[(hour >= 17) & (hour < 22)] = 620.0
    frame["electricity_price"] = price

    landing_profile = np.full(hours, 0.95)
    landing_profile[(hour >= 0) & (hour < 7)] = 0.78
    landing_profile[(hour >= 17) & (hour < 22)] = 1.00
    frame["landing_demand_factor"] = landing_profile
    frame["grid_absorption_factor"] = 1.0

    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    compute_pool = 420.0 + 60.0 * np.sin(2.0 * np.pi * (t - 5.0) / 24.0)
    frame["national_compute_demand_mw_it"] = np.clip(compute_pool, 250.0, None)
    frame["national_compute_flexible_fraction"] = 0.35
    frame["national_compute_price_cny_per_mwh_it"] = 1_500.0
    frame["hydrogen_demand"] = 12_000.0
    return frame


def _with_markets(
    base: pd.DataFrame,
    *,
    compute_price: float,
    hydrogen_flexible_fraction: float | None = None,
    compute_flexible_fraction: float | None = None,
) -> pd.DataFrame:
    frame = base.copy()
    frame["national_compute_price_cny_per_mwh_it"] = compute_price
    if compute_flexible_fraction is not None:
        frame["national_compute_flexible_fraction"] = compute_flexible_fraction
    if hydrogen_flexible_fraction is not None:
        frame["hydrogen_demand"] *= hydrogen_flexible_fraction
    return frame


def _scenario(base, scenario_id: str, *, hydrogen_price: float, distance_km: float = 200.0):
    return base.model_copy(
        update={
            "scenario_id": scenario_id,
            "offshore_distance_km": distance_km,
            "hydrogen_price_multiplier": hydrogen_price / 30.0,
            "tx_outage_hours": 0,
            "fiber_outage_hours": 0,
            "wind_event_hours": 0,
            "price_event_hours": 0,
        }
    )


def _design(
    design_id: str,
    strategy: str,
    *,
    tx: float,
    landing: float,
    flexible: bool,
    hub: bool,
    pv: float = 0.0,
    wave: float = 0.0,
) -> S7Design:
    return S7Design(
        design_id=design_id,
        strategy=strategy,
        wind_capacity_mw=1_000.0,
        pv_capacity_mw=pv,
        wave_capacity_mw=wave,
        tx_capacity_mw=tx,
        landing_grid_limit_mw=landing,
        include_hub_common_assets=hub,
        enable_flexible_assets=flexible,
    )


def _selected_assets(result: S7CounterfactualResult) -> str:
    selected: list[str] = []
    for key, label in (
        ("battery_power_mw", "BESS"),
        ("electrolyzer_power_mw", "H2"),
        ("hydrogen_storage_kg", "H2-store"),
        ("fuel_cell_power_mw", "FC"),
        ("compute_it_capacity_mw", "Compute"),
    ):
        if float(result.kpis[key]) > 1e-5:
            selected.append(label)
    return "+".join(selected) if selected else "none"


def _record(result: S7CounterfactualResult) -> dict[str, object]:
    return {**result.kpis, "selected_assets": _selected_assets(result)}


def _counterfactual_record(
    result: S7CounterfactualResult,
    direct: S7CounterfactualResult,
) -> dict[str, object]:
    record = _record(result)
    record.update(
        {
            "incremental_full_net_value_vs_direct_cny_per_year": (
                result.kpis["full_project_net_annual_value_cny"]
                - direct.kpis["full_project_net_annual_value_cny"]
            ),
            "incremental_capex_vs_direct_cny": (
                result.kpis["full_project_capex_cny"]
                - direct.kpis["full_project_capex_cny"]
            ),
            "avoided_curtailment_vs_direct_mwh": (
                direct.kpis["curtailment_mwh"] - result.kpis["curtailment_mwh"]
            ),
            "utilization_gain_vs_direct_percentage_points": 100.0
            * (
                result.kpis["renewable_utilization_rate"]
                - direct.kpis["renewable_utilization_rate"]
            ),
        }
    )
    return record


def _without_assets(config, *, config_id: str, tx_capacity_mw: float):
    return config.model_copy(
        update={
            "config_id": config_id,
            "tx_capacity_mw": tx_capacity_mw,
            "battery_power_mw": 0.0,
            "battery_energy_mwh": 0.0,
            "electrolyzer_power_mw": 0.0,
            "hydrogen_storage_kg": 0.0,
            "hydrogen_export_capacity_kg_per_h": 0.0,
            "fuel_cell_power_mw": 0.0,
            "compute_it_capacity_mw": 0.0,
            "subsea_fiber_service_capacity_mw_it": 0.0,
            "initial_battery_soc_fraction": 0.0,
            "initial_hydrogen_inventory_fraction": 0.0,
            "battery_reserve_power_mw": 0.0,
            "battery_reserve_duration_h": 0.0,
        }
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def _plot_pain(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = frame["absorption_case"].tolist()
    axes[0].bar(labels, 100.0 * frame["curtailment_rate"], color="#c44e52")
    axes[0].set_ylabel("Curtailment (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, frame["constrained_grid_hours"], color="#4c72b0")
    axes[1].set_ylabel("Landing-constrained hours")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Observed-national calibration and local landing stress cases")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_strategies(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    labels = frame["strategy"].tolist()
    axes[0].bar(
        labels,
        frame["incremental_full_net_value_vs_direct_cny_per_year"] / 1e6,
        color="#55a868",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Incremental full net value (CNY million/yr)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, 100.0 * frame["renewable_utilization_rate"], color="#4c72b0")
    axes[1].set_ylabel("Renewable utilization (%)")
    axes[1].set_ylim(0.0, 102.0)
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Same-resource counterfactual strategies at a 450 MW landing point")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_frontier(frame: pd.DataFrame, path: Path) -> None:
    values = frame.pivot(
        index="hydrogen_price_cny_per_kg",
        columns="compute_price_cny_per_mwh_it",
        values="incremental_full_net_value_vs_direct_cny_per_year",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(values.to_numpy() / 1e6, origin="lower", aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(values.columns)), [f"{value:g}" for value in values.columns])
    ax.set_yticks(range(len(values.index)), [f"{value:g}" for value in values.index])
    ax.set_xlabel("Compute service value (CNY/MWh-IT)")
    ax.set_ylabel("Delivered hydrogen price (CNY/kg)")
    ax.set_title("Integrated hub incremental full net value (CNY million/yr)")
    for row in range(len(values.index)):
        for column in range(len(values.columns)):
            ax.text(
                column,
                row,
                f"{values.iloc[row, column] / 1e6:.0f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="CNY million/yr")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_allocation(result: S7CounterfactualResult, path: Path) -> None:
    renewable = float(result.kpis["renewable_generation_mwh"])
    export_send = float(result.planning.hourly["export_send_mw"].sum())
    compute = float(result.kpis["dc_facility_energy_mwh"])
    hydrogen = float(result.kpis["electrolyzer_energy_mwh"])
    curtailment = float(result.kpis["curtailment_mwh"])
    accounted = export_send + compute + hydrogen + curtailment
    other = max(renewable - accounted, 0.0)
    labels = ["Power export", "Compute", "Hydrogen", "Curtailment", "Island/losses"]
    values = np.array([export_send, compute, hydrogen, curtailment, other]) / 1e6
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    left = 0.0
    colors = ["#4c72b0", "#8172b3", "#55a868", "#c44e52", "#ccb974"]
    for label, value, color in zip(labels, values, colors, strict=True):
        ax.barh(["Annual allocation"], [value], left=left, label=label, color=color)
        left += value
    ax.set_xlabel("TWh at offshore bus (IT service uses facility electricity)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax.set_title("Optimized allocation under the representative contract case")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(root: Path, output: Path, hours: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

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
    limits = _load_limits(root / "configs/s7_china_planning_limits.yaml")
    reference_flexible = flexible_cases["china_reference"]
    reference_common = infrastructure_cases["china_reference"]
    frame = _china_reference_timeseries(hours)

    national_landing = calibrate_landing_limit_mw(
        frame,
        wind_capacity_mw=1_000.0,
        pv_capacity_mw=0.0,
        tx_capacity_mw=700.0,
        target_utilization=0.959,
    )
    absorption_cases = {
        "observed_national_2024": national_landing,
        "local_cluster_450mw": 450.0,
        "deep_constraint_300mw": 300.0,
    }

    pain_records: list[dict[str, object]] = []
    zero = PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pain_results: dict[str, S7CounterfactualResult] = {}
    for case_id, landing in absorption_cases.items():
        result = evaluate_s7_design(
            frame,
            parameters,
            base_config,
            _scenario(base_scenario, f"s7_{case_id}", hydrogen_price=25.0),
            reference_flexible,
            reference_common,
            zero,
            _design(
                f"direct_{case_id}",
                "direct_existing",
                tx=700.0,
                landing=landing,
                flexible=False,
                hub=False,
            ),
        )
        pain_results[case_id] = result
        record = _record(result)
        record["absorption_case"] = case_id
        record["curtailment_rate"] = 1.0 - result.kpis["renewable_utilization_rate"]
        pain_records.append(record)
    pain = pd.DataFrame(pain_records)
    _write(pain, output / "s7_pain_point_diagnostics.csv")
    _plot_pain(pain, figures / "s7_pain_point_diagnostics.png")

    cluster_direct = pain_results["local_cluster_450mw"]
    strategy_specs = (
        _design(
            "cluster_direct_existing",
            "direct_existing",
            tx=700.0,
            landing=450.0,
            flexible=False,
            hub=False,
        ),
        _design(
            "cluster_cable_only",
            "cable_only",
            tx=1_000.0,
            landing=450.0,
            flexible=False,
            hub=False,
        ),
        _design(
            "cluster_grid_reinforced",
            "grid_reinforced",
            tx=1_000.0,
            landing=900.0,
            flexible=False,
            hub=False,
        ),
        _design(
            "cluster_integrated_hub",
            "integrated_hub",
            tx=700.0,
            landing=450.0,
            flexible=True,
            hub=True,
        ),
        _design(
            "cluster_hybrid",
            "hybrid_reinforcement_hub",
            tx=850.0,
            landing=650.0,
            flexible=True,
            hub=True,
        ),
    )
    strategy_records: list[dict[str, object]] = []
    strategy_results: dict[str, S7CounterfactualResult] = {}
    reference_scenario = _scenario(
        base_scenario, "s7_cluster_reference_markets", hydrogen_price=25.0
    )
    for design in strategy_specs:
        result = (
            cluster_direct
            if design.strategy == "direct_existing"
            else evaluate_s7_design(
                frame,
                parameters,
                base_config,
                reference_scenario,
                reference_flexible,
                reference_common,
                limits,
                design,
            )
        )
        strategy_results[design.strategy] = result
        strategy_records.append(_counterfactual_record(result, cluster_direct))
    strategies = pd.DataFrame(strategy_records)
    _write(strategies, output / "s7_strategy_counterfactual.csv")
    _plot_strategies(strategies, figures / "s7_strategy_counterfactual.png")

    frontier_records: list[dict[str, object]] = []
    frontier_results: dict[tuple[float, float], S7CounterfactualResult] = {}
    hub_design = _design(
        "cluster_integrated_hub_contract",
        "integrated_hub",
        tx=700.0,
        landing=450.0,
        flexible=True,
        hub=True,
    )
    for hydrogen_price in (20.0, 25.0, 30.0, 35.0):
        for compute_price in (1_000.0, 1_500.0, 2_000.0, 2_500.0):
            market_frame = _with_markets(frame, compute_price=compute_price)
            result = evaluate_s7_design(
                market_frame,
                parameters,
                base_config,
                _scenario(
                    base_scenario,
                    f"s7_frontier_c{compute_price:g}_h{hydrogen_price:g}",
                    hydrogen_price=hydrogen_price,
                ),
                reference_flexible,
                reference_common,
                limits,
                hub_design,
            )
            frontier_results[(compute_price, hydrogen_price)] = result
            record = _counterfactual_record(result, cluster_direct)
            record["compute_price_cny_per_mwh_it"] = compute_price
            record["hydrogen_price_cny_per_kg"] = hydrogen_price
            frontier_records.append(record)
    frontier = pd.DataFrame(frontier_records)
    _write(frontier, output / "s7_contract_value_frontier.csv")
    _plot_frontier(frontier, figures / "s7_contract_value_frontier.png")

    contract_result = frontier_results[(2_000.0, 30.0)]
    contract_result.planning.hourly.to_csv(
        output / "s7_representative_contract_hourly.csv", index=False, float_format="%.8g"
    )
    _plot_allocation(contract_result, figures / "s7_representative_energy_allocation.png")

    portability_records: list[dict[str, object]] = []
    for fraction in (0.15, 0.35, 0.60):
        portable_frame = _with_markets(
            frame, compute_price=2_000.0, compute_flexible_fraction=fraction
        )
        result = evaluate_s7_design(
            portable_frame,
            parameters,
            base_config,
            _scenario(
                base_scenario, f"s7_portability_{fraction:.2f}", hydrogen_price=30.0
            ),
            reference_flexible,
            reference_common,
            limits,
            hub_design,
        )
        record = _counterfactual_record(result, cluster_direct)
        record["national_compute_flexible_fraction"] = fraction
        portability_records.append(record)
    portability = pd.DataFrame(portability_records)
    _write(portability, output / "s7_compute_portability_sensitivity.csv")

    cost_records: list[dict[str, object]] = []
    contract_frame = _with_markets(frame, compute_price=2_000.0)
    for suffix in ("low", "reference", "high"):
        flex_cost = flexible_cases[f"china_{suffix}"]
        common_cost = infrastructure_cases[f"china_{suffix}"]
        direct = evaluate_s7_design(
            contract_frame,
            parameters,
            base_config,
            _scenario(base_scenario, f"s7_cost_{suffix}_direct", hydrogen_price=30.0),
            flex_cost,
            common_cost,
            zero,
            _design(
                f"cost_{suffix}_direct",
                "direct_existing",
                tx=700.0,
                landing=450.0,
                flexible=False,
                hub=False,
            ),
        )
        result = evaluate_s7_design(
            contract_frame,
            parameters,
            base_config,
            _scenario(base_scenario, f"s7_cost_{suffix}_hub", hydrogen_price=30.0),
            flex_cost,
            common_cost,
            limits,
            hub_design,
        )
        record = _counterfactual_record(result, direct)
        record["cost_case"] = suffix
        cost_records.append(record)
    cost_uncertainty = pd.DataFrame(cost_records)
    _write(cost_uncertainty, output / "s7_cost_uncertainty.csv")

    resource_records: list[dict[str, object]] = []
    resource_specs = (
        ("wind_only", 0.0, 0.0),
        ("wind_pv", 200.0, 0.0),
        ("wind_wave", 0.0, 50.0),
        ("wind_pv_wave", 200.0, 50.0),
    )
    resource_baseline = contract_result
    for resource_id, pv, wave in resource_specs:
        result = (
            resource_baseline
            if resource_id == "wind_only"
            else evaluate_s7_design(
                contract_frame,
                parameters,
                base_config,
                _scenario(
                    base_scenario, f"s7_resource_{resource_id}", hydrogen_price=30.0
                ),
                reference_flexible,
                reference_common,
                limits,
                _design(
                    f"resource_{resource_id}",
                    "integrated_hub",
                    tx=700.0,
                    landing=450.0,
                    flexible=True,
                    hub=True,
                    pv=pv,
                    wave=wave,
                ),
            )
        )
        record = _record(result)
        record["resource_case"] = resource_id
        record["renewable_output_cv"] = float(
            result.planning.hourly["renewable_available_mw"].std()
            / result.planning.hourly["renewable_available_mw"].mean()
        )
        record["incremental_full_net_vs_wind_only_cny_per_year"] = (
            result.kpis["full_project_net_annual_value_cny"]
            - resource_baseline.kpis["full_project_net_annual_value_cny"]
        )
        record["incremental_generation_vs_wind_only_mwh"] = (
            result.kpis["renewable_generation_mwh"]
            - resource_baseline.kpis["renewable_generation_mwh"]
        )
        resource_records.append(record)
    resources = pd.DataFrame(resource_records)
    _write(resources, output / "s7_resource_addon_screening.csv")

    outage_frame = contract_frame.copy()
    outage_frame["grid_absorption_limit_mw"] = 450.0 * outage_frame[
        "landing_demand_factor"
    ]
    outage_scenario = _scenario(
        base_scenario, "s7_fixed_hub_cable_outage_72h", hydrogen_price=30.0
    ).model_copy(update={"tx_outage_hours": 72, "tx_outage_start_hour": 1_000})
    fixed_hub_config = optimal_system_configuration(
        base_config, contract_result.planning, config_id="S7_fixed_contract_hub"
    )
    fixed_direct_config = _without_assets(
        base_config, config_id="S7_fixed_direct", tx_capacity_mw=700.0
    )
    fixed_hub = run_s5_dispatch(
        outage_frame, parameters, fixed_hub_config, outage_scenario
    )
    fixed_direct = run_s5_dispatch(
        outage_frame, parameters, fixed_direct_config, outage_scenario
    )
    event = slice(1_000, min(1_072, hours))
    stress_records: list[dict[str, object]] = []
    for strategy, result in (("direct", fixed_direct), ("integrated_hub", fixed_hub)):
        event_hourly = result.hourly.iloc[event]
        stress_records.append(
            {
                "stress_case": "fixed_capacity_cable_outage_72h",
                "strategy": strategy,
                "event_hours": len(event_hourly),
                "event_curtailment_mwh": float(event_hourly["curtailment_mw"].sum()),
                "event_compute_service_mwh_it": float(
                    event_hourly["spot_compute_completed_mwh_it"].sum()
                ),
                "event_hydrogen_production_kg": float(
                    event_hourly["hydrogen_production_kg"].sum()
                ),
                "event_eens_mwh": float(event_hourly["unmet_critical_load_mw"].sum()),
                "annual_operating_margin_cny": result.kpis["operating_margin_cny"],
                "max_offshore_balance_residual_mw": result.kpis[
                    "max_offshore_balance_residual_mw"
                ],
            }
        )

    reliability_frame = frame.copy()
    reliability_frame["national_compute_demand_mw_it"] = 0.0
    reliability_frame["hydrogen_demand"] = 0.0
    reliability_frame["critical_load"] = 50.0
    lull_start = min(4_000, max(hours // 2, 0))
    lull_end = min(lull_start + 720, hours)
    reliability_frame.loc[lull_start : lull_end - 1, "wind_availability"] = 0.03
    reliability_scenario = _scenario(
        base_scenario, "s7_30day_low_wind_reliability", hydrogen_price=20.0
    )
    reliability_direct = evaluate_s7_design(
        reliability_frame,
        parameters,
        base_config,
        reliability_scenario,
        reference_flexible,
        reference_common,
        zero,
        _design(
            "reliability_direct",
            "direct_existing",
            tx=700.0,
            landing=450.0,
            flexible=False,
            hub=False,
        ),
    )
    reliability_hub = evaluate_s7_design(
        reliability_frame,
        parameters,
        base_config,
        reliability_scenario,
        reference_flexible,
        reference_common,
        limits,
        _design(
            "reliability_hub",
            "integrated_hub_reliability",
            tx=700.0,
            landing=450.0,
            flexible=True,
            hub=True,
        ),
    )
    for strategy, result in (
        ("direct", reliability_direct),
        ("integrated_hub", reliability_hub),
    ):
        stress_records.append(
            {
                "stress_case": "endogenous_30day_low_wind",
                "strategy": strategy,
                "event_hours": lull_end - lull_start,
                "event_curtailment_mwh": float(
                    result.planning.hourly.iloc[lull_start:lull_end]["curtailment_mw"].sum()
                ),
                "event_compute_service_mwh_it": 0.0,
                "event_hydrogen_production_kg": float(
                    result.planning.hourly.iloc[lull_start:lull_end][
                        "hydrogen_production_kg"
                    ].sum()
                ),
                "event_eens_mwh": float(
                    result.planning.hourly.iloc[lull_start:lull_end][
                        "unmet_critical_load_mw"
                    ].sum()
                ),
                "annual_operating_margin_cny": result.planning.kpis[
                    "annualized_operating_margin_cny"
                ],
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
    stress = pd.DataFrame(stress_records)
    _write(stress, output / "s7_fixed_and_reliability_stress_tests.csv")

    reinforced = strategy_results["grid_reinforced"]
    hub_reference = strategy_results["integrated_hub"]
    cable_only = strategy_results["cable_only"]
    reference_frontier = frontier_results[(2_000.0, 30.0)]
    wind_only_resource = resources.loc[resources["resource_case"] == "wind_only"].iloc[0]
    pv_wave_resource = resources.loc[resources["resource_case"] == "wind_pv_wave"].iloc[0]
    claims = pd.DataFrame(
        [
            {
                "claim_id": "P1_national_severe_curtailment",
                "assessment": "not_supported_as_a_current_national_generalization",
                "model_metric": pain.loc[
                    pain["absorption_case"] == "observed_national_2024",
                    "curtailment_rate",
                ].iloc[0],
                "unit": "fraction",
                "interpretation": "The national reference is calibrated to 4.1% non-utilization, so severe curtailment must be demonstrated at a specific landing point.",
            },
            {
                "claim_id": "P2_local_landing_bottleneck",
                "assessment": "supported_conditionally",
                "model_metric": pain.loc[
                    pain["absorption_case"] == "local_cluster_450mw",
                    "curtailment_rate",
                ].iloc[0],
                "unit": "fraction",
                "interpretation": "A fixed 450 MW landing capability creates material local curtailment using the same resource year.",
            },
            {
                "claim_id": "P3_cable_expansion_alone",
                "assessment": "not_sufficient_when_landing_grid_is_fixed",
                "model_metric": (
                    cable_only.kpis["renewable_utilization_rate"]
                    - cluster_direct.kpis["renewable_utilization_rate"]
                ),
                "unit": "utilization_fraction_change",
                "interpretation": "Increasing the cable from 700 to 1000 MW does not remove an unchanged 450 MW mainland constraint.",
            },
            {
                "claim_id": "P4_high_cost_low_price_conflict",
                "assessment": "supported_in_reference_screening_case",
                "model_metric": cluster_direct.kpis["full_cost_recovery_ratio"],
                "unit": "revenue_to_annual_cash_cost_ratio",
                "interpretation": "The complete direct-export project does not recover the screened deep-offshore common costs at the reference price profile.",
            },
            {
                "claim_id": "V1_compute_value",
                "assessment": "supported_only_above_contract_and_portability_thresholds",
                "model_metric": reference_frontier.kpis["compute_it_capacity_mw"],
                "unit": "MW-IT",
                "interpretation": "Compute is selected at the representative contract point, but lower prices or a smaller portable workload pool can return the optimal capacity to zero.",
            },
            {
                "claim_id": "V2_hydrogen_value",
                "assessment": "supported_conditionally_after_pipeline_cost",
                "model_metric": reference_frontier.kpis["electrolyzer_power_mw"],
                "unit": "MW",
                "interpretation": "Hydrogen can absorb longer-duration surplus, but its entry depends on delivered price and an endogenous distance-dependent export pipeline.",
            },
            {
                "claim_id": "V3_integrated_hub_vs_grid_reinforcement",
                "assessment": "compare_on_value_and_utilization_not_by_presumption",
                "model_metric": (
                    hub_reference.kpis["full_project_net_annual_value_cny"]
                    - reinforced.kpis["full_project_net_annual_value_cny"]
                ),
                "unit": "CNY/year",
                "interpretation": "The preferred option changes with contracts and grid cost; the model reports both rather than declaring the hub universally superior.",
            },
            {
                "claim_id": "V4_pv_wave_addons",
                "assessment": "technical_smoothing_and_economic_value_are_separate_tests",
                "model_metric": (
                    pv_wave_resource["renewable_output_cv"]
                    - wind_only_resource["renewable_output_cv"]
                ),
                "unit": "coefficient_of_variation_change",
                "interpretation": "The synthetic lagged-wave and solar add-ons change output variability, but their economic value must also cover their incremental marine CAPEX.",
            },
        ]
    )
    _write(claims, output / "s7_claim_assessment.csv")

    output_files = sorted(
        [path for path in output.rglob("*") if path.is_file() and path.name != "analysis_manifest.json"]
    )
    manifest = {
        "phase": "S7",
        "model_version": __version__,
        "simulation_hours": hours,
        "timestamp_note": "Deterministic synthetic screening year; not site measurements",
        "national_reference_target_utilization": 0.959,
        "calibrated_nominal_landing_limit_mw": national_landing,
        "representative_distance_km": 200.0,
        "wind_full_load_hours_target": 3_300.0,
        "flexible_compute_fraction_reference": 0.35,
        "source_files": {
            str(path.relative_to(root)): _sha256(path)
            for path in (
                root / "configs/technology_parameters.csv",
                root / "configs/s7_china_evidence_register.csv",
                root / "configs/s7_china_flexible_cost_cases.csv",
                root / "configs/s7_china_infrastructure_cost_cases.csv",
                root / "configs/s7_china_planning_limits.yaml",
            )
        },
        "output_files": {
            str(path.relative_to(output)): _sha256(path) for path in output_files
        },
        "limitations": [
            "The annual profiles are deterministic synthetic screening inputs.",
            "D-grade cost ranges are not supplier quotations or a bankable estimate.",
            "The LP has perfect foresight and does not include unit commitment or seconds-scale stability.",
            "Wave output is a lagged screening proxy and not a measured hindcast.",
            "Merchant, network and reliability comparisons are reported separately to avoid double counting.",
        ],
        "representative_contract_design": {
            "compute_price_cny_per_mwh_it": 2_000.0,
            "hydrogen_price_cny_per_kg": 30.0,
            "capacities": contract_result.planning.capacities,
            "kpis": contract_result.kpis,
        },
        "configuration": {
            "flexible_cost_case": asdict(reference_flexible),
            "infrastructure_cost_case": asdict(reference_common),
            "planning_limits": asdict(limits),
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8_760)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/s7_china_value_analysis")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run(root, args.output, args.hours)


if __name__ == "__main__":
    main()
