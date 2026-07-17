# ruff: noqa: E402
"""Run Phase 9 / S8 robust capacity and value-boundary screening.

The runner is checkpointed by task because each task contains several 8760 h
linear programmes.  ``--task all`` is the one-command workflow; individual
tasks can be resumed without discarding completed CSV files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-blue-hub-s8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_s7_china_value_analysis import (
    _china_reference_timeseries,
    _design,
    _load_limits,
    _rescale_clipped,
    _scenario,
    _with_markets,
    _without_assets,
)

from blue_hub import __version__
from blue_hub.china_counterfactual_model import (
    S7CounterfactualResult,
    common_infrastructure_costs,
    evaluate_s7_design,
    load_china_infrastructure_cost_cases,
)
from blue_hub.investment_planning_model import (
    PlanningLimits,
    load_investment_cost_cases,
    optimal_system_configuration,
)
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.risk_screening_model import (
    S8CapacityCandidate,
    S8ScenarioCase,
    evaluate_capacity_candidates,
    fixed_flexible_asset_costs,
    interpolate_required_capacity,
    summarize_scenario_results,
)
from blue_hub.scarcity_dispatch_model import run_s5_dispatch

CAPACITY_COLUMNS = (
    "battery_power_mw",
    "battery_energy_mwh",
    "electrolyzer_power_mw",
    "hydrogen_storage_kg",
    "fuel_cell_power_mw",
    "compute_it_capacity_mw",
    "subsea_fiber_service_capacity_mw_it",
    "hydrogen_export_capacity_kg_per_h",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def _upsert(frame: pd.DataFrame, path: Path, keys: list[str]) -> None:
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    _write(frame, path)


def _settings(root: Path) -> dict[str, object]:
    with (root / "configs/s8_analysis_settings.yaml").open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("S8 analysis settings must be a mapping")
    return values


def _candidate_definitions(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "configs/s8_candidate_definitions.csv")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("S8 candidate identifiers must be unique")
    return frame.set_index("candidate_id", drop=False)


def _resource_case(
    base: pd.DataFrame,
    *,
    case_id: str,
    full_load_hours: float,
    phase_hours: int,
    landing_limit_mw: float,
    compute_price: float,
    compute_fraction: float,
) -> pd.DataFrame:
    frame = base.copy()
    t = np.arange(len(frame), dtype=float)
    raw = np.roll(frame["wind_cf"].to_numpy(dtype=float), phase_hours)
    raw *= 1.0 + 0.07 * np.sin(2.0 * np.pi * t / (13.0 * 24.0) + phase_hours / 24.0)
    target_gross_cf = (full_load_hours / 8_760.0) / 0.97
    frame["wind_cf"] = _rescale_clipped(raw, target_gross_cf)
    frame["grid_absorption_limit_mw"] = (
        landing_limit_mw * frame["landing_demand_factor"]
    )
    frame["national_compute_price_cny_per_mwh_it"] = compute_price
    frame["national_compute_flexible_fraction"] = compute_fraction
    frame["scenario_id"] = case_id
    return frame


def _inputs(root: Path, hours: int):
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
    common_cases = {
        item.cost_case_id: item
        for item in load_china_infrastructure_cost_cases(
            root / "configs/s7_china_infrastructure_cost_cases.csv"
        )
    }
    limits = _load_limits(root / "configs/s7_china_planning_limits.yaml")
    frame = _china_reference_timeseries(hours)
    return (
        parameters,
        base_scenario,
        base_config,
        flexible_cases,
        common_cases,
        limits,
        frame,
    )


def _planning_candidate(
    candidate_id: str,
    frame: pd.DataFrame,
    *,
    compute_price: float,
    hydrogen_price: float,
    parameters,
    base_scenario,
    base_config,
    flexible_costs,
    common_costs,
    limits: PlanningLimits,
    settings: dict[str, object],
) -> tuple[S8CapacityCandidate, S7CounterfactualResult]:
    market_frame = _with_markets(frame, compute_price=compute_price)
    design = _design(
        f"s8_{candidate_id}",
        candidate_id,
        tx=700.0,
        landing=float(settings["reference_landing_capability_mw"]),
        flexible=True,
        hub=True,
    )
    result = evaluate_s7_design(
        market_frame,
        parameters,
        base_config,
        _scenario(
            base_scenario,
            f"s8_plan_{candidate_id}",
            hydrogen_price=hydrogen_price,
            distance_km=float(settings["offshore_distance_km"]),
        ),
        flexible_costs,
        common_costs,
        limits,
        design,
    )
    config = optimal_system_configuration(
        base_config,
        result.planning,
        config_id=f"s8_fixed_{candidate_id}",
    )
    return S8CapacityCandidate(candidate_id, config, design), result


def _reliability_candidate(
    frame: pd.DataFrame,
    *,
    settings: dict[str, object],
    parameters,
    base_scenario,
    base_config,
    flexible_costs,
    common_costs,
    limits,
) -> tuple[S8CapacityCandidate, S7CounterfactualResult]:
    stress = frame.copy()
    stress["national_compute_demand_mw_it"] = 0.0
    stress["hydrogen_demand"] = 0.0
    stress["critical_load"] = float(settings["reliability_critical_load_mw"])
    start = int(settings["reliability_design_event_start_hour"])
    stop = start + 24 * int(settings["reliability_design_duration_days"])
    stress.loc[start : stop - 1, "wind_availability"] = float(
        settings["reliability_wind_availability"]
    )
    stress["grid_absorption_limit_mw"] = (
        float(settings["reference_landing_capability_mw"])
        * stress["landing_demand_factor"]
    )
    design = _design(
        "s8_reliability_reserve",
        "reliability_reserve",
        tx=700.0,
        landing=float(settings["reference_landing_capability_mw"]),
        flexible=True,
        hub=True,
    )
    result = evaluate_s7_design(
        stress,
        parameters,
        base_config,
        _scenario(
            base_scenario,
            "s8_plan_reliability_reserve",
            hydrogen_price=20.0,
            distance_km=float(settings["offshore_distance_km"]),
        ),
        flexible_costs,
        common_costs,
        limits,
        design,
    )
    config = optimal_system_configuration(
        base_config,
        result.planning,
        config_id="s8_fixed_reliability_reserve",
    )
    return S8CapacityCandidate("reliability_reserve", config, design), result


def _build_candidates(root: Path, hours: int):
    (
        parameters,
        base_scenario,
        base_config,
        flexible_cases,
        common_cases,
        limits,
        frame,
    ) = _inputs(root, hours)
    flexible = flexible_cases["china_reference"]
    common = common_cases["china_reference"]
    direct_config = _without_assets(
        base_config,
        config_id="s8_fixed_direct",
        tx_capacity_mw=700.0,
    )
    direct = S8CapacityCandidate(
        "direct",
        direct_config,
        _design(
            "s8_direct",
            "direct_existing",
            tx=700.0,
            landing=float(_settings(root)["reference_landing_capability_mw"]),
            flexible=False,
            hub=False,
        ),
    )
    compute, compute_result = _planning_candidate(
        "compute_led",
        frame,
        compute_price=2_000.0,
        hydrogen_price=20.0,
        parameters=parameters,
        base_scenario=base_scenario,
        base_config=base_config,
        flexible_costs=flexible,
        common_costs=common,
        limits=limits,
        settings=_settings(root),
    )
    hydrogen, hydrogen_result = _planning_candidate(
        "hydrogen_led",
        frame,
        compute_price=1_000.0,
        hydrogen_price=30.0,
        parameters=parameters,
        base_scenario=base_scenario,
        base_config=base_config,
        flexible_costs=flexible,
        common_costs=common,
        limits=limits,
        settings=_settings(root),
    )
    integrated, integrated_result = _planning_candidate(
        "integrated_contract",
        frame,
        compute_price=2_000.0,
        hydrogen_price=30.0,
        parameters=parameters,
        base_scenario=base_scenario,
        base_config=base_config,
        flexible_costs=flexible,
        common_costs=common,
        limits=limits,
        settings=_settings(root),
    )
    diversified_values = {
        name: max(
            float(getattr(compute.configuration, name)),
            float(getattr(hydrogen.configuration, name)),
        )
        for name in CAPACITY_COLUMNS
    }
    diversified_values["config_id"] = "s8_fixed_diversified_hedge"
    diversified_values["initial_battery_soc_fraction"] = (
        base_config.initial_battery_soc_fraction
        if diversified_values["battery_energy_mwh"] > 0.0
        else 0.0
    )
    diversified_values["initial_hydrogen_inventory_fraction"] = (
        base_config.initial_hydrogen_inventory_fraction
        if diversified_values["hydrogen_storage_kg"] > 0.0
        else 0.0
    )
    diversified_config = base_config.model_copy(update=diversified_values)
    diversified = S8CapacityCandidate(
        "diversified_hedge",
        diversified_config,
        _design(
            "s8_diversified_hedge",
            "diversified_hedge",
            tx=700.0,
            landing=float(_settings(root)["reference_landing_capability_mw"]),
            flexible=True,
            hub=True,
        ),
    )
    reliability, reliability_result = _reliability_candidate(
        frame,
        settings=_settings(root),
        parameters=parameters,
        base_scenario=base_scenario,
        base_config=base_config,
        flexible_costs=flexible,
        common_costs=common,
        limits=limits,
    )
    planning_results = {
        "compute_led": compute_result,
        "hydrogen_led": hydrogen_result,
        "integrated_contract": integrated_result,
        "reliability_reserve": reliability_result,
    }
    merchant_candidates = (direct, compute, hydrogen, integrated, diversified)
    return merchant_candidates, reliability, planning_results, _inputs(root, hours)


def _candidate_table(
    merchant: tuple[S8CapacityCandidate, ...],
    reliability: S8CapacityCandidate,
) -> pd.DataFrame:
    rows = []
    for candidate in (*merchant, reliability):
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_scope": (
                    "stress_only"
                    if candidate.candidate_id == "reliability_reserve"
                    else "merchant_risk_set"
                ),
                "strategy": candidate.design.strategy,
                **{
                    name: float(getattr(candidate.configuration, name))
                    for name in CAPACITY_COLUMNS
                },
            }
        )
    return pd.DataFrame(rows)


def _risk_cases(
    base: pd.DataFrame, base_scenario, root: Path
) -> tuple[S8ScenarioCase, ...]:
    specs = pd.read_csv(root / "configs/s8_risk_cases.csv")
    if not np.isclose(specs["scenario_weight"].sum(), 1.0, atol=1e-10):
        raise ValueError("S8 risk-case weights must sum to one")
    cases = []
    for row in specs.itertuples(index=False):
        frame = _resource_case(
            base,
            case_id=row.case_id,
            full_load_hours=row.wind_full_load_hours,
            phase_hours=int(row.resource_phase_hours),
            landing_limit_mw=row.available_landing_limit_mw,
            compute_price=row.compute_price_cny_per_mwh_it,
            compute_fraction=row.compute_flexible_fraction,
        )
        scenario = _scenario(
            base_scenario,
            f"s8_risk_{row.case_id}",
            hydrogen_price=row.hydrogen_price_cny_per_kg,
            distance_km=float(_settings(root)["offshore_distance_km"]),
        ).model_copy(
            update={"electricity_price_multiplier": row.electricity_price_multiplier}
        )
        cases.append(S8ScenarioCase(row.case_id, frame, scenario, row.scenario_weight))
    return tuple(cases)


def _plot_risk(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = summary["lower_tail_full_net_value_cny_per_year"] / 1e9
    y = summary["expected_full_net_value_cny_per_year"] / 1e9
    sizes = 70.0 + 180.0 * summary["expected_renewable_utilization_rate"]
    ax.scatter(x, y, s=sizes, alpha=0.78)
    label_offsets = {
        "integrated_contract": (7, 18),
        "compute_led": (7, 4),
        "diversified_hedge": (7, -14),
        "direct": (7, 11),
        "hydrogen_led": (7, -14),
    }
    for _, row in summary.iterrows():
        ax.annotate(
            row["candidate_id"],
            (
                row["lower_tail_full_net_value_cny_per_year"] / 1e9,
                row["expected_full_net_value_cny_per_year"] / 1e9,
            ),
            xytext=label_offsets.get(str(row["candidate_id"]), (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Worst-20% mean full net value (CNY bn/yr)")
    ax.set_ylabel("Expected full net value (CNY bn/yr)")
    ax.set_title("S8 fixed-capacity risk comparison")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_risk_aversion(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for candidate, group in frame.groupby("candidate_id"):
        group = group.sort_values("risk_aversion")
        ax.plot(
            group["risk_aversion"],
            group["risk_adjusted_score_cny_per_year"] / 1e9,
            marker="o",
            markersize=3,
            label=candidate,
        )
    ax.set_xlabel("Risk aversion weight")
    ax.set_ylabel("Risk-adjusted full net value (CNY bn/yr)")
    ax.set_title("Capacity choice sensitivity to lower-tail risk")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_candidate(
    output: Path,
    candidate: S8CapacityCandidate,
    *,
    scope: str,
    planning: S7CounterfactualResult | dict[str, object] | None = None,
) -> None:
    row = pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_scope": scope,
                "strategy": candidate.design.strategy,
                **{
                    name: float(getattr(candidate.configuration, name))
                    for name in CAPACITY_COLUMNS
                },
            }
        ]
    )
    _upsert(row, output / "s8_candidate_capacities.csv", ["candidate_id"])
    if planning is not None:
        planning_values = (
            planning.kpis
            if isinstance(planning, S7CounterfactualResult)
            else planning
        )
        _upsert(
            pd.DataFrame([{"candidate_id": candidate.candidate_id, **planning_values}]),
            output / "s8_candidate_design_year_results.csv",
            ["candidate_id"],
        )


def _candidate_from_s7_outputs(
    root: Path,
    candidate_id: str,
    base_config,
) -> tuple[S8CapacityCandidate, dict[str, object]] | None:
    s7 = root / "outputs/s7_china_value_analysis"
    settings = _settings(root)
    contract_path = s7 / "s7_contract_value_frontier.csv"
    reliability_path = s7 / "s7_fixed_and_reliability_stress_tests.csv"
    definitions = _candidate_definitions(root)
    contract_candidates = definitions.loc[
        definitions["design_source"] == "audited_S7_contract_frontier"
    ]
    if candidate_id in contract_candidates.index and contract_path.exists():
        definition = contract_candidates.loc[candidate_id]
        compute_price = float(definition["compute_price_cny_per_mwh_it"])
        hydrogen_price = float(definition["hydrogen_price_cny_per_kg"])
        frame = pd.read_csv(contract_path)
        selected = frame.loc[
            np.isclose(frame["compute_price_cny_per_mwh_it"], compute_price)
            & np.isclose(frame["hydrogen_price_cny_per_kg"], hydrogen_price)
        ]
        if len(selected) != 1:
            return None
        row = selected.iloc[0]
        values = {name: float(row[name]) for name in CAPACITY_COLUMNS if name in row.index}
        values["subsea_fiber_service_capacity_mw_it"] = values[
            "compute_it_capacity_mw"
        ]
        values.update(
            {
                "config_id": f"s8_fixed_{candidate_id}",
                "initial_battery_soc_fraction": (
                    base_config.initial_battery_soc_fraction
                    if values["battery_energy_mwh"] > 0.0
                    else 0.0
                ),
                "initial_hydrogen_inventory_fraction": (
                    base_config.initial_hydrogen_inventory_fraction
                    if values["hydrogen_storage_kg"] > 0.0
                    else 0.0
                ),
            }
        )
        config = base_config.model_copy(update=values)
        candidate = S8CapacityCandidate(
            candidate_id,
            config,
            _design(
                f"s8_{candidate_id}",
                candidate_id,
                tx=700.0,
                landing=float(settings["reference_landing_capability_mw"]),
                flexible=True,
                hub=True,
            ),
        )
        planning_record = row.to_dict()
        planning_record["design_source"] = "audited_S7_contract_frontier"
        return candidate, planning_record
    if candidate_id == "reliability_reserve" and reliability_path.exists():
        frame = pd.read_csv(reliability_path)
        selected = frame.loc[
            (frame["stress_case"] == "endogenous_30day_low_wind")
            & (frame["strategy"] == "integrated_hub")
        ]
        if len(selected) != 1:
            return None
        row = selected.iloc[0]
        values = {
            "battery_power_mw": float(row["battery_power_mw"]),
            "battery_energy_mwh": float(row["battery_energy_mwh"]),
            "electrolyzer_power_mw": float(row["electrolyzer_power_mw"]),
            "hydrogen_storage_kg": float(row["hydrogen_storage_kg"]),
            "fuel_cell_power_mw": float(row["fuel_cell_power_mw"]),
            "compute_it_capacity_mw": 0.0,
            "subsea_fiber_service_capacity_mw_it": 0.0,
            "hydrogen_export_capacity_kg_per_h": 0.0,
            "config_id": "s8_fixed_reliability_reserve",
            "initial_battery_soc_fraction": (
                base_config.initial_battery_soc_fraction
                if float(row["battery_energy_mwh"]) > 0.0
                else 0.0
            ),
            "initial_hydrogen_inventory_fraction": (
                base_config.initial_hydrogen_inventory_fraction
                if float(row["hydrogen_storage_kg"]) > 0.0
                else 0.0
            ),
        }
        config = base_config.model_copy(update=values)
        candidate = S8CapacityCandidate(
            candidate_id,
            config,
            _design(
                "s8_reliability_reserve",
                "reliability_reserve",
                tx=700.0,
                landing=float(settings["reference_landing_capability_mw"]),
                flexible=True,
                hub=True,
            ),
        )
        planning_record = row.to_dict()
        planning_record["design_source"] = "audited_S7_30day_reliability_case"
        return candidate, planning_record
    return None


def run_candidates(
    root: Path,
    output: Path,
    hours: int,
    candidate_ids: tuple[str, ...] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    definitions = _candidate_definitions(root)
    requested = candidate_ids or tuple(str(value) for value in definitions.index)
    if not set(requested) <= set(definitions.index):
        raise ValueError("candidate-id contains an unknown S8 candidate")
    (
        parameters,
        base_scenario,
        base_config,
        flexible_cases,
        common_cases,
        limits,
        frame,
    ) = _inputs(root, hours)
    flexible = flexible_cases["china_reference"]
    common = common_cases["china_reference"]
    settings = _settings(root)
    for candidate_id in requested:
        if candidate_id == "direct":
            config = _without_assets(
                base_config,
                config_id="s8_fixed_direct",
                tx_capacity_mw=700.0,
            )
            candidate = S8CapacityCandidate(
                "direct",
                config,
                _design(
                    "s8_direct",
                    "direct_existing",
                    tx=700.0,
                    landing=float(settings["reference_landing_capability_mw"]),
                    flexible=False,
                    hub=False,
                ),
            )
            _save_candidate(output, candidate, scope="merchant_risk_set")
            continue
        reused = _candidate_from_s7_outputs(root, candidate_id, base_config)
        if reused is not None:
            candidate, planning_record = reused
            _save_candidate(
                output,
                candidate,
                scope=(
                    "stress_only"
                    if candidate_id == "reliability_reserve"
                    else "merchant_risk_set"
                ),
                planning=planning_record,
            )
            continue
        if candidate_id in {"compute_led", "hydrogen_led", "integrated_contract"}:
            definition = definitions.loc[candidate_id]
            compute_price = float(definition["compute_price_cny_per_mwh_it"])
            hydrogen_price = float(definition["hydrogen_price_cny_per_kg"])
            candidate, planning = _planning_candidate(
                candidate_id,
                frame,
                compute_price=compute_price,
                hydrogen_price=hydrogen_price,
                parameters=parameters,
                base_scenario=base_scenario,
                base_config=base_config,
                flexible_costs=flexible,
                common_costs=common,
                limits=limits,
                settings=settings,
            )
            _save_candidate(
                output,
                candidate,
                scope="merchant_risk_set",
                planning=planning,
            )
            continue
        if candidate_id == "diversified_hedge":
            capacity_path = output / "s8_candidate_capacities.csv"
            if not capacity_path.exists():
                raise RuntimeError("compute_led and hydrogen_led must be saved first")
            table = pd.read_csv(capacity_path).set_index("candidate_id")
            required = {"compute_led", "hydrogen_led"}
            if not required.issubset(table.index):
                raise RuntimeError("compute_led and hydrogen_led must be saved first")
            values = {
                name: max(
                    float(table.loc["compute_led", name]),
                    float(table.loc["hydrogen_led", name]),
                )
                for name in CAPACITY_COLUMNS
            }
            values.update(
                {
                    "config_id": "s8_fixed_diversified_hedge",
                    "initial_battery_soc_fraction": (
                        base_config.initial_battery_soc_fraction
                        if values["battery_energy_mwh"] > 0.0
                        else 0.0
                    ),
                    "initial_hydrogen_inventory_fraction": (
                        base_config.initial_hydrogen_inventory_fraction
                        if values["hydrogen_storage_kg"] > 0.0
                        else 0.0
                    ),
                }
            )
            config = base_config.model_copy(update=values)
            candidate = S8CapacityCandidate(
                "diversified_hedge",
                config,
                _design(
                    "s8_diversified_hedge",
                    "diversified_hedge",
                    tx=700.0,
                    landing=float(settings["reference_landing_capability_mw"]),
                    flexible=True,
                    hub=True,
                ),
            )
            _save_candidate(output, candidate, scope="merchant_risk_set")
            continue
        candidate, planning = _reliability_candidate(
            frame,
            settings=_settings(root),
            parameters=parameters,
            base_scenario=base_scenario,
            base_config=base_config,
            flexible_costs=flexible,
            common_costs=common,
            limits=limits,
        )
        _save_candidate(
            output,
            candidate,
            scope="stress_only",
            planning=planning,
        )


def finalize_risk(root: Path, output: Path) -> None:
    settings = _settings(root)
    ledger_path = output / "s8_candidate_scenario_ledger.csv"
    ledger = pd.read_csv(ledger_path)
    capacity_table = pd.read_csv(output / "s8_candidate_capacities.csv")
    strategy_by_candidate = capacity_table.set_index("candidate_id")["strategy"]
    ledger["strategy"] = ledger["candidate_id"].map(strategy_by_candidate)
    if ledger["strategy"].isna().any():
        raise ValueError("every S8 risk candidate must have a saved strategy label")
    result = summarize_scenario_results(
        ledger,
        tail_probability=float(settings["tail_probability"]),
        risk_aversion=float(settings["risk_aversion"]),
    )
    _write(result.candidate_summary, output / "s8_candidate_risk_summary.csv")
    _write(result.scenario_results, ledger_path)
    sensitivity_rows = []
    for risk_aversion in np.linspace(0.0, 1.0, 11):
        for _, row in result.candidate_summary.iterrows():
            expected = float(row["expected_full_net_value_cny_per_year"])
            lower_tail = float(row["lower_tail_full_net_value_cny_per_year"])
            sensitivity_rows.append(
                {
                    "risk_aversion": risk_aversion,
                    "candidate_id": row["candidate_id"],
                    "risk_adjusted_score_cny_per_year": (
                        (1.0 - risk_aversion) * expected + risk_aversion * lower_tail
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["rank_at_risk_aversion"] = sensitivity.groupby("risk_aversion")[
        "risk_adjusted_score_cny_per_year"
    ].rank(method="min", ascending=False)
    _write(sensitivity, output / "s8_risk_aversion_frontier.csv")
    (output / "s8_risk_metadata.json").write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _plot_risk(result.candidate_summary, figures / "s8_risk_return.png")
    _plot_risk_aversion(
        sensitivity,
        figures / "s8_risk_aversion_frontier.png",
    )


def run_risk(
    root: Path,
    output: Path,
    hours: int,
    candidate_ids: tuple[str, ...] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    capacity_path = output / "s8_candidate_capacities.csv"
    if not capacity_path.exists():
        run_candidates(root, output, hours)
    capacity_table = pd.read_csv(capacity_path)
    merchant_table = capacity_table.loc[
        capacity_table["candidate_scope"] == "merchant_risk_set"
    ]
    available = tuple(str(value) for value in merchant_table["candidate_id"])
    requested = candidate_ids or available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown merchant-risk candidate identifiers: {unknown}")
    parameters, base_scenario, base_config, flexible_cases, common_cases, _, frame = _inputs(
        root, hours
    )
    settings = _settings(root)
    cases = _risk_cases(frame, base_scenario, root)
    for candidate_id in requested:
        row = merchant_table.loc[merchant_table["candidate_id"] == candidate_id].iloc[0]
        candidate = _candidate_from_row(
            row,
            base_config,
            direct=candidate_id == "direct",
            landing_grid_limit_mw=float(
                settings["reference_landing_capability_mw"]
            ),
        )
        partial = evaluate_capacity_candidates(
            (candidate,),
            cases,
            parameters,
            flexible_cases["china_reference"],
            common_cases["china_reference"],
            tail_probability=float(settings["tail_probability"]),
            risk_aversion=float(settings["risk_aversion"]),
        )
        _upsert(
            partial.scenario_results.drop(columns="scenario_regret_cny"),
            output / "s8_candidate_scenario_ledger.csv",
            ["candidate_id", "case_id"],
        )
    ledger = pd.read_csv(output / "s8_candidate_scenario_ledger.csv")
    completed = set(ledger["candidate_id"])
    if set(available).issubset(completed):
        finalize_risk(root, output)


def _landing_record(result, strategy: str, landing: float) -> dict[str, object]:
    return {
        "landing_grid_limit_mw": landing,
        "strategy": strategy,
        "renewable_utilization_rate": result.kpis["renewable_utilization_rate"],
        "curtailment_mwh": result.kpis["curtailment_mwh"],
        "full_project_capex_cny": result.kpis["full_project_capex_cny"],
        "full_project_net_annual_value_cny": result.kpis[
            "full_project_net_annual_value_cny"
        ],
        "compute_it_capacity_mw": result.kpis["compute_it_capacity_mw"],
        "electrolyzer_power_mw": result.kpis["electrolyzer_power_mw"],
        "hydrogen_storage_kg": result.kpis["hydrogen_storage_kg"],
        "hydrogen_export_capacity_kg_per_h": result.kpis[
            "hydrogen_export_capacity_kg_per_h"
        ],
    }


def _plot_landing(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for strategy, group in frame.groupby("strategy"):
        group = group.sort_values("landing_grid_limit_mw")
        axes[0].plot(
            group["landing_grid_limit_mw"],
            100.0 * group["renewable_utilization_rate"],
            marker="o",
            label=strategy,
        )
        axes[1].plot(
            group["landing_grid_limit_mw"],
            group["full_project_net_annual_value_cny"] / 1e9,
            marker="o",
            label=strategy,
        )
    axes[0].set_ylabel("Renewable utilization (%)")
    axes[1].set_ylabel("Full net annual value (CNY bn/yr)")
    for axis in axes:
        axis.set_xlabel("Installed landing capability (MW)")
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Landing reinforcement and flexible-hub substitution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_landing(root: Path, output: Path, hours: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    capacity_path = output / "s8_candidate_capacities.csv"
    if not capacity_path.exists():
        run_candidates(
            root,
            output,
            hours,
            ("direct", "integrated_contract"),
        )
    capacity_table = pd.read_csv(capacity_path)
    parameters, base_scenario, base_config, flexible_cases, common_cases, _, frame = _inputs(
        root, hours
    )
    flexible = flexible_cases["china_reference"]
    common = common_cases["china_reference"]
    settings = _settings(root)
    market = _with_markets(frame, compute_price=2_000.0)
    scenario = _scenario(
        base_scenario,
        "s8_landing_contract",
        hydrogen_price=30.0,
        distance_km=float(settings["offshore_distance_km"]),
    )
    landing_values = tuple(float(value) for value in settings["landing_capabilities_mw"])
    rows = []
    for landing in landing_values:
        landing_frame = market.copy()
        landing_frame["grid_absorption_limit_mw"] = (
            landing * landing_frame["landing_demand_factor"]
        )
        candidates = []
        for candidate_id in ("direct", "integrated_contract"):
            source = capacity_table.loc[
                capacity_table["candidate_id"] == candidate_id
            ].iloc[0]
            loaded = _candidate_from_row(
                source,
                base_config,
                direct=candidate_id == "direct",
                landing_grid_limit_mw=landing,
            )
            candidates.append(
                S8CapacityCandidate(
                    candidate_id,
                    loaded.configuration,
                    _design(
                        f"s8_{candidate_id}_landing_{landing:g}",
                        "direct" if candidate_id == "direct" else "fixed_integrated_hub",
                        tx=700.0,
                        landing=landing,
                        flexible=candidate_id != "direct",
                        hub=candidate_id != "direct",
                    ),
                )
            )
        evaluation = evaluate_capacity_candidates(
            tuple(candidates),
            (
                S8ScenarioCase(
                    f"landing_{landing:g}",
                    landing_frame,
                    scenario,
                    1.0,
                ),
            ),
            parameters,
            flexible,
            common,
            risk_aversion=0.0,
        )
        ledger = evaluation.scenario_results.set_index("candidate_id")
        direct_row = ledger.loc["direct"]
        hub_row = ledger.loc["integrated_contract"]
        for candidate_id, source in (
            ("direct", direct_row),
            ("fixed_integrated_hub", hub_row),
        ):
            rows.append(
                {
                    "landing_grid_limit_mw": landing,
                    "strategy": candidate_id,
                    "renewable_utilization_rate": source[
                        "renewable_utilization_rate"
                    ],
                    "curtailment_mwh": source["curtailment_mwh"],
                    "full_project_capex_cny": source["full_project_capex_cny"],
                    "full_project_net_annual_value_cny": source[
                        "full_project_net_annual_value_cny"
                    ],
                    "compute_it_capacity_mw": source["compute_it_capacity_mw"],
                    "electrolyzer_power_mw": source["electrolyzer_power_mw"],
                    "hydrogen_storage_kg": source["hydrogen_storage_kg"],
                    "hydrogen_export_capacity_kg_per_h": source[
                        "hydrogen_export_capacity_kg_per_h"
                    ],
                    "incremental_net_vs_direct_same_landing_cny_per_year": (
                        source["full_project_net_annual_value_cny"]
                        - direct_row["full_project_net_annual_value_cny"]
                    ),
                    "avoided_curtailment_vs_direct_same_landing_mwh": (
                        direct_row["curtailment_mwh"] - source["curtailment_mwh"]
                    ),
                }
            )
    result = pd.DataFrame(rows)
    direct_curve = result[result["strategy"] == "direct"].sort_values(
        "landing_grid_limit_mw"
    )
    equivalent_rows = []
    for _, row in result[result["strategy"] == "fixed_integrated_hub"].iterrows():
        target = float(row["renewable_utilization_rate"])
        direct_maximum = float(direct_curve["renewable_utilization_rate"].max())
        equivalent = (
            interpolate_required_capacity(
                direct_curve["landing_grid_limit_mw"],
                direct_curve["renewable_utilization_rate"],
                target=target,
            )
            if target <= direct_maximum + 1e-10
            else np.nan
        )
        equivalent_rows.append(
            {
                "hub_landing_grid_limit_mw": row["landing_grid_limit_mw"],
                "hub_utilization_rate": target,
                "equivalent_direct_landing_grid_limit_mw": equivalent,
                "landing_capacity_substitution_mw": (
                    equivalent - float(row["landing_grid_limit_mw"])
                    if np.isfinite(equivalent)
                    else np.nan
                ),
                "attainable_by_landing_expansion_with_700mw_cable": bool(
                    np.isfinite(equivalent)
                ),
            }
        )
    _write(result, output / "s8_landing_flexibility_frontier.csv")
    _write(pd.DataFrame(equivalent_rows), output / "s8_landing_capacity_substitution.csv")
    _plot_landing(result, figures / "s8_landing_flexibility_frontier.png")


def _contract_record(
    result: S7CounterfactualResult,
    direct: S7CounterfactualResult,
    compute_price: float,
    hydrogen_price: float,
) -> dict[str, object]:
    return {
        "compute_price_cny_per_mwh_it": compute_price,
        "hydrogen_price_cny_per_kg": hydrogen_price,
        "battery_power_mw": result.kpis["battery_power_mw"],
        "battery_energy_mwh": result.kpis["battery_energy_mwh"],
        "electrolyzer_power_mw": result.kpis["electrolyzer_power_mw"],
        "hydrogen_storage_kg": result.kpis["hydrogen_storage_kg"],
        "fuel_cell_power_mw": result.kpis["fuel_cell_power_mw"],
        "compute_it_capacity_mw": result.kpis["compute_it_capacity_mw"],
        "hydrogen_export_capacity_kg_per_h": result.kpis[
            "hydrogen_export_capacity_kg_per_h"
        ],
        "renewable_utilization_rate": result.kpis["renewable_utilization_rate"],
        "curtailment_mwh": result.kpis["curtailment_mwh"],
        "full_project_capex_cny": result.kpis["full_project_capex_cny"],
        "full_project_net_annual_value_cny": result.kpis[
            "full_project_net_annual_value_cny"
        ],
        "incremental_net_value_vs_direct_cny_per_year": (
            result.kpis["full_project_net_annual_value_cny"]
            - direct.kpis["full_project_net_annual_value_cny"]
        ),
    }


def run_contract(
    root: Path,
    output: Path,
    hours: int,
    hydrogen_prices: tuple[float, ...],
    candidate_ids: tuple[str, ...] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    capacity_path = output / "s8_candidate_capacities.csv"
    if not capacity_path.exists():
        run_candidates(root, output, hours)
    capacity_table = pd.read_csv(capacity_path)
    merchant_table = capacity_table.loc[
        capacity_table["candidate_scope"] == "merchant_risk_set"
    ]
    available = tuple(str(value) for value in merchant_table["candidate_id"])
    requested = candidate_ids or available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown contract-surface candidate identifiers: {unknown}")
    parameters, base_scenario, base_config, flexible_cases, common_cases, _, frame = _inputs(
        root, hours
    )
    flexible = flexible_cases["china_reference"]
    common = common_cases["china_reference"]
    settings = _settings(root)
    compute_prices = tuple(
        float(value) for value in settings["contract_compute_prices_cny_per_mwh_it"]
    )
    reference_landing = float(settings["reference_landing_capability_mw"])
    distance = float(settings["offshore_distance_km"])
    for candidate_id in requested:
        source = merchant_table.loc[
            merchant_table["candidate_id"] == candidate_id
        ].iloc[0]
        candidate = _candidate_from_row(
            source,
            base_config,
            direct=candidate_id == "direct",
            landing_grid_limit_mw=reference_landing,
        )
        rows = []
        if candidate_id == "direct":
            representative_frame = frame.copy()
            representative_frame["grid_absorption_limit_mw"] = (
                reference_landing * representative_frame["landing_demand_factor"]
            )
            representative = evaluate_capacity_candidates(
                (candidate,),
                (
                    S8ScenarioCase(
                        "contract_direct",
                        representative_frame,
                        _scenario(
                            base_scenario,
                            "s8_contract_direct",
                            hydrogen_price=20.0,
                            distance_km=distance,
                        ),
                        1.0,
                    ),
                ),
                parameters,
                flexible,
                common,
                risk_aversion=0.0,
            ).scenario_results.iloc[0]
            for hydrogen_price in hydrogen_prices:
                for compute_price in compute_prices:
                    rows.append(
                        {
                            **representative.drop(labels="scenario_regret_cny").to_dict(),
                            "case_id": (
                                f"contract_c{compute_price:g}_h{hydrogen_price:g}"
                            ),
                            "compute_price_cny_per_mwh_it": compute_price,
                            "hydrogen_price_cny_per_kg": hydrogen_price,
                        }
                    )
        else:
            for hydrogen_price in hydrogen_prices:
                for compute_price in compute_prices:
                    market = _with_markets(frame, compute_price=compute_price)
                    market["grid_absorption_limit_mw"] = (
                        reference_landing * market["landing_demand_factor"]
                    )
                    case_id = f"contract_c{compute_price:g}_h{hydrogen_price:g}"
                    evaluation = evaluate_capacity_candidates(
                        (candidate,),
                        (
                            S8ScenarioCase(
                                case_id,
                                market,
                                _scenario(
                                    base_scenario,
                                    f"s8_{candidate_id}_{case_id}",
                                    hydrogen_price=hydrogen_price,
                                    distance_km=distance,
                                ),
                                1.0,
                            ),
                        ),
                        parameters,
                        flexible,
                        common,
                        risk_aversion=0.0,
                    )
                    row = evaluation.scenario_results.iloc[0].drop(
                        labels="scenario_regret_cny"
                    )
                    rows.append(
                        {
                            **row.to_dict(),
                            "compute_price_cny_per_mwh_it": compute_price,
                            "hydrogen_price_cny_per_kg": hydrogen_price,
                        }
                    )
        _upsert(
            pd.DataFrame(rows),
            output / "s8_contract_candidate_ledger.csv",
            [
                "candidate_id",
                "hydrogen_price_cny_per_kg",
                "compute_price_cny_per_mwh_it",
            ],
        )


def _linear_zero_threshold(x: np.ndarray, y: np.ndarray) -> float | None:
    order = np.argsort(x)
    x, y = x[order], y[order]
    nonnegative = np.flatnonzero(y >= 0.0)
    if not len(nonnegative):
        return None
    index = int(nonnegative[0])
    if index == 0:
        return float(x[0])
    if y[index] == y[index - 1]:
        return float(x[index])
    return float(
        x[index - 1]
        + (0.0 - y[index - 1])
        * (x[index] - x[index - 1])
        / (y[index] - y[index - 1])
    )


def _plot_contract(frame: pd.DataFrame, path: Path) -> None:
    values = frame.pivot(
        index="hydrogen_price_cny_per_kg",
        columns="compute_price_cny_per_mwh_it",
        values="incremental_net_value_vs_direct_cny_per_year",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    image = ax.imshow(values.to_numpy() / 1e6, origin="lower", aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(values.columns)), [f"{value:g}" for value in values.columns])
    ax.set_yticks(range(len(values.index)), [f"{value:g}" for value in values.index])
    ax.set_xlabel("Compute service value (CNY/MWh-IT)")
    ax.set_ylabel("Delivered hydrogen price (CNY/kg)")
    ax.set_title("Integrated-hub incremental net value (CNY million/yr)")
    for row in range(len(values.index)):
        for column in range(len(values.columns)):
            ax.text(
                column,
                row,
                f"{values.iloc[row, column] / 1e6:.0f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax, label="CNY million/yr")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def finalize_contract(output: Path) -> None:
    ledger = pd.read_csv(output / "s8_contract_candidate_ledger.csv")
    pair_keys = ["hydrogen_price_cny_per_kg", "compute_price_cny_per_mwh_it"]
    expected_candidates = set(
        pd.read_csv(output / "s8_candidate_capacities.csv")
        .loc[lambda data: data["candidate_scope"] == "merchant_risk_set", "candidate_id"]
        .astype(str)
    )
    counts = ledger.groupby(pair_keys)["candidate_id"].nunique()
    if (counts < len(expected_candidates)).any():
        raise RuntimeError("contract candidate ledger is incomplete")
    direct = (
        ledger.loc[ledger["candidate_id"] == "direct", pair_keys + [
            "full_project_net_annual_value_cny"
        ]]
        .rename(
            columns={
                "full_project_net_annual_value_cny": (
                    "direct_full_project_net_annual_value_cny"
                )
            }
        )
        .set_index(pair_keys)
    )
    flexible = ledger.loc[ledger["candidate_id"] != "direct"].copy()
    best_indices = flexible.groupby(pair_keys)[
        "full_project_net_annual_value_cny"
    ].idxmax()
    frame = flexible.loc[best_indices].copy().set_index(pair_keys)
    frame = frame.join(direct).reset_index()
    frame = frame.rename(
        columns={
            "candidate_id": "best_flexible_candidate_id",
            "full_project_net_annual_value_cny": (
                "best_flexible_full_project_net_annual_value_cny"
            ),
        }
    )
    frame["incremental_net_value_vs_direct_cny_per_year"] = (
        frame["best_flexible_full_project_net_annual_value_cny"]
        - frame["direct_full_project_net_annual_value_cny"]
    )
    frame["selected_overall_candidate_id"] = np.where(
        frame["incremental_net_value_vs_direct_cny_per_year"] >= 0.0,
        frame["best_flexible_candidate_id"],
        "direct",
    )
    _write(frame, output / "s8_contract_value_surface.csv")
    threshold_rows = []
    for hydrogen_price, group in frame.groupby("hydrogen_price_cny_per_kg"):
        compute = group["compute_price_cny_per_mwh_it"].to_numpy(dtype=float)
        incremental = group[
            "incremental_net_value_vs_direct_cny_per_year"
        ].to_numpy(dtype=float)
        full = group[
            "best_flexible_full_project_net_annual_value_cny"
        ].to_numpy(dtype=float)
        threshold_rows.append(
            {
                "hydrogen_price_cny_per_kg": hydrogen_price,
                "incremental_break_even_compute_price_cny_per_mwh_it": (
                    _linear_zero_threshold(compute, incremental)
                ),
                "full_project_break_even_compute_price_cny_per_mwh_it": (
                    _linear_zero_threshold(compute, full)
                ),
            }
        )
    _write(pd.DataFrame(threshold_rows), output / "s8_contract_thresholds.csv")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _plot_contract(frame, figures / "s8_contract_value_surface.png")


def _candidate_from_row(
    row: pd.Series,
    base_config,
    *,
    direct: bool,
    landing_grid_limit_mw: float,
) -> S8CapacityCandidate:
    values = {name: float(row[name]) for name in CAPACITY_COLUMNS}
    values["config_id"] = f"s8_replay_{row['candidate_id']}"
    values["initial_battery_soc_fraction"] = (
        base_config.initial_battery_soc_fraction if values["battery_energy_mwh"] > 0.0 else 0.0
    )
    values["initial_hydrogen_inventory_fraction"] = (
        base_config.initial_hydrogen_inventory_fraction
        if values["hydrogen_storage_kg"] > 0.0
        else 0.0
    )
    config = base_config.model_copy(update=values)
    candidate_id = str(row["candidate_id"])
    return S8CapacityCandidate(
        candidate_id,
        config,
        _design(
            f"s8_replay_{candidate_id}",
            str(row["strategy"]),
            tx=700.0,
            landing=landing_grid_limit_mw,
            flexible=not direct,
            hub=not direct,
        ),
    )


def _plot_reliability(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    styles = {
        "direct": {"color": "tab:blue", "marker": "o", "linestyle": "-", "zorder": 2},
        "integrated_contract": {
            "color": "tab:orange",
            "marker": "x",
            "linestyle": "--",
            "markersize": 9,
            "zorder": 3,
        },
        "reliability_reserve": {
            "color": "tab:green",
            "marker": "s",
            "linestyle": "-",
            "zorder": 4,
        },
    }
    for candidate, group in frame.groupby("candidate_id"):
        group = group.sort_values("low_wind_duration_days")
        ax.plot(
            group["low_wind_duration_days"],
            group["event_eens_mwh"],
            label=candidate,
            **styles.get(str(candidate), {"marker": "o"}),
        )
    ax.text(
        0.985,
        0.05,
        "direct and merchant integrated results overlap",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )
    ax.set_xlabel("Low-wind duration (days)")
    ax.set_ylabel("Critical-load shortfall (MWh)")
    ax.set_title("Reliability value across low-wind duration")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_reliability(root: Path, output: Path, hours: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    capacity_path = output / "s8_candidate_capacities.csv"
    if not capacity_path.exists():
        run_candidates(
            root,
            output,
            hours,
            ("direct", "integrated_contract", "reliability_reserve"),
        )
    capacity_table = pd.read_csv(capacity_path)
    parameters, base_scenario, base_config, flexible_cases, common_cases, _, base = _inputs(
        root, hours
    )
    settings = _settings(root)
    reference_landing = float(settings["reference_landing_capability_mw"])
    selected_ids = ("direct", "integrated_contract", "reliability_reserve")
    candidates = []
    for candidate_id in selected_ids:
        row = capacity_table.loc[capacity_table["candidate_id"] == candidate_id].iloc[0]
        candidates.append(
            _candidate_from_row(
                row,
                base_config,
                direct=candidate_id == "direct",
                landing_grid_limit_mw=reference_landing,
            )
        )
    flexible = flexible_cases["china_reference"]
    common = common_cases["china_reference"]
    distance = float(settings["offshore_distance_km"])
    scenario = _scenario(
        base_scenario,
        "s8_duration_stress",
        hydrogen_price=20.0,
        distance_km=distance,
    )
    rows = []
    event_start = int(settings["reliability_event_start_hour"])
    for days in tuple(int(value) for value in settings["low_wind_durations_days"]):
        duration = days * 24
        frame = base.copy()
        frame["national_compute_demand_mw_it"] = 0.0
        frame["hydrogen_demand"] = 0.0
        frame["critical_load"] = float(settings["reliability_critical_load_mw"])
        frame["grid_absorption_limit_mw"] = (
            reference_landing * frame["landing_demand_factor"]
        )
        frame.loc[event_start : event_start + duration - 1, "wind_availability"] = float(
            settings["reliability_wind_availability"]
        )
        for candidate in candidates:
            dispatch = run_s5_dispatch(
                frame, parameters, candidate.configuration, scenario
            )
            event = dispatch.hourly.iloc[event_start : event_start + duration]
            flex_capex, flex_annual = fixed_flexible_asset_costs(
                candidate.configuration,
                flexible,
                offshore_distance_km=distance,
            )
            common_capex, common_annual = common_infrastructure_costs(
                candidate.design,
                common,
                offshore_distance_km=distance,
            )
            rows.append(
                {
                    "low_wind_duration_days": days,
                    "candidate_id": candidate.candidate_id,
                    "event_eens_mwh": float(event["unmet_critical_load_mw"].sum()),
                    "event_fuel_cell_generation_mwh": float(
                        event["fuel_cell_power_mw"].sum()
                    ),
                    "event_hydrogen_production_kg": float(
                        event["hydrogen_production_kg"].sum()
                    ),
                    "minimum_hydrogen_inventory_kg": float(
                        dispatch.hourly["hydrogen_inventory_end_kg"].min()
                    ),
                    "flexible_asset_capex_cny": float(sum(flex_capex.values())),
                    "common_hub_capex_cny": common_capex["hub_common_capex_cny"],
                    "flexible_and_hub_annual_cost_cny": float(
                        sum(flex_annual.values()) + common_annual["hub_common_annual_cost_cny"]
                    ),
                    "max_offshore_balance_residual_mw": dispatch.kpis[
                        "max_offshore_balance_residual_mw"
                    ],
                }
            )
    result = pd.DataFrame(rows)
    direct_eens = result.loc[result["candidate_id"] == "direct"].set_index(
        "low_wind_duration_days"
    )["event_eens_mwh"]
    result["avoided_eens_vs_direct_mwh"] = result.apply(
        lambda row: direct_eens.loc[row["low_wind_duration_days"]]
        - row["event_eens_mwh"],
        axis=1,
    )
    direct_cost = result.loc[result["candidate_id"] == "direct"].set_index(
        "low_wind_duration_days"
    )["flexible_and_hub_annual_cost_cny"]
    result["incremental_annual_fixed_cost_vs_direct_cny"] = result.apply(
        lambda row: row["flexible_and_hub_annual_cost_cny"]
        - direct_cost.loc[row["low_wind_duration_days"]],
        axis=1,
    )
    voll = parameters.value("unserved_critical_load_penalty")
    result["break_even_event_frequency_per_year"] = np.where(
        result["avoided_eens_vs_direct_mwh"] > 0.0,
        result["incremental_annual_fixed_cost_vs_direct_cny"]
        / (voll * result["avoided_eens_vs_direct_mwh"]),
        np.nan,
    )
    _write(result, output / "s8_low_wind_duration_reliability.csv")
    _plot_reliability(result, figures / "s8_low_wind_duration_reliability.png")


def finalize(root: Path, output: Path, hours: int) -> None:
    finalize_risk(root, output)
    finalize_contract(output)
    figures = output / "figures"
    landing_path = output / "s8_landing_flexibility_frontier.csv"
    reliability_path = output / "s8_low_wind_duration_reliability.csv"
    if landing_path.exists():
        _plot_landing(
            pd.read_csv(landing_path),
            figures / "s8_landing_flexibility_frontier.png",
        )
    if reliability_path.exists():
        _plot_reliability(
            pd.read_csv(reliability_path),
            figures / "s8_low_wind_duration_reliability.png",
        )
    settings = _settings(root)
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    inputs = (
        root / "configs/s7_china_evidence_register.csv",
        root / "configs/s7_china_flexible_cost_cases.csv",
        root / "configs/s7_china_infrastructure_cost_cases.csv",
        root / "configs/s7_china_planning_limits.yaml",
        root / "configs/s8_risk_cases.csv",
        root / "configs/s8_analysis_settings.yaml",
        root / "configs/s8_candidate_definitions.csv",
    )
    manifest = {
        "phase": "S8",
        "model_version": __version__,
        "simulation_hours": hours,
        "method": {
            "capacity_risk": "fixed capacity replay across six weighted operating cases",
            "tail_probability": settings["tail_probability"],
            "risk_aversion": settings["risk_aversion"],
            "contract_compute_prices_cny_per_mwh_it": settings[
                "contract_compute_prices_cny_per_mwh_it"
            ],
            "contract_hydrogen_prices_cny_per_kg": settings[
                "contract_hydrogen_prices_cny_per_kg"
            ],
            "landing_frontier_mw": settings["landing_capabilities_mw"],
            "low_wind_duration_days": settings["low_wind_durations_days"],
        },
        "source_files": {
            str(path.relative_to(root)): _sha256(path) for path in inputs
        },
        "output_files": {
            str(path.relative_to(output)): _sha256(path) for path in files
        },
        "limitations": [
            "All operating profiles remain deterministic synthetic screening cases.",
            "Candidate selection is discrete and is not a continuous two-stage stochastic optimum.",
            "Dispatch retains perfect foresight within each scenario.",
            (
                "Scenario weights are transparent planning judgements rather than "
                "estimated probabilities."
            ),
            "Reliability stress cases are kept outside merchant expected-value weights.",
            (
                "Project-grade conclusions require site weather, landing-grid data, "
                "quotations and contracts."
            ),
        ],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=(
            "all",
            "candidates",
            "risk",
            "risk-finalize",
            "landing",
            "contract",
            "reliability",
            "finalize",
        ),
        default="all",
    )
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/s8_robust_value_analysis")
    )
    parser.add_argument("--hydrogen-price", type=float, action="append")
    parser.add_argument("--candidate-id", action="append")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = _settings(root)
    if args.task in ("all", "candidates"):
        run_candidates(
            root,
            args.output,
            args.hours,
            tuple(args.candidate_id) if args.candidate_id else None,
        )
    if args.task in ("all", "risk"):
        run_risk(
            root,
            args.output,
            args.hours,
            tuple(args.candidate_id) if args.candidate_id else None,
        )
    if args.task == "risk-finalize":
        finalize_risk(root, args.output)
    if args.task in ("all", "landing"):
        run_landing(root, args.output, args.hours)
    if args.task == "all":
        for price in settings["contract_hydrogen_prices_cny_per_kg"]:
            run_contract(root, args.output, args.hours, (price,), None)
    elif args.task == "contract":
        prices = tuple(
            args.hydrogen_price
            or settings["contract_hydrogen_prices_cny_per_kg"]
        )
        run_contract(
            root,
            args.output,
            args.hours,
            prices,
            tuple(args.candidate_id) if args.candidate_id else None,
        )
    if args.task in ("all", "reliability"):
        run_reliability(root, args.output, args.hours)
    if args.task in ("all", "finalize"):
        finalize(root, args.output, args.hours)


if __name__ == "__main__":
    main()
