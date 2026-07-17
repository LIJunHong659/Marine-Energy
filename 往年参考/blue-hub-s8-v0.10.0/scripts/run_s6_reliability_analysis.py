"""Size long-duration hydrogen assets for a 90-day low-wind stress year."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-blue-hub")

import matplotlib.pyplot as plt
import pandas as pd

from blue_hub import __version__
from blue_hub.investment_planning_model import (
    PlanningLimits,
    load_investment_cost_cases,
    run_s6_investment_planning,
)
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_dispatch_results
from blue_hub.schemas import TechnologyParameters
from blue_hub.synthetic import generate_synthetic_timeseries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_unserved_cost(
    parameters: TechnologyParameters, value_cny_per_mwh: float
) -> TechnologyParameters:
    items = tuple(
        item.model_copy(update={"value_base": value_cny_per_mwh})
        if item.parameter == "unserved_critical_load_penalty"
        else item
        for item in parameters.items
    )
    return TechnologyParameters(items=items)


def _stress_timeseries(hours: int) -> pd.DataFrame:
    frame = generate_synthetic_timeseries(hours, scenario_id="s6_seasonal_reliability")
    frame["rigid_compute_arrival"] = 0.0
    frame["flex_compute_arrival"] = 0.0
    frame["national_compute_demand_mw_it"] = 0.0
    frame["national_compute_price_cny_per_mwh_it"] = 0.0
    frame["hydrogen_demand"] = 0.0
    frame["grid_absorption_factor"] = 1.0
    start = min(3_000, max(hours // 3, 0))
    stop = min(start + 2_160, hours)
    frame.loc[start : stop - 1, "wind_availability"] = 0.01
    return frame


def _plot(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    storage_line = axes[0].plot(
        frame["unserved_cost_cny_per_mwh"],
        frame["hydrogen_storage_kg"] / 1_000.0,
        marker="o",
        color="#0f766e",
        label="H2 storage (tonne)",
    )
    fuel_axis = axes[0].twinx()
    fuel_line = fuel_axis.plot(
        frame["unserved_cost_cny_per_mwh"],
        frame["fuel_cell_power_mw"],
        marker="s",
        color="#d97706",
        label="Fuel cell (MW)",
    )
    axes[0].set_xlabel("Critical-load interruption value (CNY/MWh)")
    axes[0].set_ylabel("Hydrogen storage (tonne)", color="#0f766e")
    fuel_axis.set_ylabel("Fuel-cell capacity (MW)", color="#d97706")
    axes[0].set_title("Long-duration capacity selected")
    axes[0].legend(storage_line + fuel_line, ["H2 storage", "Fuel cell"], frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        frame["unserved_cost_cny_per_mwh"],
        frame["direct_eens_mwh"],
        marker="o",
        label="Direct export",
        color="#b91c1c",
    )
    axes[1].plot(
        frame["unserved_cost_cny_per_mwh"],
        frame["optimized_eens_mwh"],
        marker="o",
        label="Optimized H2 return",
        color="#1d4ed8",
    )
    axes[1].set_xlabel("Critical-load interruption value (CNY/MWh)")
    axes[1].set_ylabel("EENS (MWh/year)")
    axes[1].set_title("90-day low-wind reliability outcome")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8_760)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/s6_reliability_value")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    parameters = load_parameters(root / "configs/technology_parameters.csv")
    config = load_system_configuration(
        root / "configs/s5_flexible_hub_b100_e400_h200_s500000_c200_f200_fc50.yaml"
    )
    scenario = load_scenarios(root / "configs/scenario_matrix.csv")[0].model_copy(
        update={
            "scenario_id": "s6_seasonal_reliability",
            "hydrogen_price_multiplier": 0.75,
        }
    )
    cost = {
        item.cost_case_id: item
        for item in load_investment_cost_cases(root / "configs/s6_investment_cost_cases.csv")
    }["cost_reference"]
    limits = PlanningLimits(300.0, 2_400.0, 400.0, 1_500_000.0, 100.0, 0.0)
    zero = PlanningLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    frame = _stress_timeseries(args.hours)

    records: list[dict[str, object]] = []
    base_result = None
    for unserved_cost in (5_000.0, 10_000.0, 20_000.0):
        case_parameters = _with_unserved_cost(parameters, unserved_cost)
        direct = run_s6_investment_planning(
            frame, case_parameters, config, scenario, cost, zero
        )
        result = run_s6_investment_planning(
            frame, case_parameters, config, scenario, cost, limits
        )
        if unserved_cost == 10_000.0:
            base_result = result
        records.append(
            {
                "unserved_cost_cny_per_mwh": unserved_cost,
                **result.capacities,
                "flexible_asset_capex_cny": result.kpis["flexible_asset_capex_cny"],
                "annualized_flexible_fixed_cost_cny": result.kpis[
                    "annualized_flexible_fixed_cost_cny"
                ],
                "direct_eens_mwh": direct.kpis["eens_mwh"],
                "optimized_eens_mwh": result.kpis["eens_mwh"],
                "avoided_eens_mwh": direct.kpis["eens_mwh"] - result.kpis["eens_mwh"],
                "incremental_operating_margin_cny": (
                    result.kpis["annualized_operating_margin_cny"]
                    - direct.kpis["annualized_operating_margin_cny"]
                ),
                "incremental_net_value_cny": (
                    result.kpis["net_annual_value_cny"]
                    - direct.kpis["net_annual_value_cny"]
                ),
                "fuel_cell_generation_mwh": result.kpis["fuel_cell_generation_mwh"],
                "hydrogen_production_kg": result.kpis["hydrogen_production_kg"],
                "configuration_hash": result.metadata["configuration_hash"],
            }
        )
    results = pd.DataFrame(records)
    table_path = output / "s6_reliability_value_sensitivity.csv"
    results.to_csv(table_path, index=False)
    if base_result is None:
        raise RuntimeError("base reliability result was not generated")
    export_dispatch_results(base_result, output / "base_10000_hourly")
    figure_path = figures / "s6_long_duration_reliability.png"
    _plot(results, figure_path)
    manifest = {
        "phase": "S6-reliability",
        "model_version": __version__,
        "hours": args.hours,
        "low_wind_hours": min(2_160, max(args.hours - min(3_000, args.hours // 3), 0)),
        "cost_case_id": cost.cost_case_id,
        "artifacts": {
            table_path.name: _sha256(table_path),
            str(figure_path.relative_to(output)): _sha256(figure_path),
        },
        "interpretation": (
            "stress-test value of avoiding critical-load interruption; not a market cash flow"
        ),
    }
    manifest_path = output / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"Wrote S6 reliability analysis to {output}")


if __name__ == "__main__":
    main()
