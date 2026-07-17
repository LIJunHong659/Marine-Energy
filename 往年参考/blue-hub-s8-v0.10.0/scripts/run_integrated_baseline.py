"""Run one Phase 5 / S4 integrated dispatch case and export auditable results."""

from __future__ import annotations

import argparse
from pathlib import Path

from blue_hub.integrated_dispatch_model import run_s4_dispatch
from blue_hub.loaders import (
    load_parameters,
    load_scenarios,
    load_system_configuration,
    load_timeseries,
)
from blue_hub.outputs import export_dispatch_results
from blue_hub.synthetic import generate_synthetic_timeseries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--timeseries", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/s4_integrated_b100_e400_h100_s72000_c100_f100.yaml"),
    )
    parser.add_argument("--scenario-id", default="base")
    parser.add_argument("--output", type=Path, default=Path("outputs/integrated_baseline"))
    args = parser.parse_args()

    parameters = load_parameters("configs/technology_parameters.csv")
    config = load_system_configuration(args.config)
    scenarios = {item.scenario_id: item for item in load_scenarios("configs/scenario_matrix.csv")}
    if args.scenario_id not in scenarios:
        raise KeyError(f"unknown scenario_id: {args.scenario_id}")
    timeseries = (
        load_timeseries(args.timeseries)
        if args.timeseries is not None
        else generate_synthetic_timeseries(args.hours)
    )
    result = run_s4_dispatch(timeseries, parameters, config, scenarios[args.scenario_id])
    paths = export_dispatch_results(result, args.output)
    print(
        f"S4 solved for {len(result.hourly)} hours; "
        f"margin={result.kpis['operating_margin_cny']:.2f} CNY; "
        f"H2 sales={result.kpis['hydrogen_sales_kg']:.2f} kg; "
        f"compute={result.kpis['compute_service_mwh_it']:.2f} MWh-IT"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
