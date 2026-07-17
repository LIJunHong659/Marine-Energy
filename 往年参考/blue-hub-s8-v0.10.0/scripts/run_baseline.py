"""Run the Phase 1 / S0 analytical dispatch baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from blue_hub.dispatch_model import run_s0_dispatch
from blue_hub.loaders import (
    load_parameters,
    load_scenarios,
    load_system_configuration,
    load_timeseries,
)
from blue_hub.outputs import export_s0_results
from blue_hub.synthetic import generate_synthetic_timeseries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--timeseries", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base_case.yaml"))
    parser.add_argument(
        "--parameters", type=Path, default=Path("configs/technology_parameters.csv")
    )
    parser.add_argument("--scenarios", type=Path, default=Path("configs/scenario_matrix.csv"))
    parser.add_argument("--scenario-id", default="base")
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()

    params = load_parameters(args.parameters)
    config = load_system_configuration(args.config)
    scenarios = {scenario.scenario_id: scenario for scenario in load_scenarios(args.scenarios)}
    if args.scenario_id not in scenarios:
        raise KeyError(f"unknown scenario_id: {args.scenario_id}")
    timeseries = (
        load_timeseries(args.timeseries)
        if args.timeseries is not None
        else generate_synthetic_timeseries(hours=args.hours)
    )
    result = run_s0_dispatch(timeseries, params, config, scenarios[args.scenario_id])
    paths = export_s0_results(result, args.output)
    print(
        f"S0 solved for {len(result.hourly)} hours; "
        f"land export={result.kpis['export_land_mwh']:.3f} MWh; "
        f"curtailment={result.kpis['curtailment_rate']:.3%}; "
        f"margin={result.kpis['operating_margin_cny']:.2f} CNY"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
