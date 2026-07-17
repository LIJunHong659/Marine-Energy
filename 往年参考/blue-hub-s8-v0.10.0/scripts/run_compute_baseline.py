"""Run one Phase 4 / S3 compute dispatch case and export auditable results."""

from __future__ import annotations

import argparse
from pathlib import Path

from blue_hub.compute_dispatch_model import run_s3_dispatch
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
        "--config", type=Path, default=Path("configs/s3_compute_100mw_fiber_100mw_it.yaml")
    )
    parser.add_argument("--scenario-id", default="base")
    parser.add_argument("--output", type=Path, default=Path("outputs/compute_baseline"))
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
    result = run_s3_dispatch(timeseries, parameters, config, scenarios[args.scenario_id])
    paths = export_dispatch_results(result, args.output)
    print(
        f"S3 solved for {len(result.hourly)} hours; "
        f"margin={result.kpis['operating_margin_cny']:.2f} CNY; "
        f"compute service={result.kpis['compute_service_mwh_it']:.2f} MWh-IT; "
        f"service rate={result.kpis['compute_service_rate']:.3%}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
