"""Run transparent S0 capacity-distance-price and cable-outage sensitivities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from blue_hub import __version__
from blue_hub.dispatch_model import S0DispatchResult, run_s0_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import export_s0_results
from blue_hub.synthetic import generate_synthetic_timeseries


def _float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot_sensitivity(results: pd.DataFrame, output: Path) -> None:
    figure_directory = output / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    base_price = results[results["electricity_price_multiplier"] == 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for distance, group in base_price.groupby("offshore_distance_km"):
        ordered = group.sort_values("tx_capacity_mw")
        label = f"{distance:g} km"
        axes[0].plot(
            ordered["tx_capacity_mw"],
            100.0 * ordered["curtailment_rate"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            ordered["tx_capacity_mw"],
            ordered["export_land_mwh"] / 1e6,
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Transmission capacity (MW)", ylabel="Curtailment rate (%)")
    axes[1].set(xlabel="Transmission capacity (MW)", ylabel="Land delivery (TWh/year)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_directory / "capacity_distance_sensitivity.png", dpi=180)
    plt.close(fig)


def _plot_event_cases(
    outage: S0DispatchResult,
    economic: S0DispatchResult,
    must_take: S0DispatchResult,
    output: Path,
) -> None:
    figure_directory = output / "figures"
    event_slice = slice(60, 108)
    hour = range(60, 108)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)

    outage_hourly = outage.hourly.iloc[event_slice]
    axes[0].plot(hour, outage_hourly["wind_available_mw"], label="Available wind")
    axes[0].plot(hour, outage_hourly["export_send_mw"], label="Cable export")
    axes[0].plot(hour, outage_hourly["curtailment_mw"], label="Curtailment")
    axes[0].axvspan(72, 96, color="black", alpha=0.08, label="Cable outage")
    axes[0].set_ylabel("Power (MW)")
    axes[0].set_title("24-hour cable outage")
    axes[0].legend(frameon=False, ncol=4)

    economic_hourly = economic.hourly.iloc[event_slice]
    must_take_hourly = must_take.hourly.iloc[event_slice]
    axes[1].plot(hour, economic_hourly["export_send_mw"], label="Economic export")
    axes[1].plot(
        hour,
        must_take_hourly["export_send_mw"],
        linestyle="--",
        label="Must-take export",
    )
    axes[1].axvspan(72, 96, color="tab:red", alpha=0.08, label="Negative price")
    axes[1].set(xlabel="Hour of synthetic year", ylabel="Send-side export (MW)")
    axes[1].set_title("Negative-price dispatch policy")
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_directory / "event_stress_tests.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--capacities", default="300,500,700,900,1000")
    parser.add_argument("--distances", default="50,100,150,200")
    parser.add_argument("--price-multipliers", default="0.7,1.0,1.3")
    parser.add_argument("--output", type=Path, default=Path("outputs/s0_sensitivity"))
    args = parser.parse_args()

    params = load_parameters("configs/technology_parameters.csv")
    base_config = load_system_configuration("configs/base_case.yaml")
    scenarios = {item.scenario_id: item for item in load_scenarios("configs/scenario_matrix.csv")}
    base_scenario = scenarios["base"]
    timeseries = generate_synthetic_timeseries(args.hours)

    records: list[dict[str, object]] = []
    for capacity in _float_list(args.capacities):
        for distance in _float_list(args.distances):
            for price_multiplier in _float_list(args.price_multipliers):
                config = base_config.model_copy(
                    update={
                        "config_id": f"S0_tx{capacity:g}",
                        "tx_capacity_mw": capacity,
                    }
                )
                scenario = base_scenario.model_copy(
                    update={
                        "scenario_id": (f"D{distance:g}_P{price_multiplier:g}".replace(".", "p")),
                        "offshore_distance_km": distance,
                        "electricity_price_multiplier": price_multiplier,
                    }
                )
                result = run_s0_dispatch(timeseries, params, config, scenario)
                records.append(
                    {
                        "tx_capacity_mw": capacity,
                        "offshore_distance_km": distance,
                        "electricity_price_multiplier": price_multiplier,
                        **result.kpis,
                        "configuration_hash": result.metadata["configuration_hash"],
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    sensitivity = pd.DataFrame(records).sort_values(
        ["electricity_price_multiplier", "offshore_distance_km", "tx_capacity_mw"]
    )
    sensitivity.to_csv(args.output / "s0_sensitivity.csv", index=False)
    _plot_sensitivity(sensitivity, args.output)

    if args.hours >= 96:
        outage = run_s0_dispatch(timeseries, params, base_config, scenarios["cable_outage_24h"])
        export_s0_results(outage, args.output / "cable_outage_24h")
        negative_price = run_s0_dispatch(
            timeseries, params, base_config, scenarios["negative_price_24h"]
        )
        export_s0_results(negative_price, args.output / "negative_price_economic")
        must_take_config = base_config.model_copy(update={"export_policy": "must_take"})
        must_take = run_s0_dispatch(
            timeseries, params, must_take_config, scenarios["negative_price_24h"]
        )
        export_s0_results(must_take, args.output / "negative_price_must_take")
        _plot_event_cases(outage, negative_price, must_take, args.output)
    artifacts = {
        str(path.relative_to(args.output)): _sha256(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "sensitivity_manifest.json"
    }
    (args.output / "sensitivity_manifest.json").write_text(
        json.dumps(
            {
                "model_version": __version__,
                "phase": "S0_sensitivity",
                "hours": args.hours,
                "case_count": len(sensitivity),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(sensitivity)} sensitivity cases to {args.output}")


if __name__ == "__main__":
    main()
