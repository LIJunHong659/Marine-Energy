"""Run Phase 5 / S4 fair ablations and representative integrated cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blue_hub import __version__
from blue_hub.battery_dispatch_model import run_s1_dispatch
from blue_hub.compute_dispatch_model import run_s3_dispatch
from blue_hub.dispatch_model import run_s0_dispatch
from blue_hub.hydrogen_dispatch_model import run_s2_dispatch
from blue_hub.integrated_dispatch_model import S4DispatchResult, run_s4_dispatch
from blue_hub.loaders import load_parameters, load_scenarios, load_system_configuration
from blue_hub.outputs import DispatchResult, export_dispatch_results
from blue_hub.schemas import ScenarioDefinition
from blue_hub.synthetic import generate_synthetic_timeseries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(base: ScenarioDefinition, name: str, h2: float, compute: float) -> ScenarioDefinition:
    return base.model_copy(
        update={
            "scenario_id": name,
            "hydrogen_price_multiplier": h2,
            "compute_price_multiplier": compute,
        }
    )


def _plot_mode_comparison(comparison: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(comparison["mode"], comparison["operating_margin_cny"] / 1e6, color="#0072B2")
    axis.set_ylabel("Operating margin (million CNY)")
    axis.set_title("Five fair operating modes under the shared base scenario")
    figure.tight_layout()
    figure.savefig(output / "figures" / "s4_five_mode_comparison.png", dpi=180)
    plt.close(figure)


def _plot_allocation(cases: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cases))
    export = cases["export_send_mwh"] / 1e3
    hydrogen = cases["electrolyzer_energy_mwh"] / 1e3
    compute = cases["dc_facility_energy_mwh"] / 1e3
    axis.bar(x, export, label="Electricity export", color="#0072B2")
    axis.bar(x, hydrogen, bottom=export, label="Electrolyzer", color="#D55E00")
    axis.bar(x, compute, bottom=export + hydrogen, label="Compute facility", color="#009E73")
    axis.set_xticks(x, cases["scenario_id"].str.replace("_", " "), rotation=25)
    axis.set_ylabel("Allocated energy (GWh)")
    axis.set_title("Integrated allocation across representative cases")
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output / "figures" / "s4_integrated_allocation.png", dpi=180)
    plt.close(figure)


def _plot_price_map(price_map: pd.DataFrame, output: Path) -> None:
    pivot = price_map.pivot(
        index="compute_price_multiplier",
        columns="hydrogen_price_multiplier",
        values="hydrogen_share_percent",
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0.0, vmax=100.0)
    axis.set_xticks(np.arange(len(pivot.columns)), [f"{value:.1f}" for value in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), [f"{value:.1f}" for value in pivot.index])
    axis.set_xlabel("Hydrogen-price multiplier")
    axis.set_ylabel("Compute-price multiplier")
    axis.set_title("Hydrogen share of non-export conversion energy")
    for row, compute_price in enumerate(pivot.index):
        for column, h2_price in enumerate(pivot.columns):
            axis.text(
                column,
                row,
                f"{pivot.loc[compute_price, h2_price]:.0f}%",
                ha="center",
                va="center",
                color="white",
            )
    figure.colorbar(image, ax=axis, label="Hydrogen share (%)")
    figure.tight_layout()
    figure.savefig(output / "figures" / "s4_hydrogen_compute_price_map.png", dpi=180)
    plt.close(figure)


def _case_record(result: S4DispatchResult, scenario_id: str) -> dict[str, object]:
    hourly = result.hourly
    return {
        "scenario_id": scenario_id,
        "operating_margin_cny": result.kpis["operating_margin_cny"],
        "export_send_mwh": result.kpis["export_send_mwh"],
        "electrolyzer_energy_mwh": float(hourly["electrolyzer_power_mw"].sum()),
        "dc_facility_energy_mwh": float(hourly["dc_facility_power_mw"].sum()),
        "hydrogen_sales_kg": result.kpis["hydrogen_sales_kg"],
        "hydrogen_service_rate": result.kpis["hydrogen_service_rate"],
        "compute_service_mwh_it": result.kpis["compute_service_mwh_it"],
        "compute_service_rate": result.kpis["compute_service_rate"],
        "rigid_compute_unserved_mwh_it": result.kpis["rigid_compute_unserved_mwh_it"],
        "curtailment_mwh": result.kpis["curtailment_mwh"],
        "battery_efc": result.kpis["battery_efc"],
        "configuration_hash": result.metadata["configuration_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--output", type=Path, default=Path("outputs/s4_integrated_analysis"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    parameters = load_parameters("configs/technology_parameters.csv")
    scenarios = {item.scenario_id: item for item in load_scenarios("configs/scenario_matrix.csv")}
    s0_config = load_system_configuration("configs/base_case.yaml")
    s1_config = load_system_configuration("configs/s1_battery_100mw_400mwh.yaml")
    s2_config = load_system_configuration(
        "configs/s2_battery_hydrogen_100mw_400mwh_100mw_72000kg.yaml"
    )
    s3_config = load_system_configuration("configs/s3_compute_100mw_fiber_100mw_it.yaml")
    s4_config = load_system_configuration(
        "configs/s4_integrated_b100_e400_h100_s72000_c100_f100.yaml"
    )
    timeseries = generate_synthetic_timeseries(args.hours)
    base = scenarios["base"]

    mode_results: dict[str, DispatchResult] = {
        "S0 direct electricity": run_s0_dispatch(timeseries, parameters, s0_config, base),
        "S1 battery": run_s1_dispatch(timeseries, parameters, s1_config, base),
        "S2 battery + hydrogen": run_s2_dispatch(timeseries, parameters, s2_config, base),
        "S3 compute": run_s3_dispatch(timeseries, parameters, s3_config, base),
        "S4 integrated": run_s4_dispatch(timeseries, parameters, s4_config, base),
    }
    mode_comparison = pd.DataFrame(
        [
            {
                "mode": name,
                "operating_margin_cny": result.kpis["operating_margin_cny"],
                "curtailment_mwh": result.kpis["curtailment_mwh"],
                "export_land_mwh": result.kpis["export_land_mwh"],
                "configuration_hash": result.metadata["configuration_hash"],
            }
            for name, result in mode_results.items()
        ]
    )
    mode_comparison.to_csv(args.output / "s4_five_mode_comparison.csv", index=False)

    representative_scenarios = {
        "base": base,
        "hydrogen_low_price": scenarios["hydrogen_low_price"],
        "hydrogen_high_price": scenarios["hydrogen_high_price"],
        "compute_high_price": scenarios["compute_high_price"],
        "h2_high_compute_high": _scenario(base, "h2_high_compute_high", 1.30, 1.30),
        "cable_outage_24h": scenarios["cable_outage_24h"],
        "fiber_outage_24h": scenarios["fiber_outage_24h"],
        "compound_outage_24h": scenarios["compound_outage_24h"],
    }
    case_results: dict[str, S4DispatchResult] = {}
    case_records: list[dict[str, object]] = []
    for name, scenario in representative_scenarios.items():
        result = run_s4_dispatch(timeseries, parameters, s4_config, scenario)
        case_results[name] = result
        case_records.append(_case_record(result, name))
        export_dispatch_results(result, args.output / "scenario_hourly" / name)
    representative_cases = pd.DataFrame(case_records)
    representative_cases.to_csv(args.output / "s4_representative_cases.csv", index=False)

    price_records: list[dict[str, object]] = []
    for h2_price in (0.70, 1.00, 1.30):
        for compute_price in (0.70, 1.00, 1.30):
            scenario = _scenario(
                base,
                f"price_h{h2_price:.1f}_c{compute_price:.1f}",
                h2_price,
                compute_price,
            )
            result = run_s4_dispatch(timeseries, parameters, s4_config, scenario)
            hourly = result.hourly
            h2_energy = float(hourly["electrolyzer_power_mw"].sum())
            compute_energy = float(hourly["dc_facility_power_mw"].sum())
            price_records.append(
                {
                    "hydrogen_price_multiplier": h2_price,
                    "compute_price_multiplier": compute_price,
                    "operating_margin_cny": result.kpis["operating_margin_cny"],
                    "hydrogen_sales_kg": result.kpis["hydrogen_sales_kg"],
                    "compute_service_mwh_it": result.kpis["compute_service_mwh_it"],
                    "hydrogen_energy_mwh": h2_energy,
                    "compute_facility_energy_mwh": compute_energy,
                    "hydrogen_share_percent": (
                        100.0 * h2_energy / (h2_energy + compute_energy)
                        if h2_energy + compute_energy
                        else 0.0
                    ),
                    "configuration_hash": result.metadata["configuration_hash"],
                }
            )
    price_map = pd.DataFrame(price_records)
    price_map.to_csv(args.output / "s4_hydrogen_compute_price_map.csv", index=False)

    _plot_mode_comparison(mode_comparison, args.output)
    _plot_allocation(representative_cases, args.output)
    _plot_price_map(price_map, args.output)
    artifacts = {
        str(path.relative_to(args.output)): _sha256(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "model_version": __version__,
                "phase": "S4_integrated_analysis",
                "hours": args.hours,
                "mode_count": len(mode_comparison),
                "representative_case_count": len(representative_cases),
                "price_case_count": len(price_map),
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
        f"wrote {len(mode_comparison)} modes, {len(representative_cases)} representative "
        f"cases and {len(price_map)} price cases to {args.output}"
    )


if __name__ == "__main__":
    main()
