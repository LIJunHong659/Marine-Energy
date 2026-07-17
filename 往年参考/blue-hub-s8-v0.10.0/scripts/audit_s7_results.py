"""Independently audit the saved S7 representative 8760-hour dispatch.

The planning optimiser already reports balance residuals.  This small reader
does not resolve the model; it rechecks the exported hourly table and writes a
fixed-seed random-hour sample that can be inspected in a classroom or review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blue_hub import __version__


POWER_TOLERANCE = 1e-6
# The representative hourly file is intentionally written with ``%.8g``.
# Kilogram values around 10^3 consequently carry a few 10^-5 kg/h of display
# rounding, so a 10^-3 kg check is much tighter than the exported precision.
MASS_TOLERANCE = 1e-3


def _maximum_positive(series: pd.Series) -> float:
    return float(np.maximum(series.to_numpy(dtype=float), 0.0).max())


def run(output: Path, sample_size: int, seed: int) -> None:
    hourly_path = output / "s7_representative_contract_hourly.csv"
    manifest_path = output / "analysis_manifest.json"
    hourly = pd.read_csv(hourly_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capacities = manifest["representative_contract"]["capacities"]

    checks = {
        "offshore_balance_residual_mw": float(
            hourly["offshore_balance_residual_mw"].abs().max()
        ),
        "land_balance_residual_mw": float(
            hourly["land_balance_residual_mw"].abs().max()
        ),
        "export_over_tx_available_mw": _maximum_positive(
            hourly["export_send_mw"] - hourly["tx_available_capacity_mw"]
        ),
        "compute_over_demand_mwh_it": _maximum_positive(
            hourly["spot_compute_completed_mwh_it"]
            - hourly["national_compute_demand_mwh_it"]
        ),
        "compute_over_it_capacity_mwh_it": _maximum_positive(
            hourly["spot_compute_completed_mwh_it"]
            - float(capacities["compute_it_capacity_mw"])
        ),
        "hydrogen_sale_over_demand_kg": _maximum_positive(
            hourly["hydrogen_sale_kg"] - 12_000.0
        ),
        "hydrogen_sale_over_pipeline_kg": _maximum_positive(
            hourly["hydrogen_sale_kg"]
            - float(capacities["hydrogen_export_capacity_kg_per_h"])
        ),
        "battery_energy_underflow_mwh": _maximum_positive(
            -hourly["battery_energy_end_mwh"]
        ),
        "battery_energy_over_capacity_mwh": _maximum_positive(
            hourly["battery_energy_end_mwh"]
            - float(capacities["battery_energy_mwh"])
        ),
        "hydrogen_inventory_underflow_kg": _maximum_positive(
            -hourly["hydrogen_inventory_end_kg"]
        ),
        "hydrogen_inventory_over_capacity_kg": _maximum_positive(
            hourly["hydrogen_inventory_end_kg"]
            - float(capacities["hydrogen_storage_kg"])
        ),
        "negative_curtailment_mw": _maximum_positive(-hourly["curtailment_mw"]),
        "negative_unmet_critical_load_mw": _maximum_positive(
            -hourly["unmet_critical_load_mw"]
        ),
    }
    mass_checks = {
        "hydrogen_sale_over_demand_kg",
        "hydrogen_sale_over_pipeline_kg",
        "hydrogen_inventory_underflow_kg",
        "hydrogen_inventory_over_capacity_kg",
    }
    passed = {
        name: value <= (MASS_TOLERANCE if name in mass_checks else POWER_TOLERANCE)
        for name, value in checks.items()
    }

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(hourly), size=min(sample_size, len(hourly)), replace=False))
    audit_columns = [
        "timestamp",
        "renewable_available_mw",
        "tx_available_capacity_mw",
        "export_send_mw",
        "electrolyzer_power_mw",
        "spot_compute_completed_mwh_it",
        "hydrogen_sale_kg",
        "battery_energy_end_mwh",
        "hydrogen_inventory_end_kg",
        "curtailment_mw",
        "offshore_balance_residual_mw",
        "land_balance_residual_mw",
    ]
    hourly.iloc[rows][audit_columns].to_csv(
        output / "s7_random_hour_audit.csv", index=False, float_format="%.10g"
    )

    summary = {
        "audit_type": "saved-hourly-result constraint recheck",
        "hours_checked": int(len(hourly)),
        "random_sample_size": int(len(rows)),
        "random_seed": seed,
        "tolerances": {
            "power_or_energy": POWER_TOLERANCE,
            "hydrogen_mass": MASS_TOLERANCE,
        },
        "checks": checks,
        "passed": passed,
        "all_passed": bool(all(passed.values())),
        "representative_capacities": capacities,
    }
    (output / "s7_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep the analysis manifest coupled to the saved artefacts even when this
    # independent audit is run after the primary optimisation batch.
    manifest["model_version"] = __version__
    manifest["result_audit"] = {
        "script": "scripts/audit_s7_results.py",
        "all_passed": summary["all_passed"],
        "random_seed": seed,
        "random_sample_size": int(len(rows)),
    }
    manifest["output_files"] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not summary["all_passed"]:
        raise RuntimeError("S7 saved-hourly audit failed; see s7_validation_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/s7_china_value_analysis")
    )
    parser.add_argument("--sample-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")
    run(args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
