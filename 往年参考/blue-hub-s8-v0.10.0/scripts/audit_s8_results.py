"""Audit checkpoint completeness and numerical constraints in saved S8 results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from blue_hub import __version__


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, output: Path) -> None:
    with (root / "configs/s8_analysis_settings.yaml").open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    definitions = pd.read_csv(root / "configs/s8_candidate_definitions.csv")
    merchant_ids = set(
        definitions.loc[
            definitions["candidate_scope"] == "merchant_risk_set", "candidate_id"
        ]
    )
    risk_cases = pd.read_csv(root / "configs/s8_risk_cases.csv")
    case_ids = set(risk_cases["case_id"])
    ledger = pd.read_csv(output / "s8_candidate_scenario_ledger.csv")
    summary = pd.read_csv(output / "s8_candidate_risk_summary.csv")
    contract_ledger = pd.read_csv(output / "s8_contract_candidate_ledger.csv")
    contract_surface = pd.read_csv(output / "s8_contract_value_surface.csv")
    landing = pd.read_csv(output / "s8_landing_flexibility_frontier.csv")
    reliability = pd.read_csv(output / "s8_low_wind_duration_reliability.csv")

    expected_pairs = len(settings["contract_compute_prices_cny_per_mwh_it"]) * len(
        settings["contract_hydrogen_prices_cny_per_kg"]
    )
    pair_counts = contract_ledger.groupby(
        ["hydrogen_price_cny_per_kg", "compute_price_cny_per_mwh_it"]
    )["candidate_id"].nunique()
    weight_sums = ledger.groupby("candidate_id")["scenario_weight"].sum()
    checks = {
        "risk_candidate_identifiers_complete": set(ledger["candidate_id"]) == merchant_ids,
        "risk_case_identifiers_complete": set(ledger["case_id"]) == case_ids,
        "risk_candidate_case_rows_complete": len(ledger) == len(merchant_ids) * len(case_ids),
        "risk_scenario_weights_close": bool(
            np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-10)
        ),
        "risk_summary_candidates_complete": set(summary["candidate_id"]) == merchant_ids,
        "risk_strategy_labels_complete": bool(
            ledger["strategy"].notna().all()
            and (ledger["strategy"].astype(str).str.strip() != "").all()
            and summary["strategy"].notna().all()
        ),
        "risk_regret_nonnegative": float(ledger["scenario_regret_cny"].min()) >= -1e-6,
        "risk_power_balance_passed": float(
            ledger[
                ["max_offshore_balance_residual_mw", "max_land_balance_residual_mw"]
            ].to_numpy(dtype=float).max()
        )
        <= 1e-6,
        "contract_pairs_complete": len(contract_surface) == expected_pairs,
        "contract_candidates_complete_per_pair": bool(
            (pair_counts == len(merchant_ids)).all()
        ),
        "contract_ledger_rows_complete": len(contract_ledger)
        == expected_pairs * len(merchant_ids),
        "landing_rows_complete": len(landing)
        == 2 * len(settings["landing_capabilities_mw"]),
        "reliability_rows_complete": len(reliability)
        == 3 * len(settings["low_wind_durations_days"]),
        "reliability_power_balance_passed": float(
            reliability["max_offshore_balance_residual_mw"].max()
        )
        <= 1e-6,
        "all_key_results_finite": bool(
            np.isfinite(
                summary[
                    [
                        "expected_full_net_value_cny_per_year",
                        "lower_tail_full_net_value_cny_per_year",
                        "risk_adjusted_score_cny_per_year",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
    }
    audit = {
        "phase": "S8",
        "model_version": __version__,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "counts": {
            "merchant_candidates": len(merchant_ids),
            "risk_cases": len(case_ids),
            "candidate_case_rows": len(ledger),
            "contract_pairs": expected_pairs,
            "contract_candidate_rows": len(contract_ledger),
            "landing_rows": len(landing),
            "reliability_rows": len(reliability),
        },
        "maximum_balance_residual_mw": float(
            max(
                ledger[
                    [
                        "max_offshore_balance_residual_mw",
                        "max_land_balance_residual_mw",
                    ]
                ].to_numpy(dtype=float).max(),
                reliability["max_offshore_balance_residual_mw"].max(),
            )
        ),
    }
    audit_path = output / "s8_validation_summary.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = output / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_version"] = __version__
    manifest["result_audit"] = {
        "script": "scripts/audit_s8_results.py",
        "all_passed": audit["all_passed"],
        "maximum_balance_residual_mw": audit["maximum_balance_residual_mw"],
    }
    manifest["output_files"] = {
        str(path.relative_to(output)): _hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not audit["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"S8 result audit failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/s8_robust_value_analysis")
    )
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.output)


if __name__ == "__main__":
    main()
