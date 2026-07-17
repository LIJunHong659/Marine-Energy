"""Reproducible S0 result export with hashes for all user-facing tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class DispatchResult(Protocol):
    """Read-only result interface shared by frozen dispatch result records."""

    @property
    def hourly(self) -> pd.DataFrame: ...

    @property
    def kpis(self) -> dict[str, Any]: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_dispatch_results(
    result: DispatchResult, output_directory: str | Path
) -> dict[str, Path]:
    """Write hourly, KPI and manifest artifacts and return their paths."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    hourly_path = output / "hourly_results.csv"
    kpi_path = output / "kpi_summary.csv"
    manifest_path = output / "run_manifest.json"

    result.hourly.to_csv(hourly_path, index=False)
    pd.DataFrame([result.kpis]).to_csv(kpi_path, index=False)
    manifest = {
        **result.metadata,
        "artifacts": {
            hourly_path.name: _sha256(hourly_path),
            kpi_path.name: _sha256(kpi_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"hourly": hourly_path, "kpis": kpi_path, "manifest": manifest_path}


def export_s0_results(result: DispatchResult, output_directory: str | Path) -> dict[str, Path]:
    """Backward-compatible alias for existing S0 scripts."""
    return export_dispatch_results(result, output_directory)
