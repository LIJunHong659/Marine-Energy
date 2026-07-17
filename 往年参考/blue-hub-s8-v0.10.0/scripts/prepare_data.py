"""Generate and validate deterministic Phase 0 time-series data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blue_hub import __version__
from blue_hub.provenance import configuration_hash
from blue_hub.synthetic import generate_synthetic_timeseries
from blue_hub.validation import validate_timeseries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--start", default="2026-01-01 00:00:00+08:00")
    parser.add_argument("--scenario-id", default="base")
    parser.add_argument("--output", type=Path, default=Path("data/processed/timeseries_24h.csv"))
    args = parser.parse_args()

    settings = {
        "generator": "deterministic_synthetic_v1",
        "hours": args.hours,
        "start": args.start,
        "scenario_id": args.scenario_id,
        "model_version": __version__,
    }
    frame = generate_synthetic_timeseries(
        hours=args.hours,
        start=args.start,
        scenario_id=args.scenario_id,
    )
    validate_timeseries(frame).raise_if_invalid()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata = {**settings, "configuration_hash": configuration_hash(settings)}
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(frame)} validated hourly rows to {args.output} with metadata {metadata_path}"
    )


if __name__ == "__main__":
    main()
