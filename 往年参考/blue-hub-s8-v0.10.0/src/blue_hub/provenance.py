"""Deterministic configuration fingerprints for reproducible artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def configuration_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible configuration."""
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
