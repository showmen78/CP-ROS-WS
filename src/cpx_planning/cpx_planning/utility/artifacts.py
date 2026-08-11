"""Small artifact-writing helpers shared by planning diagnostics."""

from __future__ import annotations

import csv
import json
import os
from typing import Mapping, Sequence


def ensure_artifact_dir(path: str) -> str:
    """Create and return an artifact directory path."""
    os.makedirs(str(path), exist_ok=True)
    return str(path)


def write_json_artifact(path: str, payload: Mapping[str, object]) -> str:
    """Write a deterministic JSON artifact and return its path."""
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(path)


def write_dict_csv_artifact(
    path: str,
    *,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> str:
    """Write rows to CSV while preserving an explicit field order."""
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    ordered_fields = [str(field) for field in list(fieldnames)]
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow({field: row.get(field, "") for field in ordered_fields})
    return str(path)
