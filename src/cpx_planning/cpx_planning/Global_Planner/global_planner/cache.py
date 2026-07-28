"""Persistent cache helpers for the OpenDRIVE global planner."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


CACHE_VERSION = 1


def compute_xodr_signature(xodr_path: Path) -> Dict[str, Any]:
    """Return a stable content and file-metadata signature for an XODR map."""
    path = Path(xodr_path).expanduser().resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1.0e9))),
        "sha256": digest.hexdigest(),
    }


def get_cache_paths(
    xodr_path: Path,
    cache_root: Path,
    signature: Mapping[str, Any],
) -> Dict[str, Path]:
    """Create and return all cache paths associated with one map signature."""
    map_name = Path(xodr_path).stem
    signature_hash = str(signature.get("sha256", ""))[:12]
    cache_dir = Path(cache_root).expanduser().resolve() / f"{map_name}_{signature_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    adm_file = cache_dir / "map.adm"
    return {
        "cache_dir": cache_dir,
        "adm_file": adm_file,
        "adm_config_file": Path(str(adm_file) + ".txt"),
        "metadata_file": cache_dir / "metadata.json",
        "planner_cache_file": cache_dir / "planner_cache.pkl",
    }


def load_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Load cache metadata, returning ``None`` for missing or invalid data."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def metadata_matches(
    metadata: Optional[Mapping[str, Any]],
    signature: Mapping[str, Any],
    centerline_spacing_m: float,
) -> bool:
    """Return whether metadata describes the current planner cache inputs."""
    if not isinstance(metadata, Mapping):
        return False
    try:
        spacing_matches = abs(
            float(metadata.get("centerline_spacing_m")) - float(centerline_spacing_m)
        ) <= 1.0e-9
    except (TypeError, ValueError):
        return False
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and metadata.get("xodr_signature") == dict(signature)
        and spacing_matches
    )


def load_pickle(path: Path) -> Any:
    """Load a Python planner-cache payload."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def save_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Save cache metadata as deterministic JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(dict(metadata), stream, indent=2, sort_keys=True)
        stream.write("\n")


def save_pickle(path: Path, payload: Any) -> None:
    """Save a Python planner-cache payload using the current pickle protocol."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
