"""
Configuration loading and merge utilities for the planning stack.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

import yaml


def load_yaml_file(path: str) -> Dict[str, Any]:
    """
    Intent:
        Load a YAML file into a Python dictionary.

    Inputs:
        path:
            str, filesystem path to a YAML file.

    Output:
        dict[str, Any], parsed YAML tree.

    Notes:
        - This helper enforces dictionary-root YAML files because the project
          configuration is organized as named sections (`mpc`, `tracker`, etc.).
    """

    with open(path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML root at {path} must be a mapping/dict.")
    return loaded


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Intent:
        Recursively merge two mapping trees.

    Method / Logic:
        - Scalars and lists in `override` replace values in `base`.
        - Nested dictionaries are merged key-by-key.
        - `base` is not modified; a new dictionary is returned.

    Inputs:
        base:
            Mapping[str, Any], default configuration tree.
        override:
            Mapping[str, Any], override tree applied on top of `base`.

    Output:
        dict[str, Any], merged configuration.
    """

    merged: Dict[str, Any] = deepcopy(dict(base))
    for key, override_value in dict(override).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(override_value, Mapping):
            merged[key] = deep_merge_dicts(merged[key], override_value)
        else:
            merged[key] = deepcopy(override_value)
    return merged
