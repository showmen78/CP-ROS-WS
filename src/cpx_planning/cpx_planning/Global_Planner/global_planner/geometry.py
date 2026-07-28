"""Coordinate and point helpers shared by the planner package."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

CARLA_TO_ENU_Y_SIGN = -1.0


def euclidean_distance(first_point: Sequence[float], second_point: Sequence[float]) -> float:
    """Return Euclidean distance without relying on Python 3.8 ``math.dist``."""
    return math.sqrt(
        sum((float(first) - float(second)) ** 2 for first, second in zip(first_point, second_point))
    )


def normalize_position(position: Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    """Convert a CARLA-style point into a plain `(x, y, z)` tuple.

    input: `position` (`dict[str, float] | list[float] | tuple[float, float, float]`)
    output: normalized point (`tuple[float, float, float]`)
    """
    if isinstance(position, Mapping):
        return (float(position["x"]), float(position["y"]), float(position["z"]))
    return (float(position[0]), float(position[1]), float(position[2]))


def as_carla_dict(position: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    """Return one CARLA-style point dictionary with `x`, `y`, and `z`.

    input: `position` (`dict[str, float] | list[float] | tuple[float, float, float]`)
    output: CARLA point (`dict[str, float]`)
    """
    x, y, z = normalize_position(position)
    return {"x": x, "y": y, "z": z}


def to_enu_tuple(position: Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    """Convert one CARLA-world point into the AD-map ENU convention.

    input: `position` (`dict[str, float] | list[float] | tuple[float, float, float]`)
    output: ENU point (`tuple[float, float, float]`)
    """
    x, y, z = normalize_position(position)
    return (x, CARLA_TO_ENU_Y_SIGN * y, z)


def to_carla_tuple(position: Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    """Convert one ENU point back into the CARLA-world convention.

    input: `position` (`dict[str, float] | list[float] | tuple[float, float, float]`)
    output: CARLA point (`tuple[float, float, float]`)
    """
    x, y, z = normalize_position(position)
    return (x, CARLA_TO_ENU_Y_SIGN * y, z)


def to_carla_dict(position: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    """Convert one ENU point into a CARLA-style dictionary.

    input: `position` (`dict[str, float] | list[float] | tuple[float, float, float]`)
    output: CARLA point (`dict[str, float]`)
    """
    x, y, z = to_carla_tuple(position)
    return {"x": x, "y": y, "z": z}


def points_are_close(
    first_point: Mapping[str, float] | Sequence[float],
    second_point: Mapping[str, float] | Sequence[float],
    tolerance: float = 0.05,
) -> bool:
    """Check whether two 3D points are nearly identical.

    input: two points (`dict[str, float] | tuple[float, float, float]`) and `tolerance` (`float`)
    output: whether the points are close (`bool`)
    """
    first_tuple = normalize_position(first_point)
    second_tuple = normalize_position(second_point)
    return all(abs(first_value - second_value) <= tolerance for first_value, second_value in zip(first_tuple, second_tuple))
