"""
Shared custom-map lane-center waypoint graph helpers.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping

import numpy as np


INVALID_LANE_ID = 0


def direction_key(raw_lane_id: int) -> str:
    return "positive" if int(raw_lane_id) > 0 else "negative"


def round_xy(x_m: float, y_m: float) -> tuple[float, float]:
    return round(float(x_m), 3), round(float(y_m), 3)


def custom_waypoint_graph_key(waypoint) -> tuple[int, int, int, float]:
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        round(float(getattr(waypoint, "s", 0.0)), 3),
    )


def _custom_waypoint_group_key(waypoint) -> tuple[int, int, str]:
    return int(waypoint.road_id), int(waypoint.section_id), direction_key(int(waypoint.lane_id))


def is_driving_waypoint(waypoint) -> bool:
    if waypoint is None:
        return False
    lane_type = getattr(waypoint, "lane_type", None)
    if lane_type is None:
        return True
    lane_type_name = getattr(lane_type, "name", lane_type)
    normalized_name = str(lane_type_name).strip().upper()
    return normalized_name.endswith("DRIVING")


def _same_lane_group(base_waypoint, candidate_waypoint) -> bool:
    if base_waypoint is None or candidate_waypoint is None:
        return False
    if not is_driving_waypoint(candidate_waypoint):
        return False

    base_lane_id = int(getattr(base_waypoint, "lane_id", 0))
    candidate_lane_id = int(getattr(candidate_waypoint, "lane_id", 0))
    if base_lane_id == 0 or candidate_lane_id == 0:
        return False
    if base_lane_id * candidate_lane_id < 0:
        return False

    # Only road_id is checked — NOT section_id.
    # In custom-map/OpenDRIVE, parallel lanes on the same physical road commonly
    # have different section_ids because lane sections can start at different
    # longitudinal offsets (e.g. when a road widens or narrows).  Requiring
    # section_id equality causes right()/left() neighbours
    # to be rejected, making every lane appear to be the only lane in its
    # group → canonical_lane_id_for_waypoint() returns 1 for every lane on
    # that road, so optimal_lane_id always equals ego_lane_id even when the
    # global route goes through a different lane.
    return (
        int(getattr(base_waypoint, "road_id", 0)) == int(getattr(candidate_waypoint, "road_id", 0))
    )


def canonical_lane_waypoints(waypoint) -> List[object]:
    if waypoint is None:
        return []

    rightmost_waypoint = waypoint
    while True:
        right = getattr(rightmost_waypoint, "right", None)
        if not callable(right):
            break
        right_waypoint = right()
        if not _same_lane_group(waypoint, right_waypoint):
            break
        rightmost_waypoint = right_waypoint

    lanes: List[object] = [rightmost_waypoint]
    current_waypoint = rightmost_waypoint
    while True:
        left = getattr(current_waypoint, "left", None)
        if not callable(left):
            break
        left_waypoint = left()
        if not _same_lane_group(waypoint, left_waypoint):
            break
        lanes.append(left_waypoint)
        current_waypoint = left_waypoint
    return lanes


def canonical_lane_ids_for_waypoint(waypoint) -> List[int]:
    lane_waypoints = canonical_lane_waypoints(waypoint)
    return [
        int(index) + 1
        for index, lane_waypoint in enumerate(lane_waypoints)
        if int(getattr(lane_waypoint, "lane_id", 0)) != int(INVALID_LANE_ID)
    ]


def canonical_lane_id_for_waypoint(waypoint) -> int:
    if waypoint is None:
        return int(INVALID_LANE_ID)
    raw_lane_id = int(getattr(waypoint, "lane_id", INVALID_LANE_ID))
    if int(raw_lane_id) == int(INVALID_LANE_ID):
        return int(INVALID_LANE_ID)
    lane_waypoints = canonical_lane_waypoints(waypoint)
    for lane_index, lane_waypoint in enumerate(lane_waypoints):
        if int(getattr(lane_waypoint, "lane_id", INVALID_LANE_ID)) == int(raw_lane_id):
            return int(lane_index) + 1
    return int(INVALID_LANE_ID)


def _lane_identity_key(waypoint) -> tuple[int, int]:
    return (
        int(getattr(waypoint, "road_id", 0)),
        int(getattr(waypoint, "lane_id", 0)),
    )


def lane_hop_offset(from_waypoint, to_waypoint, max_hops: int = 8) -> int | None:
    """Count the signed ``left()``/``right()`` hops from
    ``from_waypoint`` to the same physical lane as ``to_waypoint``.

    ``canonical_lane_id_for_waypoint`` numbers lanes by counting how many
    driving lanes exist AT THE QUERIED POINT (rightmost = 1, increasing
    leftward). That count is only meaningful locally: wherever the total
    lane count changes along a road (a lane merges away, a turn-only lane
    appears/disappears), the same physical lane a vehicle never left gets a
    different number before and after. Comparing two such numbers computed
    at different points along a route -- e.g. "is the vehicle's current
    lane the same as the lane the route requires it to reach?" -- silently
    breaks in that case: the numbers can differ, or coincidentally match,
    without either reflecting whether the vehicle actually changed lanes.

    This instead proves the relationship by walking real lane adjacency, so
    it stays correct regardless of how the lane count changes in between.
    Matching is by ``(road_id, lane_id)`` rather than waypoint identity or
    position, since ``left()``/``right()`` neighbours can
    land at a different ``section_id``/longitudinal offset than the target
    waypoint (see ``_same_lane_group``).

    Returns ``None`` if the two waypoints are not connected within
    ``max_hops`` lane-to-lane steps (e.g. they are on different roads, such
    as across a turn onto a cross street, or a respawn/teleport) -- callers
    should fall back to a fresh, position-only lane count in that case,
    since there is no meaningful lane identity to carry across an
    unconnected jump.
    """

    if from_waypoint is None or to_waypoint is None:
        return None
    target_key = _lane_identity_key(to_waypoint)
    if target_key == _lane_identity_key(from_waypoint):
        return 0

    left_cursor = from_waypoint
    for hop in range(1, int(max_hops) + 1):
        left = getattr(left_cursor, "left", None)
        if not callable(left):
            break
        candidate = left()
        if not _same_lane_group(from_waypoint, candidate):
            break
        left_cursor = candidate
        if _lane_identity_key(candidate) == target_key:
            return int(hop)

    right_cursor = from_waypoint
    for hop in range(1, int(max_hops) + 1):
        right = getattr(right_cursor, "right", None)
        if not callable(right):
            break
        candidate = right()
        if not _same_lane_group(from_waypoint, candidate):
            break
        right_cursor = candidate
        if _lane_identity_key(candidate) == target_key:
            return -int(hop)

    return None


class StableLaneIdTracker:
    """Track a vehicle's canonical lane id continuously across ticks.

    See ``lane_hop_offset`` for why a freshly recomputed
    ``canonical_lane_id_for_waypoint`` on every tick is unstable. This
    tracker only changes its reported id when it can prove, via real lane
    adjacency, that the vehicle's raw lane actually changed -- so the id
    stays a stable identity for the whole time the vehicle occupies the
    same physical lane, independent of how many lanes exist locally.
    """

    def __init__(self) -> None:
        self._lane_id: int | None = None
        self._waypoint = None

    def update(self, waypoint) -> int:
        if waypoint is None:
            return int(self._lane_id or 1)
        if self._waypoint is not None and self._lane_id is not None:
            hop = lane_hop_offset(self._waypoint, waypoint)
            if hop is not None:
                self._lane_id = int(self._lane_id) + int(hop)
                self._waypoint = waypoint
                return int(self._lane_id)
        # First update, or the vehicle's raw lane is not connected to the
        # previously tracked one within a few hops (e.g. it just completed
        # a turn onto a cross street, or was respawned/teleported): there is
        # no continuity to preserve, so start fresh from a local count.
        fresh_id = int(canonical_lane_id_for_waypoint(waypoint)) or 1
        self._lane_id = int(fresh_id)
        self._waypoint = waypoint
        return int(self._lane_id)

    def reset(self) -> None:
        self._lane_id = None
        self._waypoint = None


def canonical_lane_waypoint_for_lane_id(waypoint, target_lane_id: int):
    lane_waypoints = canonical_lane_waypoints(waypoint)
    if len(lane_waypoints) == 0:
        return waypoint
    lane_index = int(target_lane_id) - 1
    if 0 <= int(lane_index) < len(lane_waypoints):
        return lane_waypoints[int(lane_index)]
    return waypoint


def raw_opendrive_lane_id_for_waypoint(waypoint) -> int:
    if waypoint is None:
        return int(INVALID_LANE_ID)
    return int(getattr(waypoint, "lane_id", INVALID_LANE_ID))


def _internal_lane_id(waypoint, lane_ids_by_group: Mapping[tuple[int, int, str], set[int]]) -> int:
    del lane_ids_by_group
