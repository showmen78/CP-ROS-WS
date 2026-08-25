"""Read-only pose-to-HD-map matching and rolling-lane diagnostics.

This module deliberately has no behavior or control dependencies.  Lane
identity is produced from pose/geometry evidence; integer lane ids are output
labels, never inputs used to infer physical left/right direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


@dataclass(frozen=True)
class LaneProjectionCandidate:
    ad_lane_id: int
    road_id: int
    section_id: int
    raw_lane_id: int
    center_x_m: float
    center_y_m: float
    heading_rad: float
    lane_width_m: float
    snap_distance_m: float
    is_in_lane: bool
    probability: float
    topology_relation: str = "unknown"


@dataclass(frozen=True)
class MatchedLaneState:
    valid: bool = False
    ad_lane_id: int = 0
    road_id: int = 0
    section_id: int = 0
    raw_lane_id: int = 0
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    heading_rad: float = 0.0
    lane_width_m: float = 0.0
    lateral_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    score: float = float("inf")
    confidence: float = 0.0
    match_reason: str = "unmatched"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DiagnosticHDMapMatcher:
    """Stateful matcher used only for diagnostics during staged migration."""

    _CONTINUITY_COST = {
        "same": 0.0,
        "longitudinal": 0.05,
        "left": 0.35,
        "right": 0.35,
        "unknown": 0.65,
        "disconnected": 2.0,
    }

    def __init__(self) -> None:
        self.previous = MatchedLaneState()
        self._pending_lane_id = 0
        self._pending_lane_frames = 0

    def reset(self) -> None:
        self.previous = MatchedLaneState()
        self._pending_lane_id = 0
        self._pending_lane_frames = 0

    def update(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        candidates: Sequence[LaneProjectionCandidate],
    ) -> MatchedLaneState:
        if not candidates:
            result = MatchedLaneState(match_reason="no_hd_map_candidates")
            self.previous = result
            return result

        ranked: list[tuple[float, LaneProjectionCandidate, float, float]] = []
        for candidate in candidates:
            dx = float(ego_x_m) - float(candidate.center_x_m)
            dy = float(ego_y_m) - float(candidate.center_y_m)
            lateral = (
                -math.sin(float(candidate.heading_rad)) * dx
                + math.cos(float(candidate.heading_rad)) * dy
            )
            heading_error = _wrap_angle(
                float(ego_heading_rad) - float(candidate.heading_rad)
            )
            half_width = max(0.5, 0.5 * float(candidate.lane_width_m))
            lateral_cost = abs(float(lateral)) / half_width
            heading_cost = abs(float(heading_error)) / math.radians(45.0)
            inside_cost = 0.0 if bool(candidate.is_in_lane) else 0.75
            probability_cost = max(0.0, 1.0 - float(candidate.probability)) * 0.25
            topology_cost = self._CONTINUITY_COST.get(
                str(candidate.topology_relation),
                self._CONTINUITY_COST["unknown"],
            )
            score = (
                1.5 * lateral_cost
                + 1.0 * heading_cost
                + inside_cost
                + probability_cost
                + topology_cost
            )
            ranked.append((float(score), candidate, float(lateral), float(heading_error)))

        ranked.sort(key=lambda item: (item[0], item[1].ad_lane_id))
        best_score, best, lateral, heading_error = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else best_score + 2.0
        margin = max(0.0, float(second_score) - float(best_score))
        confidence = max(0.0, min(1.0, 0.5 * margin + (0.5 if best.is_in_lane else 0.0)))

        # At overlapping road/connector geometry AD-map can expose two
        # in-lane candidates with almost identical scores.  Do not publish a
        # low-confidence A->B transition until B remains best long enough to
        # be physical, provided A is still a valid candidate.  High-confidence
        # transitions and segment boundaries where A disappears remain
        # immediate, so this does not delay normal longitudinal progression.
        previous_rank = next(
            (
                item
                for item in ranked
                if int(item[1].ad_lane_id) == int(self.previous.ad_lane_id)
            ),
            None,
        )
        transition_pending = bool(
            self.previous.valid
            and int(best.ad_lane_id) != int(self.previous.ad_lane_id)
            and previous_rank is not None
            and float(confidence) < 0.65
        )
        if transition_pending:
            if int(self._pending_lane_id) == int(best.ad_lane_id):
                self._pending_lane_frames += 1
            else:
                self._pending_lane_id = int(best.ad_lane_id)
                self._pending_lane_frames = 1
            if int(self._pending_lane_frames) < 8:
                held_score, held, held_lateral, held_heading_error = previous_rank
                result = MatchedLaneState(
                    valid=True,
                    ad_lane_id=int(held.ad_lane_id),
                    road_id=int(held.road_id),
                    section_id=int(held.section_id),
                    raw_lane_id=int(held.raw_lane_id),
                    center_x_m=float(held.center_x_m),
                    center_y_m=float(held.center_y_m),
                    heading_rad=float(held.heading_rad),
                    lane_width_m=float(held.lane_width_m),
                    lateral_offset_m=float(held_lateral),
                    heading_error_rad=float(held_heading_error),
                    score=float(held_score),
                    confidence=float(confidence),
                    match_reason="pose_geometry_transition_hysteresis",
                )
                self.previous = result
                return result
        else:
            self._pending_lane_id = 0
            self._pending_lane_frames = 0

        result = MatchedLaneState(
            valid=True,
            ad_lane_id=int(best.ad_lane_id),
            road_id=int(best.road_id),
            section_id=int(best.section_id),
            raw_lane_id=int(best.raw_lane_id),
            center_x_m=float(best.center_x_m),
            center_y_m=float(best.center_y_m),
            heading_rad=float(best.heading_rad),
            lane_width_m=float(best.lane_width_m),
            lateral_offset_m=float(lateral),
            heading_error_rad=float(heading_error),
            score=float(best_score),
            confidence=float(confidence),
            match_reason=(
                "pose_geometry_topology_history"
                if self.previous.valid
                else "pose_geometry_initial"
            ),
        )
        self._pending_lane_id = 0
        self._pending_lane_frames = 0
        self.previous = result
        return result


def topology_relation(
    *,
    candidate_lane_id: int,
    previous_lane_id: int,
    previous_corridors: Mapping[int, Sequence[int]],
) -> str:
    """Classify a candidate relative to the previous rolling lane frame."""

    candidate = int(candidate_lane_id)
    if candidate != 0 and candidate == int(previous_lane_id):
        return "same"
    for offset, lane_ids in dict(previous_corridors or {}).items():
        if candidate not in {int(value) for value in list(lane_ids or [])}:
            continue
        if int(offset) == 0:
            return "longitudinal"
        if int(offset) > 0:
            return "left"
        return "right"
    return "unknown" if not previous_corridors else "disconnected"


def local_lane_frame_invariants(
    *,
    matched_lane_id: int,
    corridors: Mapping[int, Sequence[int]],
    target_lane_id: int = 0,
    reported_target_offset: int = 0,
) -> list[str]:
    """Return diagnostic violations without modifying planner behavior."""

    violations: list[str] = []
    current = {int(value) for value in list(corridors.get(0, []) or [])}
    if int(matched_lane_id) == 0:
        violations.append("map_match_invalid")
    elif int(matched_lane_id) not in current:
        violations.append("matched_lane_missing_from_current_corridor")
    if int(target_lane_id) != 0:
        observed_offsets = [
            int(offset)
            for offset, lane_ids in dict(corridors or {}).items()
            if int(target_lane_id) in {int(value) for value in list(lane_ids or [])}
        ]
        if not observed_offsets:
            violations.append("route_target_missing_from_local_frame")
        elif int(reported_target_offset) not in observed_offsets:
            violations.append("route_target_offset_mismatch")
    return violations
