"""Persistent maneuver ownership between behavior and reference sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence


_LANE_CHANGE = {"lane_change_left", "lane_change_right"}
_TURN = {"intersection_turn_left", "intersection_turn_right"}


@dataclass
class ManeuverPlan:
    maneuver_id: str
    maneuver_type: str
    direction: str
    phase: str
    source_lane_id: int
    target_lane_id: int
    geometry: list[dict[str, object]] = field(default_factory=list)
    progress_index: int = 0
    geometry_revision: int = 0
    reference_source: str = ""


@dataclass(frozen=True)
class ManeuverReferenceResult:
    reference_samples: list[dict[str, object]]
    destination_state: list[float]
    debug: dict[str, object]


class ManeuverManager:
    """Keep one geometric owner across lane-change/approach/turn phases.

    Behavior may change phase and SpeedPlanner may replace the velocity
    profile, but neither operation replaces the maneuver identity. Incoming
    geometry is joined to the retained path with a short C1 handoff instead
    of exposing a generator boundary directly to MPC.
    """

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        self.config = dict(config or {})
        self.active_plan: ManeuverPlan | None = None
        self._next_id = 1
        self._last_output: list[dict[str, object]] = []
        # A reference-source switch (e.g. the winning candidate flipping from
        # a full-speed to a slowed-down variant) can change the reference's
        # own commanded speed -- and therefore its forward extent/point
        # count -- by a large amount in one step. Blending only the first 8
        # points (the old default) isn't enough to hide a jump that size; it
        # just compresses the discontinuity into a shorter, sharper ramp
        # that still reads to MPC's lane-center cost as a near-step change.
        # Spreading the handoff over more points trades a slightly longer
        # transition for actually removing that step.
        self.continuous_handoff_blend_count = max(
            1,
            int(self.config.get("maneuver_continuous_handoff_blend_count", 8)),
        )

    def reset(self, reason: str = "reset") -> None:
        del reason
        self.active_plan = None
        self._last_output = []

    def retained_turn_continuation(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        target_speed_mps: float,
        count: int,
    ) -> list[dict[str, object]]:
        """Return the forward part of the currently owned turn geometry.

        ScenarioManager intentionally keeps the turn authoritative during
        TURN_EXIT_STABILIZATION.  Candidate generation can nevertheless have
        a one-frame map/footprint contract miss after the route has advanced.
        The already accepted maneuver geometry is the continuous reference
        for that transient; rebuilding a fresh route or emergency-stop path
        here changes ownership and produces a control spike.
        """

        plan = self.active_plan
        if plan is None or str(plan.maneuver_type) != "intersection_turn":
            return []
        retained, retained_index = self._forward_window(
            plan.geometry,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            count=max(2, int(count)),
            start_index=int(plan.progress_index),
        )
        if not retained:
            return []
        plan.progress_index = max(int(plan.progress_index), int(retained_index))
        result = self._apply_velocity_profile(
            retained,
            [
                {"speed_ref_mps": float(target_speed_mps)}
                for _ in retained
            ],
        )
        return [dict(sample) for sample in result]

    def update(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
        destination_state: Sequence[float],
        decision: str,
        behavior_fsm_state: str,
        current_lane_id: int,
        target_lane_id: int,
        ego_x_m: float,
        ego_y_m: float,
        reference_source: str,
        route_current_option: str = "",
        route_next_maneuver: str = "",
        stop_goal_active: bool = False,
        lane_change_commitment_active: bool | None = None,
    ) -> ManeuverReferenceResult:
        incoming = [dict(sample) for sample in list(reference_samples or [])]
        normalized_decision = str(decision or "").strip().lower()
        phase = self._phase(normalized_decision, behavior_fsm_state, stop_goal_active)
        normalized_route_next = (
            str(route_next_maneuver or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        route_advanced_to_lane_change = normalized_route_next in {
            "lane_change_left",
            "lane_change_right",
            "change_lane_left",
            "change_lane_right",
        }
        released_debug: dict[str, object] = {}
        if (
            self.active_plan is not None
            and self.active_plan.maneuver_type == "intersection_turn"
            and bool(route_advanced_to_lane_change)
            and normalized_decision not in _TURN
        ):
            released_debug = {
                "maneuver_geometry_release_reason": "route_advanced_to_lane_change",
                "maneuver_geometry_released_id": str(
                    self.active_plan.maneuver_id
                ),
            }
            self.active_plan = None
            self._last_output = []
        if (
            self.active_plan is not None
            and self.active_plan.maneuver_type in {"lane_change", "lane_change_to_turn"}
            and normalized_decision not in _LANE_CHANGE
            and lane_change_commitment_active is False
        ):
            released_debug = {
                "maneuver_geometry_release_reason": "lane_change_commitment_complete",
                "maneuver_geometry_released_id": str(
                    self.active_plan.maneuver_id
                ),
            }
            self.active_plan = None
            self._last_output = []
        direction = self._direction(
            normalized_decision,
            route_current_option,
            route_next_maneuver,
        )
        should_start = normalized_decision in _LANE_CHANGE | _TURN
        should_continue = bool(
            self.active_plan is not None
            and (
                bool(stop_goal_active)
                or phase in {"LANE_CHANGE", "STABILIZATION", "TURN", "STOP"}
                or self._route_still_requires_direction(
                    self.active_plan.direction,
                    route_current_option,
                    route_next_maneuver,
                )
            )
        )

        if self.active_plan is None and should_start and incoming:
            # "chained with a route turn" must come from the ROUTE's own
            # upcoming-maneuver hints, not from `decision` -- a plain
            # lane_change_left/right decision trivially contains "left"/
            # "right" itself, so using `direction` (which folds `decision`
            # into the same text match) here always finds a match and no
            # lane change was ever classified as the plain "lane_change"
            # type, even ones with no upcoming turn at all (e.g. a local
            # static-obstacle avoidance lane change). That silently routed
            # every lane change through the turn-chained geometry path.
            route_turn_direction = self._route_direction(
                route_current_option, route_next_maneuver
            )
            maneuver_type = (
                "lane_change_to_turn"
                if normalized_decision in _LANE_CHANGE and route_turn_direction
                else "lane_change"
                if normalized_decision in _LANE_CHANGE
                else "intersection_turn"
            )
            self.active_plan = ManeuverPlan(
                maneuver_id=f"maneuver-{self._next_id}",
                maneuver_type=maneuver_type,
                direction=direction,
                phase=phase,
                source_lane_id=int(current_lane_id),
                target_lane_id=int(target_lane_id),
                geometry=self._geometry_only(incoming),
                geometry_revision=1,
                reference_source=str(reference_source),
            )
            self._next_id += 1

        if self.active_plan is None:
            self._last_output = []
            inactive_debug = self._inactive_debug()
            inactive_debug.update(released_debug)
            return ManeuverReferenceResult(
                reference_samples=incoming,
                destination_state=list(destination_state or []),
                debug=inactive_debug,
            )

        if not should_continue and not should_start:
            completed_id = self.active_plan.maneuver_id
            self.active_plan = None
            self._last_output = []
            debug = self._inactive_debug()
            debug.update({
                "maneuver_geometry_release_reason": "route_maneuver_complete",
                "maneuver_geometry_released_id": completed_id,
            })
            return ManeuverReferenceResult(
                reference_samples=incoming,
                destination_state=list(destination_state or []),
                debug=debug,
            )

        plan = self.active_plan
        plan.phase = phase
        if int(target_lane_id):
            plan.target_lane_id = int(target_lane_id)
        retained, retained_index = self._forward_window(
            plan.geometry,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            count=max(len(incoming), 2),
            start_index=plan.progress_index,
        )
        source_changed = bool(
            str(reference_source) and str(reference_source) != plan.reference_source
        )
        window_start_index = retained_index
        if incoming:
            if retained and bool(stop_goal_active):
                geometry = retained[:len(incoming)]
            elif retained and source_changed:
                geometry = self._continuous_handoff(
                    retained,
                    incoming,
                    blend_count=int(self.continuous_handoff_blend_count),
                )
            else:
                geometry = self._geometry_only(incoming)
            plan.geometry = self._extend_geometry(geometry, incoming)
            plan.geometry_revision += int(source_changed)
            plan.reference_source = str(reference_source or plan.reference_source)
            # The rebuilt geometry's index 0 is re-anchored near ego (it is
            # built from retained[0]/incoming[0], the point nearest ego
            # found above), so the window search below must restart at 0
            # instead of inheriting retained_index, which is an index into
            # the old (pre-rebuild) array and can point past the end of a
            # shorter rebuilt one.
            window_start_index = 0

        window, window_index = self._forward_window(
            plan.geometry,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            count=max(1, len(incoming)),
            start_index=window_start_index,
        )
        plan.progress_index = int(window_index)
        if not window:
            window = self._geometry_only(incoming)
        output = self._apply_velocity_profile(window, incoming)
        destination = self._destination(output, destination_state, plan.target_lane_id)
        first_point_jump_m, first_heading_jump_deg = self._frame_jump(output)
        self._last_output = [dict(sample) for sample in output]
        return ManeuverReferenceResult(
            reference_samples=output,
            destination_state=destination,
            debug={
                "maneuver_geometry_active": True,
                "maneuver_geometry_id": str(plan.maneuver_id),
                "maneuver_geometry_type": str(plan.maneuver_type),
                "maneuver_geometry_direction": str(plan.direction),
                "maneuver_geometry_phase": str(plan.phase),
                "maneuver_geometry_revision": int(plan.geometry_revision),
                "maneuver_geometry_source_changed": bool(source_changed),
                "maneuver_geometry_owner": "ManeuverManager",
                "maneuver_geometry_point_count": int(len(plan.geometry)),
                "maneuver_first_point_jump_m": float(first_point_jump_m),
                "maneuver_first_heading_jump_deg": float(
                    first_heading_jump_deg
                ),
            },
        )

    @staticmethod
    def _inactive_debug() -> dict[str, object]:
        return {
            "maneuver_geometry_active": False,
            "maneuver_geometry_id": "",
            "maneuver_geometry_type": "",
            "maneuver_geometry_direction": "",
            "maneuver_geometry_phase": "IDLE",
            "maneuver_geometry_revision": 0,
            "maneuver_geometry_source_changed": False,
            "maneuver_geometry_owner": "",
            "maneuver_geometry_point_count": 0,
            "maneuver_first_point_jump_m": 0.0,
            "maneuver_first_heading_jump_deg": 0.0,
        }

    @staticmethod
    def _phase(decision: str, fsm: str, stop: bool) -> str:
        if stop:
            return "STOP"
        normalized_fsm = str(fsm or "").upper()
        if "STABILIZATION" in normalized_fsm:
            return "STABILIZATION"
        if decision in _LANE_CHANGE:
            return "LANE_CHANGE"
        if decision in _TURN:
            return "TURN"
        return "APPROACH"

    @staticmethod
    def _direction(decision: str, current: str, upcoming: str) -> str:
        text = " ".join((str(decision), str(current), str(upcoming))).lower()
        if "right" in text:
            return "right"
        if "left" in text:
            return "left"
        return ""

    @staticmethod
    def _route_direction(current: str, upcoming: str) -> str:
        """Return a direction only for an explicit intersection turn hint.

        ``CHANGELANERIGHT`` and ``Lane Change Right`` contain the word
        ``right`` but are lateral maneuvers, not evidence of a following
        right turn.  Treating them as turns changes a plain lane-change
        reference into ``lane_change_to_turn`` and creates a short curved
        trajectory even when the global route crosses the junction straight.
        """
        current_option = str(current or "").strip().upper().replace("_", "")
        upcoming_option = (
            str(upcoming or "")
            .strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )
        if current_option == "RIGHT":
            return "right"
        if current_option == "LEFT":
            return "left"
        if upcoming_option in {"right", "turn right", "right turn"}:
            return "right"
        if upcoming_option in {"left", "turn left", "left turn"}:
            return "left"
        return ""

    @staticmethod
    def _route_still_requires_direction(direction: str, current: str, upcoming: str) -> bool:
        if not direction:
            return False
        return direction in f"{current} {upcoming}".lower()

    @staticmethod
    def _xy(sample: Mapping[str, object]) -> tuple[float, float]:
        return (
            float(sample.get("x_ref_m", sample.get("x", 0.0))),
            float(sample.get("y_ref_m", sample.get("y", 0.0))),
        )

    @classmethod
    def _geometry_only(cls, samples: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        result = [dict(sample) for sample in list(samples or [])]
        return cls._recompute_geometry(result)

    @classmethod
    def _forward_window(
        cls,
        geometry,
        *,
        ego_x_m: float,
        ego_y_m: float,
        count: int,
        start_index: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return a forward window plus the resolved (monotonic) start index.

        The nearest-point search is bounded to a span starting at
        ``start_index`` and never resolves to an index before it, instead of
        an unconstrained global nearest-neighbor search over the whole
        geometry array. Callers must only carry the returned index forward
        into a later search over the SAME (unmodified) array; once the array
        is rebuilt/replaced, restart the search at 0 instead (a rebuilt
        array's index 0 is always re-anchored near ego by construction).
        """

        points = [dict(sample) for sample in list(geometry or [])]
        if not points:
            return [], max(0, int(start_index))
        start = min(max(0, int(start_index)), len(points) - 1)
        search_span = max(int(count) * 2, 20)
        search_end = min(len(points), start + search_span)
        nearest = min(
            range(start, search_end),
            key=lambda index: (
                cls._xy(points[index])[0] - ego_x_m
            ) ** 2 + (cls._xy(points[index])[1] - ego_y_m) ** 2,
        )
        resolved_index = max(start, nearest)
        return (
            points[resolved_index:resolved_index + max(1, int(count))],
            int(resolved_index),
        )

    @classmethod
    def _continuous_handoff(cls, retained, incoming, blend_count: int = 8):
        count = min(len(retained), len(incoming))
        blend_count = min(count, max(1, int(blend_count)))
        output: list[dict[str, object]] = []
        for index in range(len(incoming)):
            sample = dict(incoming[index])
            if index < blend_count:
                old_x, old_y = cls._xy(retained[index])
                new_x, new_y = cls._xy(sample)
                alpha = (index + 1.0) / float(blend_count + 1.0)
                sample["x_ref_m"] = (1.0 - alpha) * old_x + alpha * new_x
                sample["y_ref_m"] = (1.0 - alpha) * old_y + alpha * new_y
                sample["x"] = sample["x_ref_m"]
                sample["y"] = sample["y_ref_m"]
            output.append(sample)
        return cls._recompute_geometry(output)

    @classmethod
    def _extend_geometry(cls, base, incoming):
        result = [dict(sample) for sample in list(base or [])]
        source = [dict(sample) for sample in list(incoming or [])]
        if result and source:
            # Continue after the incoming point nearest the retained
            # endpoint.  Iterating from source[0] appends the whole path a
            # second time whenever base already contains incoming, creating
            # a terminal->start loop and an artificial ~180 degree heading
            # reversal once the forward window reaches that seam.
            lx, ly = cls._xy(result[-1])
            nearest = min(
                range(len(source)),
                key=lambda index: (
                    cls._xy(source[index])[0] - lx
                ) ** 2 + (cls._xy(source[index])[1] - ly) ** 2,
            )
            source = source[nearest + 1 :]
        for sample in source:
            if not result:
                result.append(dict(sample))
                continue
            x, y = cls._xy(sample)
            lx, ly = cls._xy(result[-1])
            if math.hypot(x - lx, y - ly) >= 0.20:
                result.append(dict(sample))
        return cls._recompute_geometry(result)

    @classmethod
    def _recompute_geometry(cls, samples):
        result = [dict(sample) for sample in list(samples or [])]
        for index, sample in enumerate(result):
            if len(result) == 1:
                heading = float(sample.get("heading_rad", 0.0))
            else:
                first = result[max(0, index - 1)]
                second = result[min(len(result) - 1, index + 1)]
                x0, y0 = cls._xy(first)
                x1, y1 = cls._xy(second)
                heading = math.atan2(y1 - y0, x1 - x0)
            sample["heading_rad"] = float(heading)
            sample["psi_ref"] = float(heading)
        return result

    @staticmethod
    def _apply_velocity_profile(geometry, incoming):
        result = [dict(sample) for sample in list(geometry or [])]
        source = list(incoming or [])
        # `result` is the persisted geometry window (may span many ticks
        # unchanged); `source` is this tick's freshly planned speed
        # profile, which is very often shorter (e.g. speed_planner only
        # extends a few points ahead) or simply a different length than
        # the geometry window. Plain index alignment (source[i]) then
        # silently repeats source's LAST speed value for every geometry
        # point beyond len(source), regardless of how far along the
        # locked maneuver that point actually is -- decoupling commanded
        # speed from lateral progress (e.g. a stale following-cap speed
        # from early in a lane change getting stamped onto points already
        # well into the target lane). Match by each sample's own
        # `lane_change_progress` instead when both sides carry it: that
        # tag is a physical/time position along the maneuver shared by
        # both arrays, not an incidental array offset.
        source_has_progress = any("lane_change_progress" in item for item in source)
        for index, sample in enumerate(result):
            if not source:
                speed = 0.0
            elif source_has_progress and "lane_change_progress" in sample:
                target_progress = float(sample.get("lane_change_progress", 0.0) or 0.0)
                speed_sample = min(
                    source,
                    key=lambda item: abs(
                        float(item.get("lane_change_progress", 0.0) or 0.0)
                        - target_progress
                    ),
                )
                speed = float(speed_sample.get(
                    "speed_ref_mps",
                    speed_sample.get("v_ref_mps", speed_sample.get("speed_mps", 0.0)),
                ) or 0.0)
            else:
                speed_sample = source[min(index, len(source) - 1)]
                speed = float(speed_sample.get(
                    "speed_ref_mps",
                    speed_sample.get("v_ref_mps", speed_sample.get("speed_mps", 0.0)),
                ) or 0.0)
            sample["speed_ref_mps"] = speed
            sample["v_ref_mps"] = speed
            sample["speed_mps"] = speed
        return result

    @classmethod
    def _destination(cls, reference, fallback, lane_id):
        if not reference:
            return list(fallback or [])
        terminal = reference[-1]
        x, y = cls._xy(terminal)
        speed = float(terminal.get("speed_ref_mps", 0.0) or 0.0)
        return [x, y, speed, float(terminal.get("heading_rad", 0.0)), int(lane_id)]

    def _frame_jump(self, output) -> tuple[float, float]:
        if not output or not self._last_output:
            return 0.0, 0.0
        x, y = self._xy(output[0])
        old_x, old_y = self._xy(self._last_output[0])
        heading = float(output[0].get("heading_rad", 0.0))
        old_heading = float(self._last_output[0].get("heading_rad", 0.0))
        heading_delta = math.atan2(
            math.sin(heading - old_heading),
            math.cos(heading - old_heading),
        )
        return math.hypot(x - old_x, y - old_y), abs(math.degrees(heading_delta))
