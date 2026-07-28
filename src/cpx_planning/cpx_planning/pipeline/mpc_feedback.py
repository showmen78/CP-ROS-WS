"""Behavior-layer memory of recent MPC feasibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class MPCFeedbackRecord:
    decision: str
    target_lane_id: int
    status: str
    reason: str
    timestamp_s: float
    consecutive_failures: int


class BehaviorMPCFeedback:
    """Short-term feedback from MPC solve status into behavior candidates."""

    def __init__(
        self,
        *,
        hold_s: float = 1.5,
        min_failures: int = 1,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.hold_s = max(0.0, float(hold_s))
        self.min_failures = max(1, int(min_failures))
        self._records: Dict[str, MPCFeedbackRecord] = {}
        self._last_key = ""

    def candidate_feedback(self, *, current_time_s: float) -> Dict[str, object]:
        if not self.enabled:
            return {"blocked_lane_ids": [], "summary": "mpc_feedback_disabled"}
        self._expire(current_time_s=float(current_time_s))
        blocked = [
            int(record.target_lane_id)
            for record in self._records.values()
            if int(record.consecutive_failures) >= int(self.min_failures)
        ]
        blocked = sorted(set(blocked))
        if not blocked:
            return {"blocked_lane_ids": [], "summary": "mpc_feedback_clear"}
        return {
            "blocked_lane_ids": blocked,
            "summary": "mpc_feedback_blocked_lanes:" + ",".join(str(v) for v in blocked),
        }

    def record_result(
        self,
        *,
        decision: str,
        target_lane_id: int,
        status: str,
        reason: str,
        timestamp_s: float,
        success: bool,
    ) -> str:
        if not self.enabled:
            return "mpc_feedback_disabled"
        key = self._key(decision=decision, target_lane_id=int(target_lane_id))
        self._last_key = str(key)
        if bool(success):
            self._records.pop(key, None)
            return f"mpc_feedback_success:{key}"
        previous = self._records.get(key)
        failures = 1 if previous is None else int(previous.consecutive_failures) + 1
        self._records[key] = MPCFeedbackRecord(
            decision=str(decision),
            target_lane_id=int(target_lane_id),
            status=str(status),
            reason=str(reason),
            timestamp_s=float(timestamp_s),
            consecutive_failures=int(failures),
        )
        return f"mpc_feedback_failure:{key}:count={int(failures)}"

    @property
    def active_records(self) -> List[Dict[str, object]]:
        return [
            {
                "decision": record.decision,
                "target_lane_id": int(record.target_lane_id),
                "status": record.status,
                "reason": record.reason,
                "timestamp_s": float(record.timestamp_s),
                "consecutive_failures": int(record.consecutive_failures),
            }
            for record in self._records.values()
        ]

    def _expire(self, *, current_time_s: float) -> None:
        if self.hold_s <= 0.0:
            self._records.clear()
            return
        expired = [
            key
            for key, record in self._records.items()
            if float(current_time_s) - float(record.timestamp_s) > float(self.hold_s)
        ]
        for key in expired:
            self._records.pop(key, None)

    @staticmethod
    def _key(*, decision: str, target_lane_id: int) -> str:
        return f"{str(decision).strip().lower()}:{int(target_lane_id)}"
