"""Top-level CP-X planning pipeline orchestration."""

from __future__ import annotations

from typing import Any

from .output import PlannerOutput


class CPXPlanningPipeline:
    """Keep the same pipeline delegate used by the active OpenCDA planner."""

    def __init__(self, bridge: Any):
        self.bridge = bridge

    def run_step(self) -> PlannerOutput:
        """Run one planner cycle through the same bridge method used by OpenCDA."""
        return self.bridge._run_full_cpx_pipeline_step()
