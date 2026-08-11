"""Top-level CP-X planning pipeline orchestration."""

from __future__ import annotations

from typing import Any

from cpx_planning.pipeline.output import PlannerOutput


class CPXPlanningPipeline:
    """Run the full CP-X pipeline for one OpenCDA tick.

    The bridge owns OpenCDA lifecycle concerns.  This class is the explicit
    planning-module entrypoint used by the bridge:

    ``PlannerInputFrame -> behavior/reference/MPC -> PlannerOutput``.

    OpenCDA lifecycle details stay behind the bridge's public
    ``execute_planning_pipeline`` port.  No pipeline code calls bridge-private
    methods.
    """

    def __init__(self, bridge: Any):
        self.bridge = bridge

    def run_step(self) -> PlannerOutput:
        return self.bridge.execute_planning_pipeline()
