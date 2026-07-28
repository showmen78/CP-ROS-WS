"""Top-level CP-X planning pipeline orchestration."""

from __future__ import annotations

from typing import Any

#from opencda.planning_module.pipeline.output import PlannerOutput

from .output import PlannerOutput

class CPXPlanningPipeline:
    """Run the full CP-X pipeline for one OpenCDA tick.

    The bridge owns OpenCDA lifecycle concerns.  This class is the explicit
    planning-module entrypoint used by the bridge:

    ``PlannerInputFrame -> behavior/reference/MPC -> PlannerOutput``.

    The first integrated version delegates the detailed stages to methods on
    ``CPXMPCPlannerBridge`` so we preserve the already-tested behavior while
    exposing the correct pipeline boundary.  The stage internals can then be
    moved here incrementally without changing OpenCDA integration.
    """

    def __init__(self, bridge: Any):
        self.bridge = bridge

    def run_step(self) -> PlannerOutput:
        return self.bridge._run_full_cpx_pipeline_step()
