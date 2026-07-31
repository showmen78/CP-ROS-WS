"""CARLA- and OpenCDA-independent entry point for the active CP-X pipeline."""

from __future__ import annotations

from cpx_planning.pipeline.planner_pipeline import CPXPlanningPipeline


class CPXMPCPlannerBridge(CPXPlanningPipeline):
    """Keep the authoritative OpenCDA bridge class and method names at the ROS boundary."""

    def _run_full_cpx_pipeline_step(self, adapter_output):
        """Run the full CP-X pipeline for the PlannerInputFrame received through ROS."""
        return self.run_planning_cycle(adapter_output)

    def run_step(self, adapter_output):
        """Expose the same single-step entry point while accepting the ROS adapter output explicitly."""
        return self._run_full_cpx_pipeline_step(adapter_output)
