"""Optional method-level timing for planner performance comparisons.

The profiler only wraps existing callables and records their wall-clock time.
It does not change planner arguments, return values, state, or call order.
"""

from __future__ import annotations

from collections import defaultdict
import functools
import importlib
import threading
import time


class PlannerStageProfiler:
    """Collect comparable OpenCDA and ROS timings for selected planner methods."""

    def __init__(self, enabled=False, external_recorder=None):
        self.enabled = bool(enabled)
        self.external_recorder = external_recorder
        self._totals_ms = defaultdict(float)
        self._counts = defaultdict(int)
        self._maximum_ms = defaultdict(float)
        self._wrapped = set()
        self._lock = threading.RLock()
        self._local = threading.local()

    def _record(self, label, duration_ms):
        if not self.enabled:
            return
        label = str(label)
        duration_ms = max(0.0, float(duration_ms))
        with self._lock:
            self._totals_ms[label] += duration_ms
            self._counts[label] += 1
            self._maximum_ms[label] = max(float(self._maximum_ms[label]), duration_ms)
        if self.external_recorder is not None:
            self.external_recorder(label, duration_ms)

    def wrap_method(self, owner, method_name, label):
        """Time one existing bound method without changing what it does."""
        if not self.enabled or owner is None:
            return False
        key = (id(owner), str(method_name))
        if key in self._wrapped:
            return False
        original = getattr(owner, str(method_name), None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def timed_method(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._record(str(label), (time.perf_counter() - started) * 1000.0)

        try:
            setattr(owner, str(method_name), timed_method)
        except Exception:
            return False
        self._wrapped.add(key)
        return True

    def wrap_module_function(self, module, function_name, label):
        """Time one existing module function without changing its behavior."""
        if not self.enabled or module is None:
            return False
        key = (id(module), str(function_name))
        if key in self._wrapped:
            return False
        original = getattr(module, str(function_name), None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def timed_function(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._record(str(label), (time.perf_counter() - started) * 1000.0)

        setattr(module, str(function_name), timed_function)
        self._wrapped.add(key)
        return True

    def instrument_global_planner(self, planner):
        """Measure the main custom-map lookup and stored-route operations."""
        for method_name, label in (
            ("get_waypoint", "profile.global_planner.nearest_waypoint_lookup"),
            ("get_waypoint_candidates", "profile.global_planner.waypoint_candidate_lookup"),
            ("get_local_lane_context", "profile.global_planner.local_lane_context"),
            ("get_local_lane_graph", "profile.global_planner.local_lane_graph"),
            ("get_current_route_info", "profile.global_planner.current_route_info"),
            ("_nearest_stored_route_index", "profile.global_planner.route_index_search"),
            ("plan_route_from_locations", "profile.global_planner.plan_route"),
        ):
            self.wrap_method(planner, method_name, label)

    def instrument_reference_map(self, reference_map):
        """Measure the waypoint lookup used by local reference generation."""
        self.wrap_method(reference_map, "get_waypoint", "profile.reference_map.waypoint_lookup")

    def instrument_route_manager(self, route_manager):
        """Measure route progress and geometry extraction operations."""
        for method_name, label in (
            ("geometry_route_points", "profile.route_manager.geometry_extraction"),
            ("route_points", "profile.route_manager.route_points_extraction"),
            ("get_route_info", "profile.route_manager.route_info"),
            ("sync_carla_route_progress", "profile.route_manager.route_progress_sync"),
            ("upcoming_turn", "profile.route_manager.upcoming_turn_search"),
            ("carla_route_alignment", "profile.route_manager.route_alignment"),
        ):
            self.wrap_method(route_manager, method_name, label)

    def instrument_reference_generator(self, generator):
        """Measure the main reference geometry and interpolation methods."""
        for method_name, label in (
            ("_current_lane_center_reference_samples", "profile.reference.lane_center_generation"),
            ("_ego_anchored_lane_recovery_reference_samples", "profile.reference.lane_recovery_generation"),
            ("_ego_anchored_turn_reference_samples", "profile.reference.turn_generation"),
            ("_route_aligned_reference_samples", "profile.reference.route_aligned_generation"),
            ("_smooth_reference_polyline_samples", "profile.reference.polyline_smoothing"),
            ("_interpolate_dense_pose_at_arc", "profile.reference.dense_pose_interpolation"),
            ("_interpolate_corridor_geometry", "profile.reference.corridor_interpolation"),
            ("_interpolate_corridor_geometry_at", "profile.reference.corridor_interpolation_at"),
        ):
            self.wrap_method(generator, method_name, label)

    def instrument_reference_pipeline(self, pipeline):
        """Measure candidate conditioning and final reference validation."""
        if pipeline is None or "component_interfaces" in str(pipeline.__class__.__module__):
            return
        self.wrap_method(pipeline, "condition", "profile.reference_pipeline.condition")
        self.wrap_method(pipeline, "finalize", "profile.reference_pipeline.finalize")

    def instrument_bridge(self, bridge):
        """Measure bridge-owned reference work and install shared function timers."""
        if bridge is None:
            return
        self.instrument_reference_generator(getattr(bridge, "reference_generator", None))
        self.instrument_reference_pipeline(getattr(bridge, "reference_pipeline", None))
        self.wrap_method(bridge, "_active_global_route_points", "profile.route_manager.active_route_geometry")
        self.wrap_method(bridge, "_lock_route_tracking_lane_change_reference", "profile.reference.lane_change_lock_generation")
        generator = getattr(bridge, "reference_generator", None)
        generator_module = str(getattr(getattr(generator, "__class__", None), "__module__", ""))
        package_prefix = generator_module.split(".pipeline.reference_generator", 1)[0]
        if not package_prefix or package_prefix == generator_module:
            return
        try:
            temp_destination = importlib.import_module(package_prefix + ".behavior_planner.temp_destination")
        except Exception:
            return
        for function_name, label in (
            ("project_ego_to_route", "profile.reference.route_projection"),
            ("get_lookahead_route_point", "profile.reference.route_interpolation"),
            ("_build_route_reference_samples_from_anchor_impl", "profile.reference.route_geometry_generation"),
            ("_build_lane_reference_samples_to_target", "profile.reference.lane_target_generation"),
            ("_blend_reference_samples", "profile.reference.lane_change_blending"),
            ("_build_reference_samples_impl", "profile.reference.reference_samples_generation"),
        ):
            self.wrap_module_function(temp_destination, function_name, label)

    def instrument_mpc(self, mpc):
        """Separate candidate-probe solver time from the committed final solve."""
        if not self.enabled or mpc is None:
            return
        plan_key = (id(mpc), "plan_trajectory")
        original_plan = getattr(mpc, "plan_trajectory", None)
        if callable(original_plan) and plan_key not in self._wrapped:
            @functools.wraps(original_plan)
            def timed_plan(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_plan(*args, **kwargs)
                finally:
                    in_probe = int(getattr(self._local, "probe_depth", 0)) > 0
                    label = "profile.mpc.probe_solver" if in_probe else "profile.mpc.final_solver"
                    self._record(label, (time.perf_counter() - started) * 1000.0)
            setattr(mpc, "plan_trajectory", timed_plan)
            self._wrapped.add(plan_key)

        probe_key = (id(mpc), "probe_trajectory_feasibility")
        original_probe = getattr(mpc, "probe_trajectory_feasibility", None)
        if callable(original_probe) and probe_key not in self._wrapped:
            @functools.wraps(original_probe)
            def timed_probe(*args, **kwargs):
                started = time.perf_counter()
                self._local.probe_depth = int(getattr(self._local, "probe_depth", 0)) + 1
                try:
                    return original_probe(*args, **kwargs)
                finally:
                    self._local.probe_depth = max(0, int(getattr(self._local, "probe_depth", 1)) - 1)
                    self._record("profile.mpc.probe_total", (time.perf_counter() - started) * 1000.0)
            setattr(mpc, "probe_trajectory_feasibility", timed_probe)
            self._wrapped.add(probe_key)

    def summary(self, planning_cycle_count):
        """Return averages per call and per completed planning cycle."""
        cycle_count = max(1, int(planning_cycle_count))
        with self._lock:
            result = {}
            for label in sorted(self._counts):
                count = int(self._counts[label])
                total_ms = float(self._totals_ms[label])
                result[label] = {
                    "call_count": count,
                    "average_ms_per_call": total_ms / max(1, count),
                    "average_ms_per_planning_cycle": total_ms / cycle_count,
                    "maximum_ms": float(self._maximum_ms[label]),
                }
            return result

