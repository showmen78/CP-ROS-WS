"""ROS owner for the unchanged CP-X MPC solver and its warm-start state."""

from __future__ import annotations

from pathlib import Path
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import yaml

from cpx_planning.MPC.mpc import MPC
from cpx_planning.component_interfaces import ComponentServer


class MPCNode(Node):
    """Serialize PROBE and FINAL calls so they use one unchanged MPC instance."""

    def __init__(self, local_bus=None):
        super().__init__("mpc_node")
        self.declare_parameter("debug", False)
        package_root = Path(__file__).resolve().parent
        with (package_root / "MPC" / "mpc.yaml").open("r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file) or {}
        mpc_config = dict(payload.get("mpc", payload))
        with (package_root / "behavior_planner" / "behavior_planner.yaml").open("r", encoding="utf-8") as config_file:
            behavior_payload = yaml.safe_load(config_file) or {}
        mpc_config["behavior_planner_runtime"] = dict(behavior_payload.get("behavior_planner_runtime", behavior_payload))
        road_config = dict(payload.get("road", {}))
        road_config.setdefault("lane_count", 3)
        road_config.setdefault("lane_width_m", 3.5)
        self.mpc = MPC(mpc_cfg=mpc_config, road_cfg=road_config)
        self._solver_lock = threading.RLock()
        self.component_server = ComponentServer(self, "/cpx/mpc/request", "/cpx/mpc/result", self.solve, local_bus=local_bus)
        self.get_logger().info("MPC node owns the unchanged MPC solver, warm start, and cost profile.")

    def solve(self, operation, payload, _cycle_id, _header):
        """Run one metadata, PROBE, FINAL, or existing MPC utility topic request."""
        with self._solver_lock:
            return self._dispatch(str(operation), payload)

    def _dispatch(self, operation, payload):
        payload = dict(payload or {})
        if operation == "metadata":
            return self._state()
        if operation == "probe":
            result = self.mpc.probe_trajectory_feasibility(**payload)
            return {"result": result, "state": self._state()}
        if operation == "final":
            result = self.mpc.plan_trajectory(**payload)
            return {"result": result, "state": self._state()}
        if operation == "apply_mode_cost_profile":
            result = self.mpc.apply_mode_cost_profile(str(payload["profile_name"]), blend_alpha=payload.get("blend_alpha"))
            return {"result": result, "state": self._state()}
        if operation == "blend_toward_horizon_s":
            result = self.mpc.blend_toward_horizon_s(*list(payload.get("args", [])), **dict(payload.get("kwargs", {})))
            return {"result": result, "state": self._state()}
        if operation == "get_runtime_status":
            return {"result": self.mpc.get_runtime_status(), "state": self._state()}
        if operation == "get_last_cost_terms":
            return {"result": self.mpc.get_last_cost_terms(), "state": self._state()}
        raise ValueError("Unsupported MPC operation: {}".format(operation))

    def _state(self):
        """Return every MPC field read by the copied bridge after a solve."""
        names = (
            "horizon_s", "dt_s", "horizon_steps", "wheelbase_m", "trajectory_generation_period_s", "lane_width_m",
            "active_cost_profile_name", "adaptive_horizon_enabled", "_last_status", "_last_u_solution", "_last_x_solution",
            "_last_solve_time_ms", "_last_cost_terms", "_last_active_max_velocity_mps",
        )
        state = {name: getattr(self.mpc, name) for name in names if hasattr(self.mpc, name)}
        state["constraints"] = dict(vars(self.mpc.constraints))
        return state


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
