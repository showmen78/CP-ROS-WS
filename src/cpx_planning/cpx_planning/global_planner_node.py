"""The only ROS process that owns AD-map, the custom planner, and route state."""

from __future__ import annotations

import os
from pathlib import Path
import threading

from cpx_interfaces.msg import GlobalRoute, RoutePoint
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cpx_planning.component_interfaces import ComponentServer, RemoteWaypoint, encode_json, load_planner_configuration
from cpx_planning.pipeline.route_manager import CPXRouteManager
from cpx_planning.utility.global_planner import CustomGlobalPlannerAdapter, world_heading_rad


class GlobalPlannerNode(Node):
    """Answer map and route requests while keeping one shared route lifecycle."""

    def __init__(self, local_bus=None):
        super().__init__("global_planner_node")
        package_root = Path(__file__).resolve().parent
        self.declare_parameter("xodr_path", str(package_root / "Global_Planner" / "maps" / "Town06.xodr"))
        self.declare_parameter("cache_root", os.environ.get("CPX_GLOBAL_PLANNER_CACHE_ROOT", str(Path.home() / ".cache" / "cpx_planning" / "global_planner")))
        self.declare_parameter("ad_map_install_root", os.environ.get("GLOBAL_PLANNER_AD_MAP_INSTALL", ""))
        self.declare_parameter("route_sample_distance_m", 2.0)
        self.declare_parameter("debug", False)
        xodr_path = Path(str(self.get_parameter("xodr_path").value)).expanduser().resolve()
        if not xodr_path.is_file():
            raise FileNotFoundError("OpenDRIVE map not found: {}".format(xodr_path))
        ad_map_install_root = str(self.get_parameter("ad_map_install_root").value).strip()
        self.map_planner = CustomGlobalPlannerAdapter(xodr_path=str(xodr_path), cache_root=str(self.get_parameter("cache_root").value), route_sample_distance_m=float(self.get_parameter("route_sample_distance_m").value), ad_map_install_root=ad_map_install_root or None)
        self.map_planner.load()
        self.config = load_planner_configuration(package_root)
        self.config["debug"] = bool(self.get_parameter("debug").value)
        self.route_manager = CPXRouteManager(
            global_planner=self.map_planner,
            carla_map=None,
            carla_api=None,
            carla_route_sampling_resolution_m=float(self.config.get("carla_route_sampling_resolution_m", 1.0)),
            carla_reference_smoothing_passes=int(self.config.get("carla_reference_smoothing_passes", 3)),
            carla_turn_connector_smoothing_passes=int(self.config.get("carla_turn_connector_smoothing_passes", 16)),
            carla_reference_boundary_aware=bool(self.config.get("carla_reference_boundary_aware", True)),
            carla_reference_vehicle_half_width_m=float(self.config.get("reference_vehicle_half_width_m", 1.0)),
            carla_reference_boundary_margin_m=float(self.config.get("reference_contract_turn_boundary_margin_m", 0.15)),
            carla_reference_tracking_reserve_m=float(self.config.get("carla_reference_tracking_reserve_m", 0.20)),
            carla_rejoin_min_lateral_m=float(self.config.get("carla_rejoin_min_lateral_m", 0.35)),
            carla_rejoin_max_lateral_m=float(self.config.get("carla_rejoin_max_lateral_m", 3.0)),
            carla_rejoin_distance_m=float(self.config.get("carla_rejoin_distance_m", 8.0)),
            reached_distance_m=float(self.config.get("route_reached_distance_m", 3.0)),
            stale_route_lateral_m=float(self.config.get("route_stale_lateral_m", 12.0)),
        )
        self._state_lock = threading.RLock()
        self._cycle_cache = {}
        route_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.route_publisher = self.create_publisher(GlobalRoute, "/cpx/planning/global_route", route_qos)
        self.component_server = ComponentServer(self, "/cpx/global_planner/request", "/cpx/global_planner/result", self.component_call, local_bus=local_bus)
        self.get_logger().info("Global planner node loaded AD-map and owns all map and route state.")

    def component_call(self, operation, payload, cycle_id, header):
        """Dispatch the existing custom-planner and route-manager method names for one topic request."""
        cache_key = (int(cycle_id), str(operation), encode_json(payload))
        cacheable = str(operation) in {"get_waypoint", "get_waypoint_candidates", "get_local_lane_context", "get_local_lane_graph", "waypoint_left", "waypoint_right", "waypoint_next", "waypoint_previous"}
        with self._state_lock:
            if cacheable and cache_key in self._cycle_cache:
                return self._cycle_cache[cache_key]
            result = self._dispatch(str(operation), payload)
            if cacheable and int(cycle_id) > 0:
                self._cycle_cache = {key: value for key, value in self._cycle_cache.items() if key[0] == int(cycle_id)}
                self._cycle_cache[cache_key] = result
            if str(operation) == "route_manager_call" and str(dict(payload or {}).get("name", "")) in {"set_destination", "ensure_route", "reroute"} and result is not None:
                self.route_publisher.publish(self._route_message(result, int(cycle_id), header))
            return result

    def _dispatch(self, operation, payload):
        """Keep all simulator-independent map calls inside this process."""
        payload = dict(payload or {})
        if operation == "get_waypoint":
            return self._remote_waypoint(self.map_planner.get_waypoint(payload["point"]))
        if operation == "get_local_lane_context":
            return self.map_planner.get_local_lane_context(**payload)
        if operation == "get_waypoint_candidates":
            return self.map_planner.get_waypoint_candidates(payload["point"])
        if operation == "get_local_lane_graph":
            return self.map_planner.get_local_lane_graph(**payload)
        if operation == "plan_route_from_locations":
            return self.map_planner.plan_route_from_locations(**payload)
        if operation == "trace_route":
            return self.map_planner.trace_route(*list(payload.get("args", [])), **dict(payload.get("kwargs", {})))
        if operation == "get_current_route_info":
            return self.map_planner.get_current_route_info(**payload)
        if operation == "block_ad_lane_id":
            return self.map_planner.block_ad_lane_id(int(payload["ad_lane_id"]))
        if operation == "block_lane_at_position":
            return self.map_planner.block_lane_at_position(payload["position"])
        if operation.startswith("waypoint_"):
            waypoint = self._waypoint(payload["waypoint"])
            method_name = operation[len("waypoint_"):]
            method = getattr(waypoint, method_name)
            if method_name in {"next", "previous"}:
                distance_m = float(payload["distance_m"])
                return self._remote_step_results(list(method(distance_m) or []), method_name, distance_m)
            return self._remote_waypoint(method())
        if operation == "route_manager_property":
            return getattr(self.route_manager, str(payload["name"]))
        if operation == "route_manager_call":
            method = getattr(self.route_manager, str(payload["name"]))
            return method(*list(payload.get("args", [])), **dict(payload.get("kwargs", {})))
        raise ValueError("Unsupported global-planner operation: {}".format(operation))

    def _waypoint(self, data):
        """Map-match a serialized remote waypoint back to the AD-map waypoint object."""
        waypoint = self.map_planner.get_waypoint(dict(data).get("position", {}))
        if waypoint is None:
            raise RuntimeError("AD-map could not restore the requested waypoint.")
        return waypoint

    def _remote_waypoint(self, waypoint):
        """Return one waypoint with nearby lanes prefetched for local proxy traversal."""
        if waypoint is None:
            return None
        data = waypoint.to_dict()
        data["__cpx_left__"] = self._lateral_chain(waypoint, "left", 6)
        data["__cpx_right__"] = self._lateral_chain(waypoint, "right", 6)
        return RemoteWaypoint(data, None)

    def _lateral_chain(self, waypoint, side, remaining):
        """Serialize the short same-direction lane chain used by canonical lane helpers."""
        if waypoint is None or int(remaining) <= 0:
            return None
        adjacent = getattr(waypoint, str(side))()
        if adjacent is None:
            return None
        data = adjacent.to_dict()
        data["__cpx_{}__".format(side)] = self._lateral_chain(adjacent, side, int(remaining) - 1)
        return data

    def _remote_step_results(self, waypoints, direction, distance_m):
        """Return a bounded forward/backward waypoint tree in one ROS response."""
        remaining_nodes = [96]

        def bundle(waypoint, remaining_depth):
            if waypoint is None or remaining_nodes[0] <= 0:
                return None
            remaining_nodes[0] -= 1
            data = self._remote_waypoint(waypoint).to_wire_dict()
            if int(remaining_depth) > 0 and remaining_nodes[0] > 0:
                next_waypoints = list(getattr(waypoint, str(direction))(float(distance_m)) or [])
                children = []
                for next_waypoint in next_waypoints:
                    child = bundle(next_waypoint, int(remaining_depth) - 1)
                    if child is not None:
                        children.append(child)
                data["__cpx_{}_distance_m__".format(direction)] = float(distance_m)
                data["__cpx_{}__".format(direction)] = children
            return data

        return [RemoteWaypoint(data, None) for data in (bundle(waypoint, 32) for waypoint in list(waypoints or [])) if data is not None]

    @staticmethod
    def _point(point):
        return {"x": float(point.x), "y": float(point.y), "z": float(point.z)}

    def _route_message(self, summary, cycle_id, header):
        message = GlobalRoute()
        message.header = header
        message.header.frame_id = "map"
        message.cycle_id = int(cycle_id)
        message.route_found = bool(getattr(summary, "route_found", False))
        message.reason = str(getattr(summary, "debug_reason", ""))
        message.optimal_lane_id = int(getattr(summary, "optimal_lane_id", 0) or 0)
        message.next_macro_maneuver = str(getattr(summary, "next_macro_maneuver", ""))
        message.current_road_option = str(getattr(summary, "current_road_option", ""))
        message.remaining_distance_m = float(getattr(summary, "distance_to_destination_m", 0.0) or 0.0)
        for raw in list(getattr(summary, "route_waypoints", []) or []):
            point = RoutePoint()
            point.position.x = float(raw[0])
            point.position.y = float(raw[1])
            point.position.z = float(raw[2]) if len(raw) >= 3 else 0.0
            waypoint = self.map_planner.get_waypoint({"x": point.position.x, "y": point.position.y, "z": point.position.z})
            point.heading_rad = float(world_heading_rad(waypoint) or 0.0)
            point.lane_id = int(getattr(waypoint, "lane_id", 0) or 0)
            point.road_id = int(getattr(waypoint, "road_id", 0) or 0)
            point.section_id = int(getattr(waypoint, "section_id", 0) or 0)
            point.ad_lane_id = int(getattr(waypoint, "ad_lane_id", 0) or 0)
            message.points.append(point)
        return message

    def destroy_node(self):
        self.map_planner.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
