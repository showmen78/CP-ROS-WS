"""Average performance timing for the ROS CP-X planning pipeline."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import threading


def default_timing_summary_path() -> Path:
    """Keep the single timing summary in the ROS workspace root."""
    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        if parent.name == "src":
            return parent.parent / "ros_planner_timing_summary.json"
    return Path.cwd() / "ros_planner_timing_summary.json"


class CycleTimingRecorder:
    """Calculate average major-stage and sub-stage times without per-cycle files."""

    COMPONENTS = {
        "global_planner_node": {"prefix": "component.cpx.global_planner", "topics": ["/cpx/global_planner/request", "/cpx/global_planner/result"]},
        "behavior_context_node": {"prefix": "component.cpx.behavior.context", "topics": ["/cpx/behavior/context/traffic_memory/request", "/cpx/behavior/context/traffic_memory/result", "/cpx/behavior/context/scenario/request", "/cpx/behavior/context/scenario/result"]},
        "behavior_decision_node": {"prefix": "component.cpx.behavior.decision", "topics": ["/cpx/behavior/decision/request", "/cpx/behavior/decision/result"]},
        "reference_planner_node": {"prefix": "component.cpx.behavior.reference", "topics": ["/cpx/behavior/reference/maneuver/request", "/cpx/behavior/reference/maneuver/result", "/cpx/behavior/reference/finalize/request", "/cpx/behavior/reference/finalize/result"]},
        "mpc_node": {"prefix": "component.cpx.mpc", "topics": ["/cpx/mpc/request", "/cpx/mpc/result"]},
    }

    def __init__(self, *, enabled: bool, summary_path=None):
        self.enabled = bool(enabled)
        self.summary_path = Path(summary_path or default_timing_summary_path()).expanduser().resolve()
        self._totals_ms = defaultdict(float)
        self._successful_cycle_count = 0
        self._failed_cycle_count = 0
        self._last_failure_reason = ""
        self._event_times_ns = {}
        self._event_cycle_order = []
        self._derived_event_metrics = set()
        self._lock = threading.RLock()
        if self.enabled:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text("{}\n", encoding="utf-8")

    def start_cycle(self, cycle_id: int, sim_time_s: float) -> None:
        """Remember a bounded set of cycle timestamps used to separate waiting stages."""
        if not self.enabled or int(cycle_id) <= 0:
            return
        with self._lock:
            if int(cycle_id) not in self._event_times_ns:
                self._event_times_ns[int(cycle_id)] = {}
                self._event_cycle_order.append(int(cycle_id))
            while len(self._event_cycle_order) > 256:
                old_cycle_id = self._event_cycle_order.pop(0)
                self._event_times_ns.pop(old_cycle_id, None)
                self._derived_event_metrics = {key for key in self._derived_event_metrics if key[0] != old_cycle_id}

    def record_duration(self, cycle_id: int, stage: str, duration_ms: float) -> None:
        """Add one duration sample to the running average for a sub-stage."""
        if not self.enabled:
            return
        stage = str(stage)
        duration_ms = max(0.0, float(duration_ms))
        with self._lock:
            self._record_duration_locked(stage, duration_ms)

    def _record_duration_locked(self, stage: str, duration_ms: float) -> None:
        """Update one aggregate while the recorder lock is already held."""
        self._totals_ms[str(stage)] += max(0.0, float(duration_ms))

    def record_event_time(self, cycle_id: int, name: str, wall_time_ns: int) -> None:
        """Store one wall-clock event and derive cross-node delays when both ends exist."""
        if not self.enabled or int(cycle_id) <= 0 or int(wall_time_ns) <= 0:
            return
        self.start_cycle(int(cycle_id), 0.0)
        with self._lock:
            self._event_times_ns[int(cycle_id)][str(name)] = int(wall_time_ns)
            self._derive_event_metrics_locked(int(cycle_id))

    def _record_derived_once_locked(self, cycle_id: int, stage: str, start_ns: int, end_ns: int) -> None:
        """Record one cross-node duration once, ignoring missing or reversed clocks."""
        key = (int(cycle_id), str(stage))
        if key in self._derived_event_metrics or int(start_ns) <= 0 or int(end_ns) < int(start_ns):
            return
        self._derived_event_metrics.add(key)
        self._record_duration_locked(str(stage), (int(end_ns) - int(start_ns)) / 1000000.0)

    def _derive_event_metrics_locked(self, cycle_id: int) -> None:
        """Create the requested transfer, waiting, scheduling, forwarding, and output times."""
        events = dict(self._event_times_ns.get(int(cycle_id), {}))
        stream_to_inputs = {
            "localization": ("localization", "final_destination"),
            "perception": ("perception",),
            "v2x": ("v2x", "cp_obstacles"),
            "traffic_light": ("traffic_lights",),
            "cooperative": ("cooperative",),
            "safety_status": ("safety_status",),
        }
        for stream, input_names in stream_to_inputs.items():
            source_ns = int(events.get("transport.{}.source_send".format(stream), 0) or 0)
            tcp_ns = int(events.get("transport.{}.tcp_received".format(stream), 0) or 0)
            publish_start_ns = int(events.get("transport.{}.ros_publish_started".format(stream), 0) or 0)
            publish_finish_ns = int(events.get("transport.{}.ros_publish_finished".format(stream), 0) or 0)
            self._record_derived_once_locked(cycle_id, "input_transfer.{}.source_to_ros_publish".format(stream), source_ns, publish_finish_ns)
            self._record_derived_once_locked(cycle_id, "node_scheduling.input_publisher.{}.tcp_queue_to_callback".format(stream), tcp_ns, publish_start_ns)
            for input_name in input_names:
                callback_ns = int(events.get("input.{}.callback_started".format(input_name), 0) or 0)
                self._record_derived_once_locked(cycle_id, "node_scheduling.planner_input.{}.ros_publish_to_callback".format(input_name), publish_finish_ns, callback_ns)

        source_times = [int(events.get("transport.{}.source_send".format(stream), 0) or 0) for stream in stream_to_inputs]
        tcp_receive_times = [int(events.get("transport.{}.tcp_received".format(stream), 0) or 0) for stream in stream_to_inputs]
        publish_finish_times = [int(events.get("transport.{}.ros_publish_finished".format(stream), 0) or 0) for stream in stream_to_inputs]
        if all(value > 0 for value in source_times) and all(value > 0 for value in tcp_receive_times) and all(value > 0 for value in publish_finish_times):
            self._record_derived_once_locked(cycle_id, "input_transfer.all_inputs.source_to_last_ros_publish", min(source_times), max(publish_finish_times))
            self._record_derived_once_locked(cycle_id, "input_transfer.critical_path.source_to_last_tcp_receive", min(source_times), max(tcp_receive_times))
            self._record_derived_once_locked(cycle_id, "input_transfer.critical_path.last_tcp_receive_to_last_ros_publish", max(tcp_receive_times), max(publish_finish_times))

        planning_start_ns = int(events.get("planner.cycle_started", 0) or 0)
        callback_times = [int(value) for name, value in events.items() if name.startswith("input.") and name.endswith(".callback_started") and int(value) <= planning_start_ns]
        if planning_start_ns > 0 and callback_times:
            self._record_derived_once_locked(cycle_id, "callback_message_waiting.inputs_first_callback_to_cycle_start", min(callback_times), planning_start_ns)
            self._record_derived_once_locked(cycle_id, "callback_message_waiting.inputs_trigger_callback_to_cycle_start", max(callback_times), planning_start_ns)
            if all(value > 0 for value in publish_finish_times):
                self._record_derived_once_locked(cycle_id, "node_scheduling.input_critical_path", max(publish_finish_times), max(callback_times))

        output_publish_start_ns = int(events.get("output.control_topics_publish_started", 0) or 0)
        output_publish_finish_ns = int(events.get("output.control_topics_publish_finished", 0) or 0)
        output_callback_ns = int(events.get("output_forwarder.topic_callback_started", 0) or 0)
        output_tcp_start_ns = int(events.get("output_forwarder.tcp_send_started", 0) or 0)
        output_tcp_finish_ns = int(events.get("output_forwarder.tcp_send_finished", 0) or 0)
        self._record_derived_once_locked(cycle_id, "node_scheduling.output_forwarder.ros_publish_to_callback", output_publish_finish_ns, output_callback_ns)
        self._record_derived_once_locked(cycle_id, "output_publishing.output_forwarder_preparation", output_callback_ns, output_tcp_start_ns)
        self._record_derived_once_locked(cycle_id, "output_publishing.tcp_to_opencda", output_tcp_start_ns, output_tcp_finish_ns)
        self._record_derived_once_locked(cycle_id, "output_publishing.full_wall_clock", output_publish_start_ns, output_tcp_finish_ns)

    def record_value(self, cycle_id: int, name: str, value) -> None:
        """Keep only the most recent failure reason; no per-cycle values are saved."""
        if self.enabled and str(name) == "failure_reason":
            with self._lock:
                self._last_failure_reason = str(value)

    def record_external_event(self, event) -> None:
        """Add TCP and ROS-publisher timing received on /cpx/timing/events."""
        if not self.enabled or not isinstance(event, dict):
            return
        stream = str(event.get("stream", event.get("stage", "external")))
        for name, value in event.items():
            if not str(name).endswith("_ms"):
                continue
            try:
                self.record_duration(0, "transport.{}.{}".format(stream, name), float(value))
            except (TypeError, ValueError):
                pass
        cycle_id = int(event.get("cycle_id", 0) or 0)
        timestamp_fields = {
            "source_send_wall_time_ns": "transport.{}.source_send".format(stream),
            "tcp_received_wall_time_ns": "transport.{}.tcp_received".format(stream),
            "ros_publish_started_wall_time_ns": "transport.{}.ros_publish_started".format(stream),
            "ros_publish_finished_wall_time_ns": "transport.{}.ros_publish_finished".format(stream),
            "output_topic_received_wall_time_ns": "output_forwarder.topic_callback_started",
            "output_tcp_send_started_wall_time_ns": "output_forwarder.tcp_send_started",
            "output_tcp_sent_wall_time_ns": "output_forwarder.tcp_send_finished",
        }
        for field_name, event_name in timestamp_fields.items():
            try:
                self.record_event_time(cycle_id, event_name, int(event.get(field_name, 0) or 0))
            except (TypeError, ValueError):
                pass

    def finish_cycle(self, cycle_id: int, extra=None) -> None:
        """Count the completed cycle and periodically refresh the average file."""
        if not self.enabled:
            return
        success = not isinstance(extra, dict) or bool(extra.get("success", True))
        with self._lock:
            if success:
                self._successful_cycle_count += 1
            else:
                self._failed_cycle_count += 1
            completed = self._successful_cycle_count + self._failed_cycle_count
        # Refreshing the formatted summary is useful while profiling, but it
        # does not need to run once per simulated second. The final summary is
        # still written when the planner shuts down.
        if completed % 100 == 0:
            self.write_summary()

    def _sum_per_cycle(self, predicate):
        total = sum(float(value) for name, value in self._totals_ms.items() if predicate(name))
        return total / max(1, int(self._successful_cycle_count))

    def _average_per_cycle(self, stage: str) -> float:
        """Return only the requested average contribution per completed planning cycle."""
        return float(self._totals_ms.get(str(stage), 0.0)) / max(1, int(self._successful_cycle_count))

    @staticmethod
    def _timing_item(average_ms: float, item_type: str, interface: str = ""):
        """Keep every displayed timing item small and self-explanatory."""
        item = {"average_ms_per_cycle": float(average_ms), "type": str(item_type)}
        if str(interface):
            item["interface"] = str(interface)
        return item

    def write_summary(self) -> None:
        """Write only the five requested averages and their clearly named breakdowns."""
        if not self.enabled:
            return
        with self._lock:
            input_sync_ms = self._average_per_cycle("callback_message_waiting.inputs_trigger_callback_to_cycle_start")
            input_node_scheduling_ms = self._average_per_cycle("node_scheduling.input_critical_path")
            component_scheduling_ms = self._sum_per_cycle(lambda name: name.startswith("node_scheduling.component."))
            output_forwarder_preparation_ms = self._average_per_cycle("output_publishing.output_forwarder_preparation")
            output_forwarder_scheduling_ms = self._average_per_cycle("node_scheduling.output_forwarder.ros_publish_to_callback")
            ros_output_publish_ms = self._average_per_cycle("output_publishing.ros_control_topics_total")
            tcp_output_publish_ms = self._average_per_cycle("output_publishing.tcp_to_opencda")
            summary = {
                "debug_time": True,
                "unit": "milliseconds per completed planning cycle",
                "timing": {
                    "1_input_transfer": {
                        "average_ms_per_cycle": self._average_per_cycle("input_transfer.all_inputs.source_to_last_ros_publish"),
                        "type": "TCP transfer and ROS input-topic publication",
                        "what_it_means": "OpenCDA starts sending one cycle of raw inputs until all required inputs have been published as ROS topics.",
                        "availability": "Both workspaces take part. OpenCDA sends the raw data; the ROS workspace receives, converts, and publishes it.",
                        "same_in_both_planners": "Not a planner algorithm. The local OpenCDA planner does not need this transfer stage.",
                        "latest_opencda_location": "opencda/data_transmitter.py and opencda/core/common/vehicle_manager.py",
                        "ros_location": "cpx_comm_test TCP receivers and input publisher nodes",
                        "breakdown": {
                            "opencda_send_to_last_tcp_receive": self._timing_item(self._average_per_cycle("input_transfer.critical_path.source_to_last_tcp_receive"), "Actual TCP critical-path time"),
                            "last_tcp_receive_to_last_ros_input_publish": self._timing_item(self._average_per_cycle("input_transfer.critical_path.last_tcp_receive_to_last_ros_publish"), "Actual ROS conversion and publication tail"),
                        },
                        "breakdown_note": "These two parts are consecutive on the clock and do not overlap.",
                    },
                    "2_callback_and_message_waiting": {
                        "average_ms_per_cycle": float(input_sync_ms),
                        "type": "ROS callback and input-synchronization waiting",
                        "what_it_means": "Time from the final required input callback starting until the planner cycle starts.",
                        "availability": "ROS workspace only.",
                        "same_in_both_planners": "No. OpenCDA calls its planner directly and does not wait for these ROS callbacks.",
                        "latest_opencda_location": "No equivalent ROS callback stage in the local OpenCDA planner.",
                        "ros_location": "cpx_planning/planner_node.py and cpx_comm_test/data_subscriber.py",
                        "breakdown": {
                            "last_required_input_callback_to_planning_start": self._timing_item(input_sync_ms, "Actual callback preparation and lock time"),
                        },
                        "breakdown_note": "This starts after input-topic scheduling ends, so it does not overlap that category.",
                    },
                    "3_node_scheduling": {
                        "average_ms_per_cycle": float(input_node_scheduling_ms + component_scheduling_ms),
                        "type": "ROS executor or direct component-dispatch delay",
                        "what_it_means": "A message or component request is ready until the receiving node starts handling it.",
                        "availability": "ROS workspace only.",
                        "same_in_both_planners": "No. The planning methods are the same, but OpenCDA calls them directly without ROS node scheduling.",
                        "latest_opencda_location": "No ROS-node scheduling stage in the local OpenCDA planner.",
                        "ros_location": "cpx_comm_test publisher nodes, cpx_planning/planner_node.py, and cpx_planning/component_interfaces.py",
                        "breakdown": {
                            "last_ros_input_publish_to_last_planner_callback": self._timing_item(input_node_scheduling_ms, "Actual input critical-path scheduling"),
                            "global_planner_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("node_scheduling.component.cpx.global_planner.")), "Immediate in-process node dispatch", "/cpx/global_planner/request"),
                            "behavior_context_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("node_scheduling.component.cpx.behavior.context.")), "Immediate in-process node dispatch"),
                            "behavior_decision_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("node_scheduling.component.cpx.behavior.decision.")), "Immediate in-process node dispatch", "/cpx/behavior/decision/request"),
                            "reference_planner_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("node_scheduling.component.cpx.behavior.reference.")), "Immediate in-process node dispatch"),
                            "mpc_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("node_scheduling.component.cpx.mpc.")), "Immediate in-process node dispatch", "/cpx/mpc/request"),
                        },
                        "breakdown_note": "The input critical-path wait and component dispatches occur at different points in the cycle and are not overlapping callback totals.",
                    },
                    "4_inter_node_forwarding": {
                        "average_ms_per_cycle": self._sum_per_cycle(lambda name: name.startswith("inter_node_forwarding.") and name.endswith(".total")),
                        "type": "In-process planner-component request and response forwarding",
                        "what_it_means": "Preparing and returning data between the ROS node wrappers. In the fast composed launch, the data is passed directly in one process while the named topic contracts are retained. The component's actual planner calculation is excluded.",
                        "availability": "ROS workspace only. OpenCDA uses direct Python method calls.",
                        "same_in_both_planners": "The planner methods receiving the data are copied from OpenCDA; the forwarding wrappers are ROS-specific.",
                        "latest_opencda_location": "The corresponding planner methods are called directly inside CPXMPCPlannerBridge.",
                        "ros_location": "cpx_planning/component_interfaces.py",
                        "breakdown": {
                            "global_planner_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("inter_node_forwarding.cpx.global_planner.") and name.endswith(".total")), "In-process request and response forwarding"),
                            "behavior_context_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("inter_node_forwarding.cpx.behavior.context.") and name.endswith(".total")), "In-process request and response forwarding"),
                            "behavior_decision_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("inter_node_forwarding.cpx.behavior.decision.") and name.endswith(".total")), "In-process request and response forwarding"),
                            "reference_planner_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("inter_node_forwarding.cpx.behavior.reference.") and name.endswith(".total")), "In-process request and response forwarding"),
                            "mpc_node": self._timing_item(self._sum_per_cycle(lambda name: name.startswith("inter_node_forwarding.cpx.mpc.") and name.endswith(".total")), "In-process request and response forwarding"),
                        },
                    },
                    "5_output_publishing": {
                        "average_ms_per_cycle": self._average_per_cycle("output_publishing.full_wall_clock"),
                        "type": "ROS control-topic publication and TCP output transmission",
                        "what_it_means": "Publishing the final control in ROS and sending its compact copy back to OpenCDA.",
                        "availability": "Both workspaces take part. ROS publishes and sends; OpenCDA receives and applies the selected command.",
                        "same_in_both_planners": "Not a planner algorithm. The local OpenCDA planner returns control directly instead of using these ROS output topics.",
                        "latest_opencda_location": "opencda/data_receiver.py and opencda/core/common/vehicle_manager.py",
                        "ros_location": "cpx_planning/planner_node.py and cpx_comm_test/data_subscriber.py",
                        "breakdown": {
                            "publish_ros_control_topics": self._timing_item(ros_output_publish_ms, "ROS topic publication", "/cpx/planner_control_output and /control/command/control_cmd"),
                            "wait_for_output_forwarder_callback": self._timing_item(output_forwarder_scheduling_ms, "Actual ROS output callback scheduling", "/cpx/planner_control_output"),
                            "prepare_received_control_for_tcp": self._timing_item(output_forwarder_preparation_ms, "Output callback preparation"),
                            "send_control_to_opencda_over_tcp": self._timing_item(tcp_output_publish_ms, "JSON conversion and TCP socket send", "TCP port 5060"),
                        },
                        "breakdown_note": "These four parts are consecutive on the clock and do not overlap.",
                    },
                },
            }
        self.summary_path.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8")
