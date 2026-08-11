"""Apollo-style planning pipeline helpers.

This package keeps the high-level planning stages explicit:
cooperative perception input, prediction, behavior decision, and trajectory generation.
"""

from .candidate_evaluation import (
    BehaviorCandidate,
    CandidateEvaluationFrame,
    evaluate_behavior_candidates,
)
from .actuator_mapper import ActuatorCommand, CarlaActuatorMapper
from .architecture_profile import ArchitectureProfile, normalize_architecture_config
from .candidate_pipeline import (
    CandidateBehaviorIntent,
    CandidateReferenceResult,
    CandidateSelectionOutcome,
    build_candidate_intents,
    evaluate_candidate_reference,
    select_best_candidate,
    select_candidate_with_commitment,
    summarize_candidate_results,
)
from .control_buffer import MPCControlBuffer
from .decision_record import DecisionRecord, DecisionVeto, build_decision_record
from .mpc_feedback import BehaviorMPCFeedback
from .maneuver_manager import ManeuverManager, ManeuverPlan, ManeuverReferenceResult
from .prediction import PredictionFrame, build_prediction_frame
from .planner_pipeline import CPXPlanningPipeline
from .output import BehaviorCommand, PlannerDiagnostics, PlannerOutput
from .reference_contract import (
    ReferenceContract,
    ReferenceValidationResult,
    contract_from_config,
    validate_reference_contract,
)
from .reference_gate import FinalReferenceGate, FinalReferenceGateResult
from .reference_generator import (
    BoundaryRecoveryValidation,
    DrivableFootprintOccupancy,
    GeneratedReference,
    LaneCorridorOccupancy,
    ReferenceCorridorProjection,
    ReferenceGenerator,
)
from .reference_pipeline import (
    ConditionedReference,
    ReferencePipeline,
    ReferencePipelineRequest,
    ReferencePipelineResult,
)
from .traffic_light_memory import TrafficLightMemory
from .stage_contracts import ManeuverCommitment
from .route_authorization import (
    LaneChangeAuthorization,
    RouteManeuver,
    authorize_route_lane_change,
    normalize_route_maneuver,
)
from .route_manager import CPXRouteManager, RouteManagerStatus
from .safety_supervisor import SafetySupervisor
from .scenario_manager import (
    BoundaryRecoveryRequest,
    CPXScenarioDecision,
    CPXScenarioManager,
)
from .speed_planner import SpeedPlan, build_speed_plan
from .velocity_steering_adapter import (
    CarlaVelocitySteeringAdapter,
    VelocitySteeringCommand,
)
from .tracker import CPXObstacleTracker

__all__ = [
    "ActuatorCommand",
    "BehaviorMPCFeedback",
    "CarlaActuatorMapper",
    "BoundaryRecoveryRequest",
    "BoundaryRecoveryValidation",
    "DrivableFootprintOccupancy",
    "ArchitectureProfile",
    "BehaviorCandidate",
    "BehaviorCommand",
    "CandidateEvaluationFrame",
    "CandidateBehaviorIntent",
    "CandidateReferenceResult",
    "CandidateSelectionOutcome",
    "CPXRouteManager",
    "CPXObstacleTracker",
    "CPXPlanningPipeline",
    "CPXScenarioDecision",
    "CPXScenarioManager",
    "DecisionRecord",
    "DecisionVeto",
    "MPCControlBuffer",
    "ManeuverManager",
    "ManeuverPlan",
    "ManeuverReferenceResult",
    "PlannerDiagnostics",
    "PlannerOutput",
    "PredictionFrame",
    "ReferenceContract",
    "FinalReferenceGate",
    "FinalReferenceGateResult",
    "ReferenceGenerator",
    "GeneratedReference",
    "LaneCorridorOccupancy",
    "ReferenceCorridorProjection",
    "ReferencePipeline",
    "ReferencePipelineRequest",
    "ReferencePipelineResult",
    "ConditionedReference",
    "TrafficLightMemory",
    "ManeuverCommitment",
    "ReferenceValidationResult",
    "LaneChangeAuthorization",
    "RouteManagerStatus",
    "RouteManeuver",
    "SafetySupervisor",
    "SpeedPlan",
    "CarlaVelocitySteeringAdapter",
    "VelocitySteeringCommand",
    "authorize_route_lane_change",
    "build_prediction_frame",
    "build_candidate_intents",
    "build_decision_record",
    "build_speed_plan",
    "contract_from_config",
    "evaluate_behavior_candidates",
    "evaluate_candidate_reference",
    "normalize_route_maneuver",
    "normalize_architecture_config",
    "select_best_candidate",
    "select_candidate_with_commitment",
    "summarize_candidate_results",
    "validate_reference_contract",
]
