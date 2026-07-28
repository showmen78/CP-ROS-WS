"""Apollo-style planning pipeline helpers.

This package keeps the high-level planning stages explicit:
cooperative perception input, prediction, behavior decision, and trajectory generation.
"""

from .candidate_evaluation import (
    BehaviorCandidate,
    CandidateEvaluationFrame,
    evaluate_behavior_candidates,
)
from .candidate_pipeline import (
    CandidateBehaviorIntent,
    CandidateReferenceResult,
    build_candidate_intents,
    evaluate_candidate_reference,
    select_best_candidate,
    summarize_candidate_results,
)
from .control_buffer import MPCControlBuffer
from .decision_record import DecisionRecord, DecisionVeto, build_decision_record
from .mpc_feedback import BehaviorMPCFeedback
from .prediction import PredictionFrame, build_prediction_frame
from .planner_pipeline import CPXPlanningPipeline
from .output import BehaviorCommand, PlannerDiagnostics, PlannerOutput
from .reference_contract import (
    ReferenceContract,
    ReferenceValidationResult,
    contract_from_config,
    validate_reference_contract,
)
from .route_authorization import (
    LaneChangeAuthorization,
    RouteManeuver,
    authorize_route_lane_change,
    normalize_route_maneuver,
)
from .route_manager import CPXRouteManager, RouteManagerStatus
from .safety_supervisor import SafetySupervisor
from .scenario_manager import CPXScenarioDecision, CPXScenarioManager
from .speed_planner import SpeedPlan, build_speed_plan
from .tracker import CPXObstacleTracker

__all__ = [
    "BehaviorMPCFeedback",
    "BehaviorCandidate",
    "BehaviorCommand",
    "CandidateEvaluationFrame",
    "CandidateBehaviorIntent",
    "CandidateReferenceResult",
    "CPXRouteManager",
    "CPXObstacleTracker",
    "CPXPlanningPipeline",
    "CPXScenarioDecision",
    "CPXScenarioManager",
    "DecisionRecord",
    "DecisionVeto",
    "MPCControlBuffer",
    "PlannerDiagnostics",
    "PlannerOutput",
    "PredictionFrame",
    "ReferenceContract",
    "ReferenceValidationResult",
    "LaneChangeAuthorization",
    "RouteManagerStatus",
    "RouteManeuver",
    "SafetySupervisor",
    "SpeedPlan",
    "authorize_route_lane_change",
    "build_prediction_frame",
    "build_candidate_intents",
    "build_decision_record",
    "build_speed_plan",
    "contract_from_config",
    "evaluate_behavior_candidates",
    "evaluate_candidate_reference",
    "normalize_route_maneuver",
    "select_best_candidate",
    "summarize_candidate_results",
    "validate_reference_contract",
]
