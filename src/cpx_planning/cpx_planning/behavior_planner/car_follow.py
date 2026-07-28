"""IDM (Intelligent Driver Model) car-following acceleration.

Reference: Treiber, Hennecke & Helbing (2000), "Congested Traffic States in
Empirical Observations and Microscopic Simulations", Physical Review E 62(2).

Used by Apollo, SUMO, and most production AV stacks as the car-following model
for longitudinal gap regulation.  Input: ego speed, lead speed, bumper-to-bumper
gap.  Output: longitudinal acceleration in m/s^2.
"""

from __future__ import annotations

import math


def idm_acceleration(
    v: float,
    v_lead: float,
    gap_m: float,
    *,
    v_desired: float = 13.0,
    a_max: float = 2.0,
    b_comfort: float = 3.0,
    time_headway_s: float = 1.5,
    min_gap_m: float = 2.0,
    delta: float = 4.0,
) -> float:
    """Return IDM longitudinal acceleration in m/s^2.

    Positive = accelerate toward v_desired, negative = decelerate.
    Output is clamped to [-2*b_comfort, a_max].

    Args:
        v: ego speed [m/s]
        v_lead: lead vehicle speed [m/s]
        gap_m: bumper-to-bumper gap [m]
        v_desired: free-flow speed target [m/s]
        a_max: maximum acceleration [m/s^2]
        b_comfort: comfortable deceleration magnitude [m/s^2]
        time_headway_s: desired time headway [s]
        min_gap_m: standstill gap [m]
        delta: acceleration exponent (4 = standard IDM)
    """
    v = max(0.0, float(v))
    v_lead = max(0.0, float(v_lead))
    gap_m = max(0.1, float(gap_m))
    v_desired = max(0.1, float(v_desired))
    a_max = max(0.01, float(a_max))
    b_comfort = max(0.01, float(b_comfort))

    delta_v = v - v_lead
    s_star = float(min_gap_m) + max(
        0.0,
        v * float(time_headway_s)
        + v * delta_v / (2.0 * math.sqrt(float(a_max) * float(b_comfort))),
    )
    accel = float(a_max) * (
        1.0
        - (v / float(v_desired)) ** float(delta)
        - (s_star / float(gap_m)) ** 2
    )
    return float(max(-float(b_comfort) * 2.0, min(float(a_max), accel)))
