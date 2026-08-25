"""Cross-CAV cooperative arbitration for spatially-conflicting maneuvers.

Pure and CARLA-free: callers translate their own maneuver state and peer
broadcast messages into ``ResourceClaim``s plus flat (x, y) positions, and
``should_yield`` decides who proceeds. This is the shared primitive behind
every "two CPX-controlled CAVs both want to do X near each other" gate
(lane change, static-obstacle avoidance lane selection, junction entry --
see call sites in ``cpx_mpc_planner.py``), so the priority rule only needs
to be got right once.

Priority is "earliest commitment wins" (``committed_at_s``), which is the
only rule that generalizes across all of the above -- lane changes have a
natural "physically ahead" ordering, but a 90-degree road junction does
not, so ordering purely by commitment time is what both cases can agree
on. Exact simultaneous commits break the tie on ``actor_id`` so every
observer resolves the same winner independently, without needing a
side channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class ResourceClaim:
    """One CAV's claim on a shared resource.

    ``resource_id`` is caller-defined and only compared for equality, so
    callers control how coarse or specific conflicts are: a constant
    string (e.g. "lane_change") makes every claim of that kind conflict
    with every other regardless of which lane is involved; a specific
    lane/junction id restricts conflicts to CAVs contending for the exact
    same resource.
    """

    kind: str
    resource_id: str
    committed_at_s: float
    active: bool
    require_ahead: bool = True


def should_yield(
    *,
    my_claim: ResourceClaim,
    my_actor_id: int,
    my_position_xy: Tuple[float, float],
    my_heading_rad: float,
    peers: Sequence[Tuple[int, ResourceClaim, Tuple[float, float]]],
    range_m: float = 40.0,
) -> Optional[str]:
    """Return a yield reason, or ``None`` if ``my_claim`` may proceed.

    ``peers`` is ``(peer_actor_id, peer_claim, peer_position_xy)`` for every
    other CAV whose latest broadcast intent is available this tick.
    """

    forward_x = math.cos(float(my_heading_rad))
    forward_y = math.sin(float(my_heading_rad))
    for peer_actor_id, peer_claim, peer_position_xy in peers:
        if not bool(peer_claim.active):
            continue
        if str(peer_claim.kind) != str(my_claim.kind):
            continue
        if str(peer_claim.resource_id) != str(my_claim.resource_id):
            continue
        dx = float(peer_position_xy[0]) - float(my_position_xy[0])
        dy = float(peer_position_xy[1]) - float(my_position_xy[1])
        distance_m = math.hypot(dx, dy)
        if distance_m > float(range_m):
            continue
        if bool(my_claim.require_ahead):
            # A peer only has an ordering claim on ego's own maneuver if
            # it's ahead along ego's heading -- a peer already behind
            # cannot be the reason ego holds back.
            forward_distance_m = dx * forward_x + dy * forward_y
            if forward_distance_m <= 0.0:
                continue
        peer_wins = float(peer_claim.committed_at_s) < float(
            my_claim.committed_at_s
        ) or (
            float(peer_claim.committed_at_s) == float(my_claim.committed_at_s)
            and int(peer_actor_id) < int(my_actor_id)
        )
        if not peer_wins:
            continue
        return (
            "blocked_by_peer_claim:"
            f"kind={my_claim.kind}:resource={my_claim.resource_id}:"
            f"peer={peer_actor_id}:distance_m={distance_m:.1f}"
        )
    return None
