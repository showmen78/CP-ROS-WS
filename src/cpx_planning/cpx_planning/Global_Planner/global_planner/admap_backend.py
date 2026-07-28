"""Low-level AD-map helpers used by the planner core."""

from __future__ import annotations

from pathlib import Path

from .runtime import import_ad_map_access

ad = None


def _get_compatible_attribute(value, *attribute_names):
    """Read the first available AD-map 2.x/3.x binding attribute."""
    for attribute_name in attribute_names:
        if hasattr(value, attribute_name):
            return getattr(value, attribute_name)
    joined_names = ", ".join(attribute_names)
    raise AttributeError(f"None of these AD-map attributes exist: {joined_names}")


def _set_compatible_attribute(value, attribute_value, *attribute_names) -> None:
    """Write the first available AD-map 2.x/3.x binding attribute."""
    for attribute_name in attribute_names:
        if hasattr(value, attribute_name):
            setattr(value, attribute_name, attribute_value)
            return
    joined_names = ", ".join(attribute_names)
    raise AttributeError(f"None of these AD-map attributes exist: {joined_names}")


def ensure_runtime_ready(ad_map_install_root: str | Path | None = None):
    """Load the AD-map Python bindings on demand.

    input: optional install-root override (`str | Path | None`)
    output: imported AD-map module (`module`)
    """
    global ad

    if ad is None:
        ad = import_ad_map_access(ad_map_install_root)
    return ad


def to_base_value(value):
    """Turn one AD-map wrapped value into a normal Python value.

    input: `value` (`object`)
    output: unwrapped base value (`object`)
    """
    base_value = getattr(value, "toBaseType", None)
    if callable(base_value):
        return base_value()
    if base_value is not None:
        return base_value

    # The AD-map 2.3 wrappers expose integral strong types (including LaneId)
    # through Python 2's ``__long__`` hook, which Python 3 does not consult for
    # ``int(value)``.  Calling the hook directly preserves the same base value.
    legacy_long = getattr(value, "__long__", None)
    if callable(legacy_long):
        return legacy_long()
    return value


def distance_to_float(distance) -> float:
    """Convert one AD-map distance value into meters as a float.

    input: `distance` (AD-map distance-like object)
    output: distance in meters (`float`)
    """
    return float(to_base_value(distance))


def parametric_value_to_float(value) -> float:
    """Convert one AD-map parametric value into a normal float.

    input: `value` (AD-map parametric-value-like object)
    output: parametric value (`float`)
    """
    return float(to_base_value(value))


def create_enu_point(point: tuple[float, float, float]):
    """Build one AD-map ENU point from plain `x`, `y`, and `z` values.

    input: `point` (`tuple[float, float, float]`)
    output: ENU point (`ad.map.point.ENUPoint`)
    """
    enu_point = ad.map.point.ENUPoint()
    enu_point.x = ad.map.point.ENUCoordinate(float(point[0]))
    enu_point.y = ad.map.point.ENUCoordinate(float(point[1]))
    enu_point.z = ad.map.point.ENUCoordinate(float(point[2]))
    return enu_point


def enu_point_to_tuple(point) -> tuple[float, float, float]:
    """Convert one AD-map ENU point into a plain Python tuple.

    input: `point` (`ad.map.point.ENUPoint`)
    output: ENU coordinates (`tuple[float, float, float]`)
    """
    return (
        float(to_base_value(point.x)),
        float(to_base_value(point.y)),
        float(to_base_value(point.z)),
    )


def create_para_point(lane_id: int, parametric_offset: float):
    """Build one AD-map parametric lane point.

    input: `lane_id` (`int`), `parametric_offset` (`float`)
    output: lane parametric point (`ad.map.point.ParaPoint`)
    """
    para_point = ad.map.point.ParaPoint()
    _set_compatible_attribute(para_point, lane_id, "lane_id", "laneId")
    _set_compatible_attribute(
        para_point,
        ad.physics.ParametricValue(float(parametric_offset)),
        "parametric_offset",
        "parametricOffset",
    )
    return para_point


def load_open_drive_map(xodr_path: str | Path, overlap_margin: float = 0.05) -> list[int]:
    """Load one OpenDRIVE file directly into AD-map.

    input: `xodr_path` (`str | Path`), `overlap_margin` (`float`)
    output: loaded lane ids (`list[int]`)
    """
    path = Path(xodr_path)
    if not path.exists():
        raise FileNotFoundError(path)
    map_content = path.read_text(encoding="utf-8")
    try:
        initialized = ad.map.access.initFromOpenDriveContent(map_content, overlap_margin)
    except TypeError:
        intersection_type = _get_compatible_attribute(
            ad.map.intersection.IntersectionType,
            "TrafficLight",
            "TRAFFIC_LIGHT",
        )
        initialized = ad.map.access.initFromOpenDriveContent(
            map_content,
            overlap_margin,
            intersection_type,
        )
    if not initialized:
        raise RuntimeError(f"Failed to load map: {path}")
    return get_all_lane_ids()


def load_adm_map(adm_config_path: str | Path) -> bool:
    """Load one previously saved AD-map cache from disk.

    input: `adm_config_path` (`str | Path`)
    output: whether loading succeeded (`bool`)
    """
    path = Path(adm_config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not ad.map.access.init(str(path)):
        raise RuntimeError(f"Failed to load cached AD map: {path}")
    return True


def save_adm_map(adm_path: str | Path) -> bool:
    """Save the currently loaded AD map in `.adm` cache form.

    input: `adm_path` (`str | Path`)
    output: whether saving succeeded (`bool`)
    """
    save_function = getattr(ad.map.access, "saveAsAdm", None)
    if not callable(save_function):
        return False
    path = Path(adm_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not save_function(str(path), True):
        raise RuntimeError(f"Failed to save AD-map cache: {path}")
    return True


def supports_adm_cache() -> bool:
    """Return whether this AD-map binding can save a reusable ADM cache."""
    return callable(getattr(ad.map.access, "saveAsAdm", None))


def close_map() -> None:
    """Unload the current AD map singleton state.

    input: none (`None`)
    output: none (`None`)
    """
    ad.map.access.cleanup()


def get_lane(lane_id: int):
    """Return one AD-map lane object.

    input: `lane_id` (`int`)
    output: lane object (`ad.map.lane.Lane`)
    """
    return ad.map.lane.getLane(lane_id)


def get_all_lane_ids() -> list[int]:
    """Return every loaded AD lane id.

    input: none (`None`)
    output: lane ids (`list[int]`)
    """
    return [int(to_base_value(lane_id)) for lane_id in ad.map.lane.getLanes()]


def is_routeable_lane(lane_id: int) -> bool:
    """Check whether one lane can be used for routing.

    input: `lane_id` (`int`)
    output: whether the lane is routable (`bool`)
    """
    return ad.map.lane.isRouteable(get_lane(lane_id))


def get_routable_lane_ids() -> list[int]:
    """Return all loaded lanes that are routable.

    input: none (`None`)
    output: routable lane ids (`list[int]`)
    """
    return [lane_id for lane_id in get_all_lane_ids() if is_routeable_lane(lane_id)]


def is_routable_match(map_match) -> bool:
    """Check whether one map-matched lane can be routed on.

    input: `map_match` (AD-map match object)
    output: whether the lane is routable (`bool`)
    """
    return is_routeable_lane(get_match_lane_id(map_match))


def get_map_matches(enu_point, search_radius: float = 8.0) -> list:
    """Find nearby routable AD-map lane matches for one ENU position.

    input: `enu_point` (`ad.map.point.ENUPoint`), `search_radius` (`float`)
    output: routable map matches (`list`)
    """
    matcher = ad.map.match.AdMapMatching()
    matches = matcher.getMapMatchedPositions(
        enu_point,
        ad.physics.Distance(search_radius),
        ad.physics.Probability(0.0),
    )
    return [match for match in matches if is_routable_match(match)]


def get_match_distance(map_match) -> float:
    """Return the raw AD-map match distance in meters.

    input: `map_match` (AD-map match object)
    output: distance in meters (`float`)
    """
    return distance_to_float(
        _get_compatible_attribute(map_match, "matched_point_distance", "matchedPointDistance")
    )


def get_match_probability(map_match) -> float:
    """Return the AD-map match confidence as a float.

    input: `map_match` (AD-map match object)
    output: match probability (`float`)
    """
    return float(to_base_value(map_match.probability))


def get_match_lane_id(map_match) -> int:
    """Read the lane id from one AD-map match.

    input: `map_match` (AD-map match object)
    output: lane id (`int`)
    """
    lane_point = _get_compatible_attribute(map_match, "lane_point", "lanePoint")
    para_point = _get_compatible_attribute(lane_point, "para_point", "paraPoint")
    return int(to_base_value(_get_compatible_attribute(para_point, "lane_id", "laneId")))


def get_match_parametric_offset(map_match) -> float:
    """Read the matched longitudinal offset from one AD-map match.

    input: `map_match` (AD-map match object)
    output: parametric offset (`float`)
    """
    lane_point = _get_compatible_attribute(map_match, "lane_point", "lanePoint")
    para_point = _get_compatible_attribute(lane_point, "para_point", "paraPoint")
    return parametric_value_to_float(
        _get_compatible_attribute(para_point, "parametric_offset", "parametricOffset")
    )


def match_is_in_lane(map_match) -> bool:
    """Check whether the matched point lies inside the matched lane.

    input: `map_match` (AD-map match object)
    output: whether the match is in-lane (`bool`)
    """
    within_lane = getattr(ad.map.match, "isActualWithinLaneMatch", None)
    if callable(within_lane):
        return bool(within_lane(map_match))
    return map_match.type == ad.map.match.MapMatchedPositionType.LANE_IN


def sample_lane_center(para_point) -> tuple[float, float, float]:
    """Sample the lane-center ENU position for one lane parametric point.

    input: `para_point` (AD-map para point)
    output: lane-center ENU point (`tuple[float, float, float]`)
    """
    center_point = ad.map.lane.getENULanePoint(para_point, ad.physics.ParametricValue(0.5))
    return enu_point_to_tuple(center_point)


def sample_lane_center_at_offset(lane_id: int, parametric_offset: float) -> tuple[float, float, float]:
    """Sample the lane-center ENU position for one lane and offset.

    input: `lane_id` (`int`), `parametric_offset` (`float`)
    output: lane-center ENU point (`tuple[float, float, float]`)
    """
    return sample_lane_center(create_para_point(lane_id, parametric_offset))


def get_lane_length_m(lane_id: int) -> float:
    """Return the full physical length of one lane.

    input: `lane_id` (`int`)
    output: lane length in meters (`float`)
    """
    return distance_to_float(get_lane(lane_id).length)


def lane_is_positive(lane_id: int) -> bool:
    """Check whether one lane uses positive parametric travel direction.

    input: `lane_id` (`int`)
    output: whether the lane direction is positive (`bool`)
    """
    return get_lane(lane_id).direction == ad.map.lane.LaneDirection.POSITIVE


def lanes_have_same_direction(first_lane_id: int, second_lane_id: int) -> bool:
    """Check whether two lanes share the same legal driving direction.

    input: `first_lane_id` (`int`), `second_lane_id` (`int`)
    output: whether both lanes move in the same direction (`bool`)
    """
    return lane_is_positive(first_lane_id) == lane_is_positive(second_lane_id)


def get_travel_start_offset(lane_id: int) -> float:
    """Return the parametric offset where legal travel starts on one lane.

    input: `lane_id` (`int`)
    output: travel start offset (`float`)
    """
    return 0.0 if lane_is_positive(lane_id) else 1.0


def get_travel_end_offset(lane_id: int) -> float:
    """Return the parametric offset where legal travel ends on one lane.

    input: `lane_id` (`int`)
    output: travel end offset (`float`)
    """
    return 1.0 if lane_is_positive(lane_id) else 0.0


def lane_offsets_move_forward(lane_id: int, start_offset: float, end_offset: float, tolerance: float = 1e-6) -> bool:
    """Check whether an offset change follows the lane's legal direction.

    input: `lane_id` (`int`), `start_offset` (`float`), `end_offset` (`float`), `tolerance` (`float`)
    output: whether the move is forward (`bool`)
    """
    if lane_is_positive(lane_id):
        return end_offset >= start_offset - tolerance
    return end_offset <= start_offset + tolerance


def get_forward_contact_location(lane_id: int):
    """Return the contact location that corresponds to legal forward travel.

    input: `lane_id` (`int`)
    output: forward contact location (`ad.map.lane.ContactLocation`)
    """
    if lane_is_positive(lane_id):
        return ad.map.lane.ContactLocation.SUCCESSOR
    return ad.map.lane.ContactLocation.PREDECESSOR


def get_backward_contact_location(lane_id: int):
    """Return the contact location used when moving backward along a lane.

    input: `lane_id` (`int`)
    output: backward contact location (`ad.map.lane.ContactLocation`)
    """
    if lane_is_positive(lane_id):
        return ad.map.lane.ContactLocation.PREDECESSOR
    return ad.map.lane.ContactLocation.SUCCESSOR


def lane_is_intersection(lane_id: int) -> bool:
    """Check whether one lane is flagged as an intersection lane.

    input: `lane_id` (`int`)
    output: whether the lane is an intersection lane (`bool`)
    """
    return get_lane(lane_id).type == ad.map.lane.LaneType.INTERSECTION


def get_lane_heading(lane_id: int, parametric_offset: float) -> float | None:
    """Read the ENU heading of one lane-center point when available.

    input: `lane_id` (`int`), `parametric_offset` (`float`)
    output: heading in radians or no result (`float | None`)
    """
    try:
        heading = ad.map.lane.getLaneENUHeading(create_para_point(lane_id, parametric_offset))
    except Exception:
        return None
    if hasattr(heading, "mENUHeading"):
        return float(heading.mENUHeading)
    return float(to_base_value(heading))


def get_lane_width_m(lane_id: int, parametric_offset: float) -> float | None:
    """Read the physical width of one lane at the requested offset when available.

    input: `lane_id` (`int`), `parametric_offset` (`float`)
    output: lane width in meters or no result (`float | None`)
    """
    try:
        width = ad.map.lane.getWidth(get_lane(lane_id), ad.physics.ParametricValue(float(parametric_offset)))
    except Exception:
        return None
    return distance_to_float(width)


def get_contact_lane_ids(
    lane_id: int,
    location,
    require_same_direction: bool = False,
) -> list[int]:
    """Collect connected lane ids at one requested contact location.

    input: `lane_id` (`int`), `location` (`ad.map.lane.ContactLocation`), `require_same_direction` (`bool`)
    output: connected lane ids (`list[int]`)
    """
    lane = get_lane(lane_id)
    connected_lane_ids = []
    for contact_lane in get_contact_lanes(lane):
        if contact_lane.location != location:
            continue
        next_lane_id = get_contact_target_lane_id(contact_lane)
        if next_lane_id <= 0 or not is_routeable_lane(next_lane_id):
            continue
        if require_same_direction and not lanes_have_same_direction(lane_id, next_lane_id):
            continue
        connected_lane_ids.append(next_lane_id)
    return connected_lane_ids


def get_contact_lanes(lane) -> list:
    """Return lane contacts from either AD-map binding naming convention."""
    return list(_get_compatible_attribute(lane, "contact_lanes", "contactLanes"))


def get_contact_target_lane_id(contact_lane) -> int:
    """Return a contact's target lane id on AD-map 2.x or 3.x."""
    return int(to_base_value(_get_compatible_attribute(contact_lane, "to_lane", "toLane")))


def get_same_direction_adjacent_lane_id(lane_id: int, side) -> int | None:
    """Return the first valid adjacent same-direction lane on the requested side.

    input: `lane_id` (`int`), `side` (`ad.map.lane.ContactLocation`)
    output: adjacent lane id or no result (`int | None`)
    """
    lane_ids = get_contact_lane_ids(lane_id, side, require_same_direction=True)
    return lane_ids[0] if lane_ids else None


def get_opendrive_lane_info(ad_lane_id: int) -> dict[str, int | None]:
    """Decode AD-map lane ids back into OpenDRIVE-style identifiers.

    input: `ad_lane_id` (`int`)
    output: road and lane details (`dict[str, int | None]`)
    """
    ad_lane_id = int(to_base_value(ad_lane_id))
    if ad_lane_id <= 10000:
        return {
            "ad_lane_id": ad_lane_id,
            "road_id": None,
            "lane_section_index": None,
            "section_id": None,
            "lane_id": None,
        }
    road_id = ad_lane_id // 10000
    lane_section_index = ad_lane_id % 10000 // 100
    lane_id = ad_lane_id % 100 - 50
    return {
        "ad_lane_id": ad_lane_id,
        "road_id": road_id,
        "lane_section_index": lane_section_index,
        "section_id": lane_section_index,
        "lane_id": lane_id,
    }
