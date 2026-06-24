from __future__ import annotations

import re
from dataclasses import dataclass

from autonomy.contracts import MissionMode

_TARGET_SYNONYMS: dict[str, tuple[str, ...]] = {
    "vehicle": ("vehicle", "vehicles", "traffic"),
    "person": ("person", "people", "pedestrian", "pedestrians", "human", "humans"),
    "car": ("car", "cars", "vehicle", "vehicles"),
    "truck": ("truck", "trucks", "lorry", "lorries"),
    "bus": ("bus", "buses"),
    "motorcycle": ("motorcycle", "motorcycles", "bike", "bikes"),
    "bicycle": ("bicycle", "bicycles", "cycle", "cycles"),
    "dog": ("dog", "dogs"),
    "cat": ("cat", "cats"),
    "bird": ("bird", "birds"),
}

_VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
_MULTI_TARGET_INTENT_PATTERNS = (
    r"\bcount\b",
    r"\bhow many\b",
    r"\breport totals?\b",
    r"\btotal unique\b",
    r"\bsurvey\b",
    r"\bpatrol\b",
    r"\bscan\b",
    r"\bmonitor\b",
)
_COUNT_SURVEY_PATTERNS = (
    r"\bcount\b",
    r"\bhow many\b",
    r"\btotal unique\b",
    r"\breport totals?\b",
    r"\bsurvey\b",
)


@dataclass(frozen=True, slots=True)
class _TargetMatch:
    class_name: str
    position: int


def _first_pattern_position(text: str, patterns: tuple[str, ...]) -> int | None:
    positions = [
        match.start()
        for pattern in patterns
        for match in re.finditer(rf"\b{re.escape(pattern)}\b", text)
    ]
    if not positions:
        return None
    return min(positions)


def _has_multi_target_intent(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _MULTI_TARGET_INTENT_PATTERNS)


def resolve_mission_mode(task: str, priority_class: str | None = None) -> MissionMode:
    """Infer the high-level mission mode from operator text."""
    lowered = task.lower()
    if any(re.search(pattern, lowered) for pattern in _COUNT_SURVEY_PATTERNS) and (
        priority_class == "vehicle" or "road" in lowered or "traffic" in lowered
    ):
        return MissionMode.TRAFFIC_MONITOR
    if priority_class == "person":
        return MissionMode.PEDESTRIAN_WATCH
    return MissionMode.SEARCH


def resolve_cruise_altitude_m(
    mission_mode: MissionMode,
    default_altitude_m: float,
    *,
    traffic_monitor_altitude_m: float | None = None,
) -> float:
    """Choose an effective cruise altitude for the current mission mode."""
    if mission_mode == MissionMode.TRAFFIC_MONITOR:
        if traffic_monitor_altitude_m is not None:
            return float(traffic_monitor_altitude_m)
        return min(float(default_altitude_m), 3.2)
    return float(default_altitude_m)


def resolve_priority_class(task: str, target_classes: list[str]) -> str:
    """Infer the requested target class from operator text.

    Returns the synthetic ``vehicle`` class for generic road-vehicle survey tasks
    so the tracker can consider cars, trucks, and buses together.
    """
    lowered = task.lower()
    matches: list[_TargetMatch] = []
    for class_name in target_classes:
        patterns = _TARGET_SYNONYMS.get(class_name, (class_name, f"{class_name}s"))
        position = _first_pattern_position(lowered, patterns)
        if position is not None:
            matches.append(_TargetMatch(class_name=class_name, position=position))

    vehicle_position = _first_pattern_position(lowered, _TARGET_SYNONYMS["vehicle"])
    vehicle_classes_available = any(class_name in _VEHICLE_CLASSES for class_name in target_classes)
    matched_vehicle_classes = {
        match.class_name for match in matches if match.class_name in _VEHICLE_CLASSES
    }

    if vehicle_classes_available:
        if vehicle_position is not None:
            return "vehicle"
        if len(matched_vehicle_classes) > 1 and _has_multi_target_intent(lowered):
            return "vehicle"

    if matches:
        return min(matches, key=lambda match: match.position).class_name
    return target_classes[0] if target_classes else "truck"
