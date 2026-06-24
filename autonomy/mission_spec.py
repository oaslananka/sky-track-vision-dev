"""Mission task contracts and semantic completion verification.

The LLM pilot already enforces a *procedural* completion protocol
(``REPORT -> get_mission_report -> request_land``). That answers "did the pilot
run the closing sequence?" but not "did it actually accomplish the task?". A
pilot that lands without ever seeing the requested truck would still be scored
as a success.

This module closes that gap. :func:`parse_mission_spec` turns a natural-language
task into a small set of *measurable* objectives, and :class:`MissionVerifier`
checks a finished :class:`~autonomy.contracts.MissionReport` against them. The
parser is deliberately deterministic and rule-based (no LLM call) so it is cheap,
reproducible, and unit-testable, and it is conservative: a task it cannot parse
yields no objectives, leaving the existing protocol-only behaviour untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from autonomy.contracts import MissionReport

# Road-vehicle COCO classes that roll up into ``unique_vehicle_count``.
VEHICLE_CLASSES: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle"})

# Sentinel target for objectives phrased against "vehicles" generically.
ANY_VEHICLE = "__vehicle__"

# Natural-language surface forms mapped to canonical detector class names.
_CLASS_SYNONYMS: dict[str, str] = {
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "human": "person",
    "humans": "person",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "bus": "bus",
    "buses": "bus",
    "busses": "bus",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "cyclist": "bicycle",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "vehicle": ANY_VEHICLE,
    "vehicles": ANY_VEHICLE,
    "traffic": ANY_VEHICLE,
}

_DURATION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?)\b",
    re.IGNORECASE,
)
_COUNT_HINTS = ("count", "how many", "survey", "tally", "number of", "traffic")
_COUNT_NUMBER_PATTERN = re.compile(
    r"(?:at least|minimum of|min|>=)\s*(\d+)",
    re.IGNORECASE,
)


@dataclass(slots=True, frozen=True)
class MissionObjective:
    """A single measurable acceptance criterion derived from the task."""

    kind: str  # "observe" | "count" | "duration"
    description: str
    target_class: str | None = None
    min_count: int = 1
    min_duration_s: float = 0.0


@dataclass(slots=True, frozen=True)
class MissionSpec:
    """The measurable contract a mission must satisfy to count as a success."""

    raw_task: str
    objectives: tuple[MissionObjective, ...] = ()

    @property
    def is_measurable(self) -> bool:
        return bool(self.objectives)


@dataclass(slots=True, frozen=True)
class ObjectiveResult:
    objective: MissionObjective
    passed: bool
    detail: str


@dataclass(slots=True, frozen=True)
class MissionVerdict:
    """Outcome of checking a report against a :class:`MissionSpec`."""

    passed: bool
    measurable: bool
    results: tuple[ObjectiveResult, ...] = field(default_factory=tuple)
    summary: str = ""


def _observed_count(report: MissionReport, target_class: str | None) -> int:
    if target_class is None:
        return sum(report.unique_track_counts.values())
    if target_class == ANY_VEHICLE:
        if report.unique_vehicle_count:
            return report.unique_vehicle_count
        return sum(report.unique_track_counts.get(cls, 0) for cls in VEHICLE_CLASSES)
    return report.unique_track_counts.get(target_class, 0)


def _extract_classes(task_lower: str) -> list[str]:
    """Return canonical detector classes mentioned in the task, order-preserving."""
    seen: dict[str, None] = {}
    for word in re.findall(r"[a-z]+", task_lower):
        canonical = _CLASS_SYNONYMS.get(word)
        if canonical is not None and canonical not in seen:
            seen[canonical] = None
    return list(seen)


def _wants_count(task_lower: str) -> bool:
    return any(hint in task_lower for hint in _COUNT_HINTS)


def _parse_duration_s(task_lower: str) -> float:
    total = 0.0
    for value, unit in _DURATION_PATTERN.findall(task_lower):
        seconds = float(value)
        if unit.lower().startswith(("minute", "min")):
            seconds *= 60.0
        total = max(total, seconds)
    return total


def parse_mission_spec(task: str) -> MissionSpec:
    """Derive a measurable :class:`MissionSpec` from a natural-language task."""
    task_lower = task.lower()
    classes = _extract_classes(task_lower)
    wants_count = _wants_count(task_lower)
    duration_s = _parse_duration_s(task_lower)

    objectives: list[MissionObjective] = []

    if wants_count:
        min_count = 1
        number_match = _COUNT_NUMBER_PATTERN.search(task_lower)
        if number_match:
            min_count = max(1, int(number_match.group(1)))
        # Count missions target a class if named, otherwise any road vehicle.
        count_class = classes[0] if classes else ANY_VEHICLE
        objectives.append(
            MissionObjective(
                kind="count",
                description=f"count at least {min_count} unique {_label(count_class)}",
                target_class=count_class,
                min_count=min_count,
            )
        )
    else:
        for cls in classes:
            objectives.append(
                MissionObjective(
                    kind="observe",
                    description=f"observe at least one {_label(cls)}",
                    target_class=cls,
                )
            )

    if duration_s > 0.0:
        objectives.append(
            MissionObjective(
                kind="duration",
                description=f"remain on task for at least {duration_s:.0f}s",
                min_duration_s=duration_s,
            )
        )

    return MissionSpec(raw_task=task, objectives=tuple(objectives))


def _label(target_class: str | None) -> str:
    if target_class is None:
        return "objects"
    if target_class == ANY_VEHICLE:
        return "vehicles"
    return target_class


class MissionVerifier:
    """Check a finished mission report against its measurable contract."""

    def verify(self, spec: MissionSpec, report: MissionReport) -> MissionVerdict:
        if not spec.objectives:
            return MissionVerdict(
                passed=True,
                measurable=False,
                results=(),
                summary="no measurable objectives parsed from task; protocol-only completion",
            )

        results: list[ObjectiveResult] = []
        for objective in spec.objectives:
            results.append(self._check(objective, report))

        passed = all(item.passed for item in results)
        met = sum(1 for item in results if item.passed)
        summary = f"{met}/{len(results)} objectives met"
        return MissionVerdict(
            passed=passed,
            measurable=True,
            results=tuple(results),
            summary=summary,
        )

    @staticmethod
    def _check(objective: MissionObjective, report: MissionReport) -> ObjectiveResult:
        if objective.kind == "duration":
            ok = report.duration_s >= objective.min_duration_s
            return ObjectiveResult(
                objective=objective,
                passed=ok,
                detail=f"duration={report.duration_s:.1f}s "
                f"(required {objective.min_duration_s:.0f}s)",
            )
        observed = _observed_count(report, objective.target_class)
        required = objective.min_count if objective.kind == "count" else 1
        ok = observed >= required
        return ObjectiveResult(
            objective=objective,
            passed=ok,
            detail=f"observed {observed} {_label(objective.target_class)} (required {required})",
        )
