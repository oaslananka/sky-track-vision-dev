from __future__ import annotations

from skypilot.runtime_helpers import resolve_priority_class


def test_resolve_priority_class_prefers_explicit_target_in_task() -> None:
    result = resolve_priority_class(
        "Take off, scan the area, find a truck, follow it, then land.",
        ["person", "car", "truck", "bus"],
    )

    assert result == "truck"


def test_resolve_priority_class_supports_pedestrian_synonym() -> None:
    result = resolve_priority_class(
        "Scan for pedestrians and report",
        ["person", "car", "truck"],
    )

    assert result == "person"
