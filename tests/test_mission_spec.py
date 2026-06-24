from __future__ import annotations

from autonomy.contracts import MissionReport
from autonomy.mission_spec import (
    ANY_VEHICLE,
    MissionVerifier,
    parse_mission_spec,
)


def _report(
    *,
    duration_s: float = 0.0,
    track_counts: dict[str, int] | None = None,
    vehicle_count: int = 0,
) -> MissionReport:
    return MissionReport(
        mission_id="m",
        mode="SEARCH",
        state_transitions=[],
        target_ids_seen=[],
        completion_progress=1.0,
        success=True,
        completion_reason="completed",
        duration_s=duration_s,
        unique_track_counts=track_counts or {},
        unique_vehicle_count=vehicle_count,
    )


def test_parse_extracts_observe_and_duration_objectives() -> None:
    spec = parse_mission_spec("Find a person, follow for 20 seconds, then land.")

    kinds = {obj.kind for obj in spec.objectives}
    classes = {obj.target_class for obj in spec.objectives if obj.kind == "observe"}
    durations = [obj.min_duration_s for obj in spec.objectives if obj.kind == "duration"]

    assert "observe" in kinds
    assert "person" in classes
    assert durations == [20.0]


def test_parse_minutes_duration() -> None:
    spec = parse_mission_spec("Patrol and follow the truck for 2 minutes")
    durations = [obj.min_duration_s for obj in spec.objectives if obj.kind == "duration"]
    assert durations == [120.0]


def test_parse_count_mission_uses_vehicle_target_by_default() -> None:
    spec = parse_mission_spec("Count the traffic on the road and report")

    count_objs = [obj for obj in spec.objectives if obj.kind == "count"]
    assert len(count_objs) == 1
    assert count_objs[0].target_class == ANY_VEHICLE
    assert count_objs[0].min_count == 1


def test_parse_count_with_explicit_minimum() -> None:
    spec = parse_mission_spec("Survey the area and count at least 5 cars")
    count_objs = [obj for obj in spec.objectives if obj.kind == "count"]
    assert count_objs[0].target_class == "car"
    assert count_objs[0].min_count == 5


def test_unparseable_task_yields_no_objectives() -> None:
    spec = parse_mission_spec("Do something interesting out there")
    assert spec.is_measurable is False


def test_verifier_fails_when_required_class_never_seen() -> None:
    spec = parse_mission_spec("Find a truck and follow it")
    verdict = MissionVerifier().verify(spec, _report(track_counts={"car": 3}))

    assert verdict.measurable is True
    assert verdict.passed is False


def test_verifier_passes_when_objectives_met() -> None:
    spec = parse_mission_spec("Find a person, follow for 10 seconds")
    report = _report(duration_s=12.0, track_counts={"person": 1})

    verdict = MissionVerifier().verify(spec, report)

    assert verdict.passed is True
    assert "2/2" in verdict.summary


def test_verifier_count_uses_unique_vehicle_count() -> None:
    spec = parse_mission_spec("Count vehicles")
    verdict = MissionVerifier().verify(spec, _report(vehicle_count=4))
    assert verdict.passed is True


def test_verifier_duration_requires_elapsed_time() -> None:
    spec = parse_mission_spec("Follow the bus for 30 seconds")
    short = _report(duration_s=5.0, track_counts={"bus": 1})
    long = _report(duration_s=31.0, track_counts={"bus": 1})

    assert MissionVerifier().verify(spec, short).passed is False
    assert MissionVerifier().verify(spec, long).passed is True


def test_unmeasurable_spec_passes_protocol_only() -> None:
    spec = parse_mission_spec("freeform exploration")
    verdict = MissionVerifier().verify(spec, _report())
    assert verdict.passed is True
    assert verdict.measurable is False
