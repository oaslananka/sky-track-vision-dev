from __future__ import annotations

from autonomy.contracts import (
    MissionContext,
    MissionMode,
    MissionState,
    SafetyEvaluation,
    SafetyState,
)
from autonomy.reporting import EventReporter
from tests.conftest import make_target


def test_event_reporter_finalize_handles_transition_entries_after_update() -> None:
    reporter = EventReporter()
    mission = MissionContext(
        mode=MissionMode.SEARCH,
        state=MissionState.SCAN,
        priority_class="truck",
        target_id=7,
        progress=0.4,
        elapsed_s=2.0,
        operator_text="scan",
    )
    safety = SafetyEvaluation(
        state=SafetyState.PATH_CLEAR,
        blocked_directions=[],
        allow_forward=True,
        allow_descent=True,
        reason="clear",
    )

    reporter.update(mission, make_target(track_id=7), safety)
    report = reporter.finalize(success=True, reason="completed")

    assert report.success is True
    assert len(report.state_transitions) == 1
    assert report.state_transitions[0][0] == MissionState.SCAN.value
    assert 7 in report.target_ids_seen
