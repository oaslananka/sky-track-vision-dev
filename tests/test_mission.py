from __future__ import annotations

import pytest

from autonomy.contracts import MissionContext, MissionMode, MissionState
from autonomy.mission import InvalidTransitionError, MissionFSM
from tests.conftest import make_target


def test_mission_rejects_invalid_transition() -> None:
    fsm = MissionFSM(initial_state=MissionState.IDLE)

    with pytest.raises(InvalidTransitionError):
        fsm.transition(MissionState.REPORT)


def test_mission_follow_intent_requires_confirmed_target() -> None:
    fsm = MissionFSM(initial_state=MissionState.TRACK)
    context = MissionContext(
        mode=MissionMode.PEDESTRIAN_WATCH,
        state=MissionState.TRACK,
        priority_class="person",
        target_id=None,
        progress=0.2,
        elapsed_s=1.0,
        operator_text="follow person",
    )

    intent = fsm.get_motion_intent(context, make_target(frames_tracked=5))

    assert intent.primitive.value == "FOLLOW"
    assert intent.target_id == 7


def test_mission_track_without_confirmed_target_falls_back_to_hover() -> None:
    fsm = MissionFSM(initial_state=MissionState.TRACK)
    context = MissionContext(
        mode=MissionMode.PEDESTRIAN_WATCH,
        state=MissionState.TRACK,
        priority_class="person",
        target_id=None,
        progress=0.0,
        elapsed_s=0.1,
        operator_text="follow person",
    )

    intent = fsm.get_motion_intent(context, make_target(frames_tracked=1))

    assert intent.primitive.value == "HOVER"


def test_mission_allows_transition_from_track_to_orbit() -> None:
    fsm = MissionFSM(initial_state=MissionState.TRACK)

    fsm.transition(MissionState.ORBIT, reason="test")

    assert fsm.state == MissionState.ORBIT


def test_forced_scan_recovery_records_transition_log() -> None:
    """Forced-SCAN fallback in get_motion_intent must add an entry to _transition_log."""
    import time

    from autonomy.contracts import MissionContext, MissionMode

    fsm = MissionFSM(initial_state=MissionState.REACQUIRE)
    # Artificially expire the state timeout so check_timeout_recovery returns a recovery state.
    object.__setattr__(fsm, "_state_entered_at", time.monotonic() - 9999.0)
    # Cooldown must be expired too.
    object.__setattr__(fsm, "_timeout_recovery_cooldown", time.monotonic() - 9999.0)

    context = MissionContext(
        mode=MissionMode.PEDESTRIAN_WATCH,
        state=MissionState.REACQUIRE,
        priority_class=None,
        target_id=None,
        progress=0.0,
        elapsed_s=0.0,
        operator_text="",
    )
    # REACQUIRE times out → recovery = SCAN.  get_motion_intent triggers forced transition.
    fsm.get_motion_intent(context, None)

    state_names = [entry[0] for entry in fsm._transition_log]
    assert MissionState.SCAN.value in state_names


def test_same_state_transition_is_silent_no_op() -> None:
    """Transitioning to the current state must not raise and must not append to transition_log."""
    fsm = MissionFSM(initial_state=MissionState.SCAN)
    initial_log_len = len(fsm._transition_log)

    fsm.transition(MissionState.SCAN, reason="redundant")

    assert fsm.state == MissionState.SCAN
    assert len(fsm._transition_log) == initial_log_len
