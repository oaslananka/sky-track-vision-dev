from __future__ import annotations

import pytest

from autonomy.contracts import MissionContext, MissionMode, MissionState, MotionPrimitive
from autonomy.mission import MissionFSM


def _context(state: MissionState) -> MissionContext:
    return MissionContext(
        mode=MissionMode.SEARCH,
        state=state,
        priority_class=None,
        target_id=None,
        progress=0.0,
        elapsed_s=0.0,
        operator_text="",
    )


@pytest.mark.parametrize(
    "state",
    [
        MissionState.IDLE,
        MissionState.SCAN,
        MissionState.TRACK,
        MissionState.REACQUIRE,
        MissionState.MONITOR,
        MissionState.ORBIT,
        MissionState.REPORT,
        MissionState.BLOCKED,
    ],
)
def test_emergency_is_reachable_from_every_state_via_graph(state: MissionState) -> None:
    fsm = MissionFSM(initial_state=state)

    fsm.transition(MissionState.EMERGENCY, reason="test")

    assert fsm.state is MissionState.EMERGENCY


def test_emergency_method_is_unconditional_and_idempotent() -> None:
    fsm = MissionFSM(initial_state=MissionState.ORBIT)

    changed = fsm.emergency(reason="geofence")
    again = fsm.emergency(reason="geofence")

    assert changed is True
    assert again is False  # already in EMERGENCY -> no-op
    assert fsm.state is MissionState.EMERGENCY


def test_emergency_recovers_only_to_idle() -> None:
    fsm = MissionFSM(initial_state=MissionState.TRACK)
    fsm.emergency(reason="battery")

    fsm.transition(MissionState.IDLE, reason="recovered")

    assert fsm.state is MissionState.IDLE


def test_emergency_motion_intent_is_safe_hover() -> None:
    fsm = MissionFSM(initial_state=MissionState.TRACK)
    fsm.emergency(reason="timeout")

    intent = fsm.get_motion_intent(_context(MissionState.EMERGENCY), target=None)

    assert intent.primitive is MotionPrimitive.HOVER
