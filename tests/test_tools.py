from __future__ import annotations

import asyncio

from autonomy.contracts import MissionState
from autonomy.energy import BatteryModel
from autonomy.mission import MissionFSM
from autonomy.mission_spec import parse_mission_spec
from autonomy.reporting import EventReporter
from autonomy.watchdog import MissionWatchdog
from config.settings import WatchdogConfig
from skypilot.tools import ToolDispatcher


class FakeBridge:
    def __init__(self) -> None:
        self.hover_requested = False
        self.return_home_requested = False
        self.land_requested = False
        self.takeoff_calls = 0
        self.airborne = False

    def request_hover(self) -> None:
        self.hover_requested = True

    def takeoff(self) -> None:
        self.takeoff_calls += 1
        return None

    def is_airborne(self) -> bool:
        return self.airborne

    def land(self) -> None:
        self.land_requested = True
        return None

    def move_to_altitude(self, altitude_m: float) -> None:
        del altitude_m
        return None

    def return_to_home(self) -> None:
        self.return_home_requested = True
        return None

    @property
    def home_position(self) -> tuple[float, float, float] | None:
        return (0.0, 0.0, 0.0)


def test_request_follow_rejects_unconfirmed_or_wrong_class_target() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(),
        lambda: {
            "priority_class": "truck",
            "target": {
                "track_id": 21,
                "class_name": "car",
                "is_confirmed": True,
                "priority_match": False,
            },
        },
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_follow({}))

    assert result["ok"] is False
    assert result["mission_state"] == "SCAN"


def test_request_follow_uses_path_resolution_from_idle() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.IDLE),
        lambda: {
            "priority_class": "truck",
            "target": {
                "track_id": 21,
                "class_name": "truck",
                "is_confirmed": True,
                "priority_match": True,
            },
        },
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_follow({}))

    assert result["ok"] is True
    assert result["mission_state"] == "TRACK"


def test_wait_seconds_fails_fast_when_tracking_priority_target_is_lost() -> None:
    scene = {
        "priority_class": "truck",
        "target": {
            "track_id": 5,
            "class_name": "truck",
            "is_confirmed": False,
            "priority_match": False,
        },
    }
    dispatcher = ToolDispatcher(
        MissionFSM(),
        lambda: scene,
        FakeBridge(),
        EventReporter(),
    )
    dispatcher._fsm.transition(MissionState.SCAN, reason="setup")
    dispatcher._fsm.transition(MissionState.TRACK, reason="setup")

    result = asyncio.run(dispatcher._wait_seconds({"seconds": 5}))

    assert result["ok"] is False
    assert result["mission_state"] == "SCAN"
    assert "tracking interrupted" in result["message"].lower()


def test_wait_seconds_rejects_when_not_in_active_mission_phase() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.IDLE),
        lambda: {
            "priority_class": "truck",
            "target": {
                "track_id": 55,
                "class_name": "truck",
                "is_confirmed": True,
                "priority_match": True,
            },
        },
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._wait_seconds({"seconds": 30}))

    assert result["ok"] is False
    assert result["mission_state"] == "IDLE"
    assert (
        "TRACK" in result["message"]
        and "SCAN" in result["message"]
        and "MONITOR" in result["message"]
    )


def test_request_return_home_rejects_outside_idle_or_report() -> None:
    bridge = FakeBridge()
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.SCAN),
        lambda: {"priority_class": "truck", "target": {}},
        bridge,
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_return_home({}))

    assert result["ok"] is False
    assert bridge.return_home_requested is False


def test_request_land_rejects_outside_idle_or_report() -> None:
    bridge = FakeBridge()
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.SCAN),
        lambda: {"priority_class": "truck", "target": {}},
        bridge,
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_land({}))

    assert result["ok"] is False
    assert bridge.land_requested is False


def test_set_mission_state_routes_track_to_report_via_monitor() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.TRACK),
        lambda: {"priority_class": "truck", "target": {}},
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._set_mission_state({"state": "REPORT"}))

    assert result["ok"] is True
    assert result["mission_state"] == "REPORT"


def test_set_mission_state_reaches_orbit_from_blocked_via_path_resolution() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.BLOCKED),
        lambda: {"priority_class": "truck", "target": {}},
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._set_mission_state({"state": "ORBIT"}))

    assert result["ok"] is True
    assert result["mission_state"] == "ORBIT"


def test_request_scan_rejected_when_priority_target_is_locked_in_track() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.TRACK),
        lambda: {
            "priority_class": "truck",
            "target": {
                "track_id": 91,
                "class_name": "truck",
                "is_confirmed": True,
                "priority_match": True,
            },
        },
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_scan({}))

    assert result["ok"] is False
    assert result["mission_state"] == "TRACK"


def test_set_mission_state_rejects_track_to_scan_while_lock_is_active() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.TRACK),
        lambda: {
            "priority_class": "truck",
            "target": {
                "track_id": 73,
                "class_name": "truck",
                "is_confirmed": True,
                "priority_match": True,
            },
        },
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._set_mission_state({"state": "SCAN"}))

    assert result["ok"] is False
    assert result["mission_state"] == "TRACK"


def test_request_scan_rejected_during_brief_tracking_loss_grace() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.TRACK),
        lambda: {
            "priority_class": "truck",
            "target": {
                "track_id": 42,
                "class_name": "truck",
                "is_confirmed": False,
                "priority_match": True,
                "frames_since_seen": 2,
            },
        },
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_scan({}))

    assert result["ok"] is False
    assert result["mission_state"] == "TRACK"


def test_request_scan_uses_path_resolution_from_monitor() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.MONITOR),
        lambda: {
            "priority_class": "truck",
            "target": {
                "track_id": 42,
                "class_name": "truck",
                "is_confirmed": False,
                "priority_match": False,
                "frames_since_seen": 6,
            },
        },
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_scan({}))

    assert result["ok"] is True
    assert result["mission_state"] == "SCAN"


def test_request_takeoff_rejected_when_drone_is_already_airborne() -> None:
    bridge = FakeBridge()
    bridge.airborne = True
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.SCAN),
        lambda: {"priority_class": "truck", "target": {}},
        bridge,
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_takeoff({}))

    assert result["ok"] is False
    assert bridge.takeoff_calls == 0


def test_wait_seconds_returns_scene_heartbeat_on_success() -> None:
    scene = {
        "priority_class": "truck",
        "target": {
            "track_id": 5,
            "class_name": "truck",
            "is_confirmed": True,
            "priority_match": True,
        },
    }
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.SCAN),
        lambda: scene,
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._wait_seconds({"seconds": 1}))

    assert result["ok"] is True
    assert result["scene_state"] == scene


def test_request_move_to_altitude_rejects_nonnumeric_argument() -> None:
    """float() conversion of a non-numeric LLM argument must return ok=False, not raise."""
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.SCAN),
        lambda: {"mission_mode": "SEARCH"},
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_move_to_altitude({"altitude_m": "high"}))

    assert result["ok"] is False
    assert "altitude_m" in result["message"].lower()


def test_request_move_to_altitude_rejected_in_traffic_monitor_mode() -> None:
    """Altitude override is blocked when mission_mode is TRAFFIC_MONITOR."""
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.SCAN),
        lambda: {"mission_mode": "TRAFFIC_MONITOR"},
        FakeBridge(),
        EventReporter(),
    )

    result = asyncio.run(dispatcher._request_move_to_altitude({"altitude_m": 1.5}))

    assert result["ok"] is False
    assert "TRAFFIC_MONITOR" in result["message"]


def test_transition_along_path_returns_error_dict_on_partial_failure() -> None:
    """_transition_along_path must return an error dict when an intermediate step fails."""

    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.IDLE),
        lambda: {"mission_mode": "SEARCH"},
        FakeBridge(),
        EventReporter(),
    )

    # IDLE -> REPORT is not a valid transition; the path would fail.
    # Use a path with an impossible intermediate step.
    result = dispatcher._transition_along_path(
        [MissionState.REPORT],  # IDLE -> REPORT is not allowed
        reason="test",
    )

    assert result is not None
    assert result["ok"] is False
    assert "mission_state" in result


class _EnvelopeBridge:
    home_position = (0.0, 0.0, -5.0)

    def read_sensor_snapshot(self, refresh: bool = True) -> object:
        del refresh

        class _Tele:
            position_ned = (0.0, 0.0, -5.0)

        class _Snap:
            telemetry = _Tele()

        return _Snap()

    def request_hover(self) -> None:
        return None


def test_get_mission_progress_reports_unmet_objectives() -> None:
    dispatcher = ToolDispatcher(
        MissionFSM(initial_state=MissionState.SCAN),
        lambda: {"target": {}},
        FakeBridge(),
        EventReporter(),
        spec=parse_mission_spec("Find a truck and follow it"),
    )

    result = asyncio.run(dispatcher._get_mission_progress({}))

    assert result["ok"] is True
    data = result["data"]
    assert data["measurable"] is True
    assert data["all_objectives_met"] is False  # no truck observed yet
    assert any("truck" in obj["description"] for obj in data["objectives"])


def test_low_battery_aborts_wait_into_emergency() -> None:
    fsm = MissionFSM(initial_state=MissionState.SCAN)
    dispatcher = ToolDispatcher(
        fsm,
        lambda: {"target": {}},
        _EnvelopeBridge(),
        EventReporter(),
        watchdog=MissionWatchdog(WatchdogConfig(battery_rtl_fraction=0.5)),
        battery_source=BatteryModel(0.001),  # drains to empty immediately
    )

    result = asyncio.run(dispatcher._wait_seconds({"seconds": 2}))

    assert result["ok"] is False
    assert result["watchdog_trigger"] == "battery"
    assert fsm.state is MissionState.EMERGENCY
