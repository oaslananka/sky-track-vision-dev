from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import pytest

from autonomy.contracts import (
    LidarReading,
    MissionMode,
    MissionState,
    ProximityReading,
    SafetyState,
    SensorSnapshot,
    TelemetryReading,
    VelocityCmd,
)
from autonomy.mission import MissionFSM
from autonomy.safety import SafetyEvaluator
from config.settings import PilotConfig, SafetyConfig, VisionConfig
from skypilot.pilot_display import PilotDisplay, _LLMLogTail
from tests.conftest import make_target, make_telemetry


def _snapshot(*, altitude_m: float, down_m: float, timestamp_ns: int) -> SensorSnapshot:
    return SensorSnapshot(
        lidar=LidarReading(point_count=10, cluster_count=0, min_distance_m=5.0),
        proximity=ProximityReading(
            front_m=20.0,
            rear_m=20.0,
            left_m=20.0,
            right_m=20.0,
            down_m=down_m,
            available=True,
        ),
        telemetry=TelemetryReading(
            position_ned=(0.0, 0.0, -altitude_m),
            velocity_ned=(0.0, 0.0, 0.0),
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            altitude_m=altitude_m,
            gps_valid=True,
        ),
        timestamp_ns=timestamp_ns,
        missing_features=[],
    )


class _FakeSensorReader:
    def __init__(self, stale: SensorSnapshot, fresh: SensorSnapshot) -> None:
        self.last_snapshot = stale
        self._fresh = fresh
        self.read_calls = 0

    def read(self) -> SensorSnapshot:
        self.read_calls += 1
        self.last_snapshot = self._fresh
        return self._fresh


class _FakeBridge:
    def __init__(self) -> None:
        self._safety = SafetyEvaluator(SafetyConfig())
        self.move_calls: list[VelocityCmd] = []
        self.takeoff_calls = 0
        self.move_to_altitude_calls: list[float] = []

    def move(self, cmd: VelocityCmd, snapshot: SensorSnapshot | None = None) -> bool:
        del snapshot
        self.move_calls.append(cmd)
        return True

    def takeoff(self) -> None:
        self.takeoff_calls += 1

    def move_to_altitude(self, altitude_m: float) -> None:
        self.move_to_altitude_calls.append(altitude_m)


class _FakeController:
    def resolve(self, *args: object, **kwargs: object) -> VelocityCmd:
        del args, kwargs
        return VelocityCmd(0.25, 0.0, 0.0, 0.0, 0.1, "ibvs")

    def reset(self) -> None:
        return None


class _FakeScanController:
    def resolve(self, *args: object, **kwargs: object) -> VelocityCmd:
        del args, kwargs
        return VelocityCmd(0.65, 0.2, 0.0, 0.04, 0.1, "scan")

    def reset(self) -> None:
        return None


class _FakePerceptionWorker:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class _FakeReporter:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record_runtime_snapshot(
        self,
        mission: Any,
        *,
        detections: object,
        target: object,
        safety_reason: str | None,
        frame: object = None,
        timestamp_ns: object = None,
    ) -> None:
        self.records.append(
            {
                "mission_state": mission.state.value,
                "detections": detections,
                "target": target,
                "safety_reason": safety_reason,
                "frame": frame,
                "timestamp_ns": timestamp_ns,
            }
        )


@pytest.mark.usefixtures("monkeypatch")
def test_pilot_display_prefers_fresh_snapshot_over_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _snapshot(
        altitude_m=1.6,
        down_m=0.08,
        timestamp_ns=time.time_ns() - 40_000_000_000,
    )
    fresh = _snapshot(
        altitude_m=4.1,
        down_m=4.1,
        timestamp_ns=time.time_ns(),
    )
    sensor_reader = _FakeSensorReader(stale=stale, fresh=fresh)
    bridge = _FakeBridge()

    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._mission_id = "mission-test"
    display._priority_class = "truck"
    display._controller = _FakeController()  # type: ignore[assignment]
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._bridge = bridge
    display._sensor_reader = sensor_reader
    display._safety = bridge._safety
    display._last_target_seen_at = time.monotonic()
    display._last_recover_at = 0.0
    display._stuck_since = None
    display._unstick_flip = 1.0
    display._lock = threading.Lock()
    display._detections = []
    display._target = None
    display._last_cmd = None
    display._last_snapshot = None
    display._last_safety_reason = "waiting"
    display._last_safety_state = SafetyState.PATH_CLEAR
    display._tick_id = 1
    display._mission_start_time = time.monotonic()
    display._sensor_read_interval = 1

    monkeypatch.setattr("skypilot.pilot_display.time.sleep", lambda _seconds: None)

    display._execute_control(target=None, frame_w=640, frame_h=480)

    assert sensor_reader.read_calls == 1
    assert display._last_snapshot is fresh
    assert display._last_safety_state == SafetyState.PATH_CLEAR
    assert bridge.takeoff_calls == 0
    assert bridge.move_to_altitude_calls == []
    assert len(bridge.move_calls) == 1


def test_pilot_display_skips_altitude_recovery_during_startup_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(cruise_altitude_m=4.0, altitude_recovery_grace_s=3.0)
    display._mission_id = "mission-test"
    display._tick_id = 1
    display._last_recover_at = 0.0
    display._mission_start_time = 10.0
    display._altitude_recovery_grace_until = 13.0

    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 11.0)

    should_recover = display._should_recover_altitude(
        make_telemetry(altitude_m=1.7),
        MissionState.SCAN,
    )

    assert should_recover is False


def test_pilot_display_samples_high_frequency_debug_ticks() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(debug_tick_log_interval=5)
    display._tick_id = 1

    assert display._should_log_tick_debug() is True

    display._tick_id = 2
    assert display._should_log_tick_debug() is False

    display._tick_id = 5
    assert display._should_log_tick_debug() is True


def test_pilot_display_throttles_perception_inference_when_target_is_stable() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._vision_cfg = VisionConfig(track_inference_interval=3)
    display._frame_index = 2

    assert display._should_run_perception_inference(None) is False

    display._frame_index = 3
    assert display._should_run_perception_inference(None) is True

    display._frame_index = 2

    stable_target = make_target(frames_tracked=5)
    stable_target.frames_since_seen = 0

    assert display._should_run_perception_inference(stable_target) is False

    display._frame_index = 3
    assert display._should_run_perception_inference(stable_target) is True

    display._frame_index = 2


def test_pilot_display_records_runtime_snapshot_for_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = _snapshot(
        altitude_m=4.1,
        down_m=4.1,
        timestamp_ns=time.time_ns(),
    )
    sensor_reader = _FakeSensorReader(stale=fresh, fresh=fresh)
    bridge = _FakeBridge()
    reporter = _FakeReporter()
    target = make_target(track_id=12, frames_tracked=5)

    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._vision_cfg = VisionConfig()
    display._mission_id = "mission-test"
    display._priority_class = "car"
    display._controller = _FakeController()  # type: ignore[assignment]
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._bridge = bridge
    display._sensor_reader = sensor_reader
    display._safety = bridge._safety
    display._reporter = reporter
    display._last_target_seen_at = time.monotonic()
    display._last_recover_at = 0.0
    display._stuck_since = None
    display._unstick_flip = 1.0
    display._lock = threading.Lock()
    display._detections = []
    display._target = None
    display._last_cmd = None
    display._last_snapshot = None
    display._last_safety_reason = "clear"
    display._last_safety_state = SafetyState.PATH_CLEAR
    display._tick_id = 1
    display._mission_start_time = time.monotonic()
    display._sensor_read_interval = 1

    monkeypatch.setattr("skypilot.pilot_display.time.sleep", lambda _seconds: None)

    display._execute_control(
        target=target,
        frame_w=640,
        frame_h=480,
        detections=[target.detection],
    )

    assert reporter.records
    assert reporter.records[0]["mission_state"] in {"SCAN", "TRACK"}
    detections = reporter.records[0]["detections"]
    assert isinstance(detections, list) and len(detections) == 1
    assert reporter.records[0]["frame"] is None


def test_pilot_display_throttles_perception_in_tracking_when_lock_is_stable() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._vision_cfg = VisionConfig(track_inference_interval=4)
    display._fsm = MissionFSM(initial_state=MissionState.TRACK)

    stable_target = make_target(frames_tracked=6)
    stable_target.frames_since_seen = 0

    display._frame_index = 1
    assert display._should_run_perception_inference(stable_target) is False

    display._frame_index = 2
    assert display._should_run_perception_inference(stable_target) is True


def test_pilot_display_runs_full_rate_in_tracking_when_target_motion_is_high() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._vision_cfg = VisionConfig(track_inference_interval=4)
    display._fsm = MissionFSM(initial_state=MissionState.TRACK)
    display._frame_index = 1

    fast_target = make_target(frames_tracked=8)
    fast_target.frames_since_seen = 0
    fast_target.velocity_estimate = (2.0, 0.0)

    assert display._should_run_perception_inference(fast_target) is True


def test_pilot_display_runs_full_rate_in_tracking_when_target_near_edge() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._vision_cfg = VisionConfig(track_inference_interval=4)
    display._fsm = MissionFSM(initial_state=MissionState.TRACK)
    display._frame_index = 1

    edge_target = make_target(frames_tracked=8, center=(40.0, 240.0), frame_width=640)
    edge_target.frames_since_seen = 0

    assert display._should_run_perception_inference(edge_target) is True


def test_pilot_display_runs_perception_every_frame_when_safety_is_constrained() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._vision_cfg = VisionConfig(track_inference_interval=4)
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._frame_index = 2
    display._last_safety_state = SafetyState.OBSTACLE_AHEAD

    assert display._should_run_perception_inference(None) is True


def test_pilot_display_promotes_scan_to_track_when_priority_target_is_locked() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._mission_id = "mission-test"
    display._mission_mode = MissionMode.SEARCH
    display._priority_class = "truck"
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._tick_id = 1

    target = make_target(frames_tracked=5)
    target.detection.class_name = "truck"

    display._promote_locked_priority_target(target)

    assert display._fsm.state == MissionState.TRACK


def test_pilot_display_promotes_scan_to_track_for_bus_when_priority_is_truck() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._mission_id = "mission-test"
    display._mission_mode = MissionMode.SEARCH
    display._priority_class = "truck"
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._tick_id = 1

    target = make_target(frames_tracked=5)
    target.detection.class_name = "bus"

    display._promote_locked_priority_target(target)

    assert display._fsm.state == MissionState.TRACK


def test_pilot_display_does_not_promote_scan_to_track_for_car_when_priority_is_truck() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._mission_id = "mission-test"
    display._mission_mode = MissionMode.SEARCH
    display._priority_class = "truck"
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._tick_id = 1

    target = make_target(frames_tracked=5)
    target.detection.class_name = "car"

    display._promote_locked_priority_target(target)

    assert display._fsm.state == MissionState.SCAN


def test_pilot_display_does_not_promote_scan_to_track_in_traffic_monitor_mode() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._mission_id = "mission-test"
    display._mission_mode = MissionMode.TRAFFIC_MONITOR
    display._priority_class = "car"
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._tick_id = 1

    target = make_target(frames_tracked=5)
    target.detection.class_name = "car"

    display._promote_locked_priority_target(target)

    assert display._fsm.state == MissionState.SCAN


def test_pilot_display_forces_track_back_to_scan_in_traffic_monitor_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    sensor_reader = _FakeSensorReader(stale=fresh, fresh=fresh)
    bridge = _FakeBridge()

    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._vision_cfg = VisionConfig()
    display._mission_id = "mission-test"
    display._mission_mode = MissionMode.TRAFFIC_MONITOR
    display._priority_class = "car"
    display._controller = _FakeController()  # type: ignore[assignment]
    display._fsm = MissionFSM(initial_state=MissionState.TRACK)
    display._bridge = bridge
    display._sensor_reader = sensor_reader
    display._safety = bridge._safety
    display._last_target_seen_at = time.monotonic()
    display._last_recover_at = 0.0
    display._stuck_since = None
    display._unstick_flip = 1.0
    display._lock = threading.Lock()
    display._detections = []
    display._target = None
    display._last_cmd = None
    display._last_snapshot = None
    display._last_safety_reason = "clear"
    display._last_safety_state = SafetyState.PATH_CLEAR
    display._tick_id = 1
    display._mission_start_time = time.monotonic()
    display._sensor_read_interval = 1

    target = make_target(frames_tracked=10)
    target.detection.class_name = "car"

    monkeypatch.setattr("skypilot.pilot_display.time.sleep", lambda _seconds: None)

    display._execute_control(target=target, frame_w=640, frame_h=480)

    assert display._fsm.state == MissionState.SCAN


def test_pilot_display_traffic_monitor_scan_searches_in_place_when_road_signal_missing() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._tick_id = 10
    display._latest_nav_frame = None
    display._cached_road_bias = 0.0
    display._cached_road_confidence = 0.0
    display._cached_road_center_offset = 0.0
    display._last_road_bias_tick = 0
    display._road_seek_direction = 1.0

    cmd = display._traffic_monitor_scan_command(make_telemetry(altitude_m=3.2))

    assert cmd.source == "scan_road_search"
    assert cmd.vy == pytest.approx(0.0)
    assert cmd.vx == pytest.approx(display._cfg.traffic_monitor_road_search_speed)
    assert abs(cmd.yaw_rate) == pytest.approx(display._cfg.traffic_monitor_road_search_yaw_rate)


def test_pilot_display_traffic_monitor_scan_follows_visible_road_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._latest_nav_frame = None

    monkeypatch.setattr(
        display,
        "_scan_road_guidance_from_frame",
        lambda _frame: (0.22, 0.06, -0.18),
    )

    cmd = display._traffic_monitor_scan_command(make_telemetry(altitude_m=3.2))

    assert cmd.source == "scan_road_follow"
    assert cmd.vx == pytest.approx(display._cfg.traffic_monitor_road_follow_speed)
    assert cmd.vy == pytest.approx(0.0)
    assert abs(cmd.yaw_rate) > 0.01
    assert cmd.duration_s == pytest.approx(display._cfg.traffic_monitor_command_duration_s)


def test_pilot_display_traffic_monitor_scan_slows_when_vehicle_is_visible() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._tick_id = 10
    display._latest_nav_frame = None
    display._cached_road_bias = 0.0
    display._cached_road_confidence = 0.04
    display._cached_road_center_offset = 0.08
    display._last_road_bias_tick = 10
    display._road_seek_direction = 1.0

    target = make_target(frames_tracked=8)
    target.frames_since_seen = 0

    cmd = display._traffic_monitor_scan_command(make_telemetry(altitude_m=2.6), target=target)

    assert cmd.source == "scan_road_follow"
    assert cmd.vx == pytest.approx(0.28)
    assert cmd.duration_s == pytest.approx(display._cfg.traffic_monitor_command_duration_s)


def test_pilot_display_holds_redundant_traffic_monitor_commands_until_window_expires() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._mission_mode = MissionMode.TRAFFIC_MONITOR
    display._last_applied_cmd = VelocityCmd(0.42, 0.0, 0.0, 0.02, 0.24, "scan_road_follow")
    display._next_command_apply_at = 10.0

    same_cmd = VelocityCmd(0.42, 0.0, 0.0, 0.02, 0.24, "scan_road_follow")

    assert display._should_hold_traffic_monitor_command(same_cmd, 9.7) is True
    assert display._should_hold_traffic_monitor_command(same_cmd, 10.1) is False


def test_pilot_display_replaces_generic_scan_with_road_scan_in_traffic_monitor_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = _snapshot(altitude_m=3.2, down_m=3.2, timestamp_ns=time.time_ns())
    sensor_reader = _FakeSensorReader(stale=fresh, fresh=fresh)
    bridge = _FakeBridge()

    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig()
    display._vision_cfg = VisionConfig()
    display._mission_id = "mission-test"
    display._mission_mode = MissionMode.TRAFFIC_MONITOR
    display._priority_class = "car"
    display._controller = _FakeScanController()  # type: ignore[assignment]
    display._fsm = MissionFSM(initial_state=MissionState.SCAN)
    display._bridge = bridge
    display._sensor_reader = sensor_reader
    display._safety = bridge._safety
    display._last_target_seen_at = time.monotonic()
    display._last_recover_at = 0.0
    display._stuck_since = None
    display._unstick_flip = 1.0
    display._lock = threading.Lock()
    display._detections = []
    display._target = None
    display._last_cmd = None
    display._last_snapshot = None
    display._last_safety_reason = "clear"
    display._last_safety_state = SafetyState.PATH_CLEAR
    display._tick_id = 1
    display._reacquire_yaw_bias = 0.0
    display._reacquire_bias_until = 0.0
    display._scan_escape_until = 0.0
    display._latest_nav_frame = None
    display._cached_road_bias = 0.0
    display._cached_road_confidence = 0.0
    display._cached_road_center_offset = 0.0
    display._last_road_bias_tick = 0
    display._road_seek_direction = 1.0
    display._mission_start_time = time.monotonic()
    display._sensor_read_interval = 1
    display._altitude_recovery_grace_until = time.monotonic() - 1.0

    monkeypatch.setattr(
        display,
        "_traffic_monitor_scan_command",
        lambda telemetry, target=None: VelocityCmd(0.12, 0.0, 0.0, 0.2, 0.1, "scan_road_search"),
    )

    display._execute_control(target=None, frame_w=640, frame_h=480)

    assert bridge.move_calls[-1].source == "scan_road_search"


def test_pilot_display_window_init_falls_back_when_opencv_gui_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._window_enabled = True

    monkeypatch.setattr(
        "skypilot.pilot_display.cv2.namedWindow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cv2.error("highgui", "namedWindow", "not implemented")
        ),
    )

    assert display._initialize_window() is False


def test_llm_log_tail_formats_and_limits_recent_messages() -> None:
    tail = _LLMLogTail(max_lines=2, max_chars=24)

    llm_logger = logging.getLogger("skytrackvision.skypilot.llm")
    non_llm_logger = logging.getLogger("skytrackvision.skypilot.tools")

    record_1 = llm_logger.makeRecord(
        llm_logger.name,
        logging.DEBUG,
        __file__,
        1,
        "Sending LLM request",
        args=(),
        exc_info=None,
        extra={"event_name": "llm.request", "event_fields": {"model": "gpt-5-nano"}},
    )
    record_2 = llm_logger.makeRecord(
        llm_logger.name,
        logging.DEBUG,
        __file__,
        2,
        "Received LLM response",
        args=(),
        exc_info=None,
        extra={"event_name": "llm.response", "event_fields": {"latency_ms": 231.4}},
    )
    record_3 = non_llm_logger.makeRecord(
        non_llm_logger.name,
        logging.DEBUG,
        __file__,
        3,
        "Bridge move",
        args=(),
        exc_info=None,
        extra={"event_name": "bridge.command", "event_fields": {}},
    )
    record_4 = llm_logger.makeRecord(
        llm_logger.name,
        logging.WARNING,
        __file__,
        4,
        "LLM API error, retrying",
        args=(),
        exc_info=None,
        extra={"event_name": "llm.retry", "event_fields": {"attempt": 1}},
    )

    tail.handle(record_1)
    tail.handle(record_2)
    tail.handle(record_3)
    tail.handle(record_4)

    lines = tail.recent_lines()
    assert len(lines) == 2
    assert lines[0].startswith("<- ")
    assert lines[1].startswith("~ ")
    assert all("Bridge move" not in line for line in lines)


def test_llm_log_tail_includes_message_previews_and_tool_names() -> None:
    tail = _LLMLogTail(max_lines=4, max_chars=120)
    llm_logger = logging.getLogger("skytrackvision.skypilot.llm")

    request_record = llm_logger.makeRecord(
        llm_logger.name,
        logging.DEBUG,
        __file__,
        10,
        "Sending LLM request",
        args=(),
        exc_info=None,
        extra={
            "event_name": "llm.request",
            "event_fields": {
                "model": "gpt-5-nano",
                "tool_count": 12,
                "user_preview": "find and follow the nearest truck",
            },
        },
    )

    response_record = llm_logger.makeRecord(
        llm_logger.name,
        logging.DEBUG,
        __file__,
        11,
        "Received LLM response",
        args=(),
        exc_info=None,
        extra={
            "event_name": "llm.response",
            "event_fields": {
                "latency_ms": 412.7,
                "tool_calls": 2,
                "assistant_preview": "switching to scan and locking truck target",
                "tool_names": ["set_mode", "follow_target"],
            },
        },
    )

    tail.handle(request_record)
    tail.handle(response_record)

    lines = tail.recent_lines()
    assert len(lines) == 2
    assert 'ask:"find and follow' in lines[0]
    assert 'say:"switching to scan' in lines[1]
    assert "tools:set_mode,follow_target" in lines[1]


def test_pilot_display_motion_insight_reports_direction_and_heading() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    target = make_target(frames_tracked=6)
    target.velocity_estimate = (2.0, -1.0)

    speed_px, heading_deg, motion_label = display._target_motion_insight(target)

    assert speed_px > 2.2
    assert heading_deg < 0.0
    assert motion_label == "RIGHT-UP"


def test_pilot_display_applies_reacquire_scan_bias_when_recently_lost_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._reacquire_yaw_bias = 1.0
    display._reacquire_bias_until = 100.0

    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 10.0)

    base = VelocityCmd(0.1, 0.0, 0.0, 0.3, 0.1, "scan")
    adjusted = display._apply_reacquire_scan_bias(base, MissionState.SCAN)

    assert adjusted.source == "scan_biased"
    assert adjusted.yaw_rate > base.yaw_rate


def test_pilot_display_scan_escape_engages_after_long_target_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(
        scan_escape_trigger_s=5.0,
        scan_escape_duration_s=2.0,
        scan_escape_forward_speed=1.1,
        scan_escape_yaw_rate=0.02,
    )
    display._mission_id = "mission-test"
    display._tick_id = 3
    display._scan_escape_until = 0.0
    display._last_target_seen_at = 10.0

    telemetry = TelemetryReading(
        position_ned=(0.0, 0.0, -4.0),
        velocity_ned=(0.0, 0.0, 0.0),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_m=4.0,
        gps_valid=True,
    )
    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 20.0)

    cmd = display._fallback_patrol_cmd(MissionState.SCAN, telemetry)

    assert cmd.source == "scan_escape"
    assert cmd.vx == pytest.approx(1.1)
    assert abs(cmd.yaw_rate) <= 0.03


def test_pilot_display_scan_escape_does_not_engage_when_target_recent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(scan_escape_trigger_s=5.0, scan_straight_trigger_s=2.0)
    display._scan_escape_until = 0.0
    display._last_target_seen_at = 18.5

    telemetry = TelemetryReading(
        position_ned=(0.0, 0.0, -4.0),
        velocity_ned=(0.0, 0.0, 0.0),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_m=4.0,
        gps_valid=True,
    )
    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 20.0)

    cmd = display._fallback_patrol_cmd(MissionState.SCAN, telemetry)

    assert cmd.source == "scan_patrol"


def test_pilot_display_scan_straight_engages_before_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(
        scan_straight_trigger_s=2.0,
        scan_straight_forward_speed=0.92,
        scan_straight_yaw_rate=0.01,
        scan_escape_trigger_s=6.0,
    )
    display._scan_escape_until = 0.0
    display._last_target_seen_at = 10.0

    telemetry = TelemetryReading(
        position_ned=(0.0, 0.0, -4.0),
        velocity_ned=(0.0, 0.0, 0.0),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_m=4.0,
        gps_valid=True,
    )
    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 13.0)

    cmd = display._fallback_patrol_cmd(MissionState.SCAN, telemetry)

    assert cmd.source == "scan_straight"
    assert cmd.vx == pytest.approx(0.92)
    assert cmd.vy == pytest.approx(0.0)
    assert abs(cmd.yaw_rate) <= 0.02


def test_pilot_display_scan_escape_takes_precedence_when_triggered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(
        scan_straight_trigger_s=2.0,
        scan_escape_trigger_s=5.0,
        scan_escape_forward_speed=1.05,
    )
    display._mission_id = "mission-test"
    display._tick_id = 8
    display._scan_escape_until = 0.0
    display._last_target_seen_at = 10.0

    telemetry = TelemetryReading(
        position_ned=(0.0, 0.0, -4.0),
        velocity_ned=(0.0, 0.0, 0.0),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_m=4.0,
        gps_valid=True,
    )
    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 16.0)

    cmd = display._fallback_patrol_cmd(MissionState.SCAN, telemetry)

    assert cmd.source == "scan_escape"
    assert cmd.vx == pytest.approx(1.05)


def test_pilot_display_track_fallback_uses_forward_search_not_scan_patrol() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(
        scan_straight_forward_speed=0.9,
        scan_straight_yaw_rate=0.02,
    )
    display._scan_escape_until = 0.0
    display._last_target_seen_at = 0.0

    telemetry = TelemetryReading(
        position_ned=(0.0, 0.0, -4.0),
        velocity_ned=(0.0, 0.0, 0.0),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_m=4.0,
        gps_valid=True,
    )

    cmd = display._fallback_patrol_cmd(MissionState.TRACK, telemetry)

    assert cmd.source == "track_search"
    assert cmd.vx >= 0.8
    assert cmd.vy == pytest.approx(0.0)
    assert abs(cmd.yaw_rate) <= 0.1


def test_pilot_display_turns_in_place_when_front_obstacle_blocks_scan() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(scan_yaw_rate=0.06)
    display._last_safety_state = SafetyState.OBSTACLE_AHEAD

    snapshot = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    snapshot.proximity.front_m = 0.9
    snapshot.proximity.left_m = 0.7
    snapshot.proximity.right_m = 2.5
    base = VelocityCmd(0.8, 0.0, 0.0, 0.05, 0.1, "scan")

    adjusted = display._apply_safety_turn_if_needed(base, snapshot, MissionState.SCAN)

    assert adjusted.source == "safety_turn"
    assert adjusted.vx == pytest.approx(0.0)
    assert adjusted.vy == pytest.approx(0.0)
    assert adjusted.vz == pytest.approx(0.0)
    assert adjusted.yaw_rate > 0.0


def test_pilot_display_disables_unstick_when_safety_is_blocking() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(unstick_timeout_s=0.2)
    display._last_safety_state = SafetyState.OBSTACLE_AHEAD
    display._stuck_since = time.monotonic() - 1.0
    display._unstick_flip = 1.0

    cmd, used = display._apply_unstick_if_needed(
        VelocityCmd(0.6, 0.1, 0.0, 0.1, 0.1, "scan"),
        make_telemetry(velocity_ned=(0.0, 0.0, 0.0), altitude_m=4.0),
        MissionState.SCAN,
    )

    assert used is False
    assert cmd.source == "scan"
    assert display._stuck_since is None


def test_pilot_display_keeps_track_during_brief_target_loss_within_sticky_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    fresh = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    sensor_reader = _FakeSensorReader(stale=stale, fresh=fresh)
    bridge = _FakeBridge()

    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(reacquire_timeout_s=0.5)
    display._vision_cfg = VisionConfig(sticky_lock_timeout_frames=10)
    display._mission_id = "mission-test"
    display._priority_class = "truck"
    display._controller = _FakeController()  # type: ignore[assignment]
    display._fsm = MissionFSM(initial_state=MissionState.TRACK)
    display._bridge = bridge
    display._sensor_reader = sensor_reader
    display._safety = bridge._safety
    display._last_target_seen_at = 10.0
    display._last_recover_at = 0.0
    display._stuck_since = None
    display._unstick_flip = 1.0
    display._lock = threading.Lock()
    display._detections = []
    display._target = None
    display._last_cmd = None
    display._last_snapshot = None
    display._last_safety_reason = "waiting"
    display._last_safety_state = SafetyState.PATH_CLEAR
    display._tick_id = 1
    display._reacquire_yaw_bias = 0.0
    display._reacquire_bias_until = 0.0
    display._scan_escape_until = 0.0
    display._latest_nav_frame = None
    display._cached_road_bias = 0.0
    display._last_road_bias_tick = 0
    display._mission_start_time = time.monotonic()
    display._sensor_read_interval = 1

    target = make_target(frames_tracked=20)
    target.is_confirmed = False
    target.frames_since_seen = 3

    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 20.0)

    display._execute_control(target=target, frame_w=640, frame_h=480)

    assert display._fsm.state == MissionState.TRACK


def test_pilot_display_keeps_confirmed_track_beyond_two_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    fresh = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    sensor_reader = _FakeSensorReader(stale=stale, fresh=fresh)
    bridge = _FakeBridge()

    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(reacquire_timeout_s=0.5)
    display._vision_cfg = VisionConfig(sticky_lock_timeout_frames=10)
    display._mission_id = "mission-test"
    display._priority_class = "truck"
    display._controller = _FakeController()  # type: ignore[assignment]
    display._fsm = MissionFSM(initial_state=MissionState.TRACK)
    display._bridge = bridge
    display._sensor_reader = sensor_reader
    display._safety = bridge._safety
    display._last_target_seen_at = 10.0
    display._last_recover_at = 0.0
    display._stuck_since = None
    display._unstick_flip = 1.0
    display._lock = threading.Lock()
    display._detections = []
    display._target = None
    display._last_cmd = None
    display._last_snapshot = None
    display._last_safety_reason = "waiting"
    display._last_safety_state = SafetyState.PATH_CLEAR
    display._tick_id = 1
    display._reacquire_yaw_bias = 0.0
    display._reacquire_bias_until = 0.0
    display._scan_escape_until = 0.0
    display._latest_nav_frame = None
    display._cached_road_bias = 0.0
    display._last_road_bias_tick = 0
    display._track_lock_started_at = 10.0
    display._mission_start_time = time.monotonic()
    display._sensor_read_interval = 1

    target = make_target(track_id=41, frames_tracked=25)
    target.detection.class_name = "truck"
    target.is_confirmed = True
    target.frames_since_seen = 0

    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 20.0)

    display._execute_control(target=target, frame_w=640, frame_h=480)

    assert display._fsm.state == MissionState.TRACK


def test_pilot_display_enters_reacquire_when_target_loss_exceeds_sticky_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    fresh = _snapshot(altitude_m=4.0, down_m=4.0, timestamp_ns=time.time_ns())
    sensor_reader = _FakeSensorReader(stale=stale, fresh=fresh)
    bridge = _FakeBridge()

    display = PilotDisplay.__new__(PilotDisplay)
    display._cfg = PilotConfig(reacquire_timeout_s=0.5)
    display._vision_cfg = VisionConfig(sticky_lock_timeout_frames=2)
    display._mission_id = "mission-test"
    display._priority_class = "truck"
    display._controller = _FakeController()  # type: ignore[assignment]
    display._fsm = MissionFSM(initial_state=MissionState.TRACK)
    display._bridge = bridge
    display._sensor_reader = sensor_reader
    display._safety = bridge._safety
    display._last_target_seen_at = 10.0
    display._last_recover_at = 0.0
    display._stuck_since = None
    display._unstick_flip = 1.0
    display._lock = threading.Lock()
    display._detections = []
    display._target = None
    display._last_cmd = None
    display._last_snapshot = None
    display._last_safety_reason = "waiting"
    display._last_safety_state = SafetyState.PATH_CLEAR
    display._tick_id = 1
    display._reacquire_yaw_bias = 0.0
    display._reacquire_bias_until = 0.0
    display._scan_escape_until = 0.0
    display._latest_nav_frame = None
    display._cached_road_bias = 0.0
    display._last_road_bias_tick = 0
    display._mission_start_time = time.monotonic()
    display._sensor_read_interval = 1

    target = make_target(frames_tracked=20)
    target.is_confirmed = False
    target.frames_since_seen = 5

    monkeypatch.setattr("skypilot.pilot_display.time.monotonic", lambda: 20.0)

    display._execute_control(target=target, frame_w=640, frame_h=480)

    assert display._fsm.state == MissionState.REACQUIRE


def test_pilot_display_starts_and_stops_perception_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._running = False
    display._llm_tail = _LLMLogTail(max_lines=2)
    display._llm_logger = logging.getLogger("skytrackvision.skypilot.llm")
    display._perception_worker = _FakePerceptionWorker()  # type: ignore[assignment]
    display._window_enabled = False
    display._thread = None

    class _FakeCameraThread:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    display._cam_thread = _FakeCameraThread()  # type: ignore[assignment]

    monkeypatch.setattr(PilotDisplay, "_run_loop", lambda self: None)

    display.start()
    display.stop()

    assert display._perception_worker.started == 1  # type: ignore[attr-defined]
    assert display._perception_worker.stopped == 1  # type: ignore[attr-defined]


def test_pilot_display_fit_text_to_width_truncates_long_line() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._text_cache = {}
    long_line = "-> gpt-5-nano request tools=12 ask: very long command text that should clip"

    fitted = display._fit_text_to_width(long_line, 180, scale=0.41)

    assert fitted.endswith("...")
    assert len(fitted) < len(long_line)


def test_pilot_display_fit_text_to_width_keeps_short_line() -> None:
    display = PilotDisplay.__new__(PilotDisplay)
    display._text_cache = {}
    short_line = "<- response 450ms"

    fitted = display._fit_text_to_width(short_line, 400, scale=0.41)

    assert fitted == short_line
