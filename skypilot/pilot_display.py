from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections import Counter, deque
from typing import Any

import cv2
import numpy as np

from airsim_control.camera import DroneCameraStream
from airsim_control.camera_buffer import CameraBuffer, CameraThread
from autonomy.contracts import (
    Detection,
    MissionContext,
    MissionMode,
    MissionState,
    SafetyState,
    SensorSnapshot,
    TelemetryReading,
    TrackedTarget,
    VelocityCmd,
)
from autonomy.follow_controller import FollowController
from autonomy.ibvs import IBVSController
from autonomy.mission import MissionFSM
from autonomy.safety import SafetyEvaluator
from autonomy.targeting import is_priority_compatible
from config.runtime_logging import log_event
from config.settings import PilotConfig, VisionConfig
from ui.recorder import RealtimeRecorder
from vision.detector import MultiClassDetector
from vision.perception_worker import PerceptionWorker
from vision.tracker import KalmanTracker

logger = logging.getLogger("skytrackvision.skypilot.display")

# ── Colors (BGR) ──────────────────────────────────────────────────
_GREEN = (48, 214, 106)
_YELLOW = (0, 220, 255)
_RED = (60, 60, 255)
_CYAN = (255, 220, 0)
_WHITE = (240, 240, 240)
_DARK = (18, 18, 18)
_ORANGE = (40, 140, 255)
_MAGENTA = (255, 60, 200)
_LIME = (80, 255, 80)
_DIM = (160, 160, 160)

_SAFETY_COLORS = {
    SafetyState.PATH_CLEAR: _GREEN,
    SafetyState.OBSTACLE_AHEAD: _RED,
    SafetyState.OBSTACLE_CLUSTER: _ORANGE,
    SafetyState.ALTITUDE_LOW: _ORANGE,
    SafetyState.LANDING_CAUTION: _YELLOW,
    SafetyState.SAFETY_OVERRIDE: _RED,
    SafetyState.REPOSITION_SUGGEST: _YELLOW,
}

_STATE_COLORS = {
    MissionState.IDLE: _DIM,
    MissionState.SCAN: _CYAN,
    MissionState.TRACK: _GREEN,
    MissionState.REACQUIRE: _ORANGE,
    MissionState.MONITOR: _YELLOW,
    MissionState.ORBIT: _MAGENTA,
    MissionState.REPORT: _WHITE,
    MissionState.BLOCKED: _RED,
}


class PilotDisplay:
    """Real-time pilot HUD + autonomous control loop.

    Runs YOLO detection, Kalman tracking, and FSM-based movement in a background thread.
    The LLM sets FSM state via tools → this loop physically moves the drone.
    """

    _WINDOW_NAME = "SkyTrackVision - Pilot HUD"

    def __init__(
        self,
        camera: DroneCameraStream,
        vision_cfg: VisionConfig,
        pilot_cfg: PilotConfig,
        ibvs: IBVSController,
        fsm: MissionFSM,
        bridge: Any,
        sensor_reader: Any,
        reporter: Any | None = None,
        *,
        mission_id: str,
        priority_class: str = "truck",
        mission_mode: MissionMode = MissionMode.SEARCH,
        record_path: str | None = None,
        record_fps: int = 30,
    ) -> None:
        self._camera = camera
        self._cfg = pilot_cfg
        self._vision_cfg = vision_cfg
        self._mission_id = mission_id
        self._priority_class = priority_class
        self._mission_mode = mission_mode
        self._detector: MultiClassDetector | None = None
        self._tracker: KalmanTracker | None = None
        self._perception_worker = PerceptionWorker(vision_cfg, priority_class=priority_class)
        self._controller = FollowController(ibvs, pilot_cfg)
        self._fsm = fsm
        self._bridge = bridge
        self._sensor_reader = sensor_reader
        self._reporter = reporter
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps = _FPSCounter()
        self._safety = getattr(bridge, "_safety", None)
        self._last_target_seen_at = time.monotonic()
        self._last_recover_at = 0.0
        self._stuck_since: float | None = None
        self._unstick_flip = 1.0
        self._mission_start_time = time.monotonic()
        self._altitude_recovery_grace_until = (
            self._mission_start_time + self._cfg.altitude_recovery_grace_s
        )

        # Camera thread decoupling — fetch frames off the main loop
        self._cam_buf = CameraBuffer()
        self._cam_thread = CameraThread(camera, self._cam_buf)

        # Shared state
        self._lock = threading.Lock()
        self._detections: list[Detection] = []
        self._target: TrackedTarget | None = None
        self._last_cmd: VelocityCmd | None = None
        self._last_applied_cmd: VelocityCmd | None = None
        self._last_snapshot: SensorSnapshot | None = None
        self._last_safety_reason = "waiting for telemetry"
        self._last_safety_state = SafetyState.PATH_CLEAR
        self._tick_id = 0
        self._frame_index = 0
        self._cached_detections: list[Detection] = []
        self._window_enabled = True
        self._llm_tail = _LLMLogTail()
        self._llm_logger = logging.getLogger("skytrackvision.skypilot.llm")
        self._reacquire_yaw_bias = 0.0
        self._reacquire_bias_until = 0.0
        self._scan_escape_until = 0.0
        self._track_lock_started_at: float | None = None
        self._latest_nav_frame: np.ndarray | None = None
        self._cached_road_bias = 0.0
        self._cached_road_confidence = 0.0
        self._cached_road_center_offset = 0.0
        self._last_road_bias_tick = 0
        self._road_seek_direction = 1.0
        self._panel_mask: np.ndarray | None = None
        self._panel_mask_shape: tuple[int, int] = (0, 0)
        self._text_cache: dict[tuple[str, int, float, int], str] = {}
        self._sensor_read_interval = (
            self._cfg.traffic_monitor_sensor_read_interval
            if mission_mode == MissionMode.TRAFFIC_MONITOR
            else 3
        )
        self._hud_display_scale = 0.5  # downscale HUD for display
        self._components_started = False
        self._next_command_apply_at = 0.0
        self._recorder: RealtimeRecorder | None = (
            RealtimeRecorder(record_path, fps=record_fps) if record_path is not None else None
        )

    def _priority_label(self) -> str:
        if self._priority_class == "vehicle":
            return "ANY VEHICLE"
        return self._priority_class.upper()

    @property
    def target(self) -> TrackedTarget | None:
        with self._lock:
            return self._target

    @property
    def detections(self) -> list[Detection]:
        with self._lock:
            return list(self._detections)

    def _start_components(self) -> None:
        if getattr(self, "_components_started", False):
            return
        if self._llm_tail not in self._llm_logger.handlers:
            self._llm_logger.addHandler(self._llm_tail)
        perception_worker = getattr(self, "_perception_worker", None)
        if perception_worker is not None:
            perception_worker.start()
        self._cam_thread.start()
        self._components_started = True

    def _cleanup_components(self) -> None:
        if not getattr(self, "_components_started", False):
            return
        self._cam_thread.stop()
        perception_worker = getattr(self, "_perception_worker", None)
        if perception_worker is not None:
            perception_worker.stop()
        llm_logger = getattr(self, "_llm_logger", None)
        llm_tail = getattr(self, "_llm_tail", None)
        if llm_logger is not None and llm_tail in llm_logger.handlers:
            llm_logger.removeHandler(llm_tail)
        recorder = getattr(self, "_recorder", None)
        if recorder is not None:
            recorder.close()
        if self._window_enabled:
            with contextlib.suppress(cv2.error):
                cv2.destroyAllWindows()
        self._components_started = False

    def _run_loop_thread(self) -> None:
        try:
            self._run_loop()
        finally:
            self._cleanup_components()
            self._thread = None
            logger.info("Pilot HUD display stopped")

    def start(self) -> None:
        if self._running:
            return
        self._start_components()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop_thread, daemon=True, name="PilotHUD")
        self._thread.start()
        logger.info("Pilot HUD display started")

    def run_foreground(self) -> None:
        if self._running:
            return
        self._start_components()
        self._running = True
        self._thread = threading.current_thread()
        logger.info("Pilot HUD display started")
        try:
            self._run_loop()
        finally:
            self._cleanup_components()
            self._thread = None
            logger.info("Pilot HUD display stopped")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        elif self._thread is None:
            self._cleanup_components()
            logger.info("Pilot HUD display stopped")

    def _initialize_window(self) -> bool:
        try:
            cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._WINDOW_NAME, 1024, 576)
            return True
        except cv2.error as exc:
            log_event(
                logger,
                logging.WARNING,
                "hud.window_unavailable",
                "OpenCV GUI backend unavailable; continuing without HUD window",
                mission_id=getattr(self, "_mission_id", "unknown"),
                reason=str(exc),
            )
            return False

    def _run_loop(self) -> None:
        self._window_enabled = self._initialize_window()

        while self._running:
            try:
                packet = self._cam_buf.get(timeout=0.033)
                if packet is None or packet.frame is None:
                    time.sleep(self._cfg.empty_frame_sleep_s)
                    if self._window_enabled:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self._running = False
                    continue
                frame = packet.frame
                self._latest_nav_frame = frame

                h, w = frame.shape[:2]

                # ── Detection + tracking ──────────────────────
                self._frame_index += 1
                run_perception = self._should_run_perception_inference(self.target)
                perception_worker = getattr(self, "_perception_worker", None)
                if perception_worker is not None:
                    if run_perception:
                        perception_worker.submit_frame(frame)
                    latest = perception_worker.latest()
                    if latest.timestamp_ns > 0:
                        detections = list(latest.detections)
                        target = latest.target
                        self._cached_detections = detections
                    else:
                        detections = list(self._cached_detections)
                        target = None
                else:
                    if self._detector is None or self._tracker is None:
                        self._detector = MultiClassDetector(self._vision_cfg)
                        self._tracker = KalmanTracker(self._vision_cfg)
                    if run_perception:
                        detections = self._detector.track(frame)
                        self._cached_detections = detections
                        target = self._tracker.update(
                            detections,
                            priority_class=self._priority_class,
                            frame_size=(w, h),
                            frame=frame,
                        )
                    else:
                        detections = list(self._cached_detections)
                        target = self._tracker.update(
                            [],
                            priority_class=self._priority_class,
                            frame_size=(w, h),
                            frame=frame,
                        )
                self._tick_id += 1
                with self._lock:
                    self._detections = detections
                    self._target = target
                if self._should_log_tick_debug():
                    speed_px, heading_deg, motion_label = self._target_motion_insight(target)
                    log_event(
                        logger,
                        logging.DEBUG,
                        "perception.tick",
                        "Perception tick completed",
                        mission_id=self._mission_id,
                        tick_id=self._tick_id,
                        fsm_state=self._fsm.state.value,
                        frame_timestamp_ns=packet.timestamp_ns,
                        detections_count=len(detections),
                        class_counts=self._summarize_classes(detections),
                        primary_target_id=target.track_id if target else None,
                        target_confirmed=bool(target and target.is_confirmed),
                        frames_since_seen=target.frames_since_seen if target else None,
                        target_motion_speed_px=round(speed_px, 3) if target else None,
                        target_motion_heading_deg=round(heading_deg, 2) if target else None,
                        target_motion_label=motion_label if target else None,
                        perception_inference=run_perception,
                    )

                # ── Autonomous control (FSM-driven movement) ──
                self._execute_control(
                    target,
                    w,
                    h,
                    detections=detections,
                    frame=frame,
                    frame_timestamp_ns=packet.timestamp_ns,
                )

                # ── Draw HUD ──────────────────────────────────
                canvas = self._draw_hud(frame, detections, target)

                if self._recorder is not None:
                    self._recorder.add(canvas)

                if self._window_enabled:
                    # Use cv2's WINDOW_NORMAL native OS scaling instead of manual downscaling
                    # to keep sub-pixel text rendering sharp.
                    cv2.imshow(self._WINDOW_NAME, canvas)

                    key = cv2.waitKey(self._cfg.hud_wait_key_ms) & 0xFF
                    if key == ord("q"):
                        self._running = False
                        break

                self._fps.tick()

            except Exception as e:
                log_event(
                    logger,
                    logging.ERROR,
                    "hud.loop",
                    "HUD loop error",
                    mission_id=self._mission_id,
                    tick_id=self._tick_id,
                    reason=str(e),
                )
                # Don't close window on error, just wait
                time.sleep(self._cfg.error_loop_sleep_s)
                if self._window_enabled:
                    cv2.waitKey(1)

    def _execute_control(
        self,
        target: TrackedTarget | None,
        frame_w: int,
        frame_h: int,
        detections: list[Detection] | None = None,
        frame: np.ndarray | None = None,
        frame_timestamp_ns: int | None = None,
    ) -> None:
        """Read FSM state and send velocity commands via the bridge."""
        state = self._fsm.state
        if state in (MissionState.IDLE, MissionState.REPORT, MissionState.BLOCKED):
            return  # No autonomous movement in these states

        mission_mode = getattr(self, "_mission_mode", MissionMode.SEARCH)

        # Build a minimal mission context for the controller
        elapsed_s = time.monotonic() - self._mission_start_time
        progress = min(1.0, elapsed_s / max(self._cfg.mission_timeout_s, 1.0))
        mission_ctx = MissionContext(
            mode=mission_mode,
            state=state,
            priority_class=self._priority_class,
            target_id=target.track_id if target else None,
            progress=progress,
            elapsed_s=elapsed_s,
            operator_text="",
        )

        intent = self._fsm.get_motion_intent(mission_ctx, target)

        # Read sensor telemetry (throttled to reduce AirSim RPC overhead)
        should_read_sensor = (
            self._tick_id % self._sensor_read_interval == 0
        ) or self._last_snapshot is None
        if should_read_sensor:
            try:
                snapshot = self._sensor_reader.read()
                telemetry = snapshot.telemetry
            except Exception:
                snapshot = self._sensor_reader.last_snapshot
                if snapshot is not None:
                    telemetry = snapshot.telemetry
                else:
                    return
            with self._lock:
                self._last_snapshot = snapshot
        else:
            with self._lock:
                snapshot = self._last_snapshot
            if snapshot is None:
                return
            telemetry = snapshot.telemetry

        if snapshot is not None:
            self._update_safety(snapshot)

        vision_cfg = getattr(self, "_vision_cfg", VisionConfig())
        sticky_lock_grace_frames = max(1, vision_cfg.sticky_lock_timeout_frames)
        if mission_mode == MissionMode.TRAFFIC_MONITOR and self._fsm.state in {
            MissionState.TRACK,
            MissionState.REACQUIRE,
        }:
            self._fsm.transition(MissionState.SCAN, reason="traffic_monitor_scan_hold")
            self._controller.reset()
        if target and (target.is_confirmed or target.frames_since_seen <= sticky_lock_grace_frames):
            self._last_target_seen_at = time.monotonic()
            self._update_reacquire_bias_from_target(target)
            if self._fsm.state == MissionState.SCAN:
                self._promote_locked_priority_target(target)
                state = self._fsm.state
                mission_ctx = MissionContext(
                    mode=mission_mode,
                    state=state,
                    priority_class=self._priority_class,
                    target_id=target.track_id,
                    progress=progress,
                    elapsed_s=elapsed_s,
                    operator_text="",
                )
            elif mission_mode != MissionMode.TRAFFIC_MONITOR and (
                self._fsm.state == MissionState.REACQUIRE
                and is_priority_compatible(self._priority_class, target.detection.class_name)
            ):
                self._fsm.transition(MissionState.TRACK, reason="reacquired_target")
                log_event(
                    logger,
                    logging.INFO,
                    "fsm.transition",
                    "Reacquired priority target, returning to TRACK",
                    mission_id=self._mission_id,
                    tick_id=self._tick_id,
                    from_state=MissionState.REACQUIRE.value,
                    to_state=MissionState.TRACK.value,
                    source="reacquired_target",
                    reason="reacquired_target",
                    fsm_state=self._fsm.state.value,
                )

        mission_ctx = MissionContext(
            mode=mission_mode,
            state=self._fsm.state,
            priority_class=self._priority_class,
            target_id=target.track_id if target else None,
            progress=progress,
            elapsed_s=elapsed_s,
            operator_text="",
        )
        self._record_runtime_snapshot(
            mission_ctx,
            detections or [],
            target,
            frame=frame,
            frame_timestamp_ns=frame_timestamp_ns,
        )
        state = self._fsm.state

        if snapshot is not None and self._should_recover_altitude(snapshot.telemetry, state):
            return

        if (
            state == MissionState.TRACK
            and (
                target is None
                or (not target.is_confirmed and target.frames_since_seen > sticky_lock_grace_frames)
            )
            and time.monotonic() - self._last_target_seen_at > self._cfg.reacquire_timeout_s
        ):
            self._fsm.transition(MissionState.REACQUIRE, reason="target_lost_reacquire")
            self._reacquire_bias_until = time.monotonic() + 3.0
            self._controller.reset()
            log_event(
                logger,
                logging.INFO,
                "fsm.transition",
                "Target lost, entering REACQUIRE mode for focused re-search",
                mission_id=self._mission_id,
                tick_id=self._tick_id,
                from_state=state.value,
                to_state=self._fsm.state.value,
                source="target_lost_reacquire",
                reason="target_lost_reacquire",
                fsm_state=self._fsm.state.value,
            )
            state = self._fsm.state
            intent = self._fsm.get_motion_intent(mission_ctx, target)

        cmd = self._controller.resolve(intent, target, snapshot, telemetry, frame_w, frame_h)
        if (
            mission_mode == MissionMode.TRAFFIC_MONITOR
            and state == MissionState.SCAN
            and cmd.source == "scan"
        ):
            cmd = self._traffic_monitor_scan_command(telemetry, target=target)
        cmd = self._apply_reacquire_scan_bias(cmd, state)
        fallback_used = False
        if cmd.source == "hover":
            cmd = self._fallback_patrol_cmd(state, telemetry)
            fallback_used = True

        cmd = self._apply_safety_turn_if_needed(cmd, snapshot, state)
        cmd, unstick_used = self._apply_unstick_if_needed(cmd, telemetry, state)
        if self._should_log_tick_debug():
            log_event(
                logger,
                logging.DEBUG,
                "control.decision",
                "Control decision produced",
                mission_id=self._mission_id,
                tick_id=self._tick_id,
                fsm_state=state.value,
                intent=intent.primitive.value,
                cmd_source=cmd.source,
                cmd_vx=round(cmd.vx, 4),
                cmd_vy=round(cmd.vy, 4),
                cmd_vz=round(cmd.vz, 4),
                cmd_yaw_rate=round(cmd.yaw_rate, 4),
                fallback_used=fallback_used,
                unstick_used=unstick_used,
                target_id=target.track_id if target else None,
                altitude_m=round(telemetry.altitude_m, 3),
            )

        with self._lock:
            self._last_cmd = cmd

        now = time.monotonic()
        if self._should_hold_traffic_monitor_command(cmd, now):
            return

        self._bridge.set_tracked_class(target.detection.class_name if target is not None else None)
        try:
            moved = self._bridge.move(cmd, snapshot=snapshot)
            if moved:
                with self._lock:
                    self._last_applied_cmd = cmd
                self._next_command_apply_at = now + max(0.08, cmd.duration_s * 0.75)
            if self._should_log_tick_debug() or not moved:
                log_event(
                    logger,
                    logging.DEBUG,
                    "control.apply",
                    "Control command applied through bridge",
                    mission_id=self._mission_id,
                    tick_id=self._tick_id,
                    fsm_state=state.value,
                    cmd_source=cmd.source,
                    applied=moved,
                )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "control.apply",
                "Control command failed through bridge",
                mission_id=self._mission_id,
                tick_id=self._tick_id,
                fsm_state=state.value,
                cmd_source=cmd.source,
                applied=False,
                reason=str(exc),
            )

    def _should_log_tick_debug(self) -> bool:
        interval = max(1, self._cfg.debug_tick_log_interval)
        return self._tick_id <= 1 or self._tick_id % interval == 0

    def _should_run_perception_inference(self, target: TrackedTarget | None) -> bool:
        mission_mode = getattr(self, "_mission_mode", MissionMode.SEARCH)
        fsm = getattr(self, "_fsm", None)
        if fsm is not None and fsm.state == MissionState.TRACK:
            if target and target.is_confirmed and target.frames_since_seen == 0:
                speed_px = float(np.hypot(*target.velocity_estimate))
                if speed_px >= self._vision_cfg.track_full_rate_motion_threshold_px:
                    return True
                edge_margin = (
                    target.frame_width * self._vision_cfg.track_full_rate_edge_margin_ratio
                )
                tx, _ty = target.smooth_center
                if tx <= edge_margin or tx >= (target.frame_width - edge_margin):
                    return True
                return self._frame_index % 2 == 0
            return True
        safety_state = getattr(self, "_last_safety_state", SafetyState.PATH_CLEAR)
        if safety_state in {
            SafetyState.OBSTACLE_AHEAD,
            SafetyState.OBSTACLE_CLUSTER,
            SafetyState.SAFETY_OVERRIDE,
            SafetyState.LANDING_CAUTION,
        }:
            return True
        if mission_mode == MissionMode.TRAFFIC_MONITOR:
            interval = max(1, getattr(self._cfg, "traffic_monitor_inference_interval", 5))
            if self._frame_index <= 1:
                return True
            if target is None:
                return self._frame_index % interval == 0
            if target.frames_since_seen > 0:
                return self._frame_index % max(2, interval - 1) == 0
            return self._frame_index % interval == 0
        interval = max(1, self._vision_cfg.track_inference_interval)
        if interval == 1 or self._frame_index <= 1:
            return True
        if target is None:
            return self._frame_index % interval == 0
        if target.frames_since_seen > 0:
            return True
        return self._frame_index % interval == 0

    def _promote_locked_priority_target(self, target: TrackedTarget) -> None:
        mission_mode = getattr(self, "_mission_mode", MissionMode.SEARCH)
        if (
            mission_mode == MissionMode.TRAFFIC_MONITOR
            or self._fsm.state != MissionState.SCAN
            or not target.is_confirmed
            or not is_priority_compatible(self._priority_class, target.detection.class_name)
        ):
            return
        self._fsm.transition(MissionState.TRACK, reason="local_target_lock")
        log_event(
            logger,
            logging.INFO,
            "fsm.transition",
            "Priority target locked locally, promoting mission to TRACK",
            mission_id=self._mission_id,
            tick_id=self._tick_id,
            from_state=MissionState.SCAN.value,
            to_state=MissionState.TRACK.value,
            source="local_target_lock",
            reason="local_target_lock",
            target_id=target.track_id,
            target_class=target.detection.class_name,
            fsm_state=self._fsm.state.value,
        )

    def _update_safety(self, snapshot: SensorSnapshot) -> None:
        evaluator = self._safety
        if not isinstance(evaluator, SafetyEvaluator):
            self._last_safety_state = SafetyState.PATH_CLEAR
            self._last_safety_reason = "safety bridge active"
            return
        evaluation = evaluator.evaluate(snapshot, True)
        self._last_safety_state = evaluation.state
        self._last_safety_reason = evaluation.reason

    def _should_recover_altitude(
        self,
        telemetry: TelemetryReading,
        state: MissionState,
    ) -> bool:
        # Don't recover in terminal states
        if state in (MissionState.REPORT, MissionState.BLOCKED, MissionState.IDLE):
            return False

        if time.monotonic() < getattr(self, "_altitude_recovery_grace_until", 0.0):
            return False

        target_alt = self._cfg.cruise_altitude_m
        min_acceptable = max(target_alt * 0.7, target_alt - 0.8)  # Within 0.8m band of target

        if telemetry.altitude_m >= min_acceptable:
            return False

        now = time.monotonic()
        # Longer cooldown to prevent recovery spam
        if now - self._last_recover_at < 8.0:
            return True

        self._last_recover_at = now
        log_event(
            logger,
            logging.WARNING,
            "recovery.action",
            "Altitude recovery triggered",
            mission_id=self._mission_id,
            tick_id=self._tick_id,
            fsm_state=state.value,
            recovery_type="altitude_recovery",
            trigger_reason="altitude_below_operating_band",
            current_value=round(telemetry.altitude_m, 3),
            target_value=round(target_alt, 3),
            min_acceptable=round(min_acceptable, 3),
            cooldown_s=8.0,
        )

        try:
            # First ensure we have control
            self._bridge.takeoff()
            self._bridge.move_to_altitude(target_alt)

            with self._lock:
                self._last_cmd = VelocityCmd(
                    0.0,
                    0.0,
                    -0.5,  # Stronger upward velocity
                    0.0,
                    2.0,  # Longer duration
                    "recovery_climb",
                )
        except Exception as e:
            log_event(
                logger,
                logging.ERROR,
                "recovery.action",
                "Altitude recovery failed",
                mission_id=self._mission_id,
                tick_id=self._tick_id,
                fsm_state=state.value,
                recovery_type="altitude_recovery",
                reason=str(e),
            )

        return True

    def _fallback_patrol_cmd(
        self,
        state: MissionState,
        telemetry: TelemetryReading,
    ) -> VelocityCmd:
        phase = time.monotonic()
        if (
            state == MissionState.SCAN
            and getattr(self, "_mission_mode", MissionMode.SEARCH) == MissionMode.TRAFFIC_MONITOR
        ):
            return self._traffic_monitor_scan_command(telemetry)
        if state == MissionState.TRACK:
            yaw_bias = float(getattr(self, "_reacquire_yaw_bias", 0.0))
            yaw = float(np.clip(yaw_bias * 0.08, -0.1, 0.1))
            return VelocityCmd(
                vx=max(0.45, self._cfg.scan_straight_forward_speed),
                vy=0.0,
                vz=self._cfg.scan_vertical_bias,
                yaw_rate=yaw,
                duration_s=self._cfg.tick_duration_s,
                source="track_search",
            )
        if state == MissionState.SCAN:
            no_target_s = phase - self._last_target_seen_at
            if phase < self._scan_escape_until:
                return VelocityCmd(
                    vx=self._cfg.scan_escape_forward_speed,
                    vy=0.0,
                    vz=self._cfg.scan_vertical_bias,
                    yaw_rate=self._cfg.scan_escape_yaw_rate,
                    duration_s=self._cfg.tick_duration_s,
                    source="scan_escape",
                )
            if no_target_s >= self._cfg.scan_escape_trigger_s:
                self._scan_escape_until = phase + self._cfg.scan_escape_duration_s
                log_event(
                    logger,
                    logging.INFO,
                    "recovery.action",
                    "Long scan without target, applying forward escape maneuver",
                    mission_id=getattr(self, "_mission_id", "unknown"),
                    tick_id=getattr(self, "_tick_id", 0),
                    fsm_state=state.value,
                    recovery_type="scan_escape",
                    trigger_reason="long_scan_without_target",
                    target_value=self._cfg.scan_escape_trigger_s,
                    cooldown_s=self._cfg.scan_escape_duration_s,
                )
                return VelocityCmd(
                    vx=self._cfg.scan_escape_forward_speed,
                    vy=0.0,
                    vz=self._cfg.scan_vertical_bias,
                    yaw_rate=self._cfg.scan_escape_yaw_rate,
                    duration_s=self._cfg.tick_duration_s,
                    source="scan_escape",
                )
            if no_target_s >= self._cfg.scan_straight_trigger_s:
                road_bias = 0.0
                if self._cfg.scan_road_bias_enabled:
                    road_bias = self._scan_road_bias_from_frame(
                        getattr(self, "_latest_nav_frame", None)
                    )
                yaw = float(
                    np.clip(
                        self._cfg.scan_straight_yaw_rate
                        + (road_bias * self._cfg.scan_road_bias_gain),
                        -self._cfg.scan_road_bias_clip,
                        self._cfg.scan_road_bias_clip,
                    )
                )
                return VelocityCmd(
                    vx=self._cfg.scan_straight_forward_speed,
                    vy=0.0,
                    vz=self._cfg.scan_vertical_bias,
                    yaw_rate=yaw,
                    duration_s=self._cfg.tick_duration_s,
                    source="scan_straight_road" if abs(road_bias) > 0.05 else "scan_straight",
                )
        lateral = self._cfg.scan_lateral_speed * np.sin(phase * 0.9)
        if state == MissionState.MONITOR:
            return VelocityCmd(
                vx=0.18,
                vy=0.12 * np.cos(phase * 0.6),
                vz=-0.02 if telemetry.altitude_m < self._cfg.cruise_altitude_m else 0.02,
                yaw_rate=self._cfg.scan_yaw_rate * 0.4,
                duration_s=self._cfg.tick_duration_s,
                source="monitor_patrol",
            )
        return VelocityCmd(
            vx=self._cfg.scan_forward_speed * 0.9,
            vy=float(lateral),
            vz=self._cfg.scan_vertical_bias,
            yaw_rate=self._cfg.scan_yaw_rate,
            duration_s=self._cfg.tick_duration_s,
            source="scan_patrol",
        )

    def _apply_unstick_if_needed(
        self,
        cmd: VelocityCmd,
        telemetry: TelemetryReading,
        state: MissionState,
    ) -> tuple[VelocityCmd, bool]:
        if state in (MissionState.IDLE, MissionState.REPORT, MissionState.BLOCKED):
            self._stuck_since = None
            return cmd, False
        if self._last_safety_state in {
            SafetyState.OBSTACLE_AHEAD,
            SafetyState.OBSTACLE_CLUSTER,
            SafetyState.SAFETY_OVERRIDE,
        }:
            self._stuck_since = None
            return cmd, False
        speed = float(np.linalg.norm(np.array(telemetry.velocity_ned, dtype=float)))
        active_motion = abs(cmd.vx) + abs(cmd.vy) + abs(cmd.vz) + abs(cmd.yaw_rate) > 0.08
        if not active_motion or speed > self._cfg.unstick_velocity_threshold:
            self._stuck_since = None
            return cmd, False
        now = time.monotonic()
        if self._stuck_since is None:
            self._stuck_since = now
            return cmd, False
        if now - self._stuck_since < self._cfg.unstick_timeout_s:
            return cmd, False
        self._stuck_since = now
        self._unstick_flip *= -1.0
        unstick_cmd = VelocityCmd(
            vx=max(cmd.vx, 0.35),
            vy=self._unstick_flip * max(abs(cmd.vy), self._cfg.scan_lateral_speed),
            vz=min(cmd.vz, -0.04),
            yaw_rate=cmd.yaw_rate if abs(cmd.yaw_rate) > 0.1 else self._cfg.scan_yaw_rate,
            duration_s=self._cfg.tick_duration_s,
            source="unstick",
        )
        log_event(
            logger,
            logging.INFO,
            "recovery.action",
            "Detected low-motion stall, injecting unstick maneuver",
            mission_id=self._mission_id,
            tick_id=self._tick_id,
            fsm_state=state.value,
            recovery_type="unstick",
            trigger_reason="low_velocity_with_active_motion",
            current_value=round(speed, 4),
            target_value=self._cfg.unstick_velocity_threshold,
            cooldown_s=self._cfg.unstick_timeout_s,
        )
        return unstick_cmd, True

    def _apply_safety_turn_if_needed(
        self,
        cmd: VelocityCmd,
        snapshot: SensorSnapshot,
        state: MissionState,
    ) -> VelocityCmd:
        if state in (MissionState.IDLE, MissionState.REPORT, MissionState.BLOCKED):
            return cmd
        if self._last_safety_state not in {
            SafetyState.OBSTACLE_AHEAD,
            SafetyState.OBSTACLE_CLUSTER,
        }:
            return cmd
        if cmd.vx <= 0.0:
            return cmd
        prox = snapshot.proximity
        if not prox.available:
            return cmd
        clearer_right = prox.right_m >= prox.left_m
        turn_rate = max(abs(cmd.yaw_rate), max(self._cfg.scan_yaw_rate * 1.5, 0.18))
        yaw_rate = turn_rate if clearer_right else -turn_rate
        return VelocityCmd(
            vx=0.0,
            vy=0.0,
            vz=0.0,
            yaw_rate=yaw_rate,
            duration_s=cmd.duration_s,
            source="safety_turn",
        )

    def _summarize_classes(self, detections: list[Detection]) -> dict[str, int]:
        return dict(Counter(detection.class_name for detection in detections))

    def _update_reacquire_bias_from_target(self, target: TrackedTarget) -> None:
        vx_px, _vy_px = target.velocity_estimate
        if abs(vx_px) < 2.5:
            return
        self._reacquire_yaw_bias = 1.0 if vx_px > 0 else -1.0

    def _record_runtime_snapshot(
        self,
        mission_ctx: MissionContext,
        detections: list[Detection],
        target: TrackedTarget | None,
        *,
        frame: np.ndarray | None = None,
        frame_timestamp_ns: int | None = None,
    ) -> None:
        reporter = getattr(self, "_reporter", None)
        if reporter is None:
            return
        record_runtime_snapshot = getattr(reporter, "record_runtime_snapshot", None)
        if not callable(record_runtime_snapshot):
            return
        observed = detections or ([target.detection] if target is not None else [])
        record_runtime_snapshot(
            mission_ctx,
            detections=observed,
            target=target,
            safety_reason=getattr(self, "_last_safety_reason", "runtime update"),
            frame=frame,
            timestamp_ns=frame_timestamp_ns,
        )

    def _apply_reacquire_scan_bias(
        self,
        cmd: VelocityCmd,
        state: MissionState,
    ) -> VelocityCmd:
        # Apply yaw bias in both SCAN and REACQUIRE states
        if state not in (MissionState.SCAN, MissionState.REACQUIRE):
            return cmd
        if state == MissionState.SCAN and not cmd.source.startswith("scan"):
            return cmd
        if time.monotonic() > self._reacquire_bias_until:
            return cmd
        if self._reacquire_yaw_bias == 0.0:
            return cmd
        # REACQUIRE gets stronger bias for more aggressive re-search
        bias_strength = 0.35 if state == MissionState.REACQUIRE else 0.18
        return VelocityCmd(
            vx=cmd.vx,
            vy=cmd.vy,
            vz=cmd.vz,
            yaw_rate=float(
                np.clip(cmd.yaw_rate + (bias_strength * self._reacquire_yaw_bias), -1.0, 1.0)
            ),
            duration_s=cmd.duration_s,
            source=cmd.source + "_biased",
        )

    def _should_hold_traffic_monitor_command(self, cmd: VelocityCmd, now: float) -> bool:
        if getattr(self, "_mission_mode", MissionMode.SEARCH) != MissionMode.TRAFFIC_MONITOR:
            return False
        if not cmd.source.startswith("scan_road"):
            return False
        previous = getattr(self, "_last_applied_cmd", None)
        next_apply_at = float(getattr(self, "_next_command_apply_at", 0.0))
        if previous is None or now >= next_apply_at:
            return False
        if previous.source != cmd.source:
            return False
        if abs(previous.vx - cmd.vx) > 0.08:
            return False
        if abs(previous.vy - cmd.vy) > 0.04:
            return False
        if abs(previous.vz - cmd.vz) > 0.03:
            return False
        return not abs(previous.yaw_rate - cmd.yaw_rate) > 0.04

    def _traffic_monitor_scan_command(
        self,
        telemetry: TelemetryReading,
        *,
        target: TrackedTarget | None = None,
    ) -> VelocityCmd:
        bias, confidence, center_offset = self._scan_road_guidance_from_frame(
            getattr(self, "_latest_nav_frame", None)
        )
        vertical = self._scan_altitude_hold(telemetry)
        dominant_error = center_offset if abs(center_offset) >= abs(bias) * 0.75 else bias
        command_duration = max(
            self._cfg.tick_duration_s,
            getattr(self._cfg, "traffic_monitor_command_duration_s", self._cfg.tick_duration_s),
        )
        follow_speed = self._cfg.traffic_monitor_road_follow_speed
        seek_speed = self._cfg.traffic_monitor_road_seek_speed
        if target is not None and target.is_confirmed and target.frames_since_seen <= 1:
            follow_speed = min(follow_speed, 0.28)
            seek_speed = min(seek_speed, 0.12)

        if confidence >= self._cfg.traffic_monitor_road_follow_confidence:
            yaw_rate = float(
                np.clip(
                    (bias * self._cfg.traffic_monitor_road_follow_yaw_gain)
                    + (center_offset * self._cfg.traffic_monitor_road_center_yaw_gain),
                    -self._cfg.scan_road_bias_clip,
                    self._cfg.scan_road_bias_clip,
                )
            )
            return VelocityCmd(
                vx=follow_speed,
                vy=0.0,
                vz=vertical,
                yaw_rate=yaw_rate,
                duration_s=command_duration,
                source="scan_road_follow",
            )

        if confidence >= self._cfg.traffic_monitor_road_seek_confidence:
            yaw_rate = float(
                np.clip(
                    dominant_error
                    * max(
                        self._cfg.traffic_monitor_road_follow_yaw_gain,
                        self._cfg.traffic_monitor_road_center_yaw_gain,
                    ),
                    -self._cfg.traffic_monitor_road_search_yaw_rate,
                    self._cfg.traffic_monitor_road_search_yaw_rate,
                )
            )
            return VelocityCmd(
                vx=seek_speed,
                vy=0.0,
                vz=vertical,
                yaw_rate=yaw_rate,
                duration_s=command_duration,
                source="scan_road_seek",
            )

        seek_direction = float(getattr(self, "_road_seek_direction", 1.0))
        if abs(seek_direction) < 0.1:
            seek_direction = 1.0 if math.sin(time.monotonic() * 0.6) >= 0.0 else -1.0
        return VelocityCmd(
            vx=self._cfg.traffic_monitor_road_search_speed,
            vy=0.0,
            vz=vertical,
            yaw_rate=seek_direction * self._cfg.traffic_monitor_road_search_yaw_rate,
            duration_s=command_duration,
            source="scan_road_search",
        )

    def _scan_altitude_hold(self, telemetry: TelemetryReading) -> float:
        mission_mode = getattr(self, "_mission_mode", MissionMode.SEARCH)
        target_alt = self._cfg.cruise_altitude_m
        deadband = 0.15
        max_vz = 0.08
        gain = 0.18
        if mission_mode == MissionMode.TRAFFIC_MONITOR:
            target_alt = getattr(self._cfg, "traffic_monitor_cruise_altitude_m", target_alt)
            deadband = getattr(self._cfg, "traffic_monitor_altitude_hold_deadband_m", 0.25)
            max_vz = getattr(self._cfg, "traffic_monitor_altitude_hold_max_vz", 0.04)
            gain = 0.12
        altitude_error = target_alt - telemetry.altitude_m
        if abs(altitude_error) < deadband:
            return 0.0
        correction = -gain * altitude_error
        return float(np.clip(correction, -max_vz, max_vz))

    def _target_motion_insight(
        self,
        target: TrackedTarget | None,
    ) -> tuple[float, float, str]:
        if target is None:
            return 0.0, 0.0, "STILL"
        vx_px, vy_px = target.velocity_estimate
        speed_px = float(np.hypot(vx_px, vy_px))
        if speed_px < 0.08:
            return speed_px, 0.0, "STILL"
        heading_deg = math.degrees(math.atan2(vy_px, vx_px))
        horizontal = "RIGHT" if vx_px > 0 else "LEFT"
        vertical = "DOWN" if vy_px > 0 else "UP"
        return speed_px, heading_deg, f"{horizontal}-{vertical}"

    def _scan_road_guidance_from_frame(
        self,
        frame: np.ndarray | None,
    ) -> tuple[float, float, float]:
        """Return road guidance as (left_right_bias, confidence, center_offset)."""
        tick_id = int(getattr(self, "_tick_id", 0))
        last_tick = int(getattr(self, "_last_road_bias_tick", 0))
        cached = float(getattr(self, "_cached_road_bias", 0.0))
        cached_confidence = float(getattr(self, "_cached_road_confidence", 0.0))
        cached_center = float(getattr(self, "_cached_road_center_offset", 0.0))
        if tick_id - last_tick < 3:
            return cached, cached_confidence, cached_center

        def _commit(
            bias: float,
            confidence: float,
            center_offset: float,
        ) -> tuple[float, float, float]:
            self._cached_road_bias = float(np.clip(bias, -1.0, 1.0))
            self._cached_road_confidence = float(max(0.0, confidence))
            self._cached_road_center_offset = float(np.clip(center_offset, -1.0, 1.0))
            self._last_road_bias_tick = tick_id
            dominant = (
                self._cached_road_center_offset
                if abs(self._cached_road_center_offset) >= abs(self._cached_road_bias) * 0.75
                else self._cached_road_bias
            )
            if (
                confidence >= self._cfg.traffic_monitor_road_seek_confidence
                and abs(dominant) >= 0.05
            ):
                self._road_seek_direction = 1.0 if dominant >= 0.0 else -1.0
            return (
                self._cached_road_bias,
                self._cached_road_confidence,
                self._cached_road_center_offset,
            )

        def _mask_guidance(mask: np.ndarray) -> tuple[float, float, float]:
            height, width = mask.shape[:2]
            if height == 0 or width == 0:
                return 0.0, 0.0, 0.0
            confidence = float(np.mean(mask)) / 255.0
            if confidence <= 0.0:
                return 0.0, 0.0, 0.0
            split = max(1, width // 2)
            left = float(np.mean(mask[:, :split])) / 255.0
            right = float(np.mean(mask[:, split:])) / 255.0
            xs = np.nonzero(mask > 0)[1]
            if xs.size == 0:
                center_offset = 0.0
            else:
                center_offset = ((float(np.mean(xs)) / max(1.0, width - 1.0)) - 0.5) * 2.0
            return left - right, confidence, center_offset

        # 1. Segmentation-based road detection.
        camera = getattr(self, "_camera", None)
        if camera is not None:
            try:
                seg = camera.get_segmentation_frame()
                if seg is not None and seg.size > 0:
                    sh, _sw = seg.shape[:2]
                    roi_seg = seg[int(sh * 0.58) :, :]
                    road_mask_seg = cv2.inRange(
                        roi_seg,
                        np.array([75, 0, 0], dtype=np.uint8),
                        np.array([85, 255, 255], dtype=np.uint8),
                    )
                    bias, confidence, center_offset = _mask_guidance(road_mask_seg)
                    if confidence >= self._cfg.traffic_monitor_road_seek_confidence:
                        return _commit(bias, confidence, center_offset)
            except Exception:
                pass

        # 2. RGB fallback focused on low-saturation asphalt-like pixels.
        if frame is None or frame.size == 0:
            return _commit(0.0, 0.0, 0.0)
        h, w = frame.shape[:2]
        if h < 20 or w < 20:
            return _commit(0.0, 0.0, 0.0)

        roi = frame[int(h * 0.58) :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(
            hsv, np.array([0, 0, 35], dtype=np.uint8), np.array([180, 55, 165], dtype=np.uint8)
        )
        road_mask = cv2.GaussianBlur(road_mask, (5, 5), 0)
        _threshold, road_mask = cv2.threshold(road_mask, 80, 255, cv2.THRESH_BINARY)
        bias, confidence, center_offset = _mask_guidance(road_mask)
        return _commit(bias, confidence * 0.85, center_offset)

    def _scan_road_bias_from_frame(self, frame: np.ndarray | None) -> float:
        """Compute a yaw bias from the visible road signal."""
        bias, _confidence, _center_offset = self._scan_road_guidance_from_frame(frame)
        return bias

    # ── HUD Rendering ────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        target: TrackedTarget | None,
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        right_panel_w = min(300, max(220, int(w * 0.30)))
        right_panel_h = 150
        right_panel_x1 = w - right_panel_w
        right_text_x = right_panel_x1 + 10
        right_text_max_w = max(40, right_panel_w - 20)

        # ── Dim overlay panel areas (cached static mask) ────────
        if self._panel_mask_shape != (h, w):
            self._panel_mask = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(self._panel_mask, (0, 0), (240, 170), _DARK, -1)
            cv2.rectangle(self._panel_mask, (right_panel_x1, 0), (w, right_panel_h), _DARK, -1)
            cv2.rectangle(self._panel_mask, (0, h - 40), (w, h), _DARK, -1)
            self._panel_mask_shape = (h, w)
        panel_mask = self._panel_mask
        if panel_mask is not None:
            cv2.addWeighted(panel_mask, 0.72, canvas, 0.28, 0, canvas)

        # ── Crosshair ────────────────────────────────────────
        cx, cy = w // 2, h // 2
        gap, arm = 10, 25
        cv2.line(canvas, (cx - arm, cy), (cx - gap, cy), _CYAN, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx + gap, cy), (cx + arm, cy), _CYAN, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - arm), (cx, cy - gap), _CYAN, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy + gap), (cx, cy + arm), _CYAN, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, _CYAN, 1, cv2.LINE_AA)

        # ── Detection boxes ──────────────────────────────────
        count_mode = (
            getattr(self, "_mission_mode", MissionMode.SEARCH) == MissionMode.TRAFFIC_MONITOR
        )
        for det in detections:
            is_primary = target is not None and det.track_id == target.track_id
            color = (_YELLOW if count_mode else _GREEN) if is_primary else _DIM
            x1, y1, x2, y2 = det.bbox
            thickness = 2 if is_primary else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            pct = f"{det.confidence:.0%}"
            label = f"{det.class_name} {pct}"
            if det.track_id is not None:
                label += f" #{det.track_id}"
            self._draw_label(canvas, label, (x1, max(18, y1 - 6)), color)

        # ── Target lock indicator ────────────────────────────
        if target and target.is_confirmed:
            tcx, tcy = map(int, target.smooth_center)
            sz = 32
            if count_mode:
                cv2.drawMarker(canvas, (tcx, tcy), _YELLOW, cv2.MARKER_CROSS, 22, 2)
                cv2.circle(canvas, (tcx, tcy), 9, _YELLOW, 1, cv2.LINE_AA)
                self._draw_label(
                    canvas,
                    f"VEHICLE OBSERVED #{target.track_id} | {target.frames_tracked}f",
                    (tcx - sz, tcy + sz + 14),
                    _YELLOW,
                )
            else:
                # Corner brackets (animated feel via bright green)
                for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    ox, oy = tcx + dx * sz, tcy + dy * sz
                    cv2.line(canvas, (ox, oy), (ox - dx * 14, oy), _LIME, 2, cv2.LINE_AA)
                    cv2.line(canvas, (ox, oy), (ox, oy - dy * 14), _LIME, 2, cv2.LINE_AA)
                cv2.circle(canvas, (tcx, tcy), 5, _LIME, -1, cv2.LINE_AA)
                cv2.circle(canvas, (tcx, tcy), 12, _LIME, 1, cv2.LINE_AA)
                self._draw_label(
                    canvas,
                    f"TARGET LOCKED #{target.track_id} | {target.frames_tracked}f",
                    (tcx - sz, tcy + sz + 14),
                    _LIME,
                )
        elif target:
            tcx, tcy = map(int, target.smooth_center)
            cv2.drawMarker(canvas, (tcx, tcy), _YELLOW, cv2.MARKER_TILTED_CROSS, 20, 2)
            self._draw_label(
                canvas,
                (
                    f"VEHICLE CANDIDATE #{target.track_id} ({target.frames_tracked}f)"
                    if count_mode
                    else f"ACQUIRING #{target.track_id} ({target.frames_tracked}f)"
                ),
                (tcx - 20, tcy + 28),
                _YELLOW,
            )

        # ── Left panel: Mission info ─────────────────────────
        state = self._fsm.state
        state_color = _STATE_COLORS.get(state, _WHITE)
        y = 20
        self._draw_label(canvas, "SKYTRACKVISION", (10, y), _CYAN)
        y += 24
        self._draw_label(canvas, f"STATE  {state.value}", (10, y), state_color)
        y += 22
        self._draw_label(
            canvas,
            f"SAFETY {self._last_safety_state.value}",
            (10, y),
            _SAFETY_COLORS.get(self._last_safety_state, _WHITE),
        )
        y += 22
        self._draw_label(canvas, self._last_safety_reason[:34], (10, y), _DIM, scale=0.48)
        y += 22

        with self._lock:
            last_cmd = self._last_cmd
            snapshot = self._last_snapshot
        if last_cmd:
            self._draw_label(canvas, f"CMD    {last_cmd.source}", (10, y), _WHITE)
            y += 22
            vel_text = f"V  x={last_cmd.vx:+.1f} y={last_cmd.vy:+.1f} z={last_cmd.vz:+.1f}"
            self._draw_label(canvas, vel_text, (10, y), _DIM)
            y += 22
            self._draw_label(canvas, f"YAW    {last_cmd.yaw_rate:+.2f} r/s", (10, y), _DIM)
            y += 22
        if snapshot:
            telemetry = snapshot.telemetry
            self._draw_label(canvas, f"ALT    {telemetry.altitude_m:.1f}m", (10, y), _WHITE)
            y += 22
            speed = np.linalg.norm(np.array(telemetry.velocity_ned, dtype=float))
            self._draw_label(canvas, f"SPEED  {speed:.2f}m/s", (10, y), _DIM)
            y += 22

        # ── Right panel: Detection summary ───────────────────
        det_count = len(detections)
        det_label = self._fit_text_to_width(f"DET  {det_count}", right_text_max_w, scale=0.45)
        self._draw_label(canvas, det_label, (right_text_x, 20), _YELLOW)
        if target and target.is_confirmed:
            if count_mode:
                target_label = self._fit_text_to_width(
                    "VEHICLE: OBSERVED", right_text_max_w, scale=0.45
                )
                self._draw_label(canvas, target_label, (right_text_x, 42), _YELLOW)
            else:
                target_label = self._fit_text_to_width(
                    "TARGET: LOCKED", right_text_max_w, scale=0.45
                )
                self._draw_label(canvas, target_label, (right_text_x, 42), _LIME)
        else:
            target_label = self._fit_text_to_width(
                "VEHICLE: SEARCHING" if count_mode else "TARGET: SEARCHING",
                right_text_max_w,
                scale=0.45,
            )
            self._draw_label(canvas, target_label, (right_text_x, 42), _YELLOW)
        pri_label = self._fit_text_to_width(
            f"PRI  {self._priority_label()}",
            right_text_max_w,
            scale=0.45,
        )
        self._draw_label(canvas, pri_label, (right_text_x, 64), _CYAN)
        roam_label = self._fit_text_to_width(
            "ROAD PATROL ACTIVE" if count_mode else "AUTO ROAM ACTIVE",
            right_text_max_w,
            scale=0.45,
        )
        self._draw_label(canvas, roam_label, (right_text_x, 86), _CYAN)

        if target:
            motion_speed_px, motion_heading_deg, motion_label = self._target_motion_insight(target)
            mot_text = self._fit_text_to_width(f"MOT {motion_label}", right_text_max_w, scale=0.4)
            self._draw_label(canvas, mot_text, (right_text_x, 108), _WHITE, scale=0.4)
            spd_text = self._fit_text_to_width(
                f"SPD {motion_speed_px:.2f}px H {motion_heading_deg:+.0f}",
                right_text_max_w,
                scale=0.38,
            )
            self._draw_label(
                canvas,
                spd_text,
                (right_text_x, 126),
                _DIM,
                scale=0.38,
            )

        self._draw_llm_stream(canvas)

        # ── Bottom bar ───────────────────────────────────────
        fps_val = self._fps.value
        bottom_y = h - 14
        self._draw_label(canvas, f"FPS {fps_val:.0f}", (10, bottom_y), _DIM)
        bottom_status = "ROAD PATROL COUNT active" if count_mode else "SCAN/FOLLOW/ORBIT via pilot"
        self._draw_label(canvas, bottom_status, (w // 2 - 110, bottom_y), _DIM)
        self._draw_label(canvas, "Q=quit", (w - 80, bottom_y), _DIM)

        # ── Frame border based on state ──────────────────────
        border = state_color
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border, 2)

        return canvas

    def _draw_llm_stream(self, canvas: np.ndarray) -> None:
        llm_tail = getattr(self, "_llm_tail", None)
        if llm_tail is None:
            return
        lines = llm_tail.recent_lines()
        if not lines:
            return

        h, w = canvas.shape[:2]
        line_h = 18
        panel_w = min(520, max(340, int(w * 0.46)))
        panel_h = 28 + (line_h * len(lines)) + 10
        x1 = 10
        y1 = max(176, h - 40 - panel_h - 8)
        x2 = min(w - 10, x1 + panel_w)
        y2 = min(h - 42, y1 + panel_h)
        if y2 <= y1 or x2 <= x1:
            return

        panel = canvas[y1:y2, x1:x2]
        if panel.size == 0:
            return
        tint = np.full_like(panel, _DARK)
        blended = cv2.addWeighted(tint, 0.62, panel, 0.38, 0)
        canvas[y1:y2, x1:x2] = blended
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _DIM, 1)

        self._draw_label(canvas, "LLM STREAM", (x1 + 8, y1 + 16), _CYAN, scale=0.42)
        text_y = y1 + 34
        max_text_w = max(20, (x2 - x1) - 16)
        for line in lines:
            color = _WHITE
            if line.startswith("->"):
                color = _CYAN
            elif line.startswith("<-"):
                color = _GREEN
            elif line.startswith("~"):
                color = _YELLOW
            elif line.startswith("!"):
                color = _RED
            fitted = self._fit_text_to_width(line, max_text_w, scale=0.41)
            self._draw_label(canvas, fitted, (x1 + 8, text_y), color, scale=0.41)
            text_y += line_h

    def _fit_text_to_width(
        self,
        text: str,
        max_width_px: int,
        *,
        scale: float,
        thickness: int = 1,
    ) -> str:
        if max_width_px <= 0:
            return ""
        # Cache results — text + layout params rarely change between frames
        cache_key = (text, max_width_px, scale, thickness)
        cached = self._text_cache.get(cache_key)
        if cached is not None:
            return cached
        font = cv2.FONT_HERSHEY_SIMPLEX
        if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width_px:
            self._text_cache[cache_key] = text
            return text
        ellipsis = "..."
        low = 0
        high = len(text)
        best = ellipsis
        while low <= high:
            mid = (low + high) // 2
            candidate = text[:mid].rstrip() + ellipsis
            width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
            if width <= max_width_px:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        self._text_cache[cache_key] = best
        # Prevent unbounded cache growth
        if len(self._text_cache) > 512:
            self._text_cache.clear()
        return best

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int] = _WHITE,
        scale: float = 0.55,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        x, y = origin
        # Thin black shadow keeps labels readable without making them look heavy.
        cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


class _FPSCounter:
    def __init__(self) -> None:
        self._prev = time.monotonic()
        self._alpha = 0.9
        self._fps = 30.0

    @property
    def value(self) -> float:
        return self._fps

    def tick(self) -> float:
        now = time.monotonic()
        dt = now - self._prev
        self._prev = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = self._alpha * self._fps + (1.0 - self._alpha) * instant
        return self._fps


class _LLMLogTail(logging.Handler):
    def __init__(self, *, max_lines: int = 4, max_chars: int = 96) -> None:
        super().__init__(level=logging.DEBUG)
        self._lines: deque[str] = deque(maxlen=max(1, max_lines))
        self._max_chars = max(16, max_chars)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("skytrackvision.skypilot.llm"):
            return
        line = self._format_line(record)
        if not line:
            return
        with self._lock:
            self._lines.append(self._truncate(line))

    def recent_lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def _format_line(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event_name", "log"))
        fields = getattr(record, "event_fields", {})
        if not isinstance(fields, dict):
            fields = {}

        if event == "llm.request":
            model = str(fields.get("model", "llm"))
            tools = fields.get("tool_count")
            preview = str(fields.get("user_preview", "")).strip()
            preview_text = f' ask:"{preview}"' if preview else ""
            return f"-> {model} request tools={tools}{preview_text}"

        if event == "llm.response":
            latency = fields.get("latency_ms")
            calls = fields.get("tool_calls")
            latency_text = f" {latency}ms" if latency is not None else ""
            calls_text = f" calls={calls}" if calls is not None else ""
            preview = str(fields.get("assistant_preview", "")).strip()
            preview_text = f' say:"{preview}"' if preview else ""
            tool_names = fields.get("tool_names")
            tool_names_text = ""
            if isinstance(tool_names, list) and tool_names:
                compact_names = ",".join(str(name) for name in tool_names[:4])
                tool_names_text = f" tools:{compact_names}"
            return f"<- response{latency_text}{calls_text}{preview_text}{tool_names_text}".strip()

        if event == "llm.retry":
            attempt = fields.get("attempt")
            prefix = f"~ retry#{attempt}" if attempt is not None else "~ retry"
            return f"{prefix} {record.getMessage()}"

        if record.levelno >= logging.ERROR:
            return f"! {record.getMessage()}"
        if record.levelno >= logging.WARNING:
            return f"~ {record.getMessage()}"
        return f"· {record.getMessage()}"

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars - 3] + "..."
