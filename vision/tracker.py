from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np

from autonomy.contracts import Detection, TargetDirection, TrackedTarget
from autonomy.targeting import is_priority_compatible
from config.settings import VisionConfig
from vision.reid import AppearanceStore
from vision.utils import resolve_direction

logger = logging.getLogger("skytrackvision.vision.tracker")


@dataclass(slots=True)
class _TrackState:
    track_id: int
    x: np.ndarray
    p: np.ndarray
    smooth_center: tuple[float, float]
    frames_tracked: int
    frames_since_seen: int
    last_class_name: str
    last_confidence: float
    last_area: float
    last_bbox_w: int = 0
    last_bbox_h: int = 0


class KalmanTracker:
    """Track a single primary target with a constant-velocity Kalman model."""

    def __init__(self, cfg: VisionConfig) -> None:
        self._cfg = cfg
        self._state: _TrackState | None = None
        self._reid = AppearanceStore()
        self._last_frame: np.ndarray | None = None
        self._f = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self._h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self._q_base = np.eye(4) * 0.05
        self._q_high_motion = np.eye(4) * 0.5
        self._r = np.eye(2) * 1.5

    def update(
        self,
        detections: list[Detection],
        priority_class: str | None,
        frame_size: tuple[int, int],
        frame: np.ndarray | None = None,
    ) -> TrackedTarget | None:
        frame_w, frame_h = frame_size
        if frame is not None:
            self._last_frame = frame
        if self._state is not None:
            self._predict()

        selected = self._select_detection(detections, priority_class)
        if selected is not None:
            self._ingest_detection(selected)
        elif self._state is not None:
            self._state.frames_since_seen += 1

        if self._state is None:
            return None
        if self._state.frames_since_seen > self._cfg.max_lost_frames:
            self._state = None
            return None
        return self._to_target(frame_w, frame_h)

    def _predict(self) -> None:
        if self._state is None:
            raise RuntimeError("_predict() called with no active track state")
        # Dynamic process noise: use higher Q for fast-moving targets
        speed = float(np.hypot(self._state.x[2, 0], self._state.x[3, 0]))
        q = self._q_high_motion if speed > 10.0 else self._q_base
        self._state.x = self._f @ self._state.x
        self._state.p = self._f @ self._state.p @ self._f.T + q

    def _ingest_detection(self, detection: Detection) -> None:
        measurement = np.array([[detection.center[0]], [detection.center[1]]])
        x1, y1, x2, y2 = detection.bbox
        det_bbox_w = max(1, x2 - x1)
        det_bbox_h = max(1, y2 - y1)
        if self._state is None or detection.track_id != self._state.track_id:
            # Warm start: carry forward velocity estimate from previous track
            prev_vx = float(self._state.x[2, 0]) if self._state else 0.0
            prev_vy = float(self._state.x[3, 0]) if self._state else 0.0
            prev_frames = self._state.frames_tracked if self._state else 0
            self._state = _TrackState(
                track_id=detection.track_id or 0,
                x=np.array([[detection.center[0]], [detection.center[1]], [prev_vx], [prev_vy]]),
                p=np.eye(4),
                smooth_center=detection.center,
                frames_tracked=max(1, prev_frames // 2),
                frames_since_seen=0,
                last_class_name=detection.class_name,
                last_confidence=detection.confidence,
                last_area=detection.area,
                last_bbox_w=det_bbox_w,
                last_bbox_h=det_bbox_h,
            )
            return
        innovation = measurement - (self._h @ self._state.x)
        try:
            s = self._h @ self._state.p @ self._h.T + self._r
            # Use solve instead of inv for numerical stability
            k = np.linalg.solve(s.T, (self._state.p @ self._h.T).T).T
            self._state.x = self._state.x + (k @ innovation)
            self._state.p = (np.eye(4) - (k @ self._h)) @ self._state.p
        except np.linalg.LinAlgError:
            logger.warning(
                "Kalman update skipped: singular covariance (track_id=%s). Resetting.",
                self._state.track_id,
            )
            self._state.p = np.eye(4)
        alpha = self._cfg.tracker_smoothing_alpha
        sx, sy = self._state.smooth_center
        self._state.smooth_center = (
            alpha * detection.center[0] + (1 - alpha) * sx,
            alpha * detection.center[1] + (1 - alpha) * sy,
        )
        self._state.frames_tracked += 1
        self._state.frames_since_seen = 0
        self._state.last_class_name = detection.class_name
        self._state.last_confidence = detection.confidence
        self._state.last_area = detection.area
        self._state.last_bbox_w = det_bbox_w
        self._state.last_bbox_h = det_bbox_h

        # Phase 2: Lock appearance on initial confirmation
        if (
            self._state.frames_tracked == self._cfg.min_confirm_frames
            and self._last_frame is not None
        ):
            self._reid.lock_appearance(self._last_frame, detection.bbox, self._state.track_id)

    def _select_detection(
        self,
        detections: list[Detection],
        priority_class: str | None,
    ) -> Detection | None:
        if not detections:
            return None
        candidates = detections
        if priority_class:
            preferred = [det for det in detections if det.class_name == priority_class]
            if preferred:
                candidates = preferred
            else:
                compatible = [
                    det
                    for det in detections
                    if is_priority_compatible(priority_class, det.class_name)
                ]
                if compatible:
                    candidates = compatible
                elif self._state is not None and is_priority_compatible(
                    priority_class, self._state.last_class_name
                ):
                    return None
                else:
                    return None
        if self._state is not None:
            same_track_candidates = [
                det
                for det in candidates
                if det.track_id is not None and det.track_id == self._state.track_id
            ]
            if same_track_candidates:
                return max(same_track_candidates, key=lambda det: det.confidence)

            predicted = self._predicted_center()
            if self._should_hold_sticky_lock(priority_class):
                # Dynamic distance threshold based on target speed
                speed = float(np.hypot(self._state.x[2, 0], self._state.x[3, 0]))
                max_dist = self._cfg.sticky_lock_max_center_distance_px + speed * 3.0
                nearby_candidates = [
                    det
                    for det in candidates
                    if self._distance_sq(det.center, predicted) <= max_dist**2
                ]
                if nearby_candidates:
                    best = min(
                        nearby_candidates,
                        key=lambda det: self._distance_sq(det.center, predicted),
                    )
                    # Phase 2: Re-ID check before accepting a track switch
                    if (
                        best.track_id != self._state.track_id
                        and self._last_frame is not None
                        and not self._reid.should_accept_switch(
                            self._last_frame,
                            best.bbox,
                            best.track_id or 0,
                            frames_since_seen=self._state.frames_since_seen,
                        )
                    ):
                        return None  # Reject: different appearance
                    return best
                return None

            return min(
                candidates,
                key=lambda det: self._distance_sq(det.center, predicted),
            )
        return max(candidates, key=lambda det: det.confidence)

    def _predicted_center(self) -> tuple[float, float]:
        if self._state is None:
            raise RuntimeError("_predicted_center() called with no active track state")
        return (float(self._state.x[0, 0]), float(self._state.x[1, 0]))

    def _should_hold_sticky_lock(self, priority_class: str | None) -> bool:
        if self._state is None:
            return False
        if priority_class is None or not is_priority_compatible(
            priority_class,
            self._state.last_class_name,
        ):
            return False
        if self._state.frames_tracked < self._cfg.min_confirm_frames:
            return False
        return self._state.frames_since_seen < self._cfg.sticky_lock_timeout_frames

    def _distance_sq(
        self,
        center: tuple[float, float],
        predicted: tuple[float, float],
    ) -> float:
        return (center[0] - predicted[0]) ** 2 + (center[1] - predicted[1]) ** 2

    def _to_target(self, frame_w: int, frame_h: int) -> TrackedTarget:
        if self._state is None:
            raise RuntimeError("_to_target() called with no active track state")
        predicted = (float(self._state.x[0, 0]), float(self._state.x[1, 0]))
        bbox_w = (
            self._state.last_bbox_w
            if self._state.last_bbox_w > 0
            else max(int(np.sqrt(self._state.last_area)), 1)
        )
        bbox_h = (
            self._state.last_bbox_h
            if self._state.last_bbox_h > 0
            else max(int(self._state.last_area / max(bbox_w, 1)), 1)
        )
        detection = Detection(
            class_name=self._state.last_class_name,
            confidence=self._state.last_confidence,
            bbox=(
                int(predicted[0] - bbox_w / 2),
                int(predicted[1] - bbox_h / 2),
                int(predicted[0] + bbox_w / 2),
                int(predicted[1] + bbox_h / 2),
            ),
            track_id=self._state.track_id,
            center=predicted,
            area=self._state.last_area,
        )
        return TrackedTarget(
            detection=detection,
            track_id=self._state.track_id,
            smooth_center=self._state.smooth_center,
            smoothed_area_ratio=self._state.last_area / max(frame_w * frame_h, 1),
            direction=cast(
                TargetDirection,
                resolve_direction(self._state.smooth_center[0], frame_w),
            ),
            frame_width=frame_w,
            frame_height=frame_h,
            frames_tracked=self._state.frames_tracked,
            frames_since_seen=self._state.frames_since_seen,
            is_confirmed=self._state.frames_tracked >= self._cfg.min_confirm_frames,
            velocity_estimate=(float(self._state.x[2, 0]), float(self._state.x[3, 0])),
            predicted_center=predicted if self._state.frames_since_seen else None,
        )


@dataclass(slots=True)
class PrimaryTrack:
    track_id: int
    center: tuple[float, float]
    last_seen_ns: int
    is_stale: bool


class TargetTrackMemory:
    """Persist the primary track between slower planner updates."""

    def __init__(self, stale_after_ns: int = 2_000_000_000) -> None:
        self._stale_after_ns = stale_after_ns
        self._primary: PrimaryTrack | None = None

    def lock_target(self, track_id: int, center: tuple[float, float], timestamp_ns: int) -> None:
        self._primary = PrimaryTrack(track_id, center, timestamp_ns, False)

    def refresh(self, target: TrackedTarget | None, timestamp_ns: int) -> None:
        if target is None:
            if self._primary and (timestamp_ns - self._primary.last_seen_ns) > self._stale_after_ns:
                self._primary = PrimaryTrack(
                    self._primary.track_id,
                    self._primary.center,
                    self._primary.last_seen_ns,
                    True,
                )
            return
        self.lock_target(target.track_id, target.smooth_center, timestamp_ns)

    def get_primary(self) -> PrimaryTrack | None:
        return self._primary
