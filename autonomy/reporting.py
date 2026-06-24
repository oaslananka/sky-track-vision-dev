from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from autonomy.contracts import (
    Detection,
    MissionContext,
    MissionReport,
    SafetyEvaluation,
    TrackedTarget,
)
from vision.object_registry import MultiObjectRegistry


class EventReporter:
    """Collect mission telemetry and emit a compact final report."""

    _MAX_LOG_ENTRIES = 5000
    _TRIM_KEEP = 2500

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mission_id = str(uuid.uuid4())
        self._started_at = time.monotonic()
        self._events: list[str] = []
        self._state_transitions: list[tuple[str, float]] = []
        self._target_ids: set[int] = set()
        self._registry = MultiObjectRegistry()
        self._last_progress = 0.0
        self._last_mode = "IDLE"
        self._last_state: str | None = None

    @property
    def mission_id(self) -> str:
        return self._mission_id

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started_at

    def registry_summary(self) -> dict[str, Any]:
        with self._lock:
            summary = self._registry.summary()
        return {
            "total_objects": summary.total_objects,
            "active_objects": summary.active_objects,
            "unique_object_counts": summary.unique_object_counts,
            "unique_vehicle_count": summary.unique_vehicle_count,
            "active_objects_view": summary.active_object_views,
            "merge_count": summary.merge_count,
        }

    def update(
        self,
        mission: MissionContext,
        target: TrackedTarget | None,
        safety: SafetyEvaluation,
        detections: list[Detection] | None = None,
        frame: Any | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        observed = detections
        if observed is None:
            observed = [target.detection] if target is not None else []
        self.record_runtime_snapshot(
            mission,
            detections=observed,
            target=target,
            safety_reason=safety.reason,
            frame=frame,
            timestamp_ns=timestamp_ns,
        )

    def record_runtime_snapshot(
        self,
        mission: MissionContext,
        *,
        detections: list[Detection],
        target: TrackedTarget | None,
        safety_reason: str,
        frame: Any | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        timestamp = time.monotonic()
        with self._lock:
            self._last_progress = mission.progress
            self._last_mode = mission.mode.value
            if self._last_state != mission.state.value:
                self._state_transitions.append((mission.state.value, timestamp))
                self._last_state = mission.state.value
            self._events.append(f"{mission.state.value}: {safety_reason}")
            if target is not None:
                self._target_ids.add(target.track_id)
            self._registry.update(
                detections,
                frame=frame,
                timestamp_ns=timestamp_ns,
            )
            self._trim_buffers()

    def _trim_buffers(self) -> None:
        if len(self._events) > self._MAX_LOG_ENTRIES:
            self._events = self._events[-self._TRIM_KEEP :]
        if len(self._state_transitions) > self._MAX_LOG_ENTRIES:
            self._state_transitions = self._state_transitions[-self._TRIM_KEEP :]

    def finalize(self, success: bool = True, reason: str = "completed") -> MissionReport:
        with self._lock:
            summary = self._registry.summary()
            return MissionReport(
                mission_id=self._mission_id,
                mode=self._last_mode,
                state_transitions=[(s, t) for s, t in self._state_transitions],
                target_ids_seen=sorted(self._target_ids),
                completion_progress=self._last_progress,
                success=success,
                completion_reason=reason,
                duration_s=self.elapsed_s,
                unique_track_counts=dict(
                    summary.unique_object_counts
                ),  # per-class breakdown (primary)
                unique_vehicle_count=summary.unique_vehicle_count,
                unique_object_counts=dict(
                    summary.unique_object_counts
                ),  # alias kept for API compatibility
                active_object_count=summary.active_objects,
                registry_merge_count=summary.merge_count,
                events=list(self._events),
            )
