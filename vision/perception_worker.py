"""Decoupled perception thread for non-blocking YOLO + tracker execution.

Phase 1: Runs object detection and Kalman tracking in a dedicated thread so
that the control loop in PilotDisplay can tick at a fixed rate (~30 Hz)
independent of YOLO inference time (~30-200 ms depending on hardware).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from autonomy.contracts import Detection, TrackedTarget
from config.runtime_logging import log_event
from config.settings import VisionConfig
from vision.detector import MultiClassDetector
from vision.tracker import KalmanTracker

logger = logging.getLogger("skytrackvision.perception_worker")


@dataclass(slots=True)
class PerceptionSnapshot:
    """Thread-safe snapshot of the latest perception results."""

    detections: list[Detection] = field(default_factory=list)
    target: TrackedTarget | None = None
    frame: np.ndarray | None = None
    timestamp_ns: int = 0
    inference_ms: float = 0.0


class PerceptionWorker:
    """Run YOLO detection + Kalman tracking in a background thread.

    The control loop reads the latest results via `latest()` without blocking.
    """

    def __init__(
        self,
        cfg: VisionConfig,
        priority_class: str | None = None,
    ) -> None:
        self._detector = MultiClassDetector(cfg)
        self._tracker = KalmanTracker(cfg)
        self._priority_class = priority_class

        self._lock = threading.Lock()
        self._snapshot = PerceptionSnapshot()
        self._pending_lock = threading.Lock()

        self._frame_event = threading.Event()
        self._pending_frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def priority_class(self) -> str | None:
        return self._priority_class

    @priority_class.setter
    def priority_class(self, value: str | None) -> None:
        self._priority_class = value

    def start(self) -> None:
        """Start the background perception thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="perception-worker", daemon=True
        )
        self._thread.start()
        log_event(logger, logging.INFO, "perception.start", "Perception worker started")

    def stop(self) -> None:
        """Stop the background perception thread."""
        self._running = False
        self._frame_event.set()  # Unblock if waiting
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log_event(logger, logging.INFO, "perception.stop", "Perception worker stopped")

    def submit_frame(self, frame: np.ndarray) -> None:
        """Submit a new camera frame for processing (non-blocking, drops old frames)."""
        with self._pending_lock:
            self._pending_frame = frame
        self._frame_event.set()

    def latest(self) -> PerceptionSnapshot:
        """Get the most recent perception results (thread-safe, non-blocking)."""
        with self._lock:
            return PerceptionSnapshot(
                detections=list(self._snapshot.detections),
                target=self._snapshot.target,
                frame=self._snapshot.frame,
                timestamp_ns=self._snapshot.timestamp_ns,
                inference_ms=self._snapshot.inference_ms,
            )

    def _run_loop(self) -> None:
        """Background loop: wait for frames, run detection + tracking."""
        while self._running:
            # Wait for a new frame (with timeout to allow clean shutdown)
            if not self._frame_event.wait(timeout=0.5):
                continue
            # Clear event BEFORE acquiring lock to prevent signal loss
            self._frame_event.clear()
            with self._pending_lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None:
                continue

            try:
                h, w = frame.shape[:2]
                t0 = time.monotonic()

                detections = self._detector.track(frame)
                target = self._tracker.update(
                    detections,
                    self._priority_class,
                    (w, h),
                    frame=frame,
                )

                inference_ms = (time.monotonic() - t0) * 1000.0

                with self._lock:
                    self._snapshot = PerceptionSnapshot(
                        detections=detections,
                        target=target,
                        frame=frame,
                        timestamp_ns=time.time_ns(),
                        inference_ms=inference_ms,
                    )

            except Exception as e:
                log_event(
                    logger,
                    logging.ERROR,
                    "perception.error",
                    "Perception inference failed",
                    reason=str(e),
                )
