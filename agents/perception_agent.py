from __future__ import annotations

import numpy as np

from autonomy.contracts import Detection, TrackedTarget
from config.settings import VisionConfig
from vision.detector import MultiClassDetector
from vision.tracker import KalmanTracker


class PerceptionAgent:
    """Facade over detection and tracking for the main runtime loop."""

    def __init__(self, cfg: VisionConfig) -> None:
        self._detector = MultiClassDetector(cfg)
        self._tracker = KalmanTracker(cfg)

    def observe(
        self,
        frame: np.ndarray,
        priority_class: str | None,
        frame_size: tuple[int, int],
    ) -> tuple[list[Detection], TrackedTarget | None]:
        detections = self._detector.track(frame)
        target = self._tracker.update(detections, priority_class, frame_size)
        return detections, target
