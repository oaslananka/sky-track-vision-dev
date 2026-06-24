from __future__ import annotations

import numpy as np

from autonomy.contracts import Detection, SceneInsight, TrackedTarget
from autonomy.scene_reasoner import SceneReasoner
from config.settings import VisionConfig
from vision.detector import MultiClassDetector
from vision.tracker import KalmanTracker, TargetTrackMemory


class VisionObserver:
    """Read detections, tracker state, and scene insight for the LLM pilot."""

    def __init__(self, cfg: VisionConfig) -> None:
        self._detector = MultiClassDetector(cfg)
        self._tracker = KalmanTracker(cfg)
        self._memory = TargetTrackMemory()
        self._scene_reasoner = SceneReasoner()

    def observe(
        self,
        frame: np.ndarray,
        *,
        priority_class: str | None,
        frame_size: tuple[int, int],
        timestamp_ns: int,
    ) -> tuple[list[Detection], TrackedTarget | None, SceneInsight]:
        detections = self._detector.track(frame)
        target = self._tracker.update(detections, priority_class, frame_size)
        self._memory.refresh(target, timestamp_ns)
        scene = self._scene_reasoner.analyze(detections)
        return detections, target, scene

    @property
    def memory(self) -> TargetTrackMemory:
        return self._memory
