from __future__ import annotations

import numpy as np

from autonomy.contracts import (
    Detection,
    MissionContext,
    SafetyEvaluation,
    SceneInsight,
    TrackedTarget,
)
from vision.annotator import FrameAnnotator


class OverlayRenderer:
    """Small wrapper around the frame annotator for classic runtime composition."""

    def __init__(self, overlay_mode: str = "SHOWCASE") -> None:
        self._annotator = FrameAnnotator(overlay_mode)

    def set_mode(self, overlay_mode: str) -> None:
        self._annotator.set_mode(overlay_mode)

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        target: TrackedTarget | None,
        scene: SceneInsight,
        mission: MissionContext,
        safety: SafetyEvaluation,
    ) -> np.ndarray:
        return self._annotator.draw(frame, detections, target, scene, mission, safety)
