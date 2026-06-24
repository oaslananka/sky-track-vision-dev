from __future__ import annotations

from autonomy.contracts import Detection, SceneInsight
from autonomy.scene_reasoner import SceneReasoner


class SceneAgent:
    """Facade over scene reasoning to keep runtime wiring compact."""

    def __init__(self) -> None:
        self._reasoner = SceneReasoner()

    def analyze(self, detections: list[Detection]) -> SceneInsight:
        return self._reasoner.analyze(detections)
