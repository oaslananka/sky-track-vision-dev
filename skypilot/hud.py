from __future__ import annotations

import cv2
import numpy as np

from autonomy.contracts import MissionContext, SafetyEvaluation, TrackedTarget


class SkyPilotHudRenderer:
    """Draw a compact pilot-oriented HUD over the active frame."""

    def draw(
        self,
        frame: np.ndarray,
        mission: MissionContext,
        safety: SafetyEvaluation,
        target: TrackedTarget | None,
    ) -> np.ndarray:
        canvas = frame.copy()
        lines = [
            f"Pilot Mode: {mission.mode.value}",
            f"Mission State: {mission.state.value}",
            f"Safety: {safety.state.value}",
            f"Target: {target.track_id if target else 'none'}",
        ]
        for index, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (18, 24 + 24 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas
