from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from autonomy.contracts import (
    Detection,
    MissionContext,
    SafetyEvaluation,
    SceneInsight,
    TrackedTarget,
)


class FrameAnnotator:
    """Render SHOWCASE and DEBUG overlays over the active frame."""

    def __init__(self, overlay_mode: str = "SHOWCASE") -> None:
        self._overlay_mode = overlay_mode
        # Color palette (BGR)
        self.color_primary = (48, 214, 106)  # Green
        self.color_secondary = (255, 189, 46)  # Orange
        self.color_danger = (68, 68, 255)  # Red
        self.color_bg = (20, 20, 20)  # Dark
        self.color_text = (255, 255, 255)  # White

    def set_mode(self, overlay_mode: str) -> None:
        self._overlay_mode = overlay_mode

    def draw(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        target: TrackedTarget | None,
        scene: SceneInsight,
        mission: MissionContext,
        safety: SafetyEvaluation,
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw top status bar (semi-transparent)
        self._draw_status_bar(canvas, mission, safety, w)

        # Draw detections
        self._draw_detections(canvas, detections, target)

        # Draw bottom info panel
        self._draw_info_panel(canvas, scene, mission, safety, detections, h, w)

        # Draw target lock indicator if tracking
        if target and target.is_confirmed:
            self._draw_target_lock(canvas, target)

        return canvas

    def _draw_status_bar(
        self,
        frame: np.ndarray,
        mission: MissionContext,
        safety: SafetyEvaluation,
        width: int,
    ) -> None:
        """Draw top status bar with mission and safety info."""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), self.color_bg, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Mission state
        state_color = self.color_primary if mission.state.value != "BLOCKED" else self.color_danger
        self._put_text_premium(frame, f"◉ {mission.state.value}", (20, 25), state_color, scale=0.7)

        # Safety status
        safety_color = (
            self.color_primary if safety.state.value == "PATH_CLEAR" else self.color_danger
        )
        self._put_text_premium(frame, f"⚡ {safety.state.value}", (20, 50), safety_color, scale=0.6)

        # Mode (right aligned)
        mode_text = f"MODE: {mission.mode.value}"
        (tw, _th), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        self._put_text_premium(
            frame, mode_text, (width - tw - 20, 35), self.color_secondary, scale=0.6
        )

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        scene: SceneInsight,
        mission: MissionContext,
        safety: SafetyEvaluation,
        detections: list[Detection],
        height: int,
        width: int,
    ) -> None:
        """Draw bottom info panel."""
        if self._overlay_mode == "SHOWCASE":
            # Simple info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, height - 50), (width, height), self.color_bg, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            self._put_text_premium(
                frame, scene.summary_text, (20, height - 20), self.color_text, scale=0.6
            )
        else:
            # DEBUG mode - more info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, height - 120), (width, height), self.color_bg, -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            counts = Counter(d.class_name for d in detections)
            self._put_text_premium(
                frame, scene.summary_text, (20, height - 95), self.color_text, scale=0.5
            )
            self._put_text_premium(
                frame,
                f"Detections: {dict(counts)}",
                (20, height - 70),
                self.color_secondary,
                scale=0.5,
            )
            self._put_text_premium(
                frame,
                f"Safety: {safety.reason}",
                (20, height - 45),
                self.color_secondary,
                scale=0.5,
            )

    def _draw_target_lock(self, frame: np.ndarray, target: TrackedTarget) -> None:
        """Draw target lock indicator (crosshair + bracket corners)."""
        cx, cy = map(int, target.smooth_center)
        color = self.color_primary

        # Center crosshair
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 24, 3)

        # Bracket corners around detection
        x1, y1, x2, y2 = target.detection.bbox
        bracket_len = 20

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 3)

        # Top-right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 3)

        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 3)

        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 3)

        # "TRACKING" label
        label = f"◈ TRACKING: {target.detection.class_name.upper()}"
        _h, w = frame.shape[:2]
        (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        self._put_text_premium(
            frame, label, (w // 2 - tw // 2, 100), self.color_primary, scale=0.7, thickness=2
        )

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        target: TrackedTarget | None,
    ) -> None:
        for detection in detections:
            is_primary = target is not None and detection.track_id == target.track_id
            color = self.color_primary if is_primary else self.color_secondary
            x1, y1, x2, y2 = detection.bbox

            # Thicker border for primary target
            thickness = 3 if is_primary else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label with confidence
            label = f"{detection.class_name.upper()} {detection.confidence:.0%}"
            if detection.track_id:
                label += f" #{detection.track_id}"

            self._put_text_premium(frame, label, (x1, max(30, y1 - 10)), color, scale=0.55)

    def _put_text_premium(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.6,
        thickness: int = 2,
    ) -> None:
        """Draw text with shadow for better readability."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = origin

        # Shadow
        cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)

        # Main text
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
