from __future__ import annotations

import time
from dataclasses import dataclass, field


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def resolve_direction(cx: float, frame_width: int, tolerance_px: float = 24.0) -> str:
    center_x = frame_width / 2
    if cx < center_x - tolerance_px:
        return "LEFT"
    if cx > center_x + tolerance_px:
        return "RIGHT"
    return "CENTERED"


@dataclass(slots=True)
class FpsCounter:
    last_ts: float = field(default_factory=time.monotonic)
    fps: float = 0.0

    def tick(self) -> float:
        now = time.monotonic()
        dt = now - self.last_ts
        self.last_ts = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps
