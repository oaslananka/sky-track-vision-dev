from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from autonomy.contracts import Detection
from config.settings import VisionConfig
from vision.perception_worker import PerceptionWorker


class _FakeDetector:
    def __init__(self, _cfg: VisionConfig) -> None:
        pass

    def track(self, _frame: np.ndarray) -> list[Detection]:
        return [
            Detection(
                class_name="truck",
                confidence=0.88,
                bbox=(20, 20, 80, 80),
                track_id=5,
                center=(50.0, 50.0),
                area=3600.0,
            )
        ]


class _FakeTracker:
    def __init__(self, _cfg: VisionConfig) -> None:
        self.frame_seen_event = threading.Event()
        self.last_frame_shape: tuple[int, int, int] | None = None

    def update(
        self,
        detections: list[Detection],
        _priority_class: str | None,
        _frame_size: tuple[int, int],
        frame: np.ndarray | None = None,
    ) -> None:
        if detections and frame is not None:
            self.last_frame_shape = frame.shape
            self.frame_seen_event.set()
        return None


def test_perception_worker_passes_frame_to_tracker_for_reid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("vision.perception_worker.MultiClassDetector", _FakeDetector)
    monkeypatch.setattr("vision.perception_worker.KalmanTracker", _FakeTracker)

    worker = PerceptionWorker(VisionConfig(), priority_class="truck")
    try:
        worker.start()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        worker.submit_frame(frame)

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            latest = worker.latest()
            if latest.timestamp_ns > 0:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("PerceptionWorker did not publish a snapshot")

        assert len(latest.detections) == 1
        tracker = worker._tracker
        assert tracker.frame_seen_event.is_set()
        assert tracker.last_frame_shape == (120, 160, 3)
    finally:
        worker.stop()
