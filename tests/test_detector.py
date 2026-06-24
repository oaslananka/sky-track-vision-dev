from __future__ import annotations

import numpy as np
import pytest

from config.settings import VisionConfig
from vision.detector import MultiClassDetector


class _FakeBoxes:
    xyxy = None
    conf = None
    cls = None
    id = None


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeYOLO:
    def __init__(self, _path: str) -> None:
        self.names = {0: "person", 1: "car", 2: "truck"}
        self.detect_calls: list[dict[str, object]] = []
        self.track_calls: list[dict[str, object]] = []
        self.warmup_calls: list[dict[str, object]] = []
        self._first_call = True

    def __call__(self, _frame: np.ndarray, **kwargs: object) -> list[_FakeResult]:
        if self._first_call:
            self.warmup_calls.append(kwargs)
            self._first_call = False
        else:
            self.detect_calls.append(kwargs)
        return [_FakeResult()]

    def track(self, _frame: np.ndarray, **kwargs: object) -> list[_FakeResult]:
        self.track_calls.append(kwargs)
        return [_FakeResult()]


def test_detector_uses_optimized_inference_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeYOLO("models/yolov8n.pt")
    monkeypatch.setattr("vision.detector.YOLO", lambda _path: fake_model)
    monkeypatch.setattr(MultiClassDetector, "_select_device", lambda self: "cuda:0")

    detector = MultiClassDetector(
        VisionConfig(
            target_classes=["car", "truck"],
            inference_imgsz=512,
            max_detections=24,
            use_half_precision=True,
        )
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.detect(frame)
    detector.track(frame)

    assert fake_model.warmup_calls[-1]["imgsz"] == 512
    assert fake_model.detect_calls[-1]["imgsz"] == 512
    assert fake_model.detect_calls[-1]["max_det"] == 24
    assert fake_model.detect_calls[-1]["half"] is True
    assert fake_model.track_calls[-1]["imgsz"] == 512
    assert fake_model.track_calls[-1]["max_det"] == 24
    assert fake_model.track_calls[-1]["half"] is True
    assert fake_model.track_calls[-1]["persist"] is True


def test_detector_prefers_configured_cuda_device_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeYOLO("models/yolov8n.pt")
    monkeypatch.setattr("vision.detector.YOLO", lambda _path: fake_model)
    monkeypatch.setattr(MultiClassDetector, "_detect_accelerators", lambda self: (True, False))

    detector = MultiClassDetector(
        VisionConfig(
            preferred_device="cuda:0",
            use_half_precision=True,
        )
    )

    assert detector._device == "cuda:0"
    assert detector._half is True


def test_detector_falls_back_to_cpu_when_cuda_preferred_but_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeYOLO("models/yolov8n.pt")
    monkeypatch.setattr("vision.detector.YOLO", lambda _path: fake_model)
    monkeypatch.setattr(MultiClassDetector, "_detect_accelerators", lambda self: (False, False))

    detector = MultiClassDetector(
        VisionConfig(
            preferred_device="cuda:0",
            use_half_precision=True,
        )
    )

    assert detector._device == "cpu"
    assert detector._half is False
