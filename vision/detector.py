from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from autonomy.contracts import Detection
from config.settings import VisionConfig
from vision.utils import bbox_area, bbox_center

logger = logging.getLogger("skytrackvision.detector")


class MultiClassDetector:
    """YOLOv8 detector and tracker facade tuned for low-latency AirSim frames."""

    def __init__(self, cfg: VisionConfig) -> None:
        self._cfg = cfg
        self._device = self._select_device()
        self._half = self._device.startswith("cuda") and cfg.use_half_precision
        self._model = YOLO(cfg.model_path)
        self._target_ids = self._resolve_class_ids(cfg.target_classes)
        logger.info("Detector initialized on device=%s half=%s", self._device, self._half)
        self._warmup()

    def _select_device(self) -> str:
        preferred = (self._cfg.preferred_device or "auto").strip().lower()
        cuda_ok, mps_ok = self._detect_accelerators()

        if preferred != "auto":
            if preferred.startswith("cuda"):
                return preferred if cuda_ok else "cpu"
            if preferred == "mps":
                return "mps" if mps_ok else "cpu"
            if preferred == "cpu":
                return "cpu"
            return "cpu"

        if cuda_ok:
            return "cuda:0"
        if mps_ok:
            return "mps"
        return "cpu"

    def _detect_accelerators(self) -> tuple[bool, bool]:
        try:
            import torch
        except Exception:
            return False, False
        cuda_ok = bool(torch.cuda.is_available())
        mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        return cuda_ok, mps_ok

    def _resolve_class_ids(self, target_classes: list[str]) -> list[int]:
        names = self._model.names
        resolved = [idx for idx, name in names.items() if name in target_classes]
        return resolved or list(names.keys())

    def _warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(
            dummy,
            classes=self._target_ids,
            conf=self._cfg.confidence_threshold,
            verbose=False,
            device=self._device,
            imgsz=self._cfg.inference_imgsz,
            max_det=self._cfg.max_detections,
            half=self._half,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame,
            classes=self._target_ids,
            conf=self._cfg.confidence_threshold,
            verbose=False,
            device=self._device,
            imgsz=self._cfg.inference_imgsz,
            max_det=self._cfg.max_detections,
            half=self._half,
        )
        return self._parse_results(results[0], with_track_ids=False)

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame,
            classes=self._target_ids,
            conf=self._cfg.confidence_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=self._device,
            imgsz=self._cfg.inference_imgsz,
            max_det=self._cfg.max_detections,
            half=self._half,
        )
        return self._parse_results(results[0], with_track_ids=True)

    def _parse_results(self, result: object, *, with_track_ids: bool) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int) if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
        classes = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
        ids = (
            boxes.id.int().cpu().numpy().tolist() if with_track_ids and boxes.id is not None else []
        )
        detections: list[Detection] = []
        for idx, bbox_array in enumerate(xyxy):
            bbox = (
                int(bbox_array[0]),
                int(bbox_array[1]),
                int(bbox_array[2]),
                int(bbox_array[3]),
            )
            detections.append(
                Detection(
                    class_name=str(self._model.names[classes[idx]]),
                    confidence=float(confs[idx]),
                    bbox=bbox,
                    track_id=(
                        int(ids[idx])
                        if idx < len(ids) and ids[idx] is not None and not np.isnan(float(ids[idx]))
                        else None
                    ),
                    center=bbox_center(bbox),
                    area=bbox_area(bbox),
                )
            )
        return detections


def ensure_model_path(cfg: VisionConfig) -> Path:
    return Path(cfg.model_path)
