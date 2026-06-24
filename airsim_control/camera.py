from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autonomy.contracts import FramePacket
from config.settings import AirSimConfig

try:
    import airsim
except Exception:  # pragma: no cover - optional runtime dependency
    airsim = None


_ROAD_SEG_ID = 80  # Fixed stencil ID for AirSimNH road meshes.


class DroneCameraStream:
    """Fetch BGR scene frames from AirSim and preserve the last valid frame as fallback."""

    def __init__(self, client: Any, cfg: AirSimConfig) -> None:
        self._client = client
        self._cfg = cfg
        self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._seg_initialized = False
        self._last_seg_frame: np.ndarray | None = None
        # Serialise all AirSim calls through one lock — the msgpack-rpc client
        # is NOT thread-safe; concurrent calls from CameraThread + control loop
        # cause Tornado to flood "Uncaught exception, closing connection." errors.
        self._client_lock = threading.Lock()

    def _init_road_segmentation(self) -> None:
        """Assign a known stencil ID to AirSimNH road meshes once per session."""
        try:
            patterns = (
                r"Road_[\w]+",
                r"road_[\w]+",
                r"Street_[\w]+",
                r"street_[\w]+",
                r".*Road.*",
                r".*road.*",
                r".*Street.*",
                r".*street.*",
                r".*Lane.*",
                r".*lane.*",
                r".*Asphalt.*",
                r".*asphalt.*",
            )
            for pattern in patterns:
                self._client.simSetSegmentationObjectID(pattern, _ROAD_SEG_ID, True)
        except Exception:
            pass
        self._seg_initialized = True

    def get_segmentation_frame(self) -> np.ndarray | None:
        """Return the segmentation frame where road pixels use the configured stencil ID."""
        if airsim is None:
            return None
        if not self._seg_initialized:
            self._init_road_segmentation()
        try:
            with self._client_lock:
                resp = self._client.simGetImages(
                    [
                        airsim.ImageRequest(
                            self._cfg.camera_name,
                            airsim.ImageType.Segmentation,
                            pixels_as_float=False,
                            compress=False,
                        )
                    ],
                    vehicle_name=self._cfg.vehicle_name,
                )[0]
            if not resp.image_data_uint8:
                return self._last_seg_frame
            w, h = resp.width, resp.height
            if w > 0 and h > 0:
                expected = w * h * 3
                if len(resp.image_data_uint8) == expected:
                    seg = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(h, w, 3)
                    self._last_seg_frame = np.ascontiguousarray(seg[:, :, ::-1])
                    return self._last_seg_frame
            buf = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            decoded: npt.NDArray[np.uint8] | None = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # type: ignore[assignment]
            if decoded is not None:
                self._last_seg_frame = decoded
            return self._last_seg_frame
        except Exception:
            return self._last_seg_frame

    def get_frame(self) -> FramePacket:
        if airsim is None:
            raise RuntimeError("airsim package is required for live camera streaming")
        response = self._client.simGetImages(
            [
                airsim.ImageRequest(
                    self._cfg.camera_name,
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=self._cfg.camera_compress,
                )
            ],
            vehicle_name=self._cfg.vehicle_name,
        )[0]
        frame = self._decode(
            response.image_data_uint8,
            width=getattr(response, "width", 0),
            height=getattr(response, "height", 0),
        )
        self._last_frame = frame
        height, width = frame.shape[:2]
        return FramePacket(
            frame=frame,
            timestamp_ns=time.time_ns(),
            width=width,
            height=height,
            camera_name=self._cfg.camera_name,
            vehicle_name=self._cfg.vehicle_name,
        )

    def _decode(self, payload: bytes, *, width: int = 0, height: int = 0) -> np.ndarray:
        if not payload:
            return self._last_frame
        if not self._cfg.camera_compress and width > 0 and height > 0:
            expected = width * height * 3
            if len(payload) == expected:
                rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
                return np.ascontiguousarray(rgb[:, :, ::-1])
        buffer = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3:
            return self._last_frame
        return frame
