from __future__ import annotations

import cv2
import numpy as np

from autonomy.contracts import (
    FramePacket,
    LidarReading,
    ProximityReading,
    SensorSnapshot,
    TelemetryReading,
    WorldSnapshot,
)
from demo.stages import DEFAULT_STAGES, DemoStage


class DemoDirector:
    """Drive AirSim-free demo stages using synthetic frames and sensor snapshots."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width = width
        self._height = height
        self._stages = DEFAULT_STAGES
        self._index = 0

    @property
    def stage(self) -> DemoStage:
        return self._stages[self._index]

    def next_stage(self) -> None:
        self._index = min(self._index + 1, len(self._stages) - 1)

    def previous_stage(self) -> None:
        self._index = max(self._index - 1, 0)

    def next_frame(self) -> WorldSnapshot:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            self.stage.banner,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if self.stage.target_visible:
            cv2.rectangle(frame, (270, 150), (360, 320), (0, 200, 255), 3)
        packet = FramePacket(
            frame=frame,
            timestamp_ns=0,
            width=self._width,
            height=self._height,
            camera_name="demo",
            vehicle_name="Drone",
        )
        return WorldSnapshot(frame=packet, sensors=self.make_fake_snapshot(), connection_ok=True)

    def make_fake_snapshot(self) -> SensorSnapshot:
        front_distance = 2.0 if self.stage.obstacle_ahead else 10.0
        return SensorSnapshot(
            lidar=LidarReading(
                point_count=32,
                cluster_count=12 if self.stage.obstacle_ahead else 0,
                min_distance_m=front_distance,
            ),
            proximity=ProximityReading(
                front_m=front_distance,
                rear_m=10.0,
                left_m=10.0,
                right_m=10.0,
                down_m=3.0,
                available=True,
            ),
            telemetry=TelemetryReading(
                position_ned=(0.0, 0.0, -3.0),
                velocity_ned=(0.0, 0.0, 0.0),
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                altitude_m=3.0,
                gps_valid=False,
            ),
            timestamp_ns=0,
            missing_features=[],
        )
