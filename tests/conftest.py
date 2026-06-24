from __future__ import annotations

import time

from autonomy.contracts import (
    Detection,
    LidarReading,
    ProximityReading,
    SensorSnapshot,
    TelemetryReading,
    TrackedTarget,
)


def make_telemetry(
    *,
    position_ned: tuple[float, float, float] = (0.0, 0.0, -3.0),
    velocity_ned: tuple[float, float, float] = (0.0, 0.0, 0.0),
    altitude_m: float = 3.0,
) -> TelemetryReading:
    return TelemetryReading(
        position_ned=position_ned,
        velocity_ned=velocity_ned,
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_m=altitude_m,
        gps_valid=True,
    )


def make_snapshot(
    *,
    front_m: float = 10.0,
    down_m: float = 5.0,
    proximity_available: bool = True,
    lidar_clusters: int = 0,
    min_distance_m: float = 10.0,
) -> SensorSnapshot:
    return SensorSnapshot(
        lidar=LidarReading(
            point_count=64,
            cluster_count=lidar_clusters,
            min_distance_m=min_distance_m,
        ),
        proximity=ProximityReading(
            front_m=front_m,
            rear_m=10.0,
            left_m=10.0,
            right_m=10.0,
            down_m=down_m,
            available=proximity_available,
        ),
        telemetry=make_telemetry(altitude_m=down_m),
        timestamp_ns=time.time_ns(),
        missing_features=[],
    )


def make_target(
    *,
    track_id: int = 7,
    center: tuple[float, float] = (320.0, 240.0),
    area: float = 5_000.0,
    frames_tracked: int = 5,
    frame_width: int = 640,
    frame_height: int = 480,
) -> TrackedTarget:
    detection = Detection(
        class_name="person",
        confidence=0.9,
        bbox=(290, 190, 350, 290),
        track_id=track_id,
        center=center,
        area=area,
    )
    return TrackedTarget(
        detection=detection,
        track_id=track_id,
        smooth_center=center,
        smoothed_area_ratio=area / (frame_width * frame_height),
        direction="CENTERED",
        frame_width=frame_width,
        frame_height=frame_height,
        frames_tracked=frames_tracked,
        frames_since_seen=0,
        is_confirmed=frames_tracked >= 3,
        velocity_estimate=(0.0, 0.0),
        predicted_center=None,
    )
