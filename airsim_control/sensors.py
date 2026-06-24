from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from airsim_control.client import get_client_rpc_lock
from autonomy.contracts import LidarReading, ProximityReading, SensorSnapshot, TelemetryReading
from config.runtime_logging import log_event
from config.settings import AirSimConfig

logger = logging.getLogger("skytrackvision.sensors")


class SensorSuiteReader:
    """Read LiDAR, range sensors, and telemetry into a single snapshot contract."""

    def __init__(self, client: Any, cfg: AirSimConfig) -> None:
        self._client = client
        self._cfg = cfg
        self._rpc_lock = get_client_rpc_lock(client)
        self.last_snapshot: SensorSnapshot | None = None
        self._read_count = 0

    def read(self) -> SensorSnapshot:
        missing: list[str] = []
        with self._rpc_lock:
            lidar = self._read_lidar()
        with self._rpc_lock:
            proximity = self._read_proximity(missing)
        with self._rpc_lock:
            telemetry = self._read_telemetry()
        snapshot = SensorSnapshot(
            lidar=lidar,
            proximity=proximity,
            telemetry=telemetry,
            timestamp_ns=time.time_ns(),
            missing_features=missing,
        )
        self.last_snapshot = snapshot
        self._read_count += 1
        interval = max(1, self._cfg.sensor_debug_log_interval)
        if (
            self._read_count == 1
            or self._read_count % interval == 0
            or not snapshot.proximity.available
            or bool(snapshot.missing_features)
        ):
            log_event(
                logger,
                logging.DEBUG,
                "telemetry.snapshot",
                "Sensor snapshot captured",
                altitude_m=round(snapshot.telemetry.altitude_m, 3),
                position_ned=list(snapshot.telemetry.position_ned),
                velocity_ned=list(snapshot.telemetry.velocity_ned),
                front_m=round(snapshot.proximity.front_m, 3),
                down_m=round(snapshot.proximity.down_m, 3),
                lidar_cluster_count=snapshot.lidar.cluster_count,
                lidar_min_distance_m=round(snapshot.lidar.min_distance_m, 3),
                proximity_available=snapshot.proximity.available,
                snapshot_timestamp_ns=snapshot.timestamp_ns,
                snapshot_age_ms=0.0,
                missing_features=snapshot.missing_features,
            )
        return snapshot

    def _read_lidar(self) -> LidarReading:
        data = self._client.getLidarData(
            lidar_name=self._cfg.lidar_name,
            vehicle_name=self._cfg.vehicle_name,
        )
        if not data.point_cloud:
            return LidarReading(point_count=0, cluster_count=0, min_distance_m=float("inf"))
        points = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        grid = np.floor(points[:, :2] / 0.5).astype(np.int32)
        cluster_count = int(np.unique(grid, axis=0).shape[0])
        min_distance = float(np.min(np.linalg.norm(points, axis=1)))
        return LidarReading(
            point_count=int(points.shape[0]),
            cluster_count=cluster_count,
            min_distance_m=min_distance,
        )

    def _read_proximity(self, missing: list[str]) -> ProximityReading:
        distances: dict[str, float] = {}
        available = True
        for direction, aliases in self._cfg.proximity_aliases.items():
            distance = self._resolve_distance(aliases)
            if distance is None:
                available = False
                missing.append(f"{direction}_proximity")
                distances[direction] = float("inf")
            else:
                distances[direction] = distance
        return ProximityReading(
            front_m=distances["front"],
            rear_m=distances["rear"],
            left_m=distances["left"],
            right_m=distances["right"],
            down_m=distances["down"],
            available=available,
        )

    def _resolve_distance(self, aliases: list[str]) -> float | None:
        for name in aliases:
            try:
                reading = self._client.getDistanceSensorData(
                    distance_sensor_name=name,
                    vehicle_name=self._cfg.vehicle_name,
                )
            except Exception:
                continue
            if reading is not None and getattr(reading, "distance", math.inf) != math.inf:
                return float(reading.distance)
        return None

    def _read_telemetry(self) -> TelemetryReading:
        state = self._client.getMultirotorState(vehicle_name=self._cfg.vehicle_name)
        kin = state.kinematics_estimated
        orientation = kin.orientation
        position = kin.position
        linear_velocity = kin.linear_velocity
        roll, pitch, yaw = self._quaternion_to_euler(
            orientation.w_val,
            orientation.x_val,
            orientation.y_val,
            orientation.z_val,
        )
        altitude_m = abs(getattr(position, "z_val", 0.0))
        return TelemetryReading(
            position_ned=(position.x_val, position.y_val, position.z_val),
            velocity_ned=(
                linear_velocity.x_val,
                linear_velocity.y_val,
                linear_velocity.z_val,
            ),
            roll_deg=math.degrees(roll),
            pitch_deg=math.degrees(pitch),
            yaw_deg=math.degrees(yaw),
            altitude_m=altitude_m,
            gps_valid=state.gps_location is not None,
        )

    def _quaternion_to_euler(
        self,
        w: float,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
