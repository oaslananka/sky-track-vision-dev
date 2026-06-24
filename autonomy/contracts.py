from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np

TargetDirection = Literal["LEFT", "RIGHT", "CENTERED"]


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray
    timestamp_ns: int
    width: int
    height: int
    camera_name: str
    vehicle_name: str


@dataclass(slots=True)
class LidarReading:
    point_count: int
    cluster_count: int
    min_distance_m: float


@dataclass(slots=True)
class ProximityReading:
    front_m: float
    rear_m: float
    left_m: float
    right_m: float
    down_m: float
    available: bool


@dataclass(slots=True)
class TelemetryReading:
    position_ned: tuple[float, float, float]
    velocity_ned: tuple[float, float, float]
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    altitude_m: float
    gps_valid: bool


@dataclass(slots=True)
class SensorSnapshot:
    lidar: LidarReading
    proximity: ProximityReading
    telemetry: TelemetryReading
    timestamp_ns: int
    missing_features: list[str]


@dataclass(slots=True)
class WorldSnapshot:
    frame: FramePacket
    sensors: SensorSnapshot
    connection_ok: bool


@dataclass(slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int | None
    center: tuple[float, float]
    area: float


@dataclass(slots=True)
class TrackedTarget:
    detection: Detection
    track_id: int
    smooth_center: tuple[float, float]
    smoothed_area_ratio: float
    direction: TargetDirection
    frame_width: int
    frame_height: int
    frames_tracked: int
    frames_since_seen: int
    is_confirmed: bool
    velocity_estimate: tuple[float, float]
    predicted_center: tuple[float, float] | None


@dataclass(slots=True)
class SceneInsight:
    class_counts: dict[str, int]
    dominant_class: str | None
    scene_state: str
    activity_score: float
    summary_text: str


class MissionMode(StrEnum):
    PEDESTRIAN_WATCH = "PEDESTRIAN_WATCH"
    TRAFFIC_MONITOR = "TRAFFIC_MONITOR"
    SEARCH = "SEARCH"
    ORBIT = "ORBIT"
    MANUAL = "MANUAL"


class MissionState(StrEnum):
    IDLE = "IDLE"
    SCAN = "SCAN"
    TRACK = "TRACK"
    REACQUIRE = "REACQUIRE"
    MONITOR = "MONITOR"
    ORBIT = "ORBIT"
    REPORT = "REPORT"
    BLOCKED = "BLOCKED"
    # Universal safe-abort state, reachable from ANY state. Entered by the
    # mission watchdog when a hard limit is breached (timeout, geofence,
    # low battery, control deadlock) so the drone can recover unattended.
    EMERGENCY = "EMERGENCY"


@dataclass(slots=True)
class MissionContext:
    mode: MissionMode
    state: MissionState
    priority_class: str | None
    target_id: int | None
    progress: float
    elapsed_s: float
    operator_text: str


@dataclass(slots=True)
class IBVSOutput:
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    pixel_error_x: float
    pixel_error_y: float
    area_error: float


class SafetyState(StrEnum):
    PATH_CLEAR = "PATH_CLEAR"
    OBSTACLE_AHEAD = "OBSTACLE_AHEAD"
    OBSTACLE_CLUSTER = "OBSTACLE_CLUSTER"
    ALTITUDE_LOW = "ALTITUDE_LOW"
    LANDING_CAUTION = "LANDING_CAUTION"
    REPOSITION_SUGGEST = "REPOSITION_SUGGEST"
    SAFETY_OVERRIDE = "SAFETY_OVERRIDE"


@dataclass(slots=True)
class SafetyEvaluation:
    state: SafetyState
    blocked_directions: list[str]
    allow_forward: bool
    allow_descent: bool
    reason: str


@dataclass(slots=True)
class MissionReport:
    mission_id: str
    mode: str
    state_transitions: list[tuple[str, float]]
    target_ids_seen: list[int]
    completion_progress: float
    success: bool
    completion_reason: str
    duration_s: float
    unique_track_counts: dict[str, int] = field(default_factory=dict)
    unique_vehicle_count: int = 0
    unique_object_counts: dict[str, int] = field(default_factory=dict)
    active_object_count: int = 0
    registry_merge_count: int = 0
    events: list[str] = field(default_factory=list)


class MotionPrimitive(StrEnum):
    HOVER = "HOVER"
    FOLLOW = "FOLLOW"
    SCAN = "SCAN"
    REACQUIRE = "REACQUIRE"
    ORBIT = "ORBIT"
    LAND = "LAND"
    CUSTOM = "CUSTOM"


@dataclass(slots=True)
class MotionIntent:
    primitive: MotionPrimitive
    target_id: int | None = None
    custom_velocity: IBVSOutput | None = None
    reason: str = ""


@dataclass(slots=True)
class VelocityCmd:
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    duration_s: float
    source: str
