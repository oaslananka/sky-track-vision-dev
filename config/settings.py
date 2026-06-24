from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class IBVSConfig:
    yaw_kp: float = 1.2
    yaw_ki: float = 0.01
    yaw_kd: float = 0.08
    yaw_integral_clamp: float = 0.5
    fwd_kp: float = 0.8
    fwd_ki: float = 0.005
    fwd_kd: float = 0.05
    fwd_integral_clamp: float = 0.4
    alt_kp: float = 0.6
    alt_ki: float = 0.002
    alt_kd: float = 0.04
    alt_integral_clamp: float = 0.3
    vx_inner_kp: float = 0.5
    vx_inner_kd: float = 0.02
    vz_inner_kp: float = 0.4
    vz_inner_kd: float = 0.02
    max_yaw_rate: float = 1.0
    max_vx: float = 3.0
    max_vy: float = 1.5
    max_vz: float = 1.5
    desired_area_ratio: float = 0.04
    derivative_filter_alpha: float = 0.2
    lateral_kp: float = 0.4
    lateral_ki: float = 0.0
    lateral_kd: float = 0.05
    lateral_activation_threshold: float = 0.3
    lateral_yaw_saturation_ratio: float = 0.8


@dataclass(slots=True)
class SafetyConfig:
    obstacle_front_threshold_m: float = 2.0
    obstacle_rear_threshold_m: float = 2.0
    obstacle_side_threshold_m: float = 1.2
    altitude_min_m: float = 1.5
    landing_caution_m: float = 0.5
    lidar_cluster_threshold: int = 200
    lidar_cluster_distance_factor: float = 0.8
    max_sensor_age_ms: float = 500.0
    max_decel: float = 2.0


@dataclass(slots=True)
class VisionConfig:
    model_path: str = "models/yolov8n.pt"
    preferred_device: str = "auto"
    confidence_threshold: float = 0.50
    inference_imgsz: int = 416
    max_detections: int = 16
    use_half_precision: bool = True
    track_inference_interval: int = 2
    target_classes: list[str] = field(
        default_factory=lambda: [
            "person",
            "car",
            "bicycle",
            "motorcycle",
            "bus",
            "truck",
        ]
    )
    tracker_smoothing_alpha: float = 0.6
    max_lost_frames: int = 20
    min_confirm_frames: int = 5
    sticky_lock_timeout_frames: int = 20
    sticky_lock_max_center_distance_px: float = 140.0
    track_full_rate_motion_threshold_px: float = 1.2
    track_full_rate_edge_margin_ratio: float = 0.15


@dataclass(slots=True)
class WatchdogConfig:
    """Mission-level hard limits that force a safe abort (EMERGENCY).

    These are *unattended-flight* guardrails: when any limit is breached the
    mission watchdog drives the FSM to EMERGENCY so the drone can recover
    without a human in the loop. They sit above the per-frame SafetyEvaluator,
    which handles immediate obstacle/altitude vetoes.
    """

    enabled: bool = True
    max_mission_duration_s: float = 600.0  # hard cap above the soft pilot timeout
    geofence_radius_m: float = 120.0  # max horizontal distance from home
    max_altitude_m: float = 60.0  # absolute ceiling
    battery_rtl_fraction: float = 0.20  # fraction at/below which we abort (None telemetry = skip)
    min_person_separation_m: float = 2.5  # advisory minimum standoff from a tracked person


@dataclass(slots=True)
class AirSimConfig:
    host: str = "127.0.0.1"
    port: int = 41451
    vehicle_name: str = "Drone"
    camera_name: str = "front_center"
    camera_compress: bool = False
    timeout_s: float = 5.0
    lidar_name: str = "LidarSensor"
    proximity_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "front": ["FrontProximity", "front", "front_center"],
            "rear": ["RearProximity", "rear"],
            "left": ["LeftProximity", "left"],
            "right": ["RightProximity", "right"],
            "down": ["DownProximity", "down"],
        }
    )
    sensor_debug_log_interval: int = 5


@dataclass(slots=True)
class PilotConfig:
    provider: str = "openai"
    model: str = "gpt-5-mini"  # Aligned with pilot.yaml; override with pilot.yaml for deployment
    log_level: str = "DEBUG"
    # LLM Generation Parameters.
    # Note: gpt-5-* chat.completions can reject non-default sampling overrides.
    llm_temperature: float = 1.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 2048  # Explicit cap on response length
    max_context_messages: int = 60  # Increased from 40 for longer missions (up to ~5 min @ 30fps)
    tool_retry_limit: int = 3
    mission_timeout_s: float = 360.0
    tick_duration_s: float = 0.1
    scan_yaw_rate: float = 0.06
    cruise_altitude_m: float = 4.0
    traffic_monitor_cruise_altitude_m: float = 2.6
    preflight_ascent_velocity: float = 2.5
    altitude_recovery_grace_s: float = 2.5
    traffic_monitor_road_follow_speed: float = 0.42
    traffic_monitor_road_seek_speed: float = 0.18
    traffic_monitor_road_search_speed: float = 0.08
    traffic_monitor_road_follow_confidence: float = 0.025
    traffic_monitor_road_seek_confidence: float = 0.008
    traffic_monitor_road_follow_yaw_gain: float = 0.24
    traffic_monitor_road_center_yaw_gain: float = 0.18
    traffic_monitor_road_search_yaw_rate: float = 0.12
    traffic_monitor_command_duration_s: float = 0.24
    traffic_monitor_sensor_read_interval: int = 1
    traffic_monitor_inference_interval: int = 5
    traffic_monitor_altitude_hold_deadband_m: float = 0.25
    traffic_monitor_altitude_hold_max_vz: float = 0.04
    scan_forward_speed: float = 0.8
    scan_lateral_speed: float = 0.35
    scan_vertical_bias: float = 0.0
    scan_straight_trigger_s: float = 1.5
    scan_straight_forward_speed: float = 0.95
    scan_straight_yaw_rate: float = 0.03
    scan_road_bias_enabled: bool = True
    scan_road_bias_gain: float = 0.35
    scan_road_bias_clip: float = 0.45
    scan_escape_trigger_s: float = 7.0
    scan_escape_duration_s: float = 2.2
    scan_escape_forward_speed: float = 1.1
    scan_escape_yaw_rate: float = 0.02
    orbit_lateral_speed: float = 0.65
    reacquire_timeout_s: float = 3.5
    unstick_velocity_threshold: float = 0.12
    unstick_timeout_s: float = 1.25
    hud_wait_key_ms: int = 1
    empty_frame_sleep_s: float = 0.01
    error_loop_sleep_s: float = 0.02
    debug_tick_log_interval: int = 5


@dataclass(slots=True)
class AppConfig:
    airsim: AirSimConfig = field(default_factory=AirSimConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    ibvs: IBVSConfig = field(default_factory=IBVSConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    mission_mode: str = "PEDESTRIAN_WATCH"
    overlay_mode: str = "SHOWCASE"
    auto_follow: bool = True
    demo_mode: bool = False


def _build_dataclass(cls: type[Any], values: dict[str, Any]) -> Any:
    instance = cls()
    kwargs: dict[str, Any] = {}
    for item in fields(instance):
        raw_value = values.get(item.name)
        if raw_value is None:
            continue
        current = getattr(instance, item.name)
        if hasattr(current, "__dataclass_fields__") and isinstance(raw_value, dict):
            kwargs[item.name] = _merge_dataclass(current, raw_value)
        else:
            kwargs[item.name] = raw_value
    return cls(**kwargs)


def _merge_dataclass(instance: Any, overrides: dict[str, Any]) -> Any:
    data: dict[str, Any] = asdict(instance)
    for field_name, value in overrides.items():
        current = data.get(field_name)
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
            data[field_name] = current
        else:
            data[field_name] = value
    return _build_dataclass(type(instance), data)


def load_app_config(path: str | Path | None = None) -> AppConfig:
    """Load the default config and merge optional YAML overrides from disk."""
    config = AppConfig()
    if path is None:
        return config
    path_obj = Path(path)
    if not path_obj.exists():
        return config
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _merge_dataclass(config, payload)
