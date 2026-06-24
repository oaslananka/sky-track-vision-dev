from __future__ import annotations

import math
import time

from autonomy.contracts import (
    MotionIntent,
    MotionPrimitive,
    SensorSnapshot,
    TelemetryReading,
    TrackedTarget,
    VelocityCmd,
)
from autonomy.ibvs import IBVSController
from config.settings import PilotConfig


class FollowController:
    """Translate mission intent into a deterministic short-horizon velocity command."""

    def __init__(self, ibvs: IBVSController, cfg: PilotConfig) -> None:
        self._ibvs = ibvs
        self._cfg = cfg
        self._scan_started_at = time.monotonic()

    def resolve(
        self,
        intent: MotionIntent,
        target: TrackedTarget | None,
        snapshot: SensorSnapshot,
        telemetry: TelemetryReading,
        frame_w: int,
        frame_h: int,
    ) -> VelocityCmd:
        # Use proximity data to attenuate forward speed near obstacles
        proximity_scale = 1.0
        if snapshot.proximity.available and snapshot.proximity.front_m < 5.0:
            # Linear ramp-down: full speed at 5m, zero at 1m
            proximity_scale = max(0.0, min(1.0, (snapshot.proximity.front_m - 1.0) / 4.0))

        match intent.primitive:
            case MotionPrimitive.FOLLOW if target and target.is_confirmed:
                output = self._ibvs.compute(target, telemetry, frame_w, frame_h)
                return VelocityCmd(
                    vx=output.vx * proximity_scale,
                    vy=output.vy,
                    vz=output.vz,
                    yaw_rate=output.yaw_rate,
                    duration_s=self._cfg.tick_duration_s,
                    source="ibvs",
                )
            case MotionPrimitive.SCAN:
                return self._scan_command(telemetry)
            case MotionPrimitive.REACQUIRE:
                return self._reacquire_command(telemetry)
            case MotionPrimitive.ORBIT if target and target.is_confirmed:
                output = self._ibvs.compute(target, telemetry, frame_w, frame_h)
                return VelocityCmd(
                    vx=0.0,
                    vy=self._cfg.orbit_lateral_speed,
                    vz=output.vz,
                    yaw_rate=output.yaw_rate,
                    duration_s=self._cfg.tick_duration_s,
                    source="orbit",
                )
            case MotionPrimitive.ORBIT:
                return self._orbit_search_command()
            case _:
                return VelocityCmd(0.0, 0.0, 0.0, 0.0, self._cfg.tick_duration_s, "hover")

    def _scan_command(self, telemetry: TelemetryReading) -> VelocityCmd:
        phase = time.monotonic() - self._scan_started_at
        forward = self._cfg.scan_forward_speed + 0.12 * math.sin(phase * 0.55)
        lateral = self._cfg.scan_lateral_speed * math.sin(phase * 0.9)
        vertical = self._scan_vertical_velocity(telemetry)
        yaw_rate = self._cfg.scan_yaw_rate + 0.08 * math.sin(phase * 0.7)
        return VelocityCmd(
            vx=forward,
            vy=lateral,
            vz=vertical,
            yaw_rate=yaw_rate,
            duration_s=self._cfg.tick_duration_s,
            source="scan",
        )

    def _reacquire_command(self, telemetry: TelemetryReading) -> VelocityCmd:
        """Focused re-search: slow forward + narrow oscillating yaw."""
        phase = time.monotonic() - self._scan_started_at
        forward = 0.3 + 0.05 * math.sin(phase * 0.8)
        yaw_rate = 0.5 * math.sin(phase * 1.4)  # Faster, narrower oscillation than SCAN
        return VelocityCmd(
            vx=forward,
            vy=0.0,
            vz=self._scan_vertical_velocity(telemetry),
            yaw_rate=yaw_rate,
            duration_s=self._cfg.tick_duration_s,
            source="reacquire",
        )

    def _orbit_search_command(self) -> VelocityCmd:
        phase = time.monotonic() - self._scan_started_at
        return VelocityCmd(
            vx=0.25 + 0.08 * math.cos(phase * 0.8),
            vy=self._cfg.orbit_lateral_speed,
            vz=0.02 * math.sin(phase * 0.5),
            yaw_rate=self._cfg.scan_yaw_rate,
            duration_s=self._cfg.tick_duration_s,
            source="orbit_search",
        )

    def _scan_vertical_velocity(self, telemetry: TelemetryReading) -> float:
        altitude_error = self._cfg.cruise_altitude_m - telemetry.altitude_m
        if abs(altitude_error) < 0.2:
            return 0.0
        correction = -0.18 * altitude_error
        return max(-0.12, min(0.12, correction))

    def reset(self) -> None:
        self._ibvs.reset()
        self._scan_started_at = time.monotonic()
