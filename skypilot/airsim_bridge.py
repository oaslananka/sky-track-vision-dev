from __future__ import annotations

import logging
import time
from dataclasses import replace

from airsim_control.movement import DroneMovementController
from airsim_control.sensors import SensorSuiteReader
from autonomy.contracts import SafetyState, SensorSnapshot, TelemetryReading, VelocityCmd
from autonomy.safety import SafetyEvaluator
from config.runtime_logging import log_event

logger = logging.getLogger("skytrackvision.skypilot.bridge")


class AirSimBridge:
    """Safety-gated movement bridge for the SkyPilot runtime."""

    def __init__(
        self,
        movement: DroneMovementController,
        sensor_reader: SensorSuiteReader,
        safety: SafetyEvaluator,
        *,
        connected: bool = True,
    ) -> None:
        self._movement = movement
        self._sensor_reader = sensor_reader
        self._safety = safety
        self._connected = connected
        self._last_veto: str | None = None
        self._tracked_class: str | None = None
        self._home_position: tuple[float, float, float] | None = None
        self._record_home()

    def set_tracked_class(self, class_name: str | None) -> None:
        """Tell the safety gate which class is currently being followed.

        Used to enforce a minimum standoff (e.g. from a person) on the next
        movement evaluation.
        """
        self._tracked_class = class_name

    def _record_home(self) -> None:
        """Save the current position as home for return-to-home."""
        try:
            snapshot = self._sensor_reader.read()
            pos = snapshot.telemetry.position_ned
            self._home_position = pos
            log_event(
                logger,
                logging.INFO,
                "mission.home",
                "Home position recorded",
                home_position=list(pos),
                altitude_m=round(snapshot.telemetry.altitude_m, 3),
            )
        except Exception:
            self._home_position = (0.0, 0.0, -3.0)
            log_event(
                logger,
                logging.WARNING,
                "mission.home",
                "Could not read home position, using default origin",
                home_position=list(self._home_position),
            )

    def move(self, cmd: VelocityCmd, snapshot: SensorSnapshot | None = None) -> bool:
        if snapshot is None:
            try:
                snapshot = self._sensor_reader.read()
            except Exception as exc:
                snapshot = self._sensor_reader.last_snapshot
                if snapshot is None:
                    raise
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.snapshot",
                    "Falling back to cached sensor snapshot",
                    reason=str(exc),
                    snapshot_age_ms=round(
                        (time.time_ns() - snapshot.timestamp_ns) / 1_000_000,
                        3,
                    ),
                )
        safety = self._safety.evaluate(snapshot, self._connected, tracked_class=self._tracked_class)
        if safety.state == SafetyState.SAFETY_OVERRIDE:
            self._movement.hover()
            self._last_veto = safety.reason
            log_event(
                logger,
                logging.WARNING,
                "bridge.command",
                "Movement vetoed by safety override",
                requested_source=cmd.source,
                requested_cmd={
                    "vx": cmd.vx,
                    "vy": cmd.vy,
                    "vz": cmd.vz,
                    "yaw_rate": cmd.yaw_rate,
                    "duration_s": cmd.duration_s,
                },
                applied=False,
                veto_reason=safety.reason,
                safety_state=safety.state.value,
                serial_busy=self._movement.serial_command_active,
            )
            return False
        if cmd.vx > 0 and not safety.allow_forward:
            self._movement.hover()
            self._last_veto = safety.reason
            log_event(
                logger,
                logging.WARNING,
                "bridge.command",
                "Forward movement vetoed by safety",
                requested_source=cmd.source,
                requested_cmd={
                    "vx": cmd.vx,
                    "vy": cmd.vy,
                    "vz": cmd.vz,
                    "yaw_rate": cmd.yaw_rate,
                    "duration_s": cmd.duration_s,
                },
                applied=False,
                veto_reason=safety.reason,
                safety_state=safety.state.value,
                serial_busy=self._movement.serial_command_active,
            )
            return False
        if cmd.vz > 0 and not safety.allow_descent:
            safe_cmd = replace(cmd, vz=0.0)
            self._movement.move_by_velocity(safe_cmd)
            self._last_veto = safety.reason
            log_event(
                logger,
                logging.WARNING,
                "bridge.command",
                "Descent clamped by safety",
                requested_source=cmd.source,
                requested_cmd={
                    "vx": cmd.vx,
                    "vy": cmd.vy,
                    "vz": cmd.vz,
                    "yaw_rate": cmd.yaw_rate,
                    "duration_s": cmd.duration_s,
                },
                mutated_cmd={
                    "vx": safe_cmd.vx,
                    "vy": safe_cmd.vy,
                    "vz": safe_cmd.vz,
                    "yaw_rate": safe_cmd.yaw_rate,
                    "duration_s": safe_cmd.duration_s,
                },
                applied=True,
                veto_reason=safety.reason,
                safety_state=safety.state.value,
                serial_busy=self._movement.serial_command_active,
            )
            return True
        self._movement.move_by_velocity(cmd)
        self._last_veto = None
        log_event(
            logger,
            logging.DEBUG,
            "bridge.command",
            "Movement command forwarded to AirSim",
            requested_source=cmd.source,
            requested_cmd={
                "vx": cmd.vx,
                "vy": cmd.vy,
                "vz": cmd.vz,
                "yaw_rate": cmd.yaw_rate,
                "duration_s": cmd.duration_s,
            },
            applied=True,
            veto_reason=None,
            safety_state=safety.state.value,
            serial_busy=self._movement.serial_command_active,
        )
        return True

    def takeoff(self) -> None:
        log_event(logger, logging.INFO, "bridge.command", "Bridge takeoff requested")
        self._movement.takeoff()

    def is_airborne(self, *, min_altitude_m: float = 0.8) -> bool:
        snapshot = self._sensor_reader.last_snapshot
        if snapshot is None:
            try:
                snapshot = self._sensor_reader.read()
            except Exception:
                return False
        return float(snapshot.telemetry.altitude_m) >= float(min_altitude_m)

    def land(self) -> None:
        log_event(logger, logging.INFO, "bridge.command", "Bridge land requested")
        self._movement.land()

    def move_to_altitude(self, altitude_m: float, velocity: float = 1.0) -> None:
        log_event(
            logger,
            logging.INFO,
            "bridge.command",
            "Bridge move_to_altitude requested",
            altitude_m=altitude_m,
            velocity=velocity,
        )
        self._movement.move_to_altitude(altitude_m, velocity=velocity)

    def return_to_home(self) -> None:
        if self._home_position is None:
            log_event(
                logger,
                logging.WARNING,
                "bridge.command",
                "No home position recorded, hovering instead",
            )
            self._movement.hover()
            return
        x, y, z = self._home_position
        log_event(
            logger,
            logging.INFO,
            "bridge.command",
            "Returning to home position",
            home_position=[x, y, z],
        )
        self._movement.move_to_position(x, y, z, velocity=2.0)

    def request_hover(self) -> None:
        log_event(logger, logging.DEBUG, "bridge.command", "Bridge hover requested")
        self._movement.hover()

    def read_sensor_snapshot(self, *, refresh: bool = True) -> SensorSnapshot:
        if refresh:
            try:
                return self._sensor_reader.read()
            except Exception as exc:
                snapshot = self._sensor_reader.last_snapshot
                if snapshot is None:
                    raise
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.snapshot",
                    "Falling back to cached snapshot for public bridge read",
                    reason=str(exc),
                    snapshot_age_ms=round((time.time_ns() - snapshot.timestamp_ns) / 1_000_000, 3),
                )
                return snapshot
        snapshot = self._sensor_reader.last_snapshot
        if snapshot is not None:
            return snapshot
        return self._sensor_reader.read()

    def get_telemetry(self, *, refresh: bool = True) -> TelemetryReading:
        return self.read_sensor_snapshot(refresh=refresh).telemetry

    @property
    def last_veto(self) -> str | None:
        return self._last_veto

    @property
    def home_position(self) -> tuple[float, float, float] | None:
        return self._home_position
