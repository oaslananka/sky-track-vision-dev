from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from airsim_control.client import get_client_rpc_lock
from autonomy.contracts import VelocityCmd
from config.runtime_logging import log_event
from config.settings import AirSimConfig

try:
    import airsim
except Exception:  # pragma: no cover - optional runtime dependency
    airsim = None

logger = logging.getLogger("skytrackvision.airsim")


class DroneMovementController:
    """Single gateway for all direct AirSim movement commands."""

    def __init__(self, client: Any, cfg: AirSimConfig) -> None:
        self._client = client
        self._cfg = cfg
        self._rpc_lock = get_client_rpc_lock(client)
        self._command_lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._serial_command_active = False

    @property
    def serial_command_active(self) -> bool:
        return self._serial_command_active

    def apply_velocity_gains(self, kp: float, ki: float, kd: float) -> None:
        if airsim is None:
            return
        gains = airsim.VelocityControllerGains(
            x_gains=airsim.PIDGains(kp, ki, kd),
            y_gains=airsim.PIDGains(kp, ki, kd),
            z_gains=airsim.PIDGains(kp * 1.2, ki, kd),
        )
        with self._rpc_lock:
            self._client.setVelocityControllerGains(gains, vehicle_name=self._cfg.vehicle_name)

    def takeoff(self) -> None:
        self._run_serial(self._client.takeoffAsync)

    def land(self) -> None:
        self._run_serial(self._client.landAsync)

    def move_to_altitude(self, altitude_m: float, velocity: float = 1.0) -> None:
        self._run_serial(
            self._client.moveToZAsync,
            -abs(altitude_m),
            velocity,
        )

    def move_to_position(self, x: float, y: float, z: float, velocity: float = 2.0) -> None:
        self._run_serial(
            self._client.moveToPositionAsync,
            x,
            y,
            z,
            velocity,
        )

    def hover(self) -> None:
        log_event(logger, logging.DEBUG, "airsim.command", "Hover requested", method="hoverAsync")
        with self._rpc_lock:
            future = self._client.hoverAsync(vehicle_name=self._cfg.vehicle_name)
            if hasattr(future, "join"):
                future.join()

    def move_by_velocity(self, cmd: VelocityCmd) -> None:
        if self._serial_command_active or airsim is None:
            log_event(
                logger,
                logging.DEBUG,
                "airsim.command",
                "Velocity command skipped",
                method="moveByVelocityAsync",
                serial_busy=self._serial_command_active,
                cmd_source=cmd.source,
            )
            return
        with self._motion_lock:
            started = time.monotonic()
            with self._rpc_lock:
                future = self._client.moveByVelocityAsync(
                    cmd.vx,
                    cmd.vy,
                    cmd.vz,
                    cmd.duration_s,
                    yaw_mode=airsim.YawMode(
                        is_rate=True,
                        yaw_or_rate=math.degrees(cmd.yaw_rate),
                    ),
                    vehicle_name=self._cfg.vehicle_name,
                )
                if hasattr(future, "join"):
                    future.join()
        log_event(
            logger,
            logging.DEBUG,
            "airsim.command",
            "Velocity command submitted",
            method="moveByVelocityAsync",
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            cmd_source=cmd.source,
            args_summary={
                "vx": cmd.vx,
                "vy": cmd.vy,
                "vz": cmd.vz,
                "yaw_rate": cmd.yaw_rate,
                "duration_s": cmd.duration_s,
            },
            ok=True,
        )

    def _run_serial(self, action: Any, *args: Any, **kwargs: Any) -> None:
        if airsim is None:
            return
        with self._command_lock:
            self._serial_command_active = True
            started = time.monotonic()
            try:
                with self._rpc_lock:
                    future = action(*args, **kwargs, vehicle_name=self._cfg.vehicle_name)
                    future.join()
                log_event(
                    logger,
                    logging.INFO,
                    "airsim.command",
                    "Serial AirSim command completed",
                    method=getattr(action, "__name__", str(action)),
                    args_summary=list(args),
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                    ok=True,
                )
            finally:
                self._serial_command_active = False
