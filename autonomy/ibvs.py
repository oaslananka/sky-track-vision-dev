from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from autonomy.contracts import IBVSOutput, TelemetryReading, TrackedTarget
from config.settings import IBVSConfig


@dataclass(slots=True)
class PIDState:
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    prev_error: float = 0.0
    integral_clamp: float = 1.0
    derivative_filter_alpha: float = 0.2
    _filtered_derivative: float = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral = float(
            np.clip(self.integral + error * dt, -self.integral_clamp, self.integral_clamp)
        )
        # Anti-windup: halve integral when error changes sign (overshoot damping)
        if self.prev_error != 0.0 and (error * self.prev_error) < 0:
            self.integral *= 0.5

        raw_derivative = (error - self.prev_error) / max(dt, 1e-4)
        # First-order IIR low-pass filter on derivative to suppress noise
        alpha = self.derivative_filter_alpha
        self._filtered_derivative = alpha * raw_derivative + (1 - alpha) * self._filtered_derivative
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * self._filtered_derivative


class IBVSController:
    """Map image-space tracking error into velocity commands with a cascade PID design.

    Features:
    - Cascade PID (outer position → inner velocity) for yaw, forward, altitude
    - Lateral assist channel activated when yaw saturates (Phase 3)
    - Body-frame velocity correction via yaw rotation (Phase 6)
    - Anti-windup integral reset on error sign change
    - First-order IIR derivative filter
    """

    def __init__(self, cfg: IBVSConfig) -> None:
        self._cfg = cfg
        deriv_alpha = cfg.derivative_filter_alpha
        self._pid_yaw = PIDState(
            cfg.yaw_kp,
            cfg.yaw_ki,
            cfg.yaw_kd,
            integral_clamp=cfg.yaw_integral_clamp,
            derivative_filter_alpha=deriv_alpha,
        )
        self._pid_fwd = PIDState(
            cfg.fwd_kp,
            cfg.fwd_ki,
            cfg.fwd_kd,
            integral_clamp=cfg.fwd_integral_clamp,
            derivative_filter_alpha=deriv_alpha,
        )
        self._pid_alt = PIDState(
            cfg.alt_kp,
            cfg.alt_ki,
            cfg.alt_kd,
            integral_clamp=cfg.alt_integral_clamp,
            derivative_filter_alpha=deriv_alpha,
        )
        self._pid_vx_inner = PIDState(
            cfg.vx_inner_kp, 0.0, cfg.vx_inner_kd, derivative_filter_alpha=deriv_alpha
        )
        self._pid_vz_inner = PIDState(
            cfg.vz_inner_kp, 0.0, cfg.vz_inner_kd, derivative_filter_alpha=deriv_alpha
        )
        # Phase 3: lateral assist PID
        self._pid_lateral = PIDState(
            cfg.lateral_kp,
            cfg.lateral_ki,
            cfg.lateral_kd,
            derivative_filter_alpha=deriv_alpha,
        )
        self._last_ts = 0.0

    def compute(
        self,
        target: TrackedTarget,
        telemetry: TelemetryReading,
        frame_w: int,
        frame_h: int,
    ) -> IBVSOutput:
        now = time.monotonic()
        dt = min(now - self._last_ts, 0.1) if self._last_ts else 0.033
        self._last_ts = now

        cx = float(np.clip(target.smooth_center[0], 0.0, frame_w))
        cy = float(np.clip(target.smooth_center[1], 0.0, frame_h))
        ex = cx - frame_w / 2
        ey = cy - frame_h / 2
        desired_area = self._cfg.desired_area_ratio * frame_w * frame_h
        ea = target.detection.area - desired_area

        ex_norm = ex / (frame_w / 2)
        ey_norm = ey / (frame_h / 2)
        ea_norm = ea / max(desired_area, 1e-6)

        # ── Outer PID loops ──
        yaw_rate_sp = self._pid_yaw.update(ex_norm, dt)
        # Negate ea_norm: when target is far (small area, ea_norm < 0),
        # we want positive vx (forward in AirSim body frame).
        vx_sp = self._pid_fwd.update(-ea_norm, dt)
        vz_sp = self._pid_alt.update(ey_norm, dt)

        yaw_rate_sp = float(np.clip(yaw_rate_sp, -self._cfg.max_yaw_rate, self._cfg.max_yaw_rate))
        vx_sp = float(np.clip(vx_sp, -self._cfg.max_vx, self._cfg.max_vx))
        vz_sp = float(np.clip(vz_sp, -self._cfg.max_vz, self._cfg.max_vz))

        # ── Phase 3: Lateral assist when yaw is near saturation ──
        max_vy = getattr(self._cfg, "max_vy", 1.5)
        yaw_saturation_ratio = getattr(self._cfg, "lateral_yaw_saturation_ratio", 0.8)
        lateral_threshold = getattr(self._cfg, "lateral_activation_threshold", 0.3)
        if (
            abs(ex_norm) > lateral_threshold
            and abs(yaw_rate_sp) > yaw_saturation_ratio * self._cfg.max_yaw_rate
        ):
            vy_sp = self._pid_lateral.update(ex_norm, dt)
            vy_sp = float(np.clip(vy_sp, -max_vy, max_vy))
        else:
            vy_sp = 0.0
            # Decay lateral integrator when not active to avoid windup
            self._pid_lateral.integral *= 0.9

        # ── Phase 6: Body-frame velocity correction ──
        # NED velocity → body frame rotation so inner PID uses correct forward velocity
        yaw_rad = math.radians(telemetry.yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        body_vx = cos_y * telemetry.velocity_ned[0] + sin_y * telemetry.velocity_ned[1]

        vx_err = vx_sp - body_vx
        vz_err = vz_sp - telemetry.velocity_ned[2]

        # ── Inner velocity loops ──
        vx_cmd = float(
            np.clip(
                vx_sp + self._pid_vx_inner.update(vx_err, dt),
                -self._cfg.max_vx,
                self._cfg.max_vx,
            )
        )
        vz_cmd = float(
            np.clip(
                vz_sp + self._pid_vz_inner.update(vz_err, dt),
                -self._cfg.max_vz,
                self._cfg.max_vz,
            )
        )

        return IBVSOutput(
            vx=vx_cmd,
            vy=vy_sp,
            vz=vz_cmd,
            yaw_rate=yaw_rate_sp,
            pixel_error_x=ex,
            pixel_error_y=ey,
            area_error=ea,
        )

    def reset(self) -> None:
        for pid in (
            self._pid_yaw,
            self._pid_fwd,
            self._pid_alt,
            self._pid_vx_inner,
            self._pid_vz_inner,
            self._pid_lateral,
        ):
            pid.integral = 0.0
            pid.prev_error = 0.0
            pid._filtered_derivative = 0.0
        self._last_ts = 0.0
