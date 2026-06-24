from __future__ import annotations

from autonomy.contracts import (
    MotionIntent,
    MotionPrimitive,
    SensorSnapshot,
    TrackedTarget,
    VelocityCmd,
)
from autonomy.ibvs import IBVSController
from config.settings import PilotConfig


class HybridController:
    """Fill the gap between slower LLM planning ticks with deterministic control."""

    def __init__(self, ibvs: IBVSController, cfg: PilotConfig) -> None:
        self._ibvs = ibvs
        self._cfg = cfg

    def tick(
        self,
        intent: MotionIntent,
        target: TrackedTarget | None,
        snapshot: SensorSnapshot,
        frame_w: int,
        frame_h: int,
    ) -> VelocityCmd:
        match intent.primitive:
            case MotionPrimitive.FOLLOW if target and target.is_confirmed:
                output = self._ibvs.compute(target, snapshot.telemetry, frame_w, frame_h)
                return VelocityCmd(
                    vx=output.vx,
                    vy=output.vy,
                    vz=output.vz,
                    yaw_rate=output.yaw_rate,
                    duration_s=self._cfg.tick_duration_s,
                    source="ibvs",
                )
            case MotionPrimitive.SCAN:
                return VelocityCmd(
                    vx=0.0,
                    vy=0.0,
                    vz=-0.1,
                    yaw_rate=self._cfg.scan_yaw_rate,
                    duration_s=self._cfg.tick_duration_s,
                    source="scan",
                )
            case _:
                return VelocityCmd(0.0, 0.0, 0.0, 0.0, self._cfg.tick_duration_s, "hover")
