from __future__ import annotations

from autonomy.contracts import MotionIntent, SensorSnapshot, TrackedTarget, VelocityCmd
from autonomy.follow_controller import FollowController


class ControlAgent:
    """Thin wrapper over the follow controller."""

    def __init__(self, controller: FollowController) -> None:
        self._controller = controller

    def plan(
        self,
        intent: MotionIntent,
        target: TrackedTarget | None,
        snapshot: SensorSnapshot,
        frame_size: tuple[int, int],
    ) -> VelocityCmd:
        return self._controller.resolve(
            intent,
            target,
            snapshot,
            snapshot.telemetry,
            frame_size[0],
            frame_size[1],
        )
