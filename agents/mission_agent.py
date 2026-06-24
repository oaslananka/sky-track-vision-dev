from __future__ import annotations

from autonomy.contracts import (
    MissionContext,
    MissionMode,
    MissionState,
    SceneInsight,
    SensorSnapshot,
)


class MissionAgent:
    """Produce a lightweight mission context from scene and sensor snapshots."""

    def __init__(self, mission_mode: str = "PEDESTRIAN_WATCH") -> None:
        self._mode = MissionMode(mission_mode)

    def update(self, scene: SceneInsight, snapshot: SensorSnapshot) -> MissionContext:
        # State is managed exclusively by the MissionFSM; agent only provides context.
        priority_class = (
            "person" if self._mode == MissionMode.PEDESTRIAN_WATCH else scene.dominant_class
        )
        return MissionContext(
            mode=self._mode,
            state=MissionState.IDLE,
            priority_class=priority_class,
            target_id=None,
            progress=min(1.0, scene.activity_score),
            elapsed_s=snapshot.timestamp_ns / 1_000_000_000,
            operator_text=scene.summary_text,
        )
