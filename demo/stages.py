from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DemoStage:
    name: str
    banner: str
    target_visible: bool
    obstacle_ahead: bool = False


DEFAULT_STAGES = [
    DemoStage("idle", "Stage 1: empty scene", target_visible=False),
    DemoStage("scan", "Stage 2: target acquired", target_visible=True),
    DemoStage("track", "Stage 3: active follow", target_visible=True),
    DemoStage("obstacle", "Stage 4: obstacle ahead", target_visible=True, obstacle_ahead=True),
    DemoStage("report", "Stage 5: mission summary", target_visible=False),
]
