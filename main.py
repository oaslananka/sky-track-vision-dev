from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from agents.control_agent import ControlAgent
from agents.mission_agent import MissionAgent
from agents.safety_agent import SafetyAgent
from agents.scene_agent import SceneAgent
from airsim_control.camera import DroneCameraStream
from airsim_control.client import AirSimConnectionManager
from airsim_control.movement import DroneMovementController
from airsim_control.sensors import SensorSuiteReader
from autonomy.contracts import (
    MissionState,
    SafetyState,
    VelocityCmd,
    WorldSnapshot,
)
from autonomy.follow_controller import FollowController
from autonomy.ibvs import IBVSController
from autonomy.mission import MissionFSM
from autonomy.reporting import EventReporter
from autonomy.safety import SafetyEvaluator
from config.settings import AppConfig, load_app_config
from demo.director import DemoDirector
from ui.overlay import OverlayRenderer
from vision.detector import MultiClassDetector
from vision.tracker import KalmanTracker
from vision.utils import FpsCounter


class SkyTrackVisionApp:
    """Classic runtime loop for AirSim-backed or demo-mode drone tracking."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._fps = FpsCounter()
        self._overlay = OverlayRenderer(cfg.overlay_mode)
        self._scene_agent = SceneAgent()
        self._mission_agent = MissionAgent(cfg.mission_mode)
        self._mission_fsm = MissionFSM(initial_state=MissionState.SCAN)
        self._reporter = EventReporter()
        self._ibvs = IBVSController(cfg.ibvs)
        self._follow_controller = FollowController(self._ibvs, cfg.pilot)
        self._control_agent = ControlAgent(self._follow_controller)
        self._safety_agent = SafetyAgent(SafetyEvaluator(cfg.safety))
        self._detector = MultiClassDetector(cfg.vision)
        self._tracker = KalmanTracker(cfg.vision)
        self._auto_follow = cfg.auto_follow
        self._overlay_visible = True
        self._demo = DemoDirector()

        self._connection: AirSimConnectionManager | None = None
        self._camera: DroneCameraStream | None = None
        self._movement: DroneMovementController | None = None
        self._sensors: SensorSuiteReader | None = None
        if not cfg.demo_mode:
            self._connection = AirSimConnectionManager(cfg.airsim)
            client = self._connection.connect()
            self._camera = DroneCameraStream(client, cfg.airsim)
            self._movement = DroneMovementController(client, cfg.airsim)
            self._sensors = SensorSuiteReader(client, cfg.airsim)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.disconnect()
        cv2.destroyAllWindows()

    def run_loop(self) -> None:
        cv2.namedWindow("SkyTrackVision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SkyTrackVision", 1280, 720)
        try:
            while True:
                tick_start = time.monotonic()
                world = self._get_world_snapshot()
                frame = world.frame.frame
                detections = self._detector.track(frame)
                target = self._tracker.update(
                    detections,
                    priority_class="person",
                    frame_size=(world.frame.width, world.frame.height),
                    frame=frame,
                )
                scene = self._scene_agent.analyze(detections)
                mission_context = self._mission_agent.update(scene, world.sensors)
                intent = self._mission_fsm.get_motion_intent(mission_context, target)
                cmd = self._control_agent.plan(
                    intent,
                    target,
                    world.sensors,
                    (world.frame.width, world.frame.height),
                )
                safety = self._safety_agent.evaluate(
                    world.sensors,
                    world.connection_ok or self._cfg.demo_mode,
                )
                safe_cmd = self._apply_safety(cmd, safety)
                if self._auto_follow and self._movement is not None:
                    self._movement.move_by_velocity(safe_cmd)
                self._reporter.update(mission_context, target, safety)
                if self._overlay_visible:
                    rendered = self._overlay.render(
                        frame,
                        detections,
                        target,
                        scene,
                        mission_context,
                        safety,
                    )
                else:
                    rendered = frame
                fps = self._fps.tick()
                cv2.putText(
                    rendered,
                    f"FPS: {fps:.1f}",
                    (rendered.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SkyTrackVision", rendered)
                if self._handle_keys(cv2.waitKey(1) & 0xFF):
                    break
                elapsed = time.monotonic() - tick_start
                if elapsed < 0.033:
                    time.sleep(0.033 - elapsed)
        finally:
            self.close()

    def _get_world_snapshot(self) -> Any:
        if (
            self._cfg.demo_mode
            or self._camera is None
            or self._sensors is None
            or self._connection is None
        ):
            return self._demo.next_frame()
        frame = self._camera.get_frame()
        sensors = self._sensors.read()
        return WorldSnapshot(
            frame=frame,
            sensors=sensors,
            connection_ok=self._connection.is_connected(),
        )

    def _apply_safety(self, cmd: VelocityCmd, safety: Any) -> VelocityCmd:
        if safety.state == SafetyState.SAFETY_OVERRIDE:
            return VelocityCmd(0.0, 0.0, 0.0, 0.0, cmd.duration_s, "safety_hover")

        # Gradient braking based on safety state
        modified_vx = cmd.vx
        modified_vz = cmd.vz

        if safety.state.value == "OBSTACLE_AHEAD" and "front" in safety.blocked_directions:
            # Gradual slowdown instead of hard stop
            if cmd.vx > 0:
                modified_vx = cmd.vx * 0.3  # 70% reduction

        if not safety.allow_forward and cmd.vx > 0:
            modified_vx = 0.0

        if not safety.allow_descent and cmd.vz > 0:
            modified_vz = 0.0

        return VelocityCmd(
            modified_vx,
            cmd.vy,
            modified_vz,
            cmd.yaw_rate,
            cmd.duration_s,
            cmd.source if modified_vx == cmd.vx and modified_vz == cmd.vz else "safety_modified",
        )

    def _handle_keys(self, key: int) -> bool:
        if key in (27, ord("q"), ord("Q")):
            return True
        if key in (ord("u"), ord("U")):
            next_mode = "DEBUG" if self._cfg.overlay_mode == "SHOWCASE" else "SHOWCASE"
            self._cfg.overlay_mode = next_mode
            self._overlay.set_mode(next_mode)
        elif key in (ord("o"), ord("O")):
            self._overlay_visible = not self._overlay_visible
        elif key in (ord("x"), ord("X")):
            self._auto_follow = not self._auto_follow
        elif key in (ord("n"), ord("N")) and self._cfg.demo_mode:
            self._demo.next_stage()
        elif key in (ord("b"), ord("B")) and self._cfg.demo_mode:
            self._demo.previous_stage()
        elif key in (ord("h"), ord("H")) and self._movement is not None:
            self._auto_follow = False
            self._movement.hover()
        elif key in (ord("g"), ord("G")) and self._movement is not None:
            self._auto_follow = False
            self._movement.land()
        elif key in (ord("w"), ord("W")) and self._movement is not None:
            self._manual_move(1.0, 0.0, 0.0, 0.0)
        elif key in (ord("s"), ord("S")) and self._movement is not None:
            self._manual_move(-1.0, 0.0, 0.0, 0.0)
        elif key in (ord("a"), ord("A")) and self._movement is not None:
            self._manual_move(0.0, -1.0, 0.0, 0.0)
        elif key in (ord("d"), ord("D")) and self._movement is not None:
            self._manual_move(0.0, 1.0, 0.0, 0.0)
        elif key in (ord("r"), ord("R")) and self._movement is not None:
            self._manual_move(0.0, 0.0, -0.5, 0.0)
        elif key in (ord("f"), ord("F")) and self._movement is not None:
            self._manual_move(0.0, 0.0, 0.5, 0.0)
        elif key in (ord("j"), ord("J")) and self._movement is not None:
            self._manual_move(0.0, 0.0, 0.0, -0.5)
        elif key in (ord("l"), ord("L")) and self._movement is not None:
            self._manual_move(0.0, 0.0, 0.0, 0.5)
        elif key in (ord("p"), ord("P")):
            image_path = Path("outputs/screenshots") / f"capture-{int(time.time())}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            current = self._demo.next_frame().frame.frame if self._cfg.demo_mode else None
            if current is not None:
                cv2.imwrite(str(image_path), current)
        return False

    def _manual_move(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        self._auto_follow = False
        if self._movement is None:
            return
        self._movement.move_by_velocity(VelocityCmd(vx, vy, vz, yaw_rate, 0.15, "manual"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SkyTrackVision runtime")
    parser.add_argument("--config", default="pilot.yaml")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    if args.demo:
        config.demo_mode = True
    app = SkyTrackVisionApp(config)
    app.run_loop()


if __name__ == "__main__":
    main()
