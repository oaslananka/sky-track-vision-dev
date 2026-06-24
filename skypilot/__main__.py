from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import threading
from collections import Counter
from dataclasses import replace

from airsim_control.camera import DroneCameraStream
from airsim_control.client import AirSimConnectionManager
from airsim_control.movement import DroneMovementController
from airsim_control.sensors import SensorSuiteReader
from autonomy.contracts import MissionState
from autonomy.ibvs import IBVSController
from autonomy.mission import MissionFSM
from autonomy.reporting import EventReporter
from autonomy.safety import SafetyEvaluator
from autonomy.targeting import is_priority_compatible
from config.runtime_logging import configure_logging, log_event
from config.settings import load_app_config
from skypilot.airsim_bridge import AirSimBridge
from skypilot.llm_client import build_llm_client
from skypilot.pilot import LLMPilot
from skypilot.pilot_display import PilotDisplay
from skypilot.runtime_helpers import (
    resolve_cruise_altitude_m,
    resolve_mission_mode,
    resolve_priority_class,
)
from skypilot.tools import ToolDispatcher


async def _execute_mission(
    llm: LLMPilot,
    task: str,
    timeout_s: float,
):
    return await asyncio.wait_for(llm.run_mission(task), timeout=timeout_s)


def run() -> None:
    parser = argparse.ArgumentParser(description="SkyPilot mission runner")
    parser.add_argument("task", nargs="?", default="Scan for pedestrians and report")
    parser.add_argument("--config", default="pilot.yaml")
    parser.add_argument("--no-hud", action="store_true", help="Disable pilot HUD window")
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    logger = configure_logging(cfg.pilot)
    logger.info("Preparing SkyPilot mission runtime")
    llm_client = build_llm_client(cfg.pilot)
    connection = AirSimConnectionManager(cfg.airsim)
    reporter = EventReporter()
    mission_id = reporter.mission_id
    fsm = MissionFSM(initial_state=MissionState.SCAN)
    display: PilotDisplay | None = None
    try:
        log_event(
            logger,
            logging.INFO,
            "mission.lifecycle",
            "Preparing SkyPilot mission runtime",
            mission_id=mission_id,
            phase="startup",
            fsm_state=fsm.state.value,
        )
        log_event(
            logger,
            logging.DEBUG,
            "airsim.connect",
            "Connecting to AirSim",
            mission_id=mission_id,
            phase="startup",
            fsm_state=fsm.state.value,
            host=cfg.airsim.host,
            port=cfg.airsim.port,
            vehicle_name=cfg.airsim.vehicle_name,
        )
        client = connection.connect()
        log_event(
            logger,
            logging.INFO,
            "airsim.connect",
            "AirSim connection established",
            mission_id=mission_id,
            phase="startup",
            fsm_state=fsm.state.value,
        )
        sensors = SensorSuiteReader(client, cfg.airsim)
        movement = DroneMovementController(client, cfg.airsim)
        safety = SafetyEvaluator(cfg.safety)
        bridge = AirSimBridge(movement, sensors, safety, connected=True)
        ibvs = IBVSController(cfg.ibvs)
        priority_class = resolve_priority_class(args.task, cfg.vision.target_classes)
        mission_mode = resolve_mission_mode(args.task, priority_class)
        runtime_pilot_cfg = replace(
            cfg.pilot,
            cruise_altitude_m=resolve_cruise_altitude_m(
                mission_mode,
                cfg.pilot.cruise_altitude_m,
                traffic_monitor_altitude_m=cfg.pilot.traffic_monitor_cruise_altitude_m,
            ),
        )
        log_event(
            logger,
            logging.INFO,
            "mission.preflight",
            "Executing preflight climb",
            mission_id=mission_id,
            phase="preflight",
            fsm_state=fsm.state.value,
            altitude_m=runtime_pilot_cfg.cruise_altitude_m,
            mission_mode=mission_mode.value,
        )
        bridge.takeoff()
        bridge.move_to_altitude(
            runtime_pilot_cfg.cruise_altitude_m,
            velocity=runtime_pilot_cfg.preflight_ascent_velocity,
        )

        # ── Start Pilot HUD display (separate client to avoid IOLoop conflict) ──
        if not args.no_hud:
            try:
                import airsim as _airsim

                hud_client = _airsim.MultirotorClient(ip=cfg.airsim.host, port=cfg.airsim.port)
                hud_client.confirmConnection()
                camera = DroneCameraStream(hud_client, cfg.airsim)
                display = PilotDisplay(
                    camera,
                    cfg.vision,
                    runtime_pilot_cfg,
                    ibvs,
                    fsm,
                    bridge,
                    sensors,
                    reporter=reporter,
                    mission_id=mission_id,
                    priority_class=priority_class,
                    mission_mode=mission_mode,
                )
                display.start()
                log_event(
                    logger,
                    logging.INFO,
                    "hud.lifecycle",
                    "Pilot HUD display enabled",
                    mission_id=mission_id,
                    phase="runtime",
                    fsm_state=fsm.state.value,
                    priority_class=priority_class,
                    mission_mode=mission_mode.value,
                    cruise_altitude_m=runtime_pilot_cfg.cruise_altitude_m,
                )
            except Exception:
                log_event(
                    logger,
                    logging.WARNING,
                    "hud.lifecycle",
                    "Could not start HUD display, continuing without it",
                    mission_id=mission_id,
                    phase="runtime",
                    fsm_state=fsm.state.value,
                )
                display = None

        def _scene_provider() -> dict[str, object]:
            registry = reporter.registry_summary()
            if display:
                target = display.target
                detections = display.detections
                class_counts = dict(
                    sorted(Counter(detection.class_name for detection in detections).items())
                )
                motion_speed_px = 0.0
                motion_heading_deg = 0.0
                motion_trend = "STILL"
                if target is not None:
                    vx_px, vy_px = target.velocity_estimate
                    motion_speed_px = float((vx_px**2 + vy_px**2) ** 0.5)
                    if motion_speed_px >= 0.08:
                        motion_heading_deg = math.degrees(math.atan2(vy_px, vx_px))
                        horiz = "RIGHT" if vx_px > 0 else "LEFT"
                        vert = "DOWN" if vy_px > 0 else "UP"
                        motion_trend = f"{horiz}-{vert}"
                return {
                    "scene_state": "active"
                    if target is not None and target.is_confirmed
                    else "scanning",
                    "detections_count": len(detections),
                    "class_counts": class_counts,
                    "object_registry": registry,
                    "mission_mode": mission_mode.value,
                    "priority_class": priority_class,
                    "target": {
                        "track_id": target.track_id,
                        "class_name": target.detection.class_name,
                        "center": list(target.smooth_center),
                        "is_confirmed": target.is_confirmed,
                        "priority_match": is_priority_compatible(
                            priority_class,
                            target.detection.class_name,
                        ),
                        "smoothed_area_ratio": target.smoothed_area_ratio,
                        "frames_tracked": target.frames_tracked,
                        "frames_since_seen": target.frames_since_seen,
                        "direction": target.direction,
                        "velocity_estimate": list(target.velocity_estimate),
                        "predicted_center": list(target.predicted_center)
                        if target.predicted_center
                        else None,
                        "motion_speed_px": motion_speed_px,
                        "motion_heading_deg": motion_heading_deg,
                        "motion_trend": motion_trend,
                    }
                    if target is not None
                    else {},
                }
            return {
                "scene_state": "unknown",
                "detections_count": 0,
                "class_counts": {},
                "object_registry": registry,
                "mission_mode": mission_mode.value,
                "priority_class": priority_class,
                "target": {},
            }

        tools = ToolDispatcher(fsm, _scene_provider, bridge, reporter)
        llm = LLMPilot(llm_client, tools, reporter, cfg.pilot)
        log_event(
            logger,
            logging.INFO,
            "mission.lifecycle",
            "Mission request accepted",
            mission_id=mission_id,
            phase="mission",
            fsm_state=fsm.state.value,
            task=args.task,
        )
        report = None
        mission_error: BaseException | None = None

        def _run_mission_worker() -> None:
            nonlocal report, mission_error
            try:
                report = asyncio.run(_execute_mission(llm, args.task, cfg.pilot.mission_timeout_s))
            except BaseException as exc:
                mission_error = exc
            finally:
                if display:
                    display.stop()

        if display and os.name == "nt":
            mission_thread = threading.Thread(
                target=_run_mission_worker,
                daemon=True,
                name="SkyPilotMission",
            )
            mission_thread.start()
            display.run_foreground()
            mission_thread.join()
        else:
            if display:
                display.start()
            _run_mission_worker()

        if mission_error is not None:
            raise mission_error
        if report is None:
            raise RuntimeError("Mission worker exited without a report")

        log_event(
            logger,
            logging.INFO,
            "mission.lifecycle",
            "Mission finished",
            mission_id=report.mission_id,
            phase="mission",
            fsm_state=fsm.state.value,
            success=report.success,
            completion_reason=report.completion_reason,
            duration_s=round(report.duration_s, 3),
        )
        print(
            f"\nMission {report.mission_id} success={report.success} "
            f"reason={report.completion_reason}"
        )
    except TimeoutError:
        log_event(
            logger,
            logging.ERROR,
            "mission.lifecycle",
            "Mission timed out",
            mission_id=mission_id,
            phase="mission",
            fsm_state=fsm.state.value,
            timeout_s=cfg.pilot.mission_timeout_s,
        )
        report = reporter.finalize(success=False, reason="timeout")
        print(f"\nMission {report.mission_id} TIMED OUT after {cfg.pilot.mission_timeout_s}s")
    except ConnectionError as exc:
        log_event(
            logger,
            logging.ERROR,
            "airsim.connect",
            "AirSim connection failed",
            mission_id=mission_id,
            phase="startup",
            fsm_state=fsm.state.value,
            reason=str(exc),
        )
        print(f"\nAirSim connection failed: {exc}")
    finally:
        if display:
            display.stop()
        log_event(
            logger,
            logging.DEBUG,
            "airsim.disconnect",
            "Disconnecting from AirSim",
            mission_id=mission_id,
            phase="shutdown",
            fsm_state=fsm.state.value,
        )
        connection.disconnect()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
