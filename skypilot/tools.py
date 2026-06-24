from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from autonomy.contracts import MissionMode, MissionState
from autonomy.mission import InvalidTransitionError, MissionFSM
from autonomy.watchdog import MissionWatchdog, WatchdogVerdict
from config.runtime_logging import log_event
from skypilot.models import PilotToolResult

ToolHandler = Callable[[dict[str, Any]], Awaitable[PilotToolResult]]
logger = logging.getLogger("skytrackvision.skypilot.tools")


class ToolDispatcher:
    """Register and dispatch task-level tools exposed to the LLM."""

    def __init__(
        self,
        fsm: MissionFSM,
        scene_provider: Callable[[], dict[str, Any]],
        bridge: Any,
        reporter: Any,
        watchdog: MissionWatchdog | None = None,
    ) -> None:
        self._fsm = fsm
        self._scene_provider = scene_provider
        self._bridge = bridge
        self._reporter = reporter
        self._watchdog = watchdog
        self._lock_loss_grace_frames = 3
        self._tools: dict[str, ToolHandler] = {
            "get_scene_state": self._get_scene_state,
            "get_target_info": self._get_target_info,
            "get_drone_status": self._get_drone_status,
            "set_mission_state": self._set_mission_state,
            "request_hover": self._request_hover,
            "request_scan": self._request_scan,
            "request_follow": self._request_follow,
            "get_mission_report": self._get_mission_report,
            "request_takeoff": self._request_takeoff,
            "request_land": self._request_land,
            "request_move_to_altitude": self._request_move_to_altitude,
            "wait_seconds": self._wait_seconds,
            "request_return_home": self._request_return_home,
        }

    def _target_payload(self) -> dict[str, Any]:
        scene = self._scene_provider()
        return dict(scene.get("target", {}))

    def _priority_class(self) -> str | None:
        scene = self._scene_provider()
        priority = scene.get("priority_class")
        return str(priority) if isinstance(priority, str) and priority else None

    def _mission_mode(self) -> MissionMode:
        scene = self._scene_provider()
        raw_mode = scene.get("mission_mode")
        if isinstance(raw_mode, MissionMode):
            return raw_mode
        if isinstance(raw_mode, str):
            try:
                return MissionMode(raw_mode)
            except ValueError:
                return MissionMode.SEARCH
        return MissionMode.SEARCH

    def _priority_target_locked(self) -> bool:
        if self._mission_mode() == MissionMode.TRAFFIC_MONITOR:
            return False
        target = self._target_payload()
        priority_class = self._priority_class()
        is_confirmed = bool(target.get("is_confirmed"))
        priority_match = bool(target.get("priority_match"))
        if priority_class and not priority_match:
            return False
        if is_confirmed:
            return True
        try:
            frames_since_seen = int(target.get("frames_since_seen", 9999))
        except (TypeError, ValueError):
            return False
        return frames_since_seen <= self._lock_loss_grace_frames

    def _check_watchdog(self) -> WatchdogVerdict | None:
        """Evaluate the mission-envelope watchdog from current telemetry.

        Returns ``None`` when no watchdog is configured or telemetry cannot be
        read, so callers can treat "unknown" as "do not interfere".
        """
        if self._watchdog is None:
            return None
        try:
            snapshot = self._bridge.read_sensor_snapshot(refresh=False)
        except Exception:
            return None
        home = getattr(self._bridge, "home_position", None)
        return self._watchdog.evaluate(
            elapsed_s=self._reporter.elapsed_s,
            position_ned=snapshot.telemetry.position_ned,
            home_ned=home,
            battery_fraction=None,
        )

    def _engage_emergency(self, verdict: WatchdogVerdict) -> PilotToolResult:
        """Drive the FSM to EMERGENCY and stop the drone after a watchdog trip."""
        self._fsm.emergency(reason=f"watchdog:{verdict.trigger}")
        with contextlib.suppress(Exception):  # best-effort safe stop
            self._bridge.request_hover()
        log_event(
            logger,
            logging.WARNING,
            "mission.watchdog",
            "Mission watchdog tripped — engaging EMERGENCY",
            mission_id=self._reporter.mission_id,
            trigger=verdict.trigger,
            reason=verdict.reason,
            fsm_state=self._fsm.state.value,
        )
        return {
            "ok": False,
            "message": (
                f"Mission aborted by safety watchdog: {verdict.reason}. The drone is "
                "holding in EMERGENCY. End the mission now: set_mission_state(state='IDLE') "
                "then request_land()."
            ),
            "mission_state": self._fsm.state.value,
            "watchdog_trigger": verdict.trigger,
        }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        def _object_schema(
            properties: dict[str, Any] | None = None,
            *,
            required: list[str] | None = None,
        ) -> dict[str, Any]:
            schema: dict[str, Any] = {
                "type": "object",
                "properties": properties or {},
                "additionalProperties": False,
            }
            if required:
                schema["required"] = required
            return schema

        schemas: dict[str, dict[str, Any]] = {
            "get_scene_state": {
                "description": (
                    "Get the current scene state including current frame class counts, "
                    "multi-object registry summary, and target info."
                ),
                "parameters": _object_schema(),
            },
            "get_target_info": {
                "description": "Get information about the currently tracked target.",
                "parameters": _object_schema(),
            },
            "get_drone_status": {
                "description": (
                    "Get current drone status including altitude, airborne state, FSM state, "
                    "telemetry, and mission elapsed time."
                ),
                "parameters": _object_schema(),
            },
            "set_mission_state": {
                "description": "Set the mission FSM to a new state.",
                "parameters": _object_schema(
                    {
                        "state": {
                            "type": "string",
                            "enum": [s.value for s in MissionState],
                            "description": "Target mission state.",
                        },
                    },
                    required=["state"],
                ),
            },
            "request_hover": {
                "description": "Request the drone to hover in place.",
                "parameters": _object_schema(),
            },
            "request_scan": {
                "description": "Request the drone to begin scanning the area by rotating.",
                "parameters": _object_schema(),
            },
            "request_follow": {
                "description": (
                    "Request the drone to follow the currently tracked target using IBVS."
                ),
                "parameters": _object_schema(),
            },
            "get_mission_report": {
                "description": (
                    "Get the current mission progress report, including cumulative unique "
                    "track counts by class and total unique vehicle count observed so far."
                ),
                "parameters": _object_schema(),
            },
            "request_takeoff": {
                "description": (
                    "Request the drone to take off from the ground. Must be called "
                    "before any other movement."
                ),
                "parameters": _object_schema(),
            },
            "request_land": {
                "description": "Request the drone to land at its current position.",
                "parameters": _object_schema(),
            },
            "request_move_to_altitude": {
                "description": "Move the drone to a specific altitude in meters.",
                "parameters": _object_schema(
                    {
                        "altitude_m": {
                            "type": "number",
                            "description": (
                                "Target altitude in meters "
                                "(positive value, e.g. 10 = 10m above ground)."
                            ),
                        },
                    },
                    required=["altitude_m"],
                ),
            },
            "wait_seconds": {
                "description": (
                    "Wait for a specified number of seconds while the current action "
                    "continues (e.g. follow target for 30 seconds)."
                ),
                "parameters": _object_schema(
                    {
                        "seconds": {
                            "type": "number",
                            "description": "Number of seconds to wait (max 60).",
                        },
                    },
                    required=["seconds"],
                ),
            },
            "request_return_home": {
                "description": "Request the drone to return to its starting (home) position.",
                "parameters": _object_schema(),
            },
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                    "strict": True,
                },
            }
            for name, schema in schemas.items()
        ]

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        started = time.monotonic()
        handler = self._tools.get(name)
        if handler is None:
            log_event(
                logger,
                logging.WARNING,
                "tool.dispatch",
                "Unknown tool requested",
                mission_id=self._reporter.mission_id,
                tool_name=name,
                arguments=arguments,
                result_ok=False,
            )
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"})
        result = await handler(arguments)
        log_event(
            logger,
            logging.DEBUG,
            "tool.dispatch",
            "Tool dispatch finished",
            mission_id=self._reporter.mission_id,
            tool_name=name,
            arguments=arguments,
            latency_ms=round((time.monotonic() - started) * 1000, 2),
            result_ok=bool(result.get("ok", False)),
            result_summary=result,
            fsm_state=self._fsm.state.value,
        )
        return json.dumps(result)

    # ── Existing tools ──────────────────────────────────────────────

    async def _get_scene_state(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        return {"ok": True, "data": self._scene_provider()}

    async def _get_target_info(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        return {"ok": True, "data": self._target_payload()}

    async def _get_drone_status(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        try:
            snapshot = self._bridge.read_sensor_snapshot()
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "mission_state": self._fsm.state.value,
            }
        telemetry = snapshot.telemetry
        data: dict[str, Any] = {
            "altitude_m": round(telemetry.altitude_m, 2),
            "position_ned": list(telemetry.position_ned),
            "velocity_ned": list(telemetry.velocity_ned),
            "yaw_deg": round(telemetry.yaw_deg, 1),
            "gps_valid": telemetry.gps_valid,
            "is_airborne": self._bridge.is_airborne(),
            "fsm_state": self._fsm.state.value,
            "mission_elapsed_s": round(self._reporter.elapsed_s, 2),
        }
        verdict = self._check_watchdog()
        if verdict is not None:
            data["mission_envelope"] = {
                "ok": not verdict.tripped,
                "trigger": verdict.trigger,
                "reason": verdict.reason,
            }
        return {"ok": True, "data": data}

    async def _set_mission_state(self, arguments: dict[str, Any]) -> PilotToolResult:
        state = MissionState(arguments["state"])
        from_state = self._fsm.state
        if self._mission_mode() == MissionMode.TRAFFIC_MONITOR and state == MissionState.TRACK:
            return {
                "ok": False,
                "message": (
                    "TRACK is disabled for traffic counting missions. Stay in SCAN or MONITOR, "
                    "continue patrolling roads, and use get_mission_report() for cumulative totals."
                ),
                "mission_state": self._fsm.state.value,
                "required_sequence": [
                    "request_scan()",
                    "wait_seconds(seconds=...)",
                    "get_mission_report()",
                ],
            }
        if (
            from_state == MissionState.TRACK
            and state == MissionState.SCAN
            and self._priority_target_locked()
        ):
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Mission state downgrade to SCAN rejected while priority target lock is active",
                mission_id=self._reporter.mission_id,
                from_state=from_state.value,
                to_state=state.value,
                source="tool_guard",
                reason="scan_override_blocked_active_lock",
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": "cannot switch to SCAN while priority target lock is active",
                "mission_state": self._fsm.state.value,
            }
        path = self._resolve_state_path(from_state, state)
        if path is None:
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Mission state request rejected because transition path is invalid",
                mission_id=self._reporter.mission_id,
                from_state=from_state.value,
                to_state=state.value,
                source="llm_tool",
                reason="invalid_transition_path",
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": f"{from_state.value} -> {state.value} is not allowed",
                "mission_state": self._fsm.state.value,
            }
        error = self._transition_along_path(path, reason="llm_tool")
        if error is not None:
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Mission state request failed during transition execution",
                mission_id=self._reporter.mission_id,
                from_state=from_state.value,
                to_state=state.value,
                source="llm_tool",
                reason=error["message"],
                fsm_state=self._fsm.state.value,
            )
            return error
        log_event(
            logger,
            logging.INFO,
            "fsm.transition",
            "Mission state updated via tool",
            mission_id=self._reporter.mission_id,
            from_state=from_state.value,
            to_state=state.value,
            source="llm_tool",
            reason="set_mission_state",
            fsm_state=state.value,
        )
        return {"ok": True, "mission_state": state.value}

    def _resolve_state_path(
        self,
        from_state: MissionState,
        to_state: MissionState,
    ) -> list[MissionState] | None:
        if from_state == to_state:
            return []
        transitions = getattr(self._fsm, "_transitions", {})
        queue: deque[tuple[MissionState, list[MissionState]]] = deque([(from_state, [])])
        visited = {from_state}
        while queue:
            current, path = queue.popleft()
            for nxt in transitions.get(current, []):
                if nxt in visited:
                    continue
                next_path = [*path, nxt]
                if nxt == to_state:
                    return next_path
                visited.add(nxt)
                queue.append((nxt, next_path))
        return None

    def _transition_along_path(
        self,
        path: list[MissionState],
        *,
        reason: str,
    ) -> PilotToolResult | None:
        """Execute a resolved FSM path one step at a time.

        Returns ``None`` when every step succeeds, or a structured error result
        if any intermediate transition is rejected. Centralising this loop keeps
        multi-hop path execution consistent and makes the resolved path auditable.
        """
        for step in path:
            try:
                self._fsm.transition(step, reason=reason)
            except InvalidTransitionError as exc:
                return {
                    "ok": False,
                    "message": str(exc),
                    "mission_state": self._fsm.state.value,
                }
        return None

    async def _request_hover(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        self._bridge.request_hover()
        log_event(
            logger,
            logging.INFO,
            "bridge.command",
            "Hover requested via tool",
            mission_id=self._reporter.mission_id,
            tool_name="request_hover",
            fsm_state=self._fsm.state.value,
        )
        return {"ok": True, "message": "hover requested"}

    async def _request_scan(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        from_state = self._fsm.state
        if from_state == MissionState.TRACK and self._priority_target_locked():
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Scan request rejected while priority target lock is active",
                mission_id=self._reporter.mission_id,
                from_state=from_state.value,
                to_state=MissionState.SCAN.value,
                source="tool_guard",
                reason="scan_override_blocked_active_lock",
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": "scan request blocked while target lock is active",
                "mission_state": self._fsm.state.value,
            }
        path = self._resolve_state_path(from_state, MissionState.SCAN)
        if path is None:
            return {
                "ok": False,
                "message": f"Cannot reach SCAN from {from_state.value}",
                "mission_state": self._fsm.state.value,
            }
        error = self._transition_along_path(path, reason="llm_tool")
        if error is not None:
            return error
        log_event(
            logger,
            logging.INFO,
            "fsm.transition",
            "Scan requested via tool",
            mission_id=self._reporter.mission_id,
            from_state=from_state.value,
            to_state=MissionState.SCAN.value,
            source="llm_tool",
            reason="request_scan",
            fsm_state=self._fsm.state.value,
        )
        return {"ok": True, "mission_state": MissionState.SCAN.value}

    async def _request_follow(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        if self._mission_mode() == MissionMode.TRAFFIC_MONITOR:
            if self._fsm.state != MissionState.SCAN:
                path = self._resolve_state_path(self._fsm.state, MissionState.SCAN)
                if path is not None:
                    for step in path:
                        self._fsm.transition(step, reason="counting_mode_follow_blocked")
            return {
                "ok": False,
                "message": (
                    "Follow is disabled for traffic counting missions. Keep scanning roads, "
                    "let the object registry count unique vehicles, and use get_mission_report() "
                    "to check totals."
                ),
                "mission_state": self._fsm.state.value,
                "required_sequence": [
                    "request_scan()",
                    "wait_seconds(seconds=10)",
                    "get_mission_report()",
                ],
            }
        target = self._target_payload()
        priority_class = self._priority_class()
        target_confirmed = bool(target.get("is_confirmed"))
        priority_match = bool(target.get("priority_match"))
        if not target_confirmed or (priority_class and not priority_match):
            if self._fsm.state != MissionState.SCAN:
                path = self._resolve_state_path(self._fsm.state, MissionState.SCAN)
                if path is not None:
                    for step in path:
                        self._fsm.transition(step, reason="follow_precondition_failed")
            log_event(
                logger,
                logging.WARNING,
                "fsm.transition",
                "Follow request rejected because priority target is not locked",
                mission_id=self._reporter.mission_id,
                from_state=self._fsm.state.value,
                to_state=MissionState.SCAN.value,
                source="tool_guard",
                reason="follow_precondition_failed",
                target=target,
                priority_class=priority_class,
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": (
                    "Follow rejected: priority target is not confirmed. "
                    "Recovery sequence: request_scan() -> wait_seconds(seconds=10) -> "
                    "get_target_info() -> request_follow() once is_confirmed=true."
                ),
                "mission_state": MissionState.SCAN.value,
                "required_sequence": [
                    "request_scan()",
                    "wait_seconds(seconds=10)",
                    "get_target_info()",
                    "request_follow()",
                ],
            }
        from_state = self._fsm.state
        path = self._resolve_state_path(from_state, MissionState.TRACK)
        if path is None:
            return {
                "ok": False,
                "message": f"Cannot reach TRACK from {from_state.value}",
                "mission_state": self._fsm.state.value,
            }
        error = self._transition_along_path(path, reason="llm_tool")
        if error is not None:
            return error
        log_event(
            logger,
            logging.INFO,
            "fsm.transition",
            "Follow requested via tool",
            mission_id=self._reporter.mission_id,
            from_state=from_state.value,
            to_state=MissionState.TRACK.value,
            source="llm_tool",
            reason="request_follow",
            fsm_state=self._fsm.state.value,
        )
        return {"ok": True, "mission_state": MissionState.TRACK.value}

    async def _get_mission_report(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        report = self._reporter.finalize()
        completion_ready = self._fsm.state == MissionState.REPORT
        return {
            "ok": True,
            "data": {
                "mission_id": report.mission_id,
                "success": report.success,
                "completion_progress": report.completion_progress,
                "completion_reason": report.completion_reason,
                "duration_s": round(report.duration_s, 3),
                "target_ids_seen": report.target_ids_seen,
                "unique_track_counts": report.unique_track_counts,
                "unique_vehicle_count": report.unique_vehicle_count,
                "unique_object_counts": report.unique_object_counts,
                "active_object_count": report.active_object_count,
                "registry_merge_count": report.registry_merge_count,
                "report_stage": "final" if completion_ready else "snapshot",
                "completion_ready": completion_ready,
                "mission_state": self._fsm.state.value,
            },
        }

    # ── New mission execution tools ─────────────────────────────────

    async def _request_takeoff(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        airborne_check = getattr(self._bridge, "is_airborne", None)
        if callable(airborne_check) and airborne_check():
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Takeoff request rejected because drone is already airborne",
                mission_id=self._reporter.mission_id,
                tool_name="request_takeoff",
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": "takeoff rejected: drone already airborne",
            }
        await asyncio.get_running_loop().run_in_executor(None, self._bridge.takeoff)
        log_event(
            logger,
            logging.INFO,
            "bridge.command",
            "Takeoff executed via tool",
            mission_id=self._reporter.mission_id,
            tool_name="request_takeoff",
            fsm_state=self._fsm.state.value,
        )
        return {"ok": True, "message": "takeoff completed"}

    async def _request_land(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        if self._fsm.state not in {MissionState.IDLE, MissionState.REPORT}:
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Land request rejected because mission is not ready to terminate",
                mission_id=self._reporter.mission_id,
                tool_name="request_land",
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": (
                    f"Landing rejected: FSM is '{self._fsm.state.value}'. "
                    "Required completion sequence: set_mission_state(state='REPORT') -> "
                    "get_mission_report() -> request_land()."
                ),
                "mission_state": self._fsm.state.value,
                "required_sequence": [
                    "set_mission_state(state='REPORT')",
                    "get_mission_report()",
                    "request_land()",
                ],
            }
        await asyncio.get_running_loop().run_in_executor(None, self._bridge.land)
        log_event(
            logger,
            logging.INFO,
            "bridge.command",
            "Land executed via tool",
            mission_id=self._reporter.mission_id,
            tool_name="request_land",
            fsm_state=self._fsm.state.value,
        )
        return {"ok": True, "message": "landing completed"}

    async def _request_move_to_altitude(self, arguments: dict[str, Any]) -> PilotToolResult:
        scene = self._scene_provider()
        if scene.get("mission_mode") == MissionMode.TRAFFIC_MONITOR.value:
            return {
                "ok": False,
                "message": (
                    "Altitude is automatically managed in TRAFFIC_MONITOR mode. "
                    "Do not call request_move_to_altitude; continue scanning roads at the "
                    "configured patrol altitude."
                ),
                "mission_state": self._fsm.state.value,
            }
        raw_altitude = arguments.get("altitude_m", 10.0)
        try:
            altitude_m = float(raw_altitude)
        except (TypeError, ValueError):
            return {
                "ok": False,
                "message": (
                    f"Invalid altitude_m argument: {raw_altitude!r}. "
                    "Provide a numeric altitude in meters (e.g. 10)."
                ),
                "mission_state": self._fsm.state.value,
            }
        altitude_m = max(2.0, min(altitude_m, 50.0))  # Clamp 2-50m
        await asyncio.get_running_loop().run_in_executor(
            None,
            self._bridge.move_to_altitude,
            altitude_m,
        )
        log_event(
            logger,
            logging.INFO,
            "bridge.command",
            "Move to altitude executed via tool",
            mission_id=self._reporter.mission_id,
            tool_name="request_move_to_altitude",
            altitude_m=altitude_m,
            fsm_state=self._fsm.state.value,
        )
        return {"ok": True, "message": f"moved to {altitude_m:.1f}m altitude"}

    async def _wait_seconds(self, arguments: dict[str, Any]) -> PilotToolResult:
        seconds = float(arguments.get("seconds", 5.0))
        seconds = max(1.0, min(seconds, 60.0))  # Clamp 1-60s
        if self._fsm.state not in {MissionState.TRACK, MissionState.SCAN, MissionState.MONITOR}:
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Wait request rejected because no active mission phase",
                mission_id=self._reporter.mission_id,
                tool_name="wait_seconds",
                wait_time_s=seconds,
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": (
                    f"wait_seconds requires TRACK, SCAN, or MONITOR. Current state: "
                    f"'{self._fsm.state.value}'. Use request_scan(), request_follow(), or "
                    "set_mission_state() to enter an active mission phase first."
                ),
                "mission_state": self._fsm.state.value,
                "required_sequence": [
                    "request_scan() or request_follow()",
                    "wait_seconds(seconds=...)",
                ],
            }
        log_event(
            logger,
            logging.INFO,
            "mission.wait",
            "Wait started via tool",
            mission_id=self._reporter.mission_id,
            tool_name="wait_seconds",
            wait_time_s=seconds,
            fsm_state=self._fsm.state.value,
        )
        started = time.monotonic()
        while (time.monotonic() - started) < seconds:
            watchdog_verdict = self._check_watchdog()
            if watchdog_verdict is not None and watchdog_verdict.tripped:
                return self._engage_emergency(watchdog_verdict)
            if self._fsm.state == MissionState.TRACK:
                target = self._target_payload()
                priority_class = self._priority_class()
                target_confirmed = bool(target.get("is_confirmed"))
                priority_match = bool(target.get("priority_match"))
                if not target_confirmed or (priority_class and not priority_match):
                    self._fsm.transition(MissionState.SCAN, reason="tracking_interrupted")
                    log_event(
                        logger,
                        logging.WARNING,
                        "mission.wait",
                        "Wait interrupted because tracking target was lost",
                        mission_id=self._reporter.mission_id,
                        tool_name="wait_seconds",
                        wait_time_s=seconds,
                        elapsed_s=round(time.monotonic() - started, 3),
                        priority_class=priority_class,
                        target=target,
                        fsm_state=self._fsm.state.value,
                    )
                    return {
                        "ok": False,
                        "message": (
                            "Tracking interrupted before wait completed. Recovery sequence: "
                            "set_mission_state(state='REACQUIRE') or request_scan(), then "
                            "wait_seconds(), then get_target_info(), then request_follow() "
                            "after target confirmation."
                        ),
                        "mission_state": MissionState.SCAN.value,
                        "required_sequence": [
                            "set_mission_state(state='REACQUIRE') or request_scan()",
                            "wait_seconds(seconds=10)",
                            "get_target_info()",
                            "request_follow()",
                        ],
                    }
            elapsed = time.monotonic() - started
            remaining = max(0.0, seconds - elapsed)
            poll_interval = 1.0 if self._fsm.state == MissionState.SCAN else 0.25
            await asyncio.sleep(min(poll_interval, remaining))
        log_event(
            logger,
            logging.INFO,
            "mission.wait",
            "Wait completed via tool",
            mission_id=self._reporter.mission_id,
            tool_name="wait_seconds",
            wait_time_s=seconds,
            fsm_state=self._fsm.state.value,
        )
        return {
            "ok": True,
            "message": f"waited {seconds:.1f} seconds",
            "scene_state": self._scene_provider(),
        }

    async def _request_return_home(self, arguments: dict[str, Any]) -> PilotToolResult:
        del arguments
        if self._fsm.state not in {MissionState.IDLE, MissionState.REPORT}:
            log_event(
                logger,
                logging.WARNING,
                "tool.guard",
                "Return-home request rejected because mission is still active",
                mission_id=self._reporter.mission_id,
                tool_name="request_return_home",
                fsm_state=self._fsm.state.value,
            )
            return {
                "ok": False,
                "message": "return_home is only allowed after mission completion",
                "mission_state": self._fsm.state.value,
            }
        await asyncio.get_running_loop().run_in_executor(None, self._bridge.return_to_home)
        home = self._bridge.home_position
        log_event(
            logger,
            logging.INFO,
            "bridge.command",
            "Return home executed via tool",
            mission_id=self._reporter.mission_id,
            tool_name="request_return_home",
            home_position=list(home) if home else None,
            fsm_state=self._fsm.state.value,
        )
        return {
            "ok": True,
            "message": "returning to home position",
            "home_position": list(home) if home else None,
        }
