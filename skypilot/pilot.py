from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from autonomy.contracts import MissionReport, MissionState
from autonomy.reporting import EventReporter
from config.runtime_logging import log_event
from config.settings import PilotConfig
from skypilot.models import ChatResponse

logger = logging.getLogger("skytrackvision.skypilot.pilot")

SYSTEM_PROMPT = """
You are SkyTrackVision's autonomous drone pilot running on GPT-5-mini.
You control the mission layer through tool calls only.

START CONDITIONS
- Runtime already performed preflight before you start.
- The drone is usually already airborne, near cruise altitude, and in SCAN.
- Do not repeat takeoff or climb unless tool feedback says it is required.

MISSION RULES
1. While the mission is active, respond with tool calls only.
2. Do not guess current state. If uncertain, call get_drone_status, get_scene_state,
   or get_target_info.
3. Read every tool result carefully.
   - ok=true: continue to the next step.
   - ok=false: read the message, adapt immediately, and retry with a better tool call.
4. request_follow requires a confirmed target.
   - If target is not confirmed: request_scan -> wait_seconds -> get_target_info -> request_follow
5. wait_seconds only works in TRACK, SCAN, or MONITOR.
6. If tracking is interrupted or the target is lost, explicitly recover:
   - set_mission_state(REACQUIRE) or request_scan
   - wait_seconds
   - get_target_info
   - request_follow only after confirmation returns
7. Use REPORT only when the mission objective is satisfied and you are ready to terminate.
8. Landing has a mandatory completion sequence and must not be skipped:
   set_mission_state(REPORT) -> get_mission_report() -> request_land()
9. request_land will fail outside REPORT or IDLE. If it fails, go to REPORT first.
10. For counting or survey missions, get_mission_report returns cumulative mission totals.
    - Use data.unique_track_counts.<class_name> for class-specific totals.
    - Use data.unique_vehicle_count for all unique road vehicles seen so far.
11. get_scene_state also returns an object_registry summary with active object counts and
    recently merged multi-object identities. Use it when you need mid-mission counting status.
12. For traffic counting missions, do not follow individual vehicles.
    - Stay in SCAN or MONITOR.
    - Let the registry count unique vehicles while you continue road patrol.
    - Use get_mission_report to read totals, then finish with REPORT -> LAND.
13. A mission report only counts as the final completion report when mission_state=REPORT
    and data.completion_ready=true. Snapshot reports taken during SCAN or MONITOR do not
    complete the mission.

MISSION PATTERNS
- Search / observe:
  get_scene_state -> request_scan -> wait_seconds -> get_target_info
- Count vehicles:
  get_scene_state -> request_scan -> wait_seconds -> get_mission_report
  Use unique_track_counts.car for car totals and repeat until the search area is covered.
- Track:
  get_target_info -> request_follow -> wait_seconds -> get_target_info
- Recover after temporary loss:
  set_mission_state(REACQUIRE) -> wait_seconds -> get_target_info
  If still not confirmed: request_scan -> wait_seconds -> get_target_info
- Complete mission:
  set_mission_state(REPORT) -> get_mission_report -> request_land

EXAMPLE
Task: "Find a person, follow for 20 seconds, then land."
1. get_drone_status
2. get_scene_state
3. request_scan
4. wait_seconds(seconds=15)
5. get_target_info
6. request_follow
7. wait_seconds(seconds=20)
8. get_target_info
9. set_mission_state(state="REPORT")
10. get_mission_report
11. request_land
"""


class ChatClient(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> ChatResponse: ...


class ToolRuntime(Protocol):
    def get_tool_schemas(self) -> list[dict[str, Any]]: ...

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str: ...


class LLMPilot:
    """Mission-level LLM loop that delegates execution to tools and the mission FSM."""

    def __init__(
        self,
        client: ChatClient,
        tools: ToolRuntime,
        reporter: EventReporter,
        cfg: PilotConfig,
    ) -> None:
        self._client = client
        self._tools = tools
        self._reporter = reporter
        self._cfg = cfg
        self._mission_report_retrieved = False
        self._landing_completed = False

    def _reset_completion_state(self) -> None:
        self._mission_report_retrieved = False
        self._landing_completed = False

    def _mission_completion_confirmed(self) -> bool:
        return self._mission_report_retrieved and self._landing_completed

    @staticmethod
    def _parse_tool_result(raw_result: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_result)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _record_tool_result(self, tool_name: str, raw_result: str) -> None:
        payload = self._parse_tool_result(raw_result)
        if not payload.get("ok"):
            return
        if tool_name == "get_mission_report":
            data = payload.get("data")
            if isinstance(data, dict) and bool(data.get("completion_ready")):
                self._mission_report_retrieved = True
        elif tool_name == "request_land":
            self._landing_completed = True

    async def run_mission(self, task: str) -> MissionReport:
        messages: list[dict[str, object]] = [{"role": "user", "content": task}]
        self._reset_completion_state()
        log_event(
            logger,
            logging.DEBUG,
            "mission.loop",
            "Mission loop started",
            mission_id=self._reporter.mission_id,
            iteration=0,
            task=task,
        )
        max_iterations = max(
            self._cfg.tool_retry_limit + self._cfg.max_context_messages,
            int(self._cfg.mission_timeout_s / 2),
        )
        for iteration in range(max_iterations):
            log_event(
                logger,
                logging.DEBUG,
                "mission.loop",
                "Mission loop iteration",
                mission_id=self._reporter.mission_id,
                iteration=iteration,
                message_count=len(messages),
            )
            response: ChatResponse = await self._client.chat(
                messages=messages,
                tools=self._tools.get_tool_schemas(),
                system=SYSTEM_PROMPT,
            )
            if not response["tool_calls"]:
                log_event(
                    logger,
                    logging.DEBUG,
                    "mission.loop",
                    "LLM returned final content without tool calls",
                    mission_id=self._reporter.mission_id,
                    iteration=iteration,
                    content_chars=len(response["content"]),
                )
                messages.append({"role": "assistant", "content": response["content"]})
                if self._mission_completion_confirmed():
                    break
                # Guard against early "done/stop" answers that skip required tools.
                log_event(
                    logger,
                    logging.WARNING,
                    "mission.loop",
                    "LLM attempted to finish before mission completion criteria were met",
                    mission_id=self._reporter.mission_id,
                    iteration=iteration,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Mission is still active. Continue with tool calls only. "
                            "Required completion sequence: set_mission_state(REPORT) -> "
                            "get_mission_report() -> request_land()."
                        ),
                    }
                )
                continue
            log_event(
                logger,
                logging.DEBUG,
                "mission.loop",
                "LLM requested tool calls",
                mission_id=self._reporter.mission_id,
                iteration=iteration,
                tool_calls=len(response["tool_calls"]),
                tool_names=[call["name"] for call in response["tool_calls"]],
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in response["tool_calls"]
                    ],
                }
            )
            for call in response["tool_calls"]:
                log_event(
                    logger,
                    logging.DEBUG,
                    "mission.loop",
                    "Dispatching tool",
                    mission_id=self._reporter.mission_id,
                    iteration=iteration,
                    tool_name=call["name"],
                    arguments=call["arguments"],
                )
                result = await self._tools.dispatch(call["name"], call["arguments"])
                self._record_tool_result(call["name"], result)
                log_event(
                    logger,
                    logging.DEBUG,
                    "mission.loop",
                    "Tool completed",
                    mission_id=self._reporter.mission_id,
                    iteration=iteration,
                    tool_name=call["name"],
                    result=result,
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call["id"],
                    }
                )
            if self._landing_completed:
                break
            if len(messages) > self._cfg.max_context_messages:
                log_event(
                    logger,
                    logging.DEBUG,
                    "mission.loop",
                    "Trimming message history",
                    mission_id=self._reporter.mission_id,
                    iteration=iteration,
                    message_count=len(messages),
                    max_context_messages=self._cfg.max_context_messages,
                )
                # Safe trim: find a cut point that doesn't orphan tool_calls / tool results.
                # Keep first message (user task) + last N messages.
                keep = self._cfg.max_context_messages - 1
                cut = len(messages) - keep
                # Walk backward from cut to find the start of a complete
                # assistant→tool(s) block so we don't leave orphans.
                while cut > 1:
                    role = messages[cut].get("role")
                    # If we're at a 'tool' message, its parent assistant is before it → go back
                    if role == "tool":
                        cut -= 1
                        continue
                    # If we're at an assistant with tool_calls, its tool results follow → go back
                    if role == "assistant" and messages[cut].get("tool_calls"):
                        cut -= 1
                        continue
                    break
                messages = [messages[0], *messages[cut:]]
        mission_success = self._mission_completion_confirmed()
        completion_reason = MissionState.REPORT.value
        if not mission_success:
            completion_reason = (
                "landed_without_report" if self._landing_completed else "mission_incomplete"
            )
        log_event(
            logger,
            logging.DEBUG,
            "mission.loop",
            "Mission loop ended, finalizing report",
            mission_id=self._reporter.mission_id,
            mission_success=mission_success,
            report_retrieved=self._mission_report_retrieved,
            landing_completed=self._landing_completed,
            completion_reason=completion_reason,
        )
        return self._reporter.finalize(success=mission_success, reason=completion_reason)
