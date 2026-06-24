from __future__ import annotations

import asyncio
from typing import Any, cast

from autonomy.mission_spec import parse_mission_spec
from autonomy.reporting import EventReporter
from config.settings import PilotConfig
from skypilot.models import ChatResponse
from skypilot.pilot import LLMPilot


class FakeClient:
    def __init__(self) -> None:
        self._responses: list[ChatResponse] = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "request_scan",
                        "arguments": {},
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "name": "request_land",
                        "arguments": {},
                    }
                ],
            },
            {"content": "done", "tool_calls": []},
        ]
        self.calls: list[list[dict[str, object]]] = []

    async def chat(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> ChatResponse:
        del tools, system
        self.calls.append(messages)
        return self._responses.pop(0)


class FakeTools:
    def __init__(self) -> None:
        self.dispatched: list[str] = []

    def get_tool_schemas(self) -> list[dict[str, object]]:
        return []

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        del arguments
        self.dispatched.append(name)
        return '{"ok": true}'


def test_llm_pilot_preserves_assistant_tool_calls_before_tool_messages() -> None:
    fake_client = FakeClient()
    pilot = LLMPilot(fake_client, FakeTools(), EventReporter(), PilotConfig())

    asyncio.run(pilot.run_mission("scan area"))

    second_call_messages = cast(FakeClient, pilot._client).calls[1]
    assistant_message = second_call_messages[1]
    tool_message = second_call_messages[2]

    assert assistant_message["role"] == "assistant"
    assert "tool_calls" in assistant_message
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"


def test_llm_pilot_ignores_premature_stop_without_completion_tools() -> None:
    class PrematureStopClient:
        def __init__(self) -> None:
            self._responses: list[ChatResponse] = [
                {"content": "stop", "tool_calls": []},
                {
                    "content": "landing",
                    "tool_calls": [
                        {
                            "id": "call_land",
                            "name": "request_land",
                            "arguments": {},
                        }
                    ],
                },
                {"content": "done", "tool_calls": []},
            ]
            self.calls = 0

        async def chat(
            self,
            *,
            messages: list[dict[str, object]],
            tools: list[dict[str, Any]],
            system: str,
        ) -> ChatResponse:
            del messages, tools, system
            self.calls += 1
            return self._responses.pop(0)

    fake_client = PrematureStopClient()
    fake_tools = FakeTools()
    pilot = LLMPilot(
        fake_client,
        fake_tools,
        EventReporter(),
        PilotConfig(max_context_messages=8, tool_retry_limit=3),
    )

    asyncio.run(pilot.run_mission("scan area"))

    assert "request_land" in fake_tools.dispatched
    # Pilot exits as soon as _landing_completed is set (line 263 in pilot.py).
    # The third response was never needed because landing was confirmed after call 2.
    assert fake_client.calls == 2


def test_reflection_injected_for_unmet_measurable_objectives() -> None:
    pilot = LLMPilot(
        FakeClient(), FakeTools(), EventReporter(), PilotConfig(reflection_interval_iters=2)
    )
    pilot._spec = parse_mission_spec("Find a truck and follow it")
    messages: list[dict[str, object]] = []

    pilot._maybe_inject_reflection(messages, iteration=2)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "UNMET" in str(messages[0]["content"])


def test_reflection_skipped_for_unmeasurable_task() -> None:
    pilot = LLMPilot(
        FakeClient(), FakeTools(), EventReporter(), PilotConfig(reflection_interval_iters=2)
    )
    pilot._spec = parse_mission_spec("scan around the area")
    messages: list[dict[str, object]] = []

    pilot._maybe_inject_reflection(messages, iteration=2)

    assert messages == []
