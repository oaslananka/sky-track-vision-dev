from __future__ import annotations

from typing import Any, TypedDict


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict[str, Any]


class ChatResponse(TypedDict):
    content: str
    tool_calls: list[ToolCall]


class PilotToolResult(TypedDict, total=False):
    ok: bool
    message: str
    mission_state: str
    home_position: list[float] | None
    data: dict[str, Any]
