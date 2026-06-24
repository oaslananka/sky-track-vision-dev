# SkyPilot — LLM Mission System

This document explains how the SkyPilot LLM runtime works and how to extend it.

## Overview

SkyPilot turns natural language mission descriptions into autonomous drone behavior:

```
"Take off, find a truck, follow it for 30 seconds, then return home and land."
```

The LLM breaks this into tool calls, each mapped to a specific drone action. The entire pipeline runs with a **safety gate** that the LLM cannot override.

## How It Works

### 1. System Prompt

The LLM receives a system prompt (from `pilot.py`) that describes:
- Available tools and their parameters
- Current mission state (FSM state, telemetry, vision data)
- Safety constraints it must respect
- Output format (tool calls only, no free-text commands)

### 2. Tool Calling Loop

```
User Task → LLM → Tool Call → ToolDispatcher → AirSim Bridge → Drone
                                     ↑
                              Scene Provider (vision feedback)
```

Each LLM response can contain one or more tool calls. The `ToolDispatcher`:
1. Validates the tool name and arguments
2. Checks FSM state guards (e.g., can't follow without being in SCAN first)
3. Executes the action through the `AirSimBridge`
4. Returns structured results back to the LLM

### 3. Scene Feedback

Between LLM turns, the `PilotDisplay` runs at ~15-30 Hz doing:
- YOLO detection + Kalman tracking
- IBVS visual servoing (if in TRACK state)
- Safety evaluation
- HUD rendering

The LLM sees aggregated scene data via the `scene_provider` callback.

## Adding New Tools

To add a new tool to the SkyPilot:

### Step 1: Define the Tool Schema

In `skypilot/tools.py`, add to the `TOOL_SCHEMAS` list:

```python
{
    "type": "function",
    "function": {
        "name": "my_new_tool",
        "description": "What this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Description"},
            },
            "required": ["param1"],
        },
    },
}
```

### Step 2: Implement the Handler

Add an async method to `ToolDispatcher`:

```python
async def _my_new_tool(self, arguments: dict[str, Any]) -> PilotToolResult:
    param1 = arguments.get("param1", "default")
    # Your logic here
    return {"ok": True, "message": "Done", "mission_state": self._fsm.state.value}
```

### Step 3: Register in Dispatch Table

Add the mapping in `ToolDispatcher.__init__`:

```python
self._handlers["my_new_tool"] = self._my_new_tool
```

### Step 4: Add Tests

Create tests in `tests/test_tools.py` verifying:
- Happy path execution
- State guard rejections
- Edge cases

## Mission Modes

| Mode | Description | Default Target |
|------|-------------|---------------|
| `PEDESTRIAN_WATCH` | Track and follow people | `person` |
| `SEARCH` | General search and follow | Configurable |
| `SURVEILLANCE` | Monitoring mode | Area-based |

## Configuration

Key settings in `pilot.yaml`:

```yaml
pilot:
  model: gpt-5-nano          # LLM model
  max_context_messages: 40    # Conversation history length
  tool_retry_limit: 3         # Max retries per tool call
  tick_duration_s: 0.5        # Control tick interval
  cruise_altitude_m: 4.0      # Default operating altitude
  scan_yaw_rate: 0.35         # Scanning rotation speed
```
