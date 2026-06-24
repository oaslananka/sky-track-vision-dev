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

In `skypilot/tools.py`, add an entry to the `schemas` dict inside
`ToolDispatcher.get_tool_schemas()` (the `_object_schema` helper builds the
JSON-schema body; the `type`/`function`/`strict` wrapper is added for you):

```python
"my_new_tool": {
    "description": "What this tool does",
    "parameters": _object_schema(
        {"param1": {"type": "string", "description": "Description"}},
        required=["param1"],
    ),
},
```

### Step 2: Implement the Handler

Add an async method to `ToolDispatcher`:

```python
async def _my_new_tool(self, arguments: dict[str, Any]) -> PilotToolResult:
    param1 = arguments.get("param1", "default")
    # Your logic here
    return {"ok": True, "message": "Done", "mission_state": self._fsm.state.value}
```

### Step 3: Register in the Dispatch Table

Add the mapping to the `self._tools` dict in `ToolDispatcher.__init__`:

```python
self._tools["my_new_tool"] = self._my_new_tool
```

### Step 4: Add Tests

Create tests in `tests/test_tools.py` verifying:
- Happy path execution
- State guard rejections
- Edge cases

## Mission Modes

Defined in `MissionMode` (`autonomy/contracts.py`):

| Mode | Description | Default Target |
|------|-------------|----------------|
| `PEDESTRIAN_WATCH` | Track and follow people | `person` |
| `TRAFFIC_MONITOR` | Patrol roads and count unique vehicles (no per-vehicle follow) | road vehicles |
| `SEARCH` | General search and follow | Configurable |
| `ORBIT` | Orbit a locked target | Configurable |
| `MANUAL` | Operator-driven | — |

## Safety, Watchdog & Completion

Three layers keep an unattended mission honest (see
[architecture.md](architecture.md) for details):

- **Per-frame safety gate** (`SafetyEvaluator`) — obstacle/altitude vetoes the LLM
  cannot override; also enforces a minimum standoff when following a person.
- **Mission watchdog** (`MissionWatchdog`) — envelope limits (timeout, geofence,
  altitude ceiling, battery) that force a universal `EMERGENCY` safe-abort.
- **Semantic completion** (`mission_spec.py`) — the task is parsed into measurable
  objectives; the mission is only a success if they are met. Call
  `get_mission_progress` to see per-objective status before finishing.

## Configuration

Key settings in `pilot.yaml` (defaults in `config/settings.py`):

```yaml
pilot:
  model: gpt-5-mini            # LLM model (alt: gpt-5-nano)
  max_context_messages: 60     # Conversation history length
  tool_retry_limit: 3          # Max retries per tool call
  reflection_interval_iters: 12 # Cadence of unmet-objective self-checks
  tick_duration_s: 0.1         # Control tick interval
  cruise_altitude_m: 4.0       # Default operating altitude
  scan_yaw_rate: 0.06          # Scanning rotation speed

watchdog:
  max_mission_duration_s: 600  # Hard cap -> EMERGENCY
  geofence_radius_m: 120       # Max horizontal distance from home
  max_altitude_m: 60           # Altitude ceiling
  battery_rtl_fraction: 0.20   # Abort at/below this charge
  battery_endurance_s: 900     # Full-charge flight time (energy model)
```
