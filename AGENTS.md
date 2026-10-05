# SkyTrackVision Agent Instructions

These instructions apply repository-wide. A nested `AGENTS.md` adds rules for its subtree; the closest applicable file wins for local implementation details.

## Repository purpose

SkyTrackVision is an educational and research framework for AirSim-based drone perception and mission orchestration. It is not a production flight-control system or safety-certified autopilot.

Do not present simulator/demo success as evidence of real-world flight readiness.

## Nested boundaries

- `autonomy/AGENTS.md` — AirSim-free mission, control, FSM, watchdog, safety, and shared contracts.
- `skypilot/AGENTS.md` — LLM mission orchestration, tool contracts, completion criteria, and the guarded simulator bridge.
- `airsim_control/AGENTS.md` — live AirSim connection, sensors, camera, and movement I/O.

Do not add one file per package. `vision`, `agents`, `config`, `demo`, and `ui` remain governed by this root file unless a materially different trust or runtime boundary emerges.

## Architecture

Preserve the three-layer design:

1. SkyPilot/LLM mission planning;
2. perception and deterministic control;
3. AirSim I/O.

`autonomy/` must remain independent of AirSim.

Typed cross-layer data belongs in `autonomy/contracts.py`. The mission FSM owns mission state. Higher-level agents provide context and requests; they do not redefine state-transition semantics.

## Safety contract

- Autonomous motion continues to pass through deterministic safety evaluation before simulator movement.
- Lost connection, stale telemetry, hard mission-envelope violations, and documented unsafe states must not be interpreted as clear/safe state.
- The `EMERGENCY` path must remain available from every mission state.
- Geofence, altitude, duration, battery, obstacle, and person-separation limits are deterministic runtime policy, not LLM suggestions.
- A new tool or movement path must preserve the same guarded control path.

## LLM boundary

LLM output is untrusted planning input.

- The model acts only through explicit tools.
- Tool arguments remain typed/validated and bounded.
- Safety-critical decisions remain in deterministic code.
- Retries, waits, tool calls, context growth, and mission duration remain bounded.
- Mission completion is based on measurable objectives, not a closing sentence from the model.
- Never commit real provider credentials or include them in fixtures/logs.

## AirSim boundary

Live AirSim requires Python 3.11 in this repository. Core/demo/test support remains Python 3.11–3.13.

CI is intentionally AirSim-free. A green CI run does not prove live AirSim, Unreal, GPU, physical sensors, or real vehicle behavior.

Keep AirSim imports guarded; do not make the simulator dependency mandatory for the AirSim-free core.

## Evidence and benchmarking

Do not invent detection accuracy, tracking robustness, FPS, mission success, zero-intervention, or safety-violation results.

Deterministic demo/benchmark outputs prove only the exact scenario exercised. Keep units, NED conventions, controller bounds, timestamps, and seeded scenario assumptions explicit.

## Configuration

`pilot.yaml` overrides typed defaults in `config/settings.py`.

New configuration should have typed validation/defaults. Keep runtime limits bounded and documented. Missing or invalid configuration must not silently disable a guard.

## Validation

Run the narrowest relevant tests first, then the full applicable checks:

```bash
python -m pytest tests/ -v
ruff check .
ruff format --check .
mypy .
python smoke_test.py
python scripts/benchmark.py --demo
```

Live AirSim validation is environment-dependent; state clearly when it was not run.

## Change discipline

- Keep diffs focused.
- Behavioral safety fixes require regression tests.
- New mission states/tools must preserve FSM/tool/documentation parity.
- Update docs/config when public or safety behavior changes.
- Do not weaken lint, typecheck, tests, smoke tests, or safety assertions to land unrelated work.

## Definition of done

A change is ready when implementation, typed contracts, deterministic safety/FSM behavior, focused tests, configuration/docs, and exact-head CI agree. Report any live AirSim/GPU/LLM validation that was not exercised.
