# SkyPilot Agent Instructions

These instructions apply to `skypilot/**` and supplement the repository root instructions.

## Boundary

SkyPilot converts natural-language mission intent into explicit reviewed tool calls. The LLM is a planner, not the source of flight-safety policy.

Real autonomous movement must continue through the mission/control layer and the existing safety-gated AirSim bridge.

## Tool contract

- The model can act only through registered tools.
- A new tool expands authority; document its preconditions, argument bounds, state impact, failure semantics, and tests.
- Validate state/enum/numeric arguments before acting.
- Movement tools must preserve the existing bridge and safety path rather than calling lower-level movement directly.
- Mission-state changes remain within the FSM contract.

## Runtime limits

Keep model retries, tool retries, waits, context size, and mission duration bounded. Provider failures must not become successful mission outcomes.

Semantic completion uses measurable mission objectives/evidence; model prose by itself is not completion evidence.

## Safety propagation

Safety vetoes and mission-watchdog trips remain authoritative over model intent. Cached telemetry is still subject to the safety evaluator's freshness rules.

Return-to-home, landing, altitude, scan/follow, and other movement requests preserve documented velocity/duration/envelope constraints.

## Observability

Log bounded mission/tool metadata needed to explain behavior. Do not record provider credentials or unnecessary private prompt content.

Safety vetoes, emergency transitions, failed tool preconditions, and completion evidence should remain diagnosable.

## Testing

Normal tests use fake/stub provider and AirSim surfaces. Cover invalid tool arguments, preconditions, safety-veto propagation, watchdog/emergency behavior, provider retry/error handling, false completion claims, and timeout bounds.

Mocked mission success is not live AirSim or real-model evidence.
