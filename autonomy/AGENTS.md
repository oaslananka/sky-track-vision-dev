# Autonomy Core Agent Instructions

These instructions apply to `autonomy/**` and supplement the repository root instructions.

## Boundary

`autonomy/` is the deterministic, AirSim-free mission/control core. It must not import `airsim` or `airsim_control`.

Cross-layer data belongs in the typed contracts in `contracts.py`. Keep simulator, network, UI, provider, and external-I/O concerns outside this package.

## Mission state

- `MissionFSM` owns mission state and allowed transitions.
- Invalid transitions remain explicit unless a documented recovery path handles them.
- `EMERGENCY` is a universal safe-abort state and must remain reachable from any state.
- Timeout/recovery logic stays bounded and covered by focused tests.

## Safety evaluator

Preserve the documented conservative behavior:

- connection loss produces a safety override;
- stale sensor data produces a safety override;
- missing proximity data does not become an optimistic clear-path result;
- repeated sensor failure escalates;
- obstacle thresholds account for velocity/stopping distance;
- person following preserves minimum separation;
- unsafe forward/descent motion remains blocked or clamped.

Unknown or unavailable safety evidence should not be interpreted as safe unless the contract explicitly marks that datum optional.

## Mission watchdog

The watchdog protects the mission envelope independently of per-frame obstacle evaluation.

Battery threshold, mission timeout, geofence, and altitude ceiling remain hard deterministic limits when enabled. A trip must be able to drive the mission to the documented emergency path.

## Control behavior

Keep units, NED conventions, controller bounds, saturation, and timing semantics explicit. New motion primitives require deterministic tests and a reviewed route through the safety layer.

## Testing

Autonomy changes should be testable without AirSim, OpenAI, GPU, network, or GUI.

Add focused tests for transition edges, unsafe/unknown sensor states, watchdog thresholds, controller limits, timeout recovery, and new contract fields.
