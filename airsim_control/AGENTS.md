# AirSim I/O Agent Instructions

These instructions apply to `airsim_control/**` and supplement the repository root instructions.

## Boundary

This package owns the live AirSim client lifecycle, camera/sensor acquisition, and direct simulator movement I/O.

Higher-level autonomous or LLM code must not bypass its normal guarded control path by creating alternate raw movement clients.

## Compatibility

The AirSim Python dependency is live-simulator-only and currently uses the repository's Python 3.11 path.

Keep imports guarded so offline/core code can run without AirSim. Missing AirSim should fail clearly when live functionality is requested, not during unrelated offline imports.

## Connection lifecycle

- Connection attempts remain bounded by retry count and timeout.
- Failed attempts clear connected state.
- API-control/arming state stays explicit.
- Disconnect makes a best effort to release the simulator state and then marks the local connection disconnected.

## Movement gateway

`DroneMovementController` is the direct simulator movement gateway.

- Preserve locking/serialization around RPC calls.
- Avoid conflicting serial and velocity commands.
- Keep NED units/signs and yaw conversion correct.
- New movement primitives require a corresponding reviewed higher-level safety/tool route before autonomous use.

## Sensors and camera

Retain timestamps used by staleness checks. Missing or malformed live sensor data must not be silently synthesized into a safe reading.

Keep acquisition bounded; avoid uncontrolled background work or unbounded buffers.

## Testing

Normal tests should use fakes and remain AirSim-free. Live simulator evidence is environment-dependent and must be reported separately from CI/unit evidence.
