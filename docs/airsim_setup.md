# AirSim Setup Guide

This guide covers setting up Microsoft AirSim with Unreal Engine for SkyTrackVision.

## Prerequisites

- **Unreal Engine 5.x** — [Download](https://www.unrealengine.com/download)
- **AirSim plugin** — [GitHub](https://github.com/microsoft/AirSim)
- **Python 3.11** (required) with the `airsim` package. The `airsim` RPC stack uses
  the stdlib `asyncore` module, removed in Python 3.12 ([PEP 594](https://peps.python.org/pep-0594/)),
  so live AirSim does **not** run on 3.12+. The SkyTrackVision core library runs on 3.11–3.13;
  only this live-simulator path is pinned to 3.11.

## Vehicle Configuration

SkyTrackVision uses a specific drone configuration defined in [`settings.json`](../settings.json). This file configures:

### Camera

| Parameter | Value | Description |
|-----------|-------|-------------|
| Resolution | 1280×720 | HD front camera |
| FOV | 80° | Wide field of view |
| Position | (0.5, 0, -0.3) | Slightly forward and down |
| Pitch | -5° | Tilted down for better ground visibility |

### Sensors

| Sensor | Type | Purpose |
|--------|------|---------|
| LidarSensor | 16-channel, 100K pts/sec | Obstacle detection and clustering |
| FrontProximity | Distance sensor | Forward obstacle distance |
| RearProximity | Distance sensor | Rear clearance |
| LeftProximity | Distance sensor | Left clearance |
| RightProximity | Distance sensor | Right clearance |
| DownProximity | Distance sensor | Altitude / ground distance |

## Setup Steps

### 1. Sync Settings

The `settings.json` must be copied to AirSim's config directory:

```bash
# Linux/macOS
./sync_settings.sh

# Windows PowerShell
./sync_settings.ps1
```

This copies `settings.json` to `~/Documents/AirSim/settings.json` with automatic backup.

### 2. Install AirSim Python Package

```bash
pip install setuptools numpy msgpack-rpc-python
pip install --no-build-isolation airsim
```

Or simply use the bundled extras file (same build notes apply):

```bash
pip install -r requirements-sim.txt
```

> **Note:** `airsim` requires its build deps (`numpy`, `msgpack-rpc-python`) installed
> first and `--no-build-isolation`, because the unmaintained wheel ships no build metadata.

### 3. Launch Unreal Engine

1. Open your Unreal Engine project with the AirSim plugin enabled
2. Press **Play** to start the simulation
3. The drone "Drone" will be spawned based on `settings.json`

### 4. Connect SkyTrackVision

```bash
# Classic runtime
python main.py

# SkyPilot runtime
python -m skypilot "Take off and scan the area"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ConnectionError: Could not connect to AirSim` | Ensure UE is running with AirSim plugin and press Play |
| Missing proximity sensors | Check that all 5 distance sensors are in `settings.json` |
| Black camera frames | Verify camera name matches `front_center` in settings |
| LiDAR returns empty | Ensure `LidarSensor` is enabled and `Range` is sufficient |

## References

- [AirSim Documentation](https://microsoft.github.io/AirSim/)
- [AirSim APIs](https://microsoft.github.io/AirSim/apis/)
- [AirSim Settings Reference](https://github.com/Microsoft/AirSim/blob/main/docs/settings.md)
