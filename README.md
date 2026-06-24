# SkyTrackVision

<div align="center">

**LLM-Powered Autonomous Drone Perception & Mission Orchestration System**

[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/oaslananka/sky-track-vision-dev/actions/workflows/ci.yml/badge.svg)](https://github.com/oaslananka/sky-track-vision-dev/actions)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/oaslananka/sky-track-vision-dev)

<br/>

_An open-source framework for building autonomous drone intelligence using YOLOv8 perception, Kalman-filtered visual tracking, cascade PID visual servoing, and LLM-based mission orchestration — all running on top of [AirSim](https://github.com/microsoft/AirSim) simulation._

<p align="center">
  <a href="https://www.buymeacoffee.com/oaslananka">
    <img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=%E2%98%95&slug=oaslananka&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff" alt="Buy me a coffee" />
  </a>
</p>

</div>

---

## 🎯 Purpose

SkyTrackVision is designed as a **learning and research platform** for developers who want to:

- **Develop agent management skills** — orchestrate multi-layered autonomous systems where an LLM pilot issues high-level commands and deterministic controllers handle real-time execution.
- **Explore AI engineering patterns** — study how perception, control, safety, and planning layers compose in a real robotics pipeline.
- **Contribute to open-source autonomy** — extend detection classes, add new mission modes, improve the IBVS controller, or build entirely new LLM tool integrations.

> This project is intentionally structured as a **demo-ready, extensible framework** rather than a production system, making it ideal for educational exploration and community-driven development.

## 🏗️ Architecture

The system is organized into **three frequency bands**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SkyPilot (LLM Layer)                       │
│  Natural language task → FSM transitions → Mission orchestration│
│  Frequency: ~1 Hz (LLM inference cadence)                      │
├─────────────────────────────────────────────────────────────────┤
│                  Perception & Control Layer                     │
│  YOLOv8 + BoT-SORT → Kalman Tracker → Cascade PID (IBVS)      │
│  Frequency: ~15-30 Hz (camera frame rate)                      │
├─────────────────────────────────────────────────────────────────┤
│                     AirSim I/O Layer                            │
│  Camera frames, LiDAR, proximity sensors, velocity commands     │
│  Frequency: ~30-60 Hz (simulation tick rate)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Package Map

| Package           | Responsibility                                               | AirSim Dependency |
| ----------------- | ------------------------------------------------------------ | ----------------- |
| `autonomy/`       | Contracts, IBVS controller, FSM, safety evaluator, reporting | ❌ None           |
| `vision/`         | YOLOv8 detector, Kalman tracker, frame annotator             | ❌ None           |
| `agents/`         | Thin facade agents (scene, control, mission, safety)         | ❌ None           |
| `airsim_control/` | Camera, sensors, movement, client management                 | ✅ Required       |
| `skypilot/`       | LLM pilot loop, tool dispatcher, AirSim bridge, HUD          | ✅ Required       |
| `config/`         | Typed dataclass configs, YAML merge, structured logging      | ❌ None           |
| `demo/`           | Synthetic frames & sensor snapshots for offline demos        | ❌ None           |
| `ui/`             | Overlay renderer for classic runtime                         | ❌ None           |

> **Key design principle:** The `autonomy/` package is **completely independent** of AirSim. All mission logic, safety gates, and IBVS control can be unit-tested with pure Python — no simulator needed.

## ⚙️ How It Works

### Classic Runtime (`main.py`)

Real-time loop that captures AirSim frames, runs YOLO detection + Kalman tracking, computes IBVS velocity commands, and applies safety-gated movement. Supports keyboard-driven manual control and a demo mode that runs without AirSim.

### SkyPilot Runtime (`python -m skypilot`)

LLM-powered mission executor. You give it a natural language task, and the GPT pilot:

1. **Plans** — converts the task into a sequence of tool calls
2. **Executes** — manages FSM transitions (SCAN → TRACK → MONITOR → REPORT)
3. **Controls** — delegates to the IBVS controller for real-time tracking
4. **Guards** — every movement passes through the `SafetyEvaluator` before reaching AirSim, while a mission-level `MissionWatchdog` enforces the flight envelope (timeout, geofence, altitude, battery) and can force a universal `EMERGENCY` safe-abort with no human in the loop
5. **Verifies** — the task is parsed into measurable acceptance criteria, and the mission is only scored a success if those objectives are met (not merely if the closing protocol ran) — see [Semantic Completion Verification](docs/architecture.md#semantic-completion-verification)

```
"Take off, scan for trucks, follow the nearest one for 30 seconds, then land."
    ↓
request_takeoff → request_move_to_altitude(4m) → request_scan → [target locked]
    → request_follow → wait_seconds(30) → set_mission_state(REPORT) → request_land
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11–3.13** for the core library, tests, and demo mode.
  - ⚠️ **Live AirSim requires Python 3.11 specifically.** The `airsim` package's RPC stack relies on the stdlib `asyncore` module, which was removed in Python 3.12 ([PEP 594](https://peps.python.org/pep-0594/)). Every AirSim import in the codebase is guarded, so the library still runs on 3.12/3.13 — only the live simulator bridge is unavailable there.
- **Unreal Engine 5.x + AirSim plugin** (for live simulation) — [AirSim Setup Guide](https://microsoft.github.io/AirSim/)
- **OpenAI API key** (for SkyPilot LLM runtime)

### Installation

```bash
# Clone the repository
git clone https://github.com/oaslananka/sky-track-vision-dev.git
cd sky-track-vision-dev

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install core dependencies (AirSim-free; works on Python 3.11–3.13)
pip install -r requirements.txt

# For development (lint, test, type-check)
pip install -r requirements-dev.txt

# For live AirSim simulation only (Python 3.11) — see file header for build notes
pip install -r requirements-sim.txt
```

### Configuration

```bash
# Copy environment template and add your API key
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# Sync AirSim settings (copies settings.json to ~/Documents/AirSim/)
./sync_settings.sh      # Linux/macOS
./sync_settings.ps1     # Windows
```

All runtime parameters are in [`pilot.yaml`](pilot.yaml). Defaults are defined as typed dataclasses in [`config/settings.py`](config/settings.py).

### Running

```bash
# Classic runtime (requires AirSim)
python main.py

# Demo mode (no AirSim needed — synthetic frames)
python main.py --demo

# SkyPilot mission (requires AirSim + OpenAI API key)
python -m skypilot "Scan for pedestrians and report"
python -m skypilot "Take off, find a truck, follow it for 30 seconds, then land"

# Quick smoke test
python smoke_test.py

# Record a demo video (rendered HUD + overlays)
python main.py --demo --record outputs/demo.mp4 --record-fps 30

# Record a SkyPilot mission HUD video (requires AirSim)
python -m skypilot "Scan for pedestrians and report" --record outputs/mission.mp4 --record-fps 30

The recorder writes the final rendered OpenCV frame, including HUD and overlays,
using a wall-clock based fixed-FPS writer so demo playback stays close to real
time even if processing jitter occurs. Use --no-hud to run SkyPilot without the
display; --record requires the HUD window and will error if combined with --no-hud.
```

### Keyboard Controls (Classic Runtime)

| Key   | Action               | Key     | Action              |
| ----- | -------------------- | ------- | ------------------- |
| `W/S` | Forward / Backward   | `R/F`   | Ascend / Descend    |
| `A/D` | Strafe Left / Right  | `J/L`   | Yaw Left / Right    |
| `X`   | Toggle Auto-Follow   | `H`     | Emergency Hover     |
| `G`   | Land                 | `U`     | Toggle Overlay Mode |
| `O`   | Toggle Overlay       | `P`     | Screenshot          |
| `N/B` | Next/Prev Demo Stage | `Q/Esc` | Quit                |

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Lint & format
ruff check .
ruff format .

# Type checking
mypy .
```

### Mission Benchmark

Score missions on success rate, **zero-intervention rate**, and safety-violation
rate — the metrics that matter for unattended autonomy:

```bash
python scripts/benchmark.py --demo   # offline, deterministic scorecard
```

Collect [`MissionTrial`](autonomy/benchmark.py) records from live runs (one per
seeded scenario) and pass them to `score_trials` to benchmark the real pilot.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where contributions are especially welcome:**

- 🎯 **New detection classes** — extend YOLO target list and add mission modes
- 🧠 **LLM tool integrations** — add new tools to the `ToolDispatcher`
- 🎮 **Control algorithms** — improve or replace the IBVS controller
- 🔐 **Safety features** — geofence, altitude ceiling, and a pluggable battery source now ship in `MissionWatchdog`; add wind estimation, or a Re-ID-based target re-acquisition channel
- 📊 **Visualization** — dashboards, 3D trajectory plots, mission replay
- 📖 **Documentation** — tutorials, architecture deep-dives, video demos

## 📁 Project Structure

```
sky-track-vision-dev/
├── main.py                  # Classic runtime entry point
├── skypilot.py              # SkyPilot convenience launcher
├── pilot.yaml               # Runtime configuration overrides
├── settings.json            # AirSim vehicle/sensor configuration
├── autonomy/                # AirSim-free core logic
│   ├── contracts.py         # Typed dataclass contracts
│   ├── ibvs.py              # Cascade PID visual servoing
│   ├── follow_controller.py # Motion primitive resolver
│   ├── mission.py           # Mission FSM with state graph (+ EMERGENCY safe-abort)
│   ├── mission_spec.py      # NL→objective parser + semantic completion verifier
│   ├── watchdog.py          # Mission-envelope watchdog (timeout/geofence/battery)
│   ├── safety.py            # Deterministic per-frame safety evaluator
│   ├── reporting.py         # Mission telemetry collector
│   └── scene_reasoner.py    # Detection-to-insight summarizer
├── vision/                  # Computer vision pipeline
│   ├── detector.py          # YOLOv8 + BoT-SORT facade
│   ├── tracker.py           # Kalman-filtered target tracker
│   ├── annotator.py         # Frame overlay renderer
│   └── utils.py             # Geometry & FPS utilities
├── agents/                  # Facade agent wrappers
├── airsim_control/          # AirSim client, camera, sensors, movement
├── skypilot/                # LLM mission orchestration
│   ├── __main__.py          # SkyPilot entry point
│   ├── pilot.py             # LLM conversation loop
│   ├── tools.py             # Tool dispatcher (14 tools)
│   ├── llm_client.py        # OpenAI adapter with retry logic
│   ├── airsim_bridge.py     # Safety-gated movement bridge
│   ├── pilot_display.py     # Real-time pilot HUD
│   └── hybrid.py            # Inter-tick deterministic controller
├── config/                  # Settings & logging
├── demo/                    # AirSim-free demo director
├── ui/                      # Classic runtime overlay
├── tests/                   # Pytest suite (172 tests)
├── .github/                 # CI/CD & issue templates
└── docs/                    # Extended documentation
```

## 🔧 Tech Stack

| Component        | Technology                                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Simulation       | [Unreal Engine](https://www.unrealengine.com/) + [AirSim](https://github.com/microsoft/AirSim)                                   |
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) with [BoT-SORT](https://github.com/NirAharon/BoT-SORT) tracking |
| Visual Servoing  | Cascade PID — Image-Based Visual Servoing (IBVS)                                                                                 |
| Target Tracking  | Constant-velocity Kalman filter with EMA smoothing                                                                               |
| Mission Planning | LLM function-calling via [OpenAI API](https://platform.openai.com/)                                                              |
| Safety           | Per-frame deterministic safety evaluator (dynamic stopping distance) + mission-envelope watchdog with universal EMERGENCY abort     |
| Verification     | Deterministic NL→objective parser with semantic completion gate                                                                  |
| Visualization    | [OpenCV](https://opencv.org/) real-time HUD                                                                                      |
| Language         | Python 3.11–3.13 with full type annotations (live AirSim: 3.11)                                                                  |

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[Microsoft AirSim](https://github.com/microsoft/AirSim)** — open-source drone/car simulation platform built on Unreal Engine
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** — YOLOv8 real-time object detection framework
- **[BoT-SORT](https://github.com/NirAharon/BoT-SORT)** — state-of-the-art multi-object tracking algorithm
- **[OpenAI](https://openai.com/)** — GPT models powering the SkyPilot mission orchestration
- **[OpenCV](https://opencv.org/)** — computer vision and image processing library

## 📧 Contact

**Maintainer:** [@oaslananka](https://github.com/oaslananka)
