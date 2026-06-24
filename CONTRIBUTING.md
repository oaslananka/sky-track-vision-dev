# Contributing to SkyTrackVision

Thank you for your interest in contributing! SkyTrackVision is an open-source learning platform for autonomous drone intelligence, and we welcome contributions of all kinds.

## 📋 Before You Start

1. **Fork** the repository and create a new branch from `main`.
2. **Read the architecture** — see the [README](README.md) for the package map and design principles.
3. **Check existing issues** — someone may already be working on a similar feature.

## 🔧 Development Setup

```bash
git clone https://github.com/oaslananka/sky-track-vision-dev.git
cd SkyTrackVision

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

pip install -r requirements-dev.txt
```

## 🧪 Running Tests

All changes must pass the existing test suite:

```bash
# Run tests
python -m pytest tests/ -v

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy .
```

## 📐 Code Style

- **Formatter/Linter:** [Ruff](https://github.com/astral-sh/ruff) — line length 100, Python 3.11 target
- **Type annotations:** All public functions must have type hints (`disallow_untyped_defs = true`)
- **Imports:** Sorted by ruff (`isort` rules enabled)
- **Docstrings:** One-line summary for classes; functions are self-documenting via types

## 🏗️ Architecture Rules

1. **`autonomy/` is AirSim-free.** Never import `airsim` or `airsim_control` from this package.
2. **Safety gate cannot be bypassed.** All movement commands must flow through `SafetyEvaluator`.
3. **FSM is the single source of truth** for mission state. Agents provide context, not state decisions.
4. **Typed contracts** — all data flowing between layers uses `autonomy/contracts.py` dataclasses.

## 🎯 Contribution Areas

| Area             | What to Do                                                                  |
| ---------------- | --------------------------------------------------------------------------- |
| 🎯 Detection     | Add new YOLO target classes, improve inference speed                        |
| 🧠 LLM Tools     | Add new tools to `skypilot/tools.py` — the LLM can only do what tools allow |
| 🎮 Control       | Improve IBVS gains, add MPC, implement obstacle avoidance planner           |
| 🔐 Safety        | Add geofencing, battery monitoring, wind estimation                         |
| 📊 Visualization | Build mission replay, 3D trajectory plots, web dashboard                    |
| 📖 Documentation | Tutorials, architecture deep-dives, video demos                             |
| 🧪 Testing       | Increase test coverage, add integration tests                               |

## 📝 Pull Request Process

1. Create a descriptive branch name: `feature/add-geofencing`, `fix/kalman-stability`
2. Write or update tests for your changes
3. Run the full test suite and ensure all checks pass
4. Write a clear PR description explaining **what** and **why**
5. Reference any related issues

## 🐛 Reporting Bugs

Use the [Bug Report](https://github.com/oaslananka/sky-track-vision-dev/issues/new?template=bug_report.md) issue template. Include:

- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, GPU info
- Relevant log output from `outputs/logs/`

## 💡 Feature Requests

Use the [Feature Request](https://github.com/oaslananka/sky-track-vision-dev/issues/new?template=feature_request.md) issue template.

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
