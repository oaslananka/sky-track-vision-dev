"""CLI over the mission scoring harness.

Usage:
    python scripts/benchmark.py --demo     # print a scorecard for the built-in demo set

The demo set illustrates the methodology offline (no AirSim, no LLM). To benchmark
real runs, collect :class:`~autonomy.benchmark.MissionTrial` records from live
missions (one per seeded scenario) and pass them to ``score_trials``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomy.benchmark import demo_trials, score_trials


def main() -> None:
    parser = argparse.ArgumentParser(description="SkyTrackVision mission benchmark")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Score the built-in demo scenario set (offline, deterministic).",
    )
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    scorecard = score_trials(demo_trials())
    print(scorecard.format())


if __name__ == "__main__":
    main()
