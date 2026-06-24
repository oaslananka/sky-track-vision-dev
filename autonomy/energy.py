"""Battery state-of-charge model.

The open-source AirSim multirotor exposes no battery telemetry, so the
:class:`~autonomy.watchdog.MissionWatchdog` would otherwise never get a battery
reading to act on. This module fills that gap with a simple, deterministic
time-based energy model: charge drains linearly from full over a configured
endurance. It gives the watchdog a real, monotonic ``battery_fraction`` so the
low-battery abort path is genuinely exercised, and it is trivially replaced by a
hardware/telemetry-backed source later via the same ``fraction`` interface.
"""

from __future__ import annotations


class BatteryModel:
    """Linear state-of-charge estimate as a function of elapsed flight time."""

    def __init__(self, endurance_s: float) -> None:
        # Guard against a zero/negative endurance turning into a divide-by-zero
        # or an always-empty battery.
        self._endurance_s = max(1.0, float(endurance_s))

    def fraction(self, elapsed_s: float) -> float:
        """Return remaining charge in ``[0.0, 1.0]`` after ``elapsed_s`` seconds."""
        remaining = 1.0 - (max(0.0, elapsed_s) / self._endurance_s)
        return max(0.0, min(1.0, remaining))
