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

from collections.abc import Callable
from typing import Protocol

from autonomy.contracts import TelemetryReading


class BatterySource(Protocol):
    """Anything that can report remaining charge for the mission watchdog."""

    def fraction(self, elapsed_s: float) -> float | None: ...


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


class TelemetryBatterySource:
    """Prefer real battery telemetry, fall back to the time-based estimate.

    When the platform exposes ``TelemetryReading.battery_remaining`` (e.g. a
    hardware or MAVLink bridge populates it), that value is used directly. When
    it is ``None`` — as with open-source AirSim — the model estimate is used.
    This is the seam for plugging in a real battery source without touching the
    watchdog: same ``fraction`` interface as :class:`BatteryModel`.
    """

    def __init__(
        self,
        get_telemetry: Callable[[], TelemetryReading],
        fallback: BatteryModel,
    ) -> None:
        self._get_telemetry = get_telemetry
        self._fallback = fallback

    def fraction(self, elapsed_s: float) -> float | None:
        try:
            telemetry = self._get_telemetry()
            measured = telemetry.battery_remaining
        except Exception:
            measured = None
        if measured is not None:
            return max(0.0, min(1.0, float(measured)))
        return self._fallback.fraction(elapsed_s)
