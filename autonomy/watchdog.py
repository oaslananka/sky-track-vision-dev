"""Mission-level watchdog that forces a safe abort under hard limits.

The per-frame :class:`~autonomy.safety.SafetyEvaluator` handles *immediate*
hazards (obstacle ahead, altitude floor, lost connection). The watchdog operates
one level up, on the *mission* envelope: how long we have flown, how far we have
strayed from home, how much battery remains. When any envelope limit is breached
it reports a trip so the runtime can drive the FSM to ``EMERGENCY`` and recover
without a human in the loop — the missing piece for genuinely unattended flight.

It is intentionally pure: ``evaluate`` takes plain numbers and returns a verdict,
with no AirSim or I/O dependency, so it is fully unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.settings import WatchdogConfig

# Trigger identifiers, ordered by the priority in which they are checked.
TRIGGER_OK = "ok"
TRIGGER_BATTERY = "battery"
TRIGGER_TIMEOUT = "timeout"
TRIGGER_GEOFENCE = "geofence"
TRIGGER_ALTITUDE = "altitude"


@dataclass(slots=True, frozen=True)
class WatchdogVerdict:
    tripped: bool
    trigger: str
    reason: str


_OK_VERDICT = WatchdogVerdict(tripped=False, trigger=TRIGGER_OK, reason="within mission envelope")


class MissionWatchdog:
    """Stateless evaluator of mission-envelope hard limits."""

    def __init__(self, cfg: WatchdogConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self,
        *,
        elapsed_s: float,
        position_ned: tuple[float, float, float],
        home_ned: tuple[float, float, float] | None,
        battery_fraction: float | None = None,
    ) -> WatchdogVerdict:
        """Return the first breached limit, or an OK verdict.

        ``battery_fraction`` is optional: when ``None`` (telemetry does not
        expose battery, as in the default AirSim setup) the battery check is
        skipped rather than guessed.
        """
        if not self._cfg.enabled:
            return _OK_VERDICT

        if battery_fraction is not None and battery_fraction <= self._cfg.battery_rtl_fraction:
            return WatchdogVerdict(
                tripped=True,
                trigger=TRIGGER_BATTERY,
                reason=(
                    f"battery {battery_fraction * 100:.0f}% at/below "
                    f"{self._cfg.battery_rtl_fraction * 100:.0f}% abort threshold"
                ),
            )

        if elapsed_s >= self._cfg.max_mission_duration_s:
            return WatchdogVerdict(
                tripped=True,
                trigger=TRIGGER_TIMEOUT,
                reason=(
                    f"mission ran {elapsed_s:.0f}s, exceeding hard cap "
                    f"{self._cfg.max_mission_duration_s:.0f}s"
                ),
            )

        altitude_m = abs(position_ned[2])  # NED Z is negative-up
        if altitude_m > self._cfg.max_altitude_m:
            return WatchdogVerdict(
                tripped=True,
                trigger=TRIGGER_ALTITUDE,
                reason=f"altitude {altitude_m:.1f}m above ceiling {self._cfg.max_altitude_m:.0f}m",
            )

        if home_ned is not None:
            dx = position_ned[0] - home_ned[0]
            dy = position_ned[1] - home_ned[1]
            distance_m = math.hypot(dx, dy)
            if distance_m > self._cfg.geofence_radius_m:
                return WatchdogVerdict(
                    tripped=True,
                    trigger=TRIGGER_GEOFENCE,
                    reason=(
                        f"{distance_m:.0f}m from home exceeds geofence radius "
                        f"{self._cfg.geofence_radius_m:.0f}m"
                    ),
                )

        return _OK_VERDICT
