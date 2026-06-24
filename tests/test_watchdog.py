from __future__ import annotations

from autonomy.watchdog import (
    TRIGGER_ALTITUDE,
    TRIGGER_BATTERY,
    TRIGGER_GEOFENCE,
    TRIGGER_TIMEOUT,
    MissionWatchdog,
)
from config.settings import WatchdogConfig

_HOME = (0.0, 0.0, -4.0)


def test_within_envelope_does_not_trip() -> None:
    wd = MissionWatchdog(WatchdogConfig())
    verdict = wd.evaluate(elapsed_s=30.0, position_ned=(5.0, 5.0, -4.0), home_ned=_HOME)
    assert verdict.tripped is False


def test_hard_timeout_trips() -> None:
    wd = MissionWatchdog(WatchdogConfig(max_mission_duration_s=300.0))
    verdict = wd.evaluate(elapsed_s=301.0, position_ned=(0.0, 0.0, -4.0), home_ned=_HOME)
    assert verdict.tripped is True
    assert verdict.trigger == TRIGGER_TIMEOUT


def test_geofence_trips_on_horizontal_distance() -> None:
    wd = MissionWatchdog(WatchdogConfig(geofence_radius_m=50.0))
    verdict = wd.evaluate(elapsed_s=10.0, position_ned=(40.0, 40.0, -4.0), home_ned=_HOME)
    assert verdict.tripped is True
    assert verdict.trigger == TRIGGER_GEOFENCE


def test_altitude_ceiling_trips() -> None:
    wd = MissionWatchdog(WatchdogConfig(max_altitude_m=50.0))
    verdict = wd.evaluate(elapsed_s=10.0, position_ned=(0.0, 0.0, -60.0), home_ned=_HOME)
    assert verdict.tripped is True
    assert verdict.trigger == TRIGGER_ALTITUDE


def test_battery_trips_when_provided_and_low() -> None:
    wd = MissionWatchdog(WatchdogConfig(battery_rtl_fraction=0.2))
    verdict = wd.evaluate(
        elapsed_s=10.0,
        position_ned=(0.0, 0.0, -4.0),
        home_ned=_HOME,
        battery_fraction=0.15,
    )
    assert verdict.tripped is True
    assert verdict.trigger == TRIGGER_BATTERY


def test_battery_skipped_when_unknown() -> None:
    wd = MissionWatchdog(WatchdogConfig(battery_rtl_fraction=0.2))
    verdict = wd.evaluate(
        elapsed_s=10.0,
        position_ned=(0.0, 0.0, -4.0),
        home_ned=_HOME,
        battery_fraction=None,
    )
    assert verdict.tripped is False


def test_disabled_watchdog_never_trips() -> None:
    wd = MissionWatchdog(WatchdogConfig(enabled=False, max_mission_duration_s=1.0))
    verdict = wd.evaluate(elapsed_s=999.0, position_ned=(999.0, 999.0, -999.0), home_ned=_HOME)
    assert verdict.tripped is False
