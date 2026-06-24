from __future__ import annotations

from autonomy.contracts import TelemetryReading
from autonomy.energy import BatteryModel, TelemetryBatterySource


def _telemetry(battery_remaining: float | None) -> TelemetryReading:
    return TelemetryReading(
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_m=0.0,
        gps_valid=True,
        battery_remaining=battery_remaining,
    )


def test_battery_full_at_start() -> None:
    assert BatteryModel(100.0).fraction(0.0) == 1.0


def test_battery_drains_linearly() -> None:
    assert abs(BatteryModel(100.0).fraction(20.0) - 0.8) < 1e-9


def test_battery_clamped_to_zero_when_depleted() -> None:
    assert BatteryModel(100.0).fraction(150.0) == 0.0


def test_battery_guards_against_zero_endurance() -> None:
    model = BatteryModel(0.0)  # clamped to 1s internally — no divide-by-zero
    assert model.fraction(0.0) == 1.0
    assert model.fraction(5.0) == 0.0


def test_battery_ignores_negative_elapsed() -> None:
    assert BatteryModel(100.0).fraction(-10.0) == 1.0


def test_telemetry_source_prefers_measured_reading() -> None:
    # Model would estimate ~0.8, but a real reading of 0.42 must win.
    source = TelemetryBatterySource(
        get_telemetry=lambda: _telemetry(0.42),
        fallback=BatteryModel(100.0),
    )
    assert source.fraction(20.0) == 0.42


def test_telemetry_source_clamps_measured_reading() -> None:
    source = TelemetryBatterySource(
        get_telemetry=lambda: _telemetry(1.7),
        fallback=BatteryModel(100.0),
    )
    assert source.fraction(0.0) == 1.0


def test_telemetry_source_falls_back_when_unmeasured() -> None:
    # battery_remaining is None (open-source AirSim) → use the model estimate.
    source = TelemetryBatterySource(
        get_telemetry=lambda: _telemetry(None),
        fallback=BatteryModel(100.0),
    )
    result = source.fraction(20.0)
    assert result is not None
    assert abs(result - 0.8) < 1e-9


def test_telemetry_source_falls_back_when_telemetry_raises() -> None:
    def _boom() -> TelemetryReading:
        raise RuntimeError("bridge offline")

    source = TelemetryBatterySource(
        get_telemetry=_boom,
        fallback=BatteryModel(100.0),
    )
    result = source.fraction(20.0)
    assert result is not None
    assert abs(result - 0.8) < 1e-9
