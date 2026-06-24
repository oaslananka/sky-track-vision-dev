from __future__ import annotations

from autonomy.energy import BatteryModel


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
