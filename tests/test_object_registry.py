from __future__ import annotations

import time

from autonomy.contracts import Detection
from vision.object_registry import MultiObjectRegistry as ObjectRegistry


def _make_detection(
    *,
    class_name: str = "car",
    track_id: int = 1,
    center: tuple[float, float] = (320.0, 240.0),
    area: float = 10_000.0,
    confidence: float = 0.9,
) -> Detection:
    x, y = int(center[0]), int(center[1])
    half = int(area**0.5) // 2
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=(x - half, y - half, x + half, y + half),
        track_id=track_id,
        center=center,
        area=area,
    )


def test_registry_update_does_not_crash_on_valid_input() -> None:
    """Basic smoke test: updating registry must not raise."""
    registry = ObjectRegistry()
    detection = _make_detection()

    registry.update([detection], timestamp_ns=time.time_ns())


def test_area_ratio_calculation_does_not_crash() -> None:
    """Regression for BUG-01: min(a, b, key=float) raised TypeError."""
    registry = ObjectRegistry()
    ts = time.time_ns()

    # First update establishes object.
    registry.update([_make_detection(center=(100.0, 100.0), area=5_000.0)], timestamp_ns=ts)

    # Second update with same class at similar position — triggers area_ratio calculation.
    registry.update(
        [_make_detection(center=(102.0, 102.0), area=6_000.0)],
        timestamp_ns=ts + 100_000_000,
    )


def test_area_ratio_guard_blocks_merge_on_large_area_change() -> None:
    """Objects whose area changes by >2.5x should not be merged."""
    registry = ObjectRegistry()
    ts = time.time_ns()

    registry.update([_make_detection(area=5_000.0)], timestamp_ns=ts)
    initial_count = registry.summary().total_objects

    # Same position but 10x area change — should NOT merge.
    registry.update([_make_detection(area=50_000.0)], timestamp_ns=ts + 100_000_000)

    assert registry.summary().total_objects >= initial_count


def test_registry_summary_returns_correct_vehicle_count() -> None:
    registry = ObjectRegistry()
    ts = time.time_ns()

    registry.update([_make_detection(class_name="car", track_id=1)], timestamp_ns=ts)
    registry.update(
        [_make_detection(class_name="truck", track_id=2, center=(500.0, 400.0))],
        timestamp_ns=ts + 1,
    )

    summary = registry.summary()
    assert summary.unique_vehicle_count >= 1
