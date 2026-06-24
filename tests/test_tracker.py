from __future__ import annotations

from autonomy.contracts import Detection
from config.settings import VisionConfig
from vision.tracker import KalmanTracker


def test_tracker_confirms_target_after_minimum_frames() -> None:
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=3))
    detection = Detection(
        class_name="person",
        confidence=0.95,
        bbox=(100, 80, 180, 220),
        track_id=1,
        center=(140.0, 150.0),
        area=11_200.0,
    )

    target = None
    for _ in range(3):
        target = tracker.update([detection], priority_class="person", frame_size=(640, 480))

    assert target is not None
    assert target.is_confirmed
    assert target.frames_tracked == 3


def test_tracker_keeps_prediction_when_detection_is_missing() -> None:
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1, max_lost_frames=3))
    detection = Detection(
        class_name="person",
        confidence=0.8,
        bbox=(100, 80, 180, 220),
        track_id=3,
        center=(140.0, 150.0),
        area=11_200.0,
    )
    tracker.update([detection], priority_class="person", frame_size=(640, 480))

    target = tracker.update([], priority_class="person", frame_size=(640, 480))

    assert target is not None
    assert target.frames_since_seen == 1
    assert target.predicted_center is not None


def test_tracker_prefers_priority_class_over_higher_confidence_noise() -> None:
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1))
    detections = [
        Detection(
            class_name="car",
            confidence=0.99,
            bbox=(10, 10, 50, 50),
            track_id=8,
            center=(30.0, 30.0),
            area=1_600.0,
        ),
        Detection(
            class_name="person",
            confidence=0.70,
            bbox=(300, 100, 360, 240),
            track_id=9,
            center=(330.0, 170.0),
            area=8_400.0,
        ),
    ]

    target = tracker.update(detections, priority_class="person", frame_size=(640, 480))

    assert target is not None
    assert target.track_id == 9


def test_tracker_does_not_jump_to_non_priority_class_when_priority_target_temporarily_missing() -> (
    None
):
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1, max_lost_frames=5))
    truck = Detection(
        class_name="truck",
        confidence=0.82,
        bbox=(300, 120, 420, 280),
        track_id=41,
        center=(360.0, 200.0),
        area=19_200.0,
    )
    car = Detection(
        class_name="car",
        confidence=0.95,
        bbox=(80, 100, 170, 220),
        track_id=99,
        center=(125.0, 160.0),
        area=10_800.0,
    )

    first = tracker.update([truck], priority_class="truck", frame_size=(640, 480))
    second = tracker.update([car], priority_class="truck", frame_size=(640, 480))

    assert first is not None
    assert second is not None
    assert second.track_id == 41
    assert second.detection.class_name == "truck"
    assert second.frames_since_seen == 1


def test_tracker_keeps_original_priority_track_when_another_same_class_target_appears() -> None:
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1, sticky_lock_timeout_frames=8))
    lead_truck = Detection(
        class_name="truck",
        confidence=0.83,
        bbox=(280, 120, 420, 300),
        track_id=12,
        center=(350.0, 210.0),
        area=25_200.0,
    )
    intruding_truck = Detection(
        class_name="truck",
        confidence=0.90,
        bbox=(330, 130, 470, 310),
        track_id=77,
        center=(400.0, 220.0),
        area=25_200.0,
    )

    first = tracker.update([lead_truck], priority_class="truck", frame_size=(640, 480))
    second = tracker.update(
        [intruding_truck, lead_truck],
        priority_class="truck",
        frame_size=(640, 480),
    )

    assert first is not None
    assert second is not None
    assert second.track_id == 12
    assert second.detection.class_name == "truck"


def test_tracker_rejects_car_when_priority_is_truck() -> None:
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1))
    detections = [
        Detection(
            class_name="car",
            confidence=0.96,
            bbox=(80, 100, 170, 220),
            track_id=99,
            center=(125.0, 160.0),
            area=10_800.0,
        ),
        Detection(
            class_name="person",
            confidence=0.82,
            bbox=(300, 120, 360, 240),
            track_id=17,
            center=(330.0, 180.0),
            area=7_200.0,
        ),
    ]

    target = tracker.update(detections, priority_class="truck", frame_size=(640, 480))

    assert target is None


def test_tracker_accepts_bus_when_priority_is_truck() -> None:
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1))
    detections = [
        Detection(
            class_name="bus",
            confidence=0.88,
            bbox=(80, 100, 190, 260),
            track_id=66,
            center=(135.0, 180.0),
            area=17_600.0,
        )
    ]

    target = tracker.update(detections, priority_class="truck", frame_size=(640, 480))

    assert target is not None
    assert target.track_id == 66
    assert target.detection.class_name == "bus"


def test_tracker_still_rejects_non_compatible_target_when_priority_is_truck() -> None:
    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1))
    detections = [
        Detection(
            class_name="person",
            confidence=0.96,
            bbox=(80, 100, 170, 220),
            track_id=99,
            center=(125.0, 160.0),
            area=10_800.0,
        )
    ]

    target = tracker.update(detections, priority_class="truck", frame_size=(640, 480))

    assert target is None


def test_tracker_retains_sticky_lock_before_switching_to_new_same_class_target() -> None:
    tracker = KalmanTracker(
        VisionConfig(min_confirm_frames=1, max_lost_frames=10, sticky_lock_timeout_frames=3)
    )
    original = Detection(
        class_name="truck",
        confidence=0.81,
        bbox=(290, 120, 430, 300),
        track_id=12,
        center=(360.0, 210.0),
        area=25_200.0,
    )
    new_truck = Detection(
        class_name="truck",
        confidence=0.90,
        bbox=(70, 100, 200, 260),
        track_id=55,
        center=(135.0, 180.0),
        area=20_800.0,
    )

    tracker.update([original], priority_class="truck", frame_size=(640, 480))
    preserved = tracker.update([new_truck], priority_class="truck", frame_size=(640, 480))
    still_preserved = tracker.update([new_truck], priority_class="truck", frame_size=(640, 480))
    third_hold = tracker.update([new_truck], priority_class="truck", frame_size=(640, 480))
    switched = tracker.update([new_truck], priority_class="truck", frame_size=(640, 480))

    assert preserved is not None
    assert still_preserved is not None
    assert third_hold is not None
    assert switched is not None
    assert preserved.track_id == 12
    assert still_preserved.track_id == 12
    assert third_hold.track_id == 12
    assert switched.track_id == 55


def test_tracker_internal_state_guards_raise_on_direct_call() -> None:
    """assert-replaced guards must raise RuntimeError, not silently fail under -O."""
    import pytest

    tracker = KalmanTracker(VisionConfig())
    # Force _state to None to simulate uninitialized tracker.
    object.__setattr__(tracker, "_state", None)

    with pytest.raises(RuntimeError, match="no active track state"):
        tracker._predict()

    with pytest.raises(RuntimeError, match="no active track state"):
        tracker._predicted_center()

    with pytest.raises(RuntimeError, match="no active track state"):
        tracker._to_target(640, 480)


def test_tracker_survives_singular_covariance_without_crash() -> None:
    """Kalman update must not raise on a degenerate (all-zero) covariance matrix."""
    import numpy as np

    tracker = KalmanTracker(VisionConfig(min_confirm_frames=1))
    detection = Detection(
        class_name="car",
        confidence=0.9,
        bbox=(50, 50, 150, 150),
        track_id=7,
        center=(100.0, 100.0),
        area=10_000.0,
    )
    # Establish track state.
    tracker.update([detection], priority_class="car", frame_size=(640, 480))
    # Corrupt the covariance to a singular zero matrix.
    assert tracker._state is not None
    tracker._state.p = np.zeros((4, 4))

    # This call must not raise even though the Kalman update has a singular matrix.
    result = tracker.update([detection], priority_class="car", frame_size=(640, 480))
    assert result is not None
