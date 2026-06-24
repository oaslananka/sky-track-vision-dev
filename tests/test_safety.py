from __future__ import annotations

from autonomy.safety import SafetyEvaluator
from config.settings import SafetyConfig
from tests.conftest import make_snapshot


def test_safety_overrides_when_connection_is_lost() -> None:
    evaluator = SafetyEvaluator(SafetyConfig())

    evaluation = evaluator.evaluate(make_snapshot(), connection_ok=False)

    assert evaluation.state.value == "SAFETY_OVERRIDE"
    assert not evaluation.allow_forward


def test_safety_first_evaluation_with_no_sensors_returns_override() -> None:
    """With no prior safe state, proximity failure must be SAFETY_OVERRIDE (not LANDING_CAUTION)."""
    evaluator = SafetyEvaluator(SafetyConfig())

    first = evaluator.evaluate(make_snapshot(proximity_available=False), connection_ok=True)

    assert first.state.value == "SAFETY_OVERRIDE"
    assert not first.allow_descent
    assert not first.allow_forward


def test_safety_applies_grace_using_last_good_state_when_proximity_fails() -> None:
    """When a prior good evaluation exists, the grace period returns it on transient failure."""
    evaluator = SafetyEvaluator(SafetyConfig())

    # Establish a good evaluation first.
    good = evaluator.evaluate(make_snapshot(), connection_ok=True)
    assert good.state.value == "PATH_CLEAR"

    # Now proximity fails transiently — should fall back to the last good state.
    first = evaluator.evaluate(make_snapshot(proximity_available=False), connection_ok=True)
    second = evaluator.evaluate(make_snapshot(proximity_available=False), connection_ok=True)
    third = evaluator.evaluate(make_snapshot(proximity_available=False), connection_ok=True)

    assert first.state.value == "PATH_CLEAR"  # last good evaluation returned
    assert second.state.value == "PATH_CLEAR"  # still within grace period (count < 3)
    assert third.state.value == "SAFETY_OVERRIDE"  # count >= 3 → override
    assert "all" in third.blocked_directions


def test_safety_blocks_forward_motion_on_front_obstacle() -> None:
    evaluator = SafetyEvaluator(SafetyConfig(obstacle_front_threshold_m=3.0))

    evaluation = evaluator.evaluate(make_snapshot(front_m=2.0), connection_ok=True)

    assert evaluation.state.value == "OBSTACLE_AHEAD"
    assert not evaluation.allow_forward


def test_safety_disallows_descent_when_altitude_is_too_low() -> None:
    evaluator = SafetyEvaluator(SafetyConfig(altitude_min_m=1.5))

    evaluation = evaluator.evaluate(make_snapshot(down_m=1.0), connection_ok=True)

    assert evaluation.state.value == "ALTITUDE_LOW"
    assert not evaluation.allow_descent


def test_safety_uses_lidar_clusters_as_forward_blocker() -> None:
    evaluator = SafetyEvaluator(SafetyConfig(lidar_cluster_threshold=4))

    evaluation = evaluator.evaluate(
        make_snapshot(lidar_clusters=5, min_distance_m=1.5),
        connection_ok=True,
    )

    assert evaluation.state.value == "OBSTACLE_AHEAD"
    assert "front" in evaluation.blocked_directions


def test_safety_ignores_distant_lidar_clusters_when_path_is_clear() -> None:
    evaluator = SafetyEvaluator(SafetyConfig(lidar_cluster_threshold=4))

    evaluation = evaluator.evaluate(
        make_snapshot(front_m=40.0, lidar_clusters=200, min_distance_m=18.0),
        connection_ok=True,
    )

    assert evaluation.state.value == "PATH_CLEAR"
    assert evaluation.allow_forward


def test_safety_allows_clear_path() -> None:
    evaluator = SafetyEvaluator(SafetyConfig())

    evaluation = evaluator.evaluate(make_snapshot(), connection_ok=True)

    assert evaluation.state.value == "PATH_CLEAR"
    assert evaluation.allow_forward
    assert evaluation.allow_descent
