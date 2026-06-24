from __future__ import annotations

import logging
import time

from autonomy.contracts import SafetyEvaluation, SafetyState, SensorSnapshot
from config.runtime_logging import log_event
from config.settings import SafetyConfig

logger = logging.getLogger("skytrackvision.safety")


class SafetyEvaluator:
    """Deterministic safety gate that can always veto unsafe motion."""

    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg
        self._proximity_fail_count = 0
        self._last_good_evaluation: SafetyEvaluation | None = None

    def evaluate(self, snapshot: SensorSnapshot, connection_ok: bool) -> SafetyEvaluation:
        if not connection_ok:
            result = SafetyEvaluation(
                state=SafetyState.SAFETY_OVERRIDE,
                blocked_directions=["all"],
                allow_forward=False,
                allow_descent=False,
                reason="AirSim connection lost",
            )
            self._log_evaluation(snapshot, result)
            return result

        # Phase 5: Sensor staleness detection
        age_ms = (time.time_ns() - snapshot.timestamp_ns) / 1_000_000
        if age_ms > self._cfg.max_sensor_age_ms:
            result = SafetyEvaluation(
                state=SafetyState.SAFETY_OVERRIDE,
                blocked_directions=["all"],
                allow_forward=False,
                allow_descent=False,
                reason=f"Sensor data stale: {age_ms:.0f}ms old",
            )
            self._log_evaluation(snapshot, result)
            return result

        if not snapshot.proximity.available:
            self._proximity_fail_count += 1
            if self._proximity_fail_count < 3:
                result = self._last_good_evaluation or SafetyEvaluation(
                    state=SafetyState.SAFETY_OVERRIDE,
                    blocked_directions=["all"],
                    allow_forward=False,
                    allow_descent=False,
                    reason="Proximity sensors unavailable (no prior safe state)",
                )
            else:
                result = SafetyEvaluation(
                    state=SafetyState.SAFETY_OVERRIDE,
                    blocked_directions=["all"],
                    allow_forward=False,
                    allow_descent=False,
                    reason="Proximity sensors unavailable",
                )
            self._log_evaluation(snapshot, result)
            return result
        self._proximity_fail_count = 0

        # Velocity-aware dynamic threshold for forward obstacles
        forward_velocity = abs(snapshot.telemetry.velocity_ned[0])  # NED X = forward
        max_decel = self._cfg.max_decel
        stopping_distance = (
            (forward_velocity**2) / (2 * max_decel) if forward_velocity > 0.1 else 0.0
        )
        dynamic_front_threshold = self._cfg.obstacle_front_threshold_m + stopping_distance

        blocked: list[str] = []
        prox = snapshot.proximity
        if prox.front_m < dynamic_front_threshold:
            blocked.append("front")
        if prox.left_m < self._cfg.obstacle_side_threshold_m:
            blocked.append("left")
        if prox.right_m < self._cfg.obstacle_side_threshold_m:
            blocked.append("right")
        lidar_cluster_near = (
            snapshot.lidar.cluster_count > self._cfg.lidar_cluster_threshold
            and snapshot.lidar.min_distance_m
            < dynamic_front_threshold * self._cfg.lidar_cluster_distance_factor
        )
        if lidar_cluster_near:
            blocked.append("front")

        # Phase 5: Rear collision check when moving backward
        backward_velocity = snapshot.telemetry.velocity_ned[0]
        if backward_velocity < -0.3 and prox.rear_m < self._cfg.obstacle_rear_threshold_m:
            blocked.append("rear")
        if prox.down_m < self._cfg.altitude_min_m:
            result = SafetyEvaluation(
                state=SafetyState.ALTITUDE_LOW,
                blocked_directions=blocked,
                allow_forward=not blocked,
                allow_descent=False,
                reason=f"Altitude critical: {prox.down_m:.1f}m",
            )
            self._log_evaluation(snapshot, result)
            return result
        if prox.down_m < self._cfg.altitude_min_m + self._cfg.landing_caution_m:
            result = SafetyEvaluation(
                state=SafetyState.LANDING_CAUTION,
                blocked_directions=blocked,
                allow_forward=not blocked,
                allow_descent=True,
                reason=f"Low altitude caution: {prox.down_m:.1f}m",
            )
            self._log_evaluation(snapshot, result)
            return result
        if blocked:
            result = SafetyEvaluation(
                state=SafetyState.OBSTACLE_AHEAD,
                blocked_directions=blocked,
                allow_forward="front" not in blocked,
                allow_descent=True,
                reason=f"Obstacle detected: {', '.join(blocked)}",
            )
            self._log_evaluation(snapshot, result)
            return result
        result = SafetyEvaluation(
            state=SafetyState.PATH_CLEAR,
            blocked_directions=[],
            allow_forward=True,
            allow_descent=True,
            reason="Path clear",
        )
        self._last_good_evaluation = result
        self._log_evaluation(snapshot, result)
        return result

    def _log_evaluation(self, snapshot: SensorSnapshot, result: SafetyEvaluation) -> None:
        log_event(
            logger,
            logging.DEBUG,
            "safety.evaluation",
            "Safety evaluation completed",
            safety_state=result.state.value,
            allow_forward=result.allow_forward,
            allow_descent=result.allow_descent,
            blocked_directions=result.blocked_directions,
            reason=result.reason,
            front_m=round(snapshot.proximity.front_m, 3),
            down_m=round(snapshot.proximity.down_m, 3),
            lidar_cluster_count=snapshot.lidar.cluster_count,
            lidar_min_distance_m=round(snapshot.lidar.min_distance_m, 3),
            snapshot_age_ms=round((time.time_ns() - snapshot.timestamp_ns) / 1_000_000, 3),
        )
