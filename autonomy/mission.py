from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from autonomy.contracts import (
    MissionContext,
    MissionMode,
    MissionState,
    MotionIntent,
    MotionPrimitive,
    TrackedTarget,
)

logger = logging.getLogger("skytrackvision.autonomy.mission")


class InvalidTransitionError(RuntimeError):
    """Raised when a state transition is not allowed by the mission graph."""


@dataclass(slots=True)
class MissionFSM:
    initial_state: MissionState = MissionState.IDLE
    _state: MissionState = field(init=False)
    _prev_state: MissionState | None = field(default=None, init=False)
    _transition_log: list[tuple[str, float, str]] = field(default_factory=list, init=False)
    _state_entered_at: float = field(default=0.0, init=False)
    _timeout_recovery_cooldown: float = field(default=0.0, init=False)

    _transitions: dict[MissionState, list[MissionState]] = field(
        default_factory=lambda: {
            MissionState.IDLE: [MissionState.SCAN, MissionState.TRACK, MissionState.EMERGENCY],
            MissionState.SCAN: [
                MissionState.TRACK,
                MissionState.REPORT,
                MissionState.BLOCKED,
                MissionState.IDLE,
                MissionState.EMERGENCY,
            ],
            MissionState.TRACK: [
                MissionState.REACQUIRE,
                MissionState.MONITOR,
                MissionState.ORBIT,
                MissionState.SCAN,
                MissionState.BLOCKED,
                MissionState.IDLE,
                MissionState.EMERGENCY,
            ],
            MissionState.REACQUIRE: [
                MissionState.TRACK,
                MissionState.SCAN,
                MissionState.BLOCKED,
                MissionState.EMERGENCY,
            ],
            MissionState.MONITOR: [
                MissionState.REPORT,
                MissionState.TRACK,
                MissionState.ORBIT,
                MissionState.BLOCKED,
                MissionState.EMERGENCY,
            ],
            MissionState.ORBIT: [
                MissionState.TRACK,
                MissionState.BLOCKED,
                MissionState.EMERGENCY,
            ],
            MissionState.REPORT: [MissionState.IDLE, MissionState.EMERGENCY],
            MissionState.BLOCKED: [MissionState.IDLE, MissionState.SCAN, MissionState.EMERGENCY],
            # EMERGENCY is a safe sink: once recovered the mission resets to IDLE.
            MissionState.EMERGENCY: [MissionState.IDLE],
        },
        init=False,
    )

    _state_timeouts: dict[MissionState, float] = field(
        default_factory=lambda: {
            MissionState.BLOCKED: 30.0,
            MissionState.SCAN: 120.0,
            MissionState.REACQUIRE: 8.0,  # Phase 4: focused re-search timeout
        },
        init=False,
    )

    def __post_init__(self) -> None:
        self._state = self.initial_state
        self._state_entered_at = time.monotonic()

    @property
    def state(self) -> MissionState:
        return self._state

    def transition(self, to: MissionState, reason: str = "") -> None:
        if to == self._state:
            logger.debug("FSM no-op: already in %s (reason=%s)", self._state.value, reason)
            return
        if to not in self._transitions.get(self._state, []):
            raise InvalidTransitionError(f"{self._state} -> {to} is not allowed")
        self._prev_state = self._state
        self._state = to
        self._state_entered_at = time.monotonic()
        self._transition_log.append((to.value, self._state_entered_at, reason))

    def emergency(self, reason: str = "") -> bool:
        """Force an unconditional transition into EMERGENCY from any state.

        Unlike :meth:`transition`, this never raises: the whole point of the
        emergency path is that it must succeed from *any* state — including
        states the transition graph does not anticipate — so an unattended
        mission can always reach a safe abort. Returns ``False`` if already in
        EMERGENCY (no-op), ``True`` if a transition was performed.
        """
        if self._state == MissionState.EMERGENCY:
            return False
        self._prev_state = self._state
        self._state = MissionState.EMERGENCY
        self._state_entered_at = time.monotonic()
        self._transition_log.append(
            (MissionState.EMERGENCY.value, self._state_entered_at, reason or "emergency")
        )
        logger.warning("EMERGENCY engaged from %s (reason=%s)", self._prev_state.value, reason)
        return True

    def check_timeout_recovery(self) -> MissionState | None:
        """Check if current state has timed out and return recovery state if needed."""
        # Cooldown to prevent immediate re-entry to timed-out state
        if time.monotonic() - self._timeout_recovery_cooldown < 10.0:
            return None

        timeout = self._state_timeouts.get(self._state)
        if timeout is None:
            return None

        time_in_state = time.monotonic() - self._state_entered_at
        if time_in_state > timeout:
            if self._state == MissionState.REACQUIRE:
                recovery_state = MissionState.SCAN  # Focused search failed → full scan
            elif self._state == MissionState.BLOCKED:
                recovery_state = MissionState.SCAN
            else:
                recovery_state = MissionState.IDLE
            self._timeout_recovery_cooldown = time.monotonic()
            logger.warning(
                "State timeout: %s -> %s (%.1fs in state, timeout=%.1fs)",
                self._state.value,
                recovery_state.value,
                time_in_state,
                timeout,
            )
            return recovery_state
        return None

    def get_motion_intent(
        self,
        context: MissionContext,
        target: TrackedTarget | None,
    ) -> MotionIntent:
        if context.mode == MissionMode.TRAFFIC_MONITOR and self._state in {
            MissionState.TRACK,
            MissionState.REACQUIRE,
            MissionState.ORBIT,
        }:
            if self._state != MissionState.SCAN:
                try:
                    self.transition(MissionState.SCAN, reason="traffic_monitor_scan_only")
                except InvalidTransitionError:
                    self._prev_state = self._state
                    self._state = MissionState.SCAN
                    self._state_entered_at = time.monotonic()
                    self._transition_log.append(
                        (
                            MissionState.SCAN.value,
                            self._state_entered_at,
                            "traffic_monitor_forced_scan",
                        )
                    )
            return MotionIntent(
                primitive=MotionPrimitive.SCAN,
                reason="traffic counting missions stay in scan patrol",
            )

        # Check for timeout recovery
        recovery_state = self.check_timeout_recovery()
        if recovery_state is not None:
            try:
                self.transition(recovery_state, reason=f"timeout_recovery from {self._state.value}")
            except InvalidTransitionError:
                # Recovery target not reachable from current state — fall back to SCAN
                if self._state not in (MissionState.IDLE, MissionState.REPORT):
                    _from = self._state.value  # capture BEFORE mutation for accurate logging
                    self._prev_state = self._state
                    self._state = MissionState.SCAN
                    self._state_entered_at = time.monotonic()
                    self._transition_log.append(
                        (MissionState.SCAN.value, self._state_entered_at, "forced_scan_recovery")
                    )
                    logger.warning(
                        "Forced state to SCAN after InvalidTransitionError (attempted %s -> %s)",
                        _from,
                        recovery_state.value,
                    )

        # Grace period for target loss - use prediction if target recently seen
        if (
            self._state == MissionState.TRACK
            and target
            and not target.is_confirmed
            and target.frames_since_seen > 0
            and target.frames_since_seen < 10
        ):
            # Target lost recently, use predicted position
            return MotionIntent(
                primitive=MotionPrimitive.FOLLOW,
                target_id=target.track_id,
                reason="tracking_prediction",
            )

        match self._state:
            case MissionState.EMERGENCY:
                # Deterministic safe stop; the runtime watchdog/pilot decides
                # whether to hold, return home, or land from here.
                return MotionIntent(primitive=MotionPrimitive.HOVER, reason="emergency safe-stop")
            case MissionState.SCAN:
                return MotionIntent(primitive=MotionPrimitive.SCAN, reason="scan area")
            case MissionState.TRACK if target and target.is_confirmed:
                return MotionIntent(
                    primitive=MotionPrimitive.FOLLOW,
                    target_id=target.track_id,
                    reason="confirmed target",
                )
            case MissionState.TRACK:
                return MotionIntent(primitive=MotionPrimitive.HOVER, reason="target not confirmed")
            case MissionState.REACQUIRE if target and target.is_confirmed:
                # Target re-found during REACQUIRE → will transition back to TRACK
                return MotionIntent(
                    primitive=MotionPrimitive.FOLLOW,
                    target_id=target.track_id,
                    reason="reacquired target",
                )
            case MissionState.REACQUIRE:
                return MotionIntent(
                    primitive=MotionPrimitive.REACQUIRE,
                    reason="focused re-search in last known direction",
                )
            case MissionState.MONITOR:
                return MotionIntent(primitive=MotionPrimitive.HOVER, reason="monitoring")
            case MissionState.BLOCKED:
                return MotionIntent(primitive=MotionPrimitive.HOVER, reason="blocked")
            case MissionState.ORBIT:
                return MotionIntent(
                    primitive=MotionPrimitive.ORBIT,
                    target_id=context.target_id,
                    reason="orbit target",
                )
            case _:
                return MotionIntent(primitive=MotionPrimitive.HOVER, reason="idle")
