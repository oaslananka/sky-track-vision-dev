"""Mission scoring harness — turn raw mission runs into a reproducible scorecard.

You cannot claim a pilot "completes missions unattended" without measuring it.
This module provides the measurement framework the project was missing: it takes
a set of :class:`MissionTrial` records (task + finished report + how many times a
human had to intervene + how many safety violations occurred) and produces an
aggregate :class:`Scorecard` with the metrics that matter for autonomy:

* **success rate** — protocol *and* semantic objectives met (re-verified here,
  not merely trusting the runner) AND zero human interventions;
* **zero-intervention rate** — the headline number for "no human in the loop";
* **safety-violation rate** — fraction of trials with a hard safety trip.

The scoring is AirSim-free and deterministic, so it is unit-testable and can wrap
either live AirSim runs or offline replays. ``scripts/benchmark.py`` is a thin CLI
over it; the live-AirSim trial collector is a small adapter left to the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autonomy.contracts import MissionReport
from autonomy.mission_spec import MissionVerifier, parse_mission_spec


@dataclass(slots=True, frozen=True)
class MissionTrial:
    """One mission run, as collected from a live or replayed mission."""

    task: str
    report: MissionReport
    interventions: int = 0  # human takeovers during the run (target: 0)
    safety_violations: int = 0  # hard safety/watchdog trips that ended the mission badly
    wall_clock_s: float = 0.0


@dataclass(slots=True, frozen=True)
class TrialOutcome:
    task: str
    success: bool
    objectives_passed: bool
    objectives_summary: str
    interventions: int
    safety_violations: int
    completion_reason: str


@dataclass(slots=True, frozen=True)
class Scorecard:
    n: int
    success_rate: float
    zero_intervention_rate: float
    safety_violation_rate: float
    mean_wall_clock_s: float
    outcomes: tuple[TrialOutcome, ...] = field(default_factory=tuple)

    def format(self) -> str:
        lines = [
            "Mission Benchmark Scorecard",
            "=" * 48,
            f"trials                : {self.n}",
            f"success rate          : {self.success_rate * 100:5.1f}%  "
            "(objectives met + 0 interventions)",
            f"zero-intervention rate: {self.zero_intervention_rate * 100:5.1f}%",
            f"safety-violation rate : {self.safety_violation_rate * 100:5.1f}%",
            f"mean wall-clock       : {self.mean_wall_clock_s:6.1f}s",
            "-" * 48,
        ]
        for outcome in self.outcomes:
            mark = "PASS" if outcome.success else "FAIL"
            lines.append(
                f"[{mark}] {outcome.task[:46]!r} "
                f"(obj: {outcome.objectives_summary}, "
                f"interventions: {outcome.interventions})"
            )
        return "\n".join(lines)


def score_trial(trial: MissionTrial, verifier: MissionVerifier | None = None) -> TrialOutcome:
    """Score a single trial by re-verifying its objectives against its report."""
    verifier = verifier or MissionVerifier()
    spec = parse_mission_spec(trial.task)
    verdict = verifier.verify(spec, trial.report)
    # Clean success demands the run reported success, the objectives genuinely
    # pass on independent re-verification, and no human had to take over.
    success = trial.report.success and verdict.passed and trial.interventions == 0
    return TrialOutcome(
        task=trial.task,
        success=success,
        objectives_passed=verdict.passed,
        objectives_summary=verdict.summary,
        interventions=trial.interventions,
        safety_violations=trial.safety_violations,
        completion_reason=trial.report.completion_reason,
    )


def score_trials(trials: list[MissionTrial]) -> Scorecard:
    """Aggregate a batch of trials into a scorecard."""
    if not trials:
        return Scorecard(
            n=0,
            success_rate=0.0,
            zero_intervention_rate=0.0,
            safety_violation_rate=0.0,
            mean_wall_clock_s=0.0,
        )
    verifier = MissionVerifier()
    outcomes = tuple(score_trial(trial, verifier) for trial in trials)
    n = len(outcomes)
    successes = sum(1 for o in outcomes if o.success)
    zero_intervention = sum(1 for o in outcomes if o.interventions == 0)
    violations = sum(1 for o in outcomes if o.safety_violations > 0)
    mean_wall = sum(trial.wall_clock_s for trial in trials) / n
    return Scorecard(
        n=n,
        success_rate=successes / n,
        zero_intervention_rate=zero_intervention / n,
        safety_violation_rate=violations / n,
        mean_wall_clock_s=mean_wall,
        outcomes=outcomes,
    )


def _demo_report(
    *,
    duration_s: float,
    track_counts: dict[str, int] | None = None,
    vehicle_count: int = 0,
    success: bool,
    reason: str,
) -> MissionReport:
    return MissionReport(
        mission_id="demo",
        mode="SEARCH",
        state_transitions=[],
        target_ids_seen=[],
        completion_progress=1.0,
        success=success,
        completion_reason=reason,
        duration_s=duration_s,
        unique_track_counts=track_counts or {},
        unique_vehicle_count=vehicle_count,
    )


def demo_trials() -> list[MissionTrial]:
    """A small, self-contained scenario set illustrating the scoring methodology."""
    return [
        MissionTrial(
            task="Find a person, follow for 20 seconds, then land",
            report=_demo_report(
                duration_s=24.0, track_counts={"person": 1}, success=True, reason="REPORT"
            ),
            interventions=0,
            wall_clock_s=24.0,
        ),
        MissionTrial(
            task="Find a truck and follow it for 30 seconds",
            report=_demo_report(
                duration_s=33.0, track_counts={"car": 2}, success=False, reason="objectives_unmet"
            ),
            interventions=0,
            wall_clock_s=33.0,
        ),
        MissionTrial(
            task="Count the traffic and report",
            report=_demo_report(duration_s=60.0, vehicle_count=7, success=True, reason="REPORT"),
            interventions=1,  # needed a nudge -> not a clean success
            wall_clock_s=61.0,
        ),
    ]
