from __future__ import annotations

from autonomy.benchmark import MissionTrial, demo_trials, score_trial, score_trials
from autonomy.contracts import MissionReport


def _report(
    *, success: bool, reason: str, duration_s: float, track_counts: dict[str, int]
) -> MissionReport:
    return MissionReport(
        mission_id="t",
        mode="SEARCH",
        state_transitions=[],
        target_ids_seen=[],
        completion_progress=1.0,
        success=success,
        completion_reason=reason,
        duration_s=duration_s,
        unique_track_counts=track_counts,
    )


def test_clean_success_requires_objectives_and_zero_interventions() -> None:
    trial = MissionTrial(
        task="Find a person, follow for 10 seconds",
        report=_report(success=True, reason="REPORT", duration_s=12.0, track_counts={"person": 1}),
        interventions=0,
    )
    assert score_trial(trial).success is True


def test_intervention_demotes_an_otherwise_passing_run() -> None:
    trial = MissionTrial(
        task="Find a person, follow for 10 seconds",
        report=_report(success=True, reason="REPORT", duration_s=12.0, track_counts={"person": 1}),
        interventions=1,
    )
    assert score_trial(trial).success is False


def test_objectives_reverified_even_if_runner_claims_success() -> None:
    # Runner says success, but the truck was never seen -> harness overrules it.
    trial = MissionTrial(
        task="Find a truck and follow it",
        report=_report(success=True, reason="REPORT", duration_s=12.0, track_counts={"car": 3}),
        interventions=0,
    )
    outcome = score_trial(trial)
    assert outcome.objectives_passed is False
    assert outcome.success is False


def test_scorecard_aggregates_rates() -> None:
    card = score_trials(demo_trials())

    assert card.n == 3
    # demo set: 1 clean success, 1 objectives_unmet, 1 success-but-intervened
    assert abs(card.success_rate - 1 / 3) < 1e-9
    assert abs(card.zero_intervention_rate - 2 / 3) < 1e-9
    assert "Scorecard" in card.format()


def test_empty_scorecard_is_safe() -> None:
    card = score_trials([])
    assert card.n == 0
    assert card.success_rate == 0.0
