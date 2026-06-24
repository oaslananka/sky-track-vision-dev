from __future__ import annotations

from autonomy.contracts import SafetyEvaluation, SensorSnapshot
from autonomy.safety import SafetyEvaluator


class SafetyAgent:
    """Facade over the deterministic safety evaluator."""

    def __init__(self, evaluator: SafetyEvaluator) -> None:
        self._evaluator = evaluator

    def evaluate(self, snapshot: SensorSnapshot, connection_ok: bool) -> SafetyEvaluation:
        return self._evaluator.evaluate(snapshot, connection_ok)
