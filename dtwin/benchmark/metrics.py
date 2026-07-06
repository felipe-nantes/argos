"""Métricas puras do benchmark, sem dependência do webapp ou do modelo."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Iterable

from .models import BenchmarkCaseResult, BenchmarkStatus, GroundTruthLabel, ModelResult


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> dict[str, float] | None:
    if total <= 0:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence deve estar entre 0 e 1")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "low": round(max(0.0, centre - margin), 4),
        "high": round(min(1.0, centre + margin), 4),
    }


def _coerce(results: Iterable[BenchmarkCaseResult | dict[str, Any]]) -> list[BenchmarkCaseResult]:
    return [item if isinstance(item, BenchmarkCaseResult) else BenchmarkCaseResult.from_mapping(item) for item in results]


def _undefined_reasons(*, positives: int, negatives: int, predicted_positive: int, decisions: int) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if not positives:
        reasons["sensitivity"] = "no_positive_cases"
    if not negatives:
        reasons["specificity"] = "no_negative_cases"
    if not predicted_positive:
        reasons["precision"] = "no_penalized_positive_predictions"
    if not decisions:
        reasons["decisions_only"] = "no_binary_decisions"
    return reasons


def compute_benchmark_metrics(
    results: Iterable[BenchmarkCaseResult | dict[str, Any]],
    *,
    minimum_sensitivity: float = 0.75,
    minimum_specificity: float = 0.75,
    confidence: float = 0.95,
) -> dict[str, Any]:
    cases = _coerce(results)
    total = len(cases)
    positives = [c for c in cases if c.truth is GroundTruthLabel.POSITIVE]
    negatives = [c for c in cases if c.truth is GroundTruthLabel.NEGATIVE]
    decisive = [c for c in cases if c.is_binary_decision]

    tp = sum(c.prediction is ModelResult.POSITIVE for c in positives)
    tn = sum(c.prediction is ModelResult.NEGATIVE for c in negatives)
    penalized_fn = len(positives) - tp
    penalized_fp = len(negatives) - tn

    d_tp = sum(c.truth is GroundTruthLabel.POSITIVE and c.prediction is ModelResult.POSITIVE for c in decisive)
    d_tn = sum(c.truth is GroundTruthLabel.NEGATIVE and c.prediction is ModelResult.NEGATIVE for c in decisive)
    d_fp = sum(c.truth is GroundTruthLabel.NEGATIVE and c.prediction is ModelResult.POSITIVE for c in decisive)
    d_fn = sum(c.truth is GroundTruthLabel.POSITIVE and c.prediction is ModelResult.NEGATIVE for c in decisive)

    categorical = {
        truth.value: {
            "POSITIVA": sum(c.truth is truth and c.prediction is ModelResult.POSITIVE for c in cases),
            "NEGATIVA": sum(c.truth is truth and c.prediction is ModelResult.NEGATIVE for c in cases),
            "INCONCLUSIVA": sum(c.truth is truth and c.status is BenchmarkStatus.INCONCLUSIVE for c in cases),
            "FAILURE": sum(c.truth is truth and c.status is BenchmarkStatus.FAILURE for c in cases),
            "TIMEOUT": sum(c.truth is truth and c.status is BenchmarkStatus.TIMEOUT for c in cases),
            "INVALID_RESPONSE": sum(c.truth is truth and c.status is BenchmarkStatus.INVALID_RESPONSE for c in cases),
        }
        for truth in GroundTruthLabel
    }

    sensitivity = _ratio(tp, len(positives))
    specificity = _ratio(tn, len(negatives))
    accuracy = _ratio(tp + tn, total)
    precision = _ratio(tp, tp + penalized_fp)
    f1 = _ratio(2 * tp, 2 * tp + penalized_fp + penalized_fn)
    coverage = _ratio(len(decisive), total)
    target_passed = bool(
        sensitivity is not None
        and specificity is not None
        and sensitivity >= minimum_sensitivity
        and specificity >= minimum_specificity
    )

    auxiliary = {
        "inconclusive_count": sum(c.status is BenchmarkStatus.INCONCLUSIVE for c in cases),
        "failure_count": sum(c.status is BenchmarkStatus.FAILURE for c in cases),
        "timeout_count": sum(c.status is BenchmarkStatus.TIMEOUT for c in cases),
        "invalid_response_count": sum(c.status is BenchmarkStatus.INVALID_RESPONSE for c in cases),
    }
    completed = (
        total
        - auxiliary["failure_count"]
        - auxiliary["timeout_count"]
        - auxiliary["invalid_response_count"]
    )
    primary = {
        "scope": "primary_all_cases",
        "scoring_policy": "non_correct_result_counts_as_group_error",
        "total_cases": total,
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "completed_reports": completed,
        "decisive_cases": len(decisive),
        "inconclusive_cases": auxiliary["inconclusive_count"],
        "failed_cases": auxiliary["failure_count"] + auxiliary["timeout_count"] + auxiliary["invalid_response_count"],
        "completion_rate": _ratio(completed, total),
        "coverage_rate": coverage,
        "inconclusive_rate": _ratio(auxiliary["inconclusive_count"], total),
        "failure_rate": _ratio(auxiliary["failure_count"] + auxiliary["timeout_count"] + auxiliary["invalid_response_count"], total),
        "penalized_binary_scoring_matrix": {"tp": tp, "tn": tn, "fp": penalized_fp, "fn": penalized_fn},
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": penalized_fp, "fn": penalized_fn},
        "categorical_confusion_matrix": categorical,
        **auxiliary,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "confidence_intervals_95": {
            "accuracy": wilson_interval(tp + tn, total, confidence),
            "sensitivity": wilson_interval(tp, len(positives), confidence),
            "specificity": wilson_interval(tn, len(negatives), confidence),
            "precision": wilson_interval(tp, tp + penalized_fp, confidence),
            "coverage": wilson_interval(len(decisive), total, confidence),
            "f1": None,
        },
        "f1_ci_method": "not_implemented",
        "undefined_reasons": _undefined_reasons(
            positives=len(positives), negatives=len(negatives),
            predicted_positive=tp + penalized_fp, decisions=len(decisive),
        ),
    }

    decisions_only = {
        "scope": "secondary_decisions_only",
        "warning": (
            "As métricas decisions-only são secundárias e podem superestimar o desempenho, "
            "pois excluem casos inconclusivos, inválidos, não respondidos ou com falha."
        ),
        "total_cases": len(decisive),
        "confusion_matrix": {"tp": d_tp, "tn": d_tn, "fp": d_fp, "fn": d_fn},
        "accuracy": _ratio(d_tp + d_tn, len(decisive)),
        "sensitivity": _ratio(d_tp, d_tp + d_fn),
        "specificity": _ratio(d_tn, d_tn + d_fp),
        "precision": _ratio(d_tp, d_tp + d_fp),
        "f1_score": _ratio(2 * d_tp, 2 * d_tp + d_fp + d_fn),
    }
    gate = {
        "scope": "primary_all_cases",
        "minimum_sensitivity": minimum_sensitivity,
        "minimum_specificity": minimum_specificity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "sensitivity_passed": sensitivity is not None and sensitivity >= minimum_sensitivity,
        "specificity_passed": specificity is not None and specificity >= minimum_specificity,
        "passed": target_passed,
    }

    # Campos legados permanecem no topo para o webapp e consumidores atuais.
    return {**primary, "primary": primary, "decisions_only": decisions_only, "decisive_only": decisions_only, "gate": gate,
            "target": {"minimum_sensitivity": minimum_sensitivity, "minimum_specificity": minimum_specificity, "met": target_passed}}
