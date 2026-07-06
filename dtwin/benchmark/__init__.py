"""Núcleo auditável do benchmark MedGemma do ARGOS."""

from .metrics import compute_benchmark_metrics, wilson_interval
from .models import (
    BenchmarkCaseResult,
    BenchmarkStatus,
    EvaluationCase,
    GroundTruthLabel,
    InferenceCase,
    ModelResult,
)

__all__ = [
    "BenchmarkCaseResult",
    "BenchmarkStatus",
    "EvaluationCase",
    "GroundTruthLabel",
    "InferenceCase",
    "ModelResult",
    "compute_benchmark_metrics",
    "wilson_interval",
]
