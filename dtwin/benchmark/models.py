"""Tipos que separam entradas de inferência e ground truth protegido."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class GroundTruthLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ModelResult(str, Enum):
    POSITIVE = "POSITIVA"
    NEGATIVE = "NEGATIVA"
    INCONCLUSIVE = "INCONCLUSIVA"


class BenchmarkStatus(str, Enum):
    DECISIVE = "decisive"
    INCONCLUSIVE = "inconclusive"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class InferenceCase:
    """Entrada sanitizada. Deliberadamente não possui label nem lesão."""

    case_id: str
    dataset: str
    input_format: str
    volume_path: Path
    organ_mask_path: Path
    manifest_path: Path
    workspace: Path
    input_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        forbidden = {"label", "lesion_mask", "lesion_mask_path", "annotations"}
        leaked = forbidden.intersection(vars(self))
        if leaked:
            raise ValueError(f"InferenceCase contém ground truth proibido: {sorted(leaked)}")


@dataclass(frozen=True)
class EvaluationCase:
    """Ground truth mantido fora do workspace entregue à inferência."""

    inference: InferenceCase
    label: GroundTruthLabel
    lesion_mask_path: Path | None = None
    annotation_manifest_path: Path | None = None
    protected_ground_truth_hashes: dict[str, str | None] = field(default_factory=dict)


@dataclass
class BenchmarkCaseResult:
    case_id: str
    dataset: str
    input_format: str
    truth: GroundTruthLabel
    status: BenchmarkStatus
    prediction: ModelResult | None = None
    confidence: str | None = None
    input_hashes: dict[str, str] = field(default_factory=dict)
    protected_ground_truth_hashes: dict[str, str | None] = field(default_factory=dict)
    durations_seconds: dict[str, float | None] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    report_path: str | None = None
    panel_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary_decision(self) -> bool:
        return self.status is BenchmarkStatus.DECISIVE and self.prediction in {
            ModelResult.POSITIVE,
            ModelResult.NEGATIVE,
        }

    @property
    def is_correct_primary(self) -> bool:
        return bool(
            (self.truth is GroundTruthLabel.POSITIVE and self.prediction is ModelResult.POSITIVE)
            or (self.truth is GroundTruthLabel.NEGATIVE and self.prediction is ModelResult.NEGATIVE)
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "input_format": self.input_format,
            "truth": self.truth.value,
            "ground_truth_label": self.truth.value.upper(),
            "prediction": self.prediction.value if self.prediction else None,
            "model_result": self.prediction.value if self.prediction else self.status.value.upper(),
            "status": self.status.value,
            "is_correct_primary": self.is_correct_primary,
            "correct": self.is_correct_primary if self.is_binary_decision else None,
            "is_binary_decision": self.is_binary_decision,
            "used_for_decisions_only": self.is_binary_decision,
            "confidence": self.confidence,
            "input_hashes": self.input_hashes,
            "protected_ground_truth_hashes": self.protected_ground_truth_hashes,
            "durations_seconds": self.durations_seconds,
            "error": {"type": self.error_type, "message": self.error_message},
            "report_path": self.report_path,
            "panel_path": self.panel_path,
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BenchmarkCaseResult":
        truth_raw = str(value.get("truth") or value.get("ground_truth_label") or "").lower()
        prediction_raw = value.get("prediction")
        status_raw = str(value.get("status") or "failure").lower()
        status_aliases = {"failed": "failure", "invalid": "invalid_response"}
        status_raw = status_aliases.get(status_raw, status_raw)
        error = value.get("error")
        durations = dict(value.get("durations_seconds") or {}) if isinstance(value.get("durations_seconds"), dict) else {}
        if not durations and isinstance(value.get("duration_seconds"), (int, float)):
            durations["total"] = float(value["duration_seconds"])
        inferred_error_type = None
        if status_raw in {"failure", "timeout", "invalid_response"}:
            inferred_error_type = status_raw.upper()
        return cls(
            case_id=str(value.get("case_id") or "unknown"),
            dataset=str(value.get("dataset") or "unknown"),
            input_format=str(value.get("input_format") or "DICOM"),
            truth=GroundTruthLabel(truth_raw),
            status=BenchmarkStatus(status_raw),
            prediction=ModelResult(str(prediction_raw).upper()) if prediction_raw else None,
            confidence=value.get("confidence"),
            input_hashes=dict(value.get("input_hashes") or {}),
            protected_ground_truth_hashes=dict(value.get("protected_ground_truth_hashes") or {}),
            durations_seconds=durations,
            error_type=(error.get("type") if isinstance(error, dict) else inferred_error_type),
            error_message=(error.get("message") if isinstance(error, dict) else str(error or "")) or None,
            report_path=value.get("report_path"),
            panel_path=value.get("panel_path"),
        )
