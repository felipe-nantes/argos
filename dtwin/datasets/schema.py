"""Schema do registry de datasets hepáticos para RAG/benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dtwin.benchmark.models import NEGATIVE_SUBTYPES, PHENOTYPE_TAGS, POSITIVE_SUBTYPES
from dtwin.core import PipelineError


REGISTRY_SCHEMA = "argos-dataset-registry-v1"
CONFIG_SCHEMA = "argos-dataset-config-v1"
RAG_CLASSES = {"negative", "positive"}
SOURCE_FORMATS = {"dicom", "nifti"}
MODALITIES = {"MR"}


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    dataset_name: str
    rag_class: str
    label: str
    source_format: str
    source_url: str
    negative_subtype: str | None = None
    positive_subtype: str | None = None
    phenotype_tags: tuple[str, ...] = ()
    modality: str = "MR"
    sequence_or_phase: str = "unknown"
    body_region: str = "abdomen_liver"
    clinical_use_allowed: bool = False
    research_only: bool = True
    has_segmentation_default: bool = False
    annotation_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.dataset_id or not self.dataset_name:
            raise PipelineError("Config de dataset exige dataset_id e dataset_name.")
        if self.rag_class not in RAG_CLASSES:
            raise PipelineError(f"rag_class inválido: {self.rag_class!r}")
        if self.source_format not in SOURCE_FORMATS:
            raise PipelineError(f"source_format inválido: {self.source_format!r}")
        if self.modality not in MODALITIES:
            raise PipelineError("Registry hepático v1 aceita somente modalidade MR.")
        if self.clinical_use_allowed is not False or self.research_only is not True:
            raise PipelineError("Registry v1 deve ser research_only=true e clinical_use_allowed=false.")
        if self.rag_class == "negative":
            if self.positive_subtype is not None:
                raise PipelineError("positive_subtype não é permitido em dataset negative.")
            if self.negative_subtype is not None and self.negative_subtype not in NEGATIVE_SUBTYPES:
                raise PipelineError(f"negative_subtype inválido: {self.negative_subtype!r}")
        if self.rag_class == "positive":
            if self.negative_subtype is not None:
                raise PipelineError("negative_subtype não é permitido em dataset positive.")
            if self.positive_subtype is not None and self.positive_subtype not in POSITIVE_SUBTYPES:
                raise PipelineError(f"positive_subtype inválido: {self.positive_subtype!r}")
        invalid_tags = [tag for tag in self.phenotype_tags if tag not in PHENOTYPE_TAGS]
        if invalid_tags:
            raise PipelineError(f"phenotype_tags inválidas: {invalid_tags}")


@dataclass(frozen=True)
class RegistryRecord:
    case_id: str
    series_id: str | None
    dataset_id: str
    dataset_name: str
    rag_class: str
    label: str
    negative_subtype: str | None
    positive_subtype: str | None
    phenotype_tags: list[str]
    modality: str
    source_format: str
    dicom_original: bool
    nifti_original: bool
    derived_from: str | None
    sequence_or_phase: str
    body_region: str
    raw_path: str
    annotation_path: str | None
    has_segmentation: bool
    source_url: str
    clinical_use_allowed: bool
    research_only: bool
    review_status: str
    limitations: list[str]
    warnings: list[str]
    metadata: dict[str, Any]
    schema: str = REGISTRY_SCHEMA

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "case_id": self.case_id,
            "series_id": self.series_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "rag_class": self.rag_class,
            "label": self.label,
            "negative_subtype": self.negative_subtype,
            "positive_subtype": self.positive_subtype,
            "phenotype_tags": list(self.phenotype_tags),
            "modality": self.modality,
            "source_format": self.source_format,
            "dicom_original": self.dicom_original,
            "nifti_original": self.nifti_original,
            "derived_from": self.derived_from,
            "sequence_or_phase": self.sequence_or_phase,
            "body_region": self.body_region,
            "raw_path": self.raw_path,
            "annotation_path": self.annotation_path,
            "has_segmentation": self.has_segmentation,
            "source_url": self.source_url,
            "clinical_use_allowed": self.clinical_use_allowed,
            "research_only": self.research_only,
            "review_status": self.review_status,
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name
