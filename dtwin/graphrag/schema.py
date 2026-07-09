"""Schema lógico e relações iniciais do metadata GraphRAG."""
from __future__ import annotations

from typing import Any

from dtwin.core import PipelineError


TARGET_FINDING = "focal_liver_lesion"
REGISTRY_SCHEMA = "argos-dataset-registry-v1"

MIMIC_RELATIONS: tuple[tuple[str, str], ...] = (
    ("prominent_hepatic_vein", "focal_liver_lesion"),
    ("vascular_structure", "focal_liver_lesion"),
    ("perfusion_alteration", "arterial_hyperenhancement"),
    ("motion_artifact", "focal_lesion"),
    ("partial_volume_effect", "focal_lesion"),
    ("focal_fat", "focal_liver_lesion"),
    ("simple_cyst", "hypovascular_lesion"),
    ("edge_of_liver_pseudolesion", "focal_liver_lesion"),
)

NEGATIVE_TARGET_RELATIONS: tuple[tuple[str, str], ...] = (
    ("normal", TARGET_FINDING),
    ("benign_anatomic_variant", TARGET_FINDING),
    ("pseudolesion_or_artifact", TARGET_FINDING),
    ("benign_non_target_finding", TARGET_FINDING),
    ("poor_quality_non_diagnostic", TARGET_FINDING),
)

CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS FOR (n:Dataset) REQUIRE n.dataset_id IS UNIQUE",
    "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
    "CREATE CONSTRAINT rag_class_name_unique IF NOT EXISTS FOR (n:RagClass) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT ground_truth_label_unique IF NOT EXISTS FOR (n:GroundTruthLabel) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT negative_subtype_unique IF NOT EXISTS FOR (n:NegativeSubtype) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT positive_subtype_unique IF NOT EXISTS FOR (n:PositiveSubtype) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT phenotype_tag_unique IF NOT EXISTS FOR (n:PhenotypeTag) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT finding_unique IF NOT EXISTS FOR (n:Finding) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT modality_unique IF NOT EXISTS FOR (n:Modality) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT source_format_unique IF NOT EXISTS FOR (n:SourceFormat) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT phase_unique IF NOT EXISTS FOR (n:Phase) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT limitation_unique IF NOT EXISTS FOR (n:Limitation) REQUIRE n.text IS UNIQUE",
)


def validate_registry_record(record: dict[str, Any]) -> None:
    if record.get("schema") != REGISTRY_SCHEMA:
        raise PipelineError("Registro de dataset possui schema inválido para GraphRAG.")
    if record.get("clinical_use_allowed") is not False or record.get("research_only") is not True:
        raise PipelineError("GraphRAG aceita somente registros research_only sem uso clínico.")
    if record.get("modality") != "MR":
        raise PipelineError("GraphRAG hepático v1 aceita somente registros MR.")
    forbidden = {"seriesinstanceuid", "studyinstanceuid", "sopinstanceuid", "patientname", "patientid"}
    leaked = _forbidden_keys(record, forbidden)
    if leaked:
        raise PipelineError(f"Registro contém metadado bruto proibido: {leaked}")


def _forbidden_keys(value: Any, forbidden: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
            if normalized in forbidden:
                found.append(str(key))
            found.extend(_forbidden_keys(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.extend(_forbidden_keys(child, forbidden))
    return found


def registry_record_to_graph_params(record: dict[str, Any]) -> dict[str, Any]:
    validate_registry_record(record)
    metadata = dict(record.get("metadata") or {})
    metadata.pop("raw_path", None)
    return {
        "case_id": record["case_id"],
        "series_id": record.get("series_id"),
        "dataset_id": record["dataset_id"],
        "dataset_name": record["dataset_name"],
        "rag_class": record["rag_class"],
        "label": record["label"],
        "negative_subtype": record.get("negative_subtype"),
        "positive_subtype": record.get("positive_subtype"),
        "phenotype_tags": list(record.get("phenotype_tags") or []),
        "modality": record.get("modality") or "MR",
        "source_format": record.get("source_format"),
        "phase": record.get("sequence_or_phase") or "unknown",
        "limitations": list(record.get("limitations") or []),
        "warnings": list(record.get("warnings") or []),
        "review_status": record.get("review_status") or "pending_review",
        "has_segmentation": bool(record.get("has_segmentation")),
        "research_only": True,
        "clinical_use_allowed": False,
        "metadata": metadata,
    }
