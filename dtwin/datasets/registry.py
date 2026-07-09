"""Ingestão de datasets públicos/locais para manifestos JSONL seguros."""
from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from dtwin.core import PipelineError

from .dicom_utils import discover_dicom_series, stable_hash
from .nifti_utils import discover_nifti_files
from .schema import CONFIG_SCHEMA, DatasetConfig, RegistryRecord, relative_path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Config de dataset inválida ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError(f"Config de dataset deve ser objeto YAML: {path}")
    return data


def _tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key) or []
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise PipelineError(f"{key} deve ser lista ou string.")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _optional(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def load_dataset_config(path: Path) -> DatasetConfig:
    data = _read_yaml(path)
    if data.get("schema") != CONFIG_SCHEMA:
        raise PipelineError(f"schema inválido em {path}: esperado {CONFIG_SCHEMA}.")
    config = DatasetConfig(
        dataset_id=str(data.get("dataset_id") or "").strip(),
        dataset_name=str(data.get("dataset_name") or "").strip(),
        rag_class=str(data.get("rag_class") or "").strip().lower(),
        label=str(data.get("label") or "").strip(),
        source_format=str(data.get("source_format") or "").strip().lower(),
        source_url=str(data.get("source_url") or "").strip(),
        negative_subtype=_optional(data.get("negative_subtype")),
        positive_subtype=_optional(data.get("positive_subtype")),
        phenotype_tags=_tuple(data, "phenotype_tags"),
        modality=str(data.get("modality") or "MR").strip().upper(),
        sequence_or_phase=str(data.get("sequence_or_phase") or "unknown").strip(),
        body_region=str(data.get("body_region") or "abdomen_liver").strip(),
        clinical_use_allowed=bool(data.get("clinical_use_allowed", False)),
        research_only=bool(data.get("research_only", True)),
        has_segmentation_default=bool(data.get("has_segmentation_default", False)),
        annotation_globs=_tuple(data, "annotation_globs"),
        exclude_globs=_tuple(data, "exclude_globs"),
        limitations=_tuple(data, "limitations"),
        warnings=_tuple(data, "warnings"),
        metadata=dict(data.get("metadata") or {}),
    )
    config.validate()
    return config


def _case_id(dataset_id: str, relative: str) -> str:
    digest = hashlib.sha256(f"{dataset_id}\0{relative}".encode("utf-8")).hexdigest()[:16]
    return f"{dataset_id}-{digest}"


def _find_annotation(root: Path, raw_path: Path, globs: tuple[str, ...]) -> Path | None:
    if not globs:
        return None
    raw_relative = raw_path.relative_to(root).as_posix()
    candidates = [path for path in root.rglob("*") if path.is_file()]
    for pattern in globs:
        patterns = (pattern, pattern[3:]) if pattern.startswith("**/") else (pattern,)
        for candidate in sorted(candidates):
            relative = candidate.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(relative, item) or fnmatch.fnmatch(candidate.name, item) for item in patterns):
                if relative != raw_relative:
                    return candidate
    return None


def _base_record(
    config: DatasetConfig,
    *,
    case_id: str,
    series_id: str | None,
    raw_path: str,
    annotation_path: str | None,
    has_segmentation: bool,
    metadata: dict[str, Any] | None = None,
) -> RegistryRecord:
    return RegistryRecord(
        case_id=case_id,
        series_id=series_id,
        dataset_id=config.dataset_id,
        dataset_name=config.dataset_name,
        rag_class=config.rag_class,
        label=config.label,
        negative_subtype=config.negative_subtype,
        positive_subtype=config.positive_subtype,
        phenotype_tags=list(config.phenotype_tags),
        modality=config.modality,
        source_format=config.source_format,
        dicom_original=config.source_format == "dicom",
        nifti_original=config.source_format == "nifti",
        derived_from=None,
        sequence_or_phase=config.sequence_or_phase,
        body_region=config.body_region,
        raw_path=raw_path,
        annotation_path=annotation_path,
        has_segmentation=has_segmentation,
        source_url=config.source_url,
        clinical_use_allowed=False,
        research_only=True,
        review_status="pending_review",
        limitations=list(config.limitations),
        warnings=list(config.warnings),
        metadata={**config.metadata, **(metadata or {})},
    )


def ingest_dicom_dataset(config: DatasetConfig, root: Path) -> list[RegistryRecord]:
    records: list[RegistryRecord] = []
    for series in discover_dicom_series(root, modality=config.modality):
        raw_relative = relative_path(root, series.series_dir)
        annotation = _find_annotation(root, series.series_dir, config.annotation_globs)
        records.append(
            _base_record(
                config,
                case_id=_case_id(config.dataset_id, raw_relative + series.series_uid_hash),
                series_id=series.series_uid_hash,
                raw_path=raw_relative,
                annotation_path=relative_path(root, annotation) if annotation else None,
                has_segmentation=bool(annotation) or config.has_segmentation_default,
                metadata={
                    "dicom_file_count": len(series.files),
                    "series_uid_sha256_prefix": series.series_uid_hash,
                    "series_description": series.series_description,
                },
            )
        )
    return records


def ingest_nifti_dataset(config: DatasetConfig, root: Path) -> list[RegistryRecord]:
    records: list[RegistryRecord] = []
    for path in discover_nifti_files(root, exclude_globs=config.exclude_globs):
        raw_relative = relative_path(root, path)
        annotation = _find_annotation(root, path, config.annotation_globs)
        records.append(
            _base_record(
                config,
                case_id=_case_id(config.dataset_id, raw_relative),
                series_id=None,
                raw_path=raw_relative,
                annotation_path=relative_path(root, annotation) if annotation else None,
                has_segmentation=bool(annotation) or config.has_segmentation_default,
                metadata={"nifti_filename": path.name},
            )
        )
    return records


def ingest_dataset_config(config: DatasetConfig, root: Path) -> list[RegistryRecord]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise PipelineError(f"Raiz do dataset não encontrada: {root}")
    if config.source_format == "dicom":
        return ingest_dicom_dataset(config, root)
    if config.source_format == "nifti":
        return ingest_nifti_dataset(config, root)
    raise PipelineError(f"source_format não suportado: {config.source_format}")


def write_jsonl(records: list[RegistryRecord], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    lines = [json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True) + "\n" for record in records]
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(out)
