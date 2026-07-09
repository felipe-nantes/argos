"""Utilitários DICOM seguros para o registry de datasets."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pydicom

from dtwin.core import PipelineError


@dataclass(frozen=True)
class DicomSeries:
    series_uid_hash: str
    modality: str
    files: tuple[Path, ...]
    series_dir: Path
    series_description: str | None = None


def stable_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def iter_candidate_dicom_files(root: Path) -> Iterable[Path]:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            yield path


def read_dicom_header(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler cabeçalho DICOM: {path}") from exc


def discover_dicom_series(root: Path, *, modality: str = "MR") -> list[DicomSeries]:
    root = Path(root)
    grouped: dict[str, list[Path]] = {}
    metadata: dict[str, dict[str, str | None]] = {}
    ignored_non_matching = 0
    malformed = 0
    for path in iter_candidate_dicom_files(root):
        try:
            ds = read_dicom_header(path)
        except PipelineError:
            malformed += 1
            continue
        current_modality = str(getattr(ds, "Modality", "") or "").upper()
        if current_modality != modality:
            ignored_non_matching += 1
            continue
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if not series_uid:
            series_uid = f"path:{path.parent.resolve()}"
        grouped.setdefault(series_uid, []).append(path)
        metadata.setdefault(
            series_uid,
            {
                "modality": current_modality,
                "series_description": str(getattr(ds, "SeriesDescription", "") or "").strip() or None,
            },
        )

    if malformed and not grouped:
        raise PipelineError(f"Nenhuma série DICOM válida encontrada em {root}.")
    series: list[DicomSeries] = []
    for series_uid, files in sorted(grouped.items(), key=lambda item: stable_hash(item[0])):
        common = Path(files[0]).parent
        series.append(
            DicomSeries(
                series_uid_hash=stable_hash(series_uid, length=24),
                modality=str(metadata[series_uid]["modality"] or modality),
                files=tuple(sorted(files)),
                series_dir=common,
                series_description=metadata[series_uid]["series_description"],
            )
        )
    if not series:
        detail = " Arquivos de outra modalidade foram ignorados." if ignored_non_matching else ""
        raise PipelineError(f"Nenhuma série DICOM MR encontrada em {root}.{detail}")
    return series
