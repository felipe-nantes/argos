"""Importadores declarativos para DICOM, NIfTI e MIDS explícito."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import SimpleITK as sitk
import yaml

from dtwin import stages
from dtwin.core import Case, PipelineError, load_profile, sha256_of
from dtwin.engine import Engine

from .hashing import input_hashes, sha256_paths
from .models import EvaluationCase, GroundTruthLabel, InferenceCase


SUPPORTED_FORMATS = {"DICOM", "NIFTI", "MIDS"}
FORBIDDEN_INFERENCE_KEYS = {"label", "lesion_mask", "lesion_mask_path", "annotations"}
FORBIDDEN_MANIFEST_KEYS = {
    "accessionnumber",
    "annotationmanifest",
    "annotations",
    "diagnosis",
    "diagnostico",
    "groundtruth",
    "label",
    "lesionmask",
    "lesionmaskpath",
    "patientbirthdate",
    "patientid",
    "patientname",
}
FORBIDDEN_MANIFEST_PREFIXES = ("patient",)


@dataclass(frozen=True)
class InferenceSource:
    case_id: str
    dataset: str
    input_format: str
    root: Path
    dicom_dir: Path | None = None
    volume_path: Path | None = None
    organ_mask_path: Path | None = None
    sanitized_manifest_path: Path | None = None


@dataclass(frozen=True)
class ProtectedGroundTruth:
    label: GroundTruthLabel
    lesion_mask_path: Path | None = None
    annotation_manifest_path: Path | None = None


@dataclass(frozen=True)
class DatasetCase:
    inference: InferenceSource
    ground_truth: ProtectedGroundTruth


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Manifesto YAML inválido ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Manifesto deve ser um objeto: {path}")
    return value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"{label} inválido ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} deve ser um objeto: {path}")
    return value


def _inside(root: Path, candidate: str | Path | None, *, required: bool = False) -> Path | None:
    if candidate in (None, ""):
        if required:
            raise PipelineError("Caminho obrigatório ausente no manifesto.")
        return None
    root = root.resolve()
    path = (root / Path(candidate)).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PipelineError(f"Caminho fora da raiz do dataset: {candidate}") from exc
    if not path.exists():
        raise PipelineError(f"Entrada não encontrada: {path}")
    return path


def _normalized_key(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _forbidden_manifest_paths(value: Any, prefix: str = "$") -> list[str]:
    """Localiza campos que podem carregar PHI ou ground truth no manifesto de inferência."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_MANIFEST_KEYS or normalized.startswith(FORBIDDEN_MANIFEST_PREFIXES):
                found.append(f"{prefix}.{key}")
            found.extend(_forbidden_manifest_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_manifest_paths(child, f"{prefix}[{index}]"))
    return found


def load_dataset_manifest(path: Path) -> list[DatasetCase]:
    """Carrega datasets e labels separados sem fazer qualquer inferência."""
    path = Path(path).resolve()
    manifest = _read_yaml(path)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise PipelineError("Manifesto deve conter uma lista não vazia em datasets.")
    loaded: list[DatasetCase] = []
    seen: set[tuple[str, str]] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise PipelineError("Definição de dataset inválida.")
        name = str(dataset.get("name") or "").strip()
        input_format = str(dataset.get("format") or "").upper()
        if not name or input_format not in SUPPORTED_FORMATS:
            raise PipelineError(f"Dataset inválido: name={name!r}, format={input_format!r}")
        root_value = dataset.get("root")
        if not root_value:
            raise PipelineError(f"Dataset {name} sem root.")
        root = (path.parent / str(root_value)).resolve() if not Path(str(root_value)).is_absolute() else Path(str(root_value)).resolve()
        if not root.is_dir():
            raise PipelineError(f"Raiz do dataset não encontrada: {root}")
        labels_value = dataset.get("labels_manifest")
        if not labels_value:
            raise PipelineError(f"Dataset {name} sem labels_manifest separado.")
        labels_path = (path.parent / str(labels_value)).resolve() if not Path(str(labels_value)).is_absolute() else Path(str(labels_value)).resolve()
        labels = _read_yaml(labels_path).get("cases")
        if not isinstance(labels, list) or not labels:
            raise PipelineError(f"Manifesto de labels vazio: {labels_path}")
        for item in labels:
            if not isinstance(item, dict):
                raise PipelineError(f"Caso inválido em {labels_path}")
            case_id = str(item.get("case_id") or "").strip()
            key = (name, case_id)
            if not case_id or key in seen:
                raise PipelineError(f"case_id ausente ou duplicado em {name}: {case_id!r}")
            seen.add(key)
            try:
                label = GroundTruthLabel(str(item.get("label") or "").lower())
            except ValueError as exc:
                raise PipelineError(f"Label inválido em {name}/{case_id}") from exc
            inference = item.get("inference") or item.get("volume") or {}
            ground_truth = item.get("ground_truth") or item.get("lesion_annotations") or {}
            if not isinstance(inference, dict) or not isinstance(ground_truth, dict):
                raise PipelineError(f"Caso {name}/{case_id} possui seções inválidas.")
            forbidden = FORBIDDEN_INFERENCE_KEYS.intersection(inference)
            if forbidden:
                raise PipelineError(f"Ground truth vazou para inference em {name}/{case_id}: {sorted(forbidden)}")
            source = InferenceSource(
                case_id=case_id,
                dataset=name,
                input_format=input_format,
                root=root,
                dicom_dir=_inside(root, inference.get("dicom_dir"), required=input_format == "DICOM"),
                volume_path=_inside(root, inference.get("volume") or inference.get("path"), required=input_format != "DICOM"),
                organ_mask_path=_inside(root, inference.get("organ_mask"), required=input_format != "DICOM"),
                sanitized_manifest_path=_inside(root, inference.get("sanitized_manifest")),
            )
            protected = ProtectedGroundTruth(
                label=label,
                lesion_mask_path=_inside(root, ground_truth.get("lesion_mask") or ground_truth.get("lesion_mask_path")),
                annotation_manifest_path=_inside(root, ground_truth.get("annotation_manifest")),
            )
            loaded.append(DatasetCase(source, protected))
    return loaded


def _geometry_signature(image: sitk.Image) -> tuple:
    return (
        image.GetDimension(), image.GetSize(), image.GetSpacing(),
        image.GetOrigin(), image.GetDirection(),
    )


def validate_geometry(volume_path: Path, mask_path: Path, tolerance: float = 1e-5) -> None:
    volume = sitk.ReadImage(str(volume_path))
    mask = sitk.ReadImage(str(mask_path))
    if volume.GetDimension() != 3 or mask.GetDimension() != 3:
        raise PipelineError("Volume e máscara hepática devem ser 3D.")
    if volume.GetSize() != mask.GetSize():
        raise PipelineError("Volume e máscara hepática possuem dimensões incompatíveis.")
    for label, first, second in (
        ("spacing", volume.GetSpacing(), mask.GetSpacing()),
        ("origin", volume.GetOrigin(), mask.GetOrigin()),
        ("direction", volume.GetDirection(), mask.GetDirection()),
    ):
        if any(abs(float(a) - float(b)) > tolerance for a, b in zip(first, second)):
            raise PipelineError(f"Volume e máscara hepática possuem {label} incompatível.")


def validate_inference_source(source: InferenceSource) -> dict[str, Any]:
    if source.input_format == "DICOM":
        if not source.dicom_dir or not source.dicom_dir.is_dir():
            raise PipelineError(f"Diretório DICOM ausente: {source.dicom_dir}")
        files = [path for path in source.dicom_dir.rglob("*") if path.is_file()]
        if not files:
            raise PipelineError(f"Diretório DICOM vazio: {source.dicom_dir}")
        return {
            "case_id": source.case_id,
            "format": source.input_format,
            "source_files": len(files),
            "source_hash": sha256_paths(files),
            "organ_mask_hash": sha256_of(source.organ_mask_path) if source.organ_mask_path else None,
        }
    if not source.volume_path or not source.organ_mask_path:
        raise PipelineError(f"{source.input_format} exige volume e organ_mask explícitos.")
    validate_geometry(source.volume_path, source.organ_mask_path)
    return {
        "case_id": source.case_id,
        "format": source.input_format,
        "geometry": _geometry_signature(sitk.ReadImage(str(source.volume_path))),
        "volume_hash": sha256_of(source.volume_path),
        "organ_mask_hash": sha256_of(source.organ_mask_path),
    }


def _sanitized_manifest(source: InferenceSource, volume: Path) -> dict[str, Any]:
    image = sitk.ReadImage(str(volume))
    medgemma_compatible = source.case_id.startswith("anon-")
    return {
        "case_id": source.case_id,
        "policy": "anonymize" if medgemma_compatible else "public_dataset_sanitized",
        "modality": "MR",
        "regulatory_state": "PESQUISA",
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "dataset": source.dataset,
        "input_format": source.input_format,
        "volume_sha256": sha256_of(volume),
        "benchmark_manifest_generated": True,
        "caveats": [
            "Manifesto sintético do benchmark para NIfTI/MIDS sem manifesto anonimizado explícito.",
            "PHI gravada nos pixels (burned-in) NÃO é detectada automaticamente; exige verificação humana.",
        ],
    }


def _candidate_sanitized_manifest(source: InferenceSource, prepared_manifest: Path | None = None) -> Path | None:
    if source.sanitized_manifest_path:
        return source.sanitized_manifest_path
    if source.input_format in {"NIFTI", "MIDS"} and source.volume_path:
        adjacent = source.volume_path.parent / "manifest.json"
        if adjacent.is_file():
            return adjacent
    if prepared_manifest and prepared_manifest.is_file():
        return prepared_manifest
    return None


def _float_lists_close(first: list[Any], second: tuple[float, ...], tolerance: float = 1e-5) -> bool:
    if len(first) != len(second):
        return False
    try:
        return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second))
    except (TypeError, ValueError):
        return False


def _validated_anonymized_manifest(source: InferenceSource, manifest_path: Path, volume: Path) -> dict[str, Any]:
    manifest = _read_json_object(manifest_path, "Manifesto sanitizado")
    forbidden = _forbidden_manifest_paths(manifest)
    if forbidden:
        raise PipelineError(
            "Manifesto sanitizado contém campo proibido ou potencial PHI: "
            f"{sorted(forbidden)}"
        )
    original_case_id = str(manifest.get("case_id") or "")
    effective_case_id = source.case_id if source.case_id.startswith("anon-") else original_case_id
    if not effective_case_id.startswith("anon-"):
        raise PipelineError("Manifesto sanitizado exige case_id anônimo ('anon-*').")
    if manifest.get("policy") != "anonymize":
        raise PipelineError("Manifesto sanitizado exige policy=anonymize.")
    if manifest.get("regulatory_state") != "PESQUISA":
        raise PipelineError("Manifesto sanitizado exige regulatory_state=PESQUISA.")
    if str(manifest.get("modality", "")).upper() not in {"MR", "MRI"}:
        raise PipelineError("Manifesto sanitizado exige modalidade RM (MR/MRI).")

    image = sitk.ReadImage(str(volume))
    expected_hash = sha256_of(volume)
    declared_hash = str(manifest.get("volume_sha256") or "")
    if declared_hash != expected_hash:
        raise PipelineError("Manifesto sanitizado possui volume_sha256 ausente ou inconsistente.")
    if "size_xyz" in manifest and [int(v) for v in manifest["size_xyz"]] != list(image.GetSize()):
        raise PipelineError("Manifesto sanitizado possui size_xyz incompatível com o volume.")
    if "spacing_xyz" in manifest and not _float_lists_close(list(manifest["spacing_xyz"]), image.GetSpacing()):
        raise PipelineError("Manifesto sanitizado possui spacing_xyz incompatível com o volume.")

    sanitized = dict(manifest)
    sanitized["case_id"] = effective_case_id
    if original_case_id and original_case_id != effective_case_id and original_case_id.startswith("anon-"):
        sanitized["source_manifest_case_id"] = original_case_id
    sanitized["policy"] = "anonymize"
    sanitized["regulatory_state"] = "PESQUISA"
    sanitized["dataset"] = source.dataset
    sanitized["input_format"] = source.input_format
    sanitized["volume_sha256"] = expected_hash
    sanitized["sanitized_manifest_sha256"] = sha256_of(manifest_path)
    return sanitized


def prepare_inference_case(
    source: InferenceSource,
    workspace: Path,
    *,
    profile_path: Path = Path("profiles/figado.yaml"),
    segment_if_missing: bool = False,
    device: str = "gpu",
) -> InferenceCase:
    """Materializa somente os arquivos permitidos em um workspace sanitizado."""
    validate_inference_source(source)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    case = Case(workspace)
    if source.input_format == "DICOM":
        if source.organ_mask_path:
            profile = load_profile(profile_path)
            stages.stage1_ingest(case, profile, source.dicom_dir, "anonymize")
            shutil.copyfile(source.organ_mask_path, case.mask_organ)
        elif segment_if_missing:
            Engine(profile_path).prepare(source.dicom_dir, workspace, policy="anonymize", device=device)
        else:
            raise PipelineError(
                f"DICOM {source.case_id} não possui organ_mask; habilite segment_if_missing."
            )
    else:
        shutil.copyfile(source.volume_path, case.volume)
        shutil.copyfile(source.organ_mask_path, case.mask_organ)
    validate_geometry(case.volume, case.mask_organ)
    manifest_source = _candidate_sanitized_manifest(
        source,
        case.manifest if source.input_format == "DICOM" else None,
    )
    manifest = (
        _validated_anonymized_manifest(source, manifest_source, case.volume)
        if manifest_source
        else _sanitized_manifest(source, case.volume)
    )
    case.write_manifest(manifest)
    hashes = input_hashes(case.volume, case.mask_organ, case.manifest)
    return InferenceCase(
        case_id=source.case_id, dataset=source.dataset, input_format=source.input_format,
        volume_path=case.volume, organ_mask_path=case.mask_organ,
        manifest_path=case.manifest, workspace=workspace, input_hashes=hashes,
    )


def attach_ground_truth(case: DatasetCase, inference: InferenceCase) -> EvaluationCase:
    """Chamado pelo avaliador somente depois que a resposta já foi persistida."""
    hashes: dict[str, str | None] = {"lesion_mask": None, "annotation_manifest": None}
    if case.ground_truth.lesion_mask_path:
        hashes["lesion_mask"] = sha256_of(case.ground_truth.lesion_mask_path)
    if case.ground_truth.annotation_manifest_path:
        hashes["annotation_manifest"] = sha256_of(case.ground_truth.annotation_manifest_path)
    return EvaluationCase(
        inference=inference,
        label=case.ground_truth.label,
        lesion_mask_path=case.ground_truth.lesion_mask_path,
        annotation_manifest_path=case.ground_truth.annotation_manifest_path,
        protected_ground_truth_hashes=hashes,
    )
