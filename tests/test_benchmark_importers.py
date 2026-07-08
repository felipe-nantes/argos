import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from dtwin.benchmark.importers import (
    attach_ground_truth,
    load_dataset_manifest,
    prepare_inference_case,
    validate_geometry,
)
from dtwin.core import PipelineError, sha256_of
from tools.make_synthetic_case import write_dicom_series


def _image(path: Path, value=1, spacing=(1.0, 1.0, 2.0)):
    image = sitk.GetImageFromArray(np.full((6, 8, 10), value, dtype=np.uint8))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def _manifests(tmp_path, fmt="NIFTI", inference=None, ground_truth=None, case_id="case-001"):
    root = tmp_path / "dataset"
    root.mkdir()
    labels = tmp_path / "labels.yaml"
    labels.write_text(yaml.safe_dump({"cases": [{
        "case_id": case_id, "label": "POSITIVE",
        "inference": inference or {"volume": "volume.nii.gz", "organ_mask": "mask.nii.gz"},
        "ground_truth": ground_truth or {},
    }]}), encoding="utf-8")
    manifest = tmp_path / "datasets.yaml"
    manifest.write_text(yaml.safe_dump({"datasets": [{
        "name": "TEST", "format": fmt, "root": "dataset", "labels_manifest": "labels.yaml",
    }]}), encoding="utf-8")
    return root, manifest


def test_nifti_importer_physically_excludes_ground_truth(tmp_path):
    root, manifest = _manifests(tmp_path, ground_truth={"lesion_mask": "lesion.nii.gz"})
    _image(root / "volume.nii.gz", 2)
    _image(root / "mask.nii.gz", 1)
    _image(root / "lesion.nii.gz", 1)
    case = load_dataset_manifest(manifest)[0]
    inference = prepare_inference_case(case.inference, tmp_path / "run" / "inference")
    assert inference.volume_path.exists() and inference.organ_mask_path.exists()
    assert not list(inference.workspace.rglob("*lesion*"))
    assert not hasattr(inference, "label") and not hasattr(inference, "lesion_mask_path")
    evaluation = attach_ground_truth(case, inference)
    assert evaluation.label.value == "positive"
    assert len(evaluation.protected_ground_truth_hashes["lesion_mask"]) == 64


def test_nifti_importer_preserves_adjacent_anonymized_manifest(tmp_path):
    root, manifest = _manifests(tmp_path, case_id="anon-benchmark001")
    _image(root / "volume.nii.gz", 2)
    _image(root / "mask.nii.gz", 1)
    (root / "manifest.json").write_text(json.dumps({
        "case_id": "anon-source001",
        "policy": "anonymize",
        "modality": "MR",
        "regulatory_state": "PESQUISA",
        "size_xyz": [10, 8, 6],
        "spacing_xyz": [1.0, 1.0, 2.0],
        "volume_sha256": sha256_of(root / "volume.nii.gz"),
        "software": "synthetic-test",
    }), encoding="utf-8")

    case = load_dataset_manifest(manifest)[0]
    inference = prepare_inference_case(case.inference, tmp_path / "run" / "inference")
    data = json.loads(inference.manifest_path.read_text(encoding="utf-8"))

    assert data["case_id"] == "anon-benchmark001"
    assert data["source_manifest_case_id"] == "anon-source001"
    assert data["policy"] == "anonymize"
    assert data["dataset"] == "TEST"
    assert data["input_format"] == "NIFTI"
    assert len(data["sanitized_manifest_sha256"]) == 64


def test_nifti_importer_rejects_sanitized_manifest_with_wrong_policy(tmp_path):
    root, manifest = _manifests(
        tmp_path,
        case_id="anon-benchmark001",
        inference={
            "volume": "volume.nii.gz",
            "organ_mask": "mask.nii.gz",
            "sanitized_manifest": "bad_manifest.json",
        },
    )
    _image(root / "volume.nii.gz", 2)
    _image(root / "mask.nii.gz", 1)
    (root / "bad_manifest.json").write_text(json.dumps({
        "case_id": "anon-source001",
        "policy": "public_dataset_sanitized",
        "modality": "MR",
        "regulatory_state": "PESQUISA",
        "volume_sha256": sha256_of(root / "volume.nii.gz"),
    }), encoding="utf-8")

    case = load_dataset_manifest(manifest)[0]
    with pytest.raises(PipelineError, match="policy=anonymize"):
        prepare_inference_case(case.inference, tmp_path / "run" / "inference")


def test_nifti_importer_rejects_sanitized_manifest_with_phi_key(tmp_path):
    root, manifest = _manifests(tmp_path, case_id="anon-benchmark001")
    _image(root / "volume.nii.gz", 2)
    _image(root / "mask.nii.gz", 1)
    (root / "manifest.json").write_text(json.dumps({
        "case_id": "anon-source001",
        "policy": "anonymize",
        "modality": "MR",
        "regulatory_state": "PESQUISA",
        "volume_sha256": sha256_of(root / "volume.nii.gz"),
        "patient_name": "Jane Example",
    }), encoding="utf-8")

    case = load_dataset_manifest(manifest)[0]
    with pytest.raises(PipelineError, match="potencial PHI"):
        prepare_inference_case(case.inference, tmp_path / "run" / "inference")


def test_nifti_importer_rejects_sanitized_manifest_with_wrong_volume_hash(tmp_path):
    root, manifest = _manifests(tmp_path, case_id="anon-benchmark001")
    _image(root / "volume.nii.gz", 2)
    _image(root / "mask.nii.gz", 1)
    (root / "manifest.json").write_text(json.dumps({
        "case_id": "anon-source001",
        "policy": "anonymize",
        "modality": "MR",
        "regulatory_state": "PESQUISA",
        "volume_sha256": "0" * 64,
    }), encoding="utf-8")

    case = load_dataset_manifest(manifest)[0]
    with pytest.raises(PipelineError, match="volume_sha256"):
        prepare_inference_case(case.inference, tmp_path / "run" / "inference")


def test_nifti_importer_generates_medgemma_compatible_manifest_for_anon_case(tmp_path):
    root, manifest = _manifests(tmp_path, case_id="anon-generated001")
    _image(root / "volume.nii.gz", 2)
    _image(root / "mask.nii.gz", 1)

    case = load_dataset_manifest(manifest)[0]
    inference = prepare_inference_case(case.inference, tmp_path / "run" / "inference")
    data = json.loads(inference.manifest_path.read_text(encoding="utf-8"))

    assert data["case_id"] == "anon-generated001"
    assert data["policy"] == "anonymize"
    assert data["benchmark_manifest_generated"] is True


def test_manifest_rejects_ground_truth_inside_inference(tmp_path):
    root, manifest = _manifests(tmp_path, inference={
        "volume": "volume.nii.gz", "organ_mask": "mask.nii.gz", "lesion_mask": "lesion.nii.gz",
    })
    _image(root / "volume.nii.gz")
    _image(root / "mask.nii.gz", spacing=(1.5, 1.0, 1.0))
    _image(root / "lesion.nii.gz")
    with pytest.raises(PipelineError, match="vazou"):
        load_dataset_manifest(manifest)


def test_nifti_geometry_mismatch_aborts(tmp_path):
    volume = tmp_path / "volume.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    _image(volume, spacing=(1, 1, 1))
    _image(mask, spacing=(2, 1, 1))
    with pytest.raises(PipelineError, match="spacing"):
        validate_geometry(volume, mask)


def test_dicom_importer_reuses_stage1_and_explicit_organ_mask(tmp_path):
    root, manifest = _manifests(
        tmp_path, fmt="DICOM",
        inference={"dicom_dir": "dicom", "organ_mask": "mask.nii.gz"},
    )
    volume = np.ones((6, 8, 10), dtype=np.int16)
    write_dicom_series(root / "dicom", volume)
    _image(root / "mask.nii.gz", spacing=(1.5, 1.0, 1.0))
    case = load_dataset_manifest(manifest)[0]
    inference = prepare_inference_case(case.inference, tmp_path / "inference")
    assert inference.input_format == "DICOM"
    assert inference.volume_path.exists()
    assert "volume_sha256" in inference.manifest_path.read_text("utf-8")
