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
from dtwin.core import PipelineError
from tools.make_synthetic_case import write_dicom_series


def _image(path: Path, value=1, spacing=(1.0, 1.0, 2.0)):
    image = sitk.GetImageFromArray(np.full((6, 8, 10), value, dtype=np.uint8))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def _manifests(tmp_path, fmt="NIFTI", inference=None, ground_truth=None):
    root = tmp_path / "dataset"
    root.mkdir()
    labels = tmp_path / "labels.yaml"
    labels.write_text(yaml.safe_dump({"cases": [{
        "case_id": "case-001", "label": "POSITIVE",
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
