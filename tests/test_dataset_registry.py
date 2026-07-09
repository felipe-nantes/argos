import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from dtwin.core import PipelineError
from dtwin.datasets.ingest import main as ingest_main
from dtwin.datasets.registry import ingest_dataset_config, load_dataset_config, write_jsonl
from tools.make_synthetic_case import write_dicom_series


def _image(path: Path, value=1):
    image = sitk.GetImageFromArray(np.full((4, 5, 6), value, dtype=np.uint8))
    sitk.WriteImage(image, str(path))


def _records(config_name: str, root: Path):
    config = load_dataset_config(Path("configs/datasets") / config_name)
    return ingest_dataset_config(config, root)


def test_dicom_mr_dataset_generates_registry_record_without_raw_uid(tmp_path):
    write_dicom_series(tmp_path / "series_mr", np.ones((3, 8, 8), dtype=np.int16), modality="MR")

    records = _records("liverhccseg.yaml", tmp_path)
    data = records[0].to_json()
    encoded = json.dumps(data, ensure_ascii=False)

    assert len(records) == 1
    assert data["dataset_id"] == "liverhccseg"
    assert data["modality"] == "MR"
    assert data["source_format"] == "dicom"
    assert data["dicom_original"] is True
    assert data["nifti_original"] is False
    assert data["research_only"] is True
    assert data["clinical_use_allowed"] is False
    assert data["positive_subtype"] == "hcc_suspicious"
    assert "1.2.826" not in encoded
    assert data["metadata"]["dicom_file_count"] == 3


def test_tcga_lihc_mr_ignores_ct_and_accepts_only_mr(tmp_path):
    write_dicom_series(tmp_path / "series_ct", np.ones((3, 8, 8), dtype=np.int16), modality="CT")
    write_dicom_series(tmp_path / "series_mr", np.ones((3, 8, 8), dtype=np.int16), modality="MR")

    records = _records("tcga_lihc_mr.yaml", tmp_path)

    assert len(records) == 1
    assert records[0].modality == "MR"
    assert records[0].dataset_id == "tcga_lihc_mr"


def test_dicom_ct_only_dataset_fails_closed_for_mr_registry(tmp_path):
    write_dicom_series(tmp_path / "series_ct", np.ones((3, 8, 8), dtype=np.int16), modality="CT")

    with pytest.raises(PipelineError, match="MR"):
        _records("tcga_lihc_mr.yaml", tmp_path)


def test_lld_mmri_nifti_never_marks_dicom_original_and_links_annotation(tmp_path):
    _image(tmp_path / "case_001.nii.gz", 2)
    _image(tmp_path / "case_001_mask.nii.gz", 1)

    records = _records("lld_mmri.yaml", tmp_path)

    assert len(records) == 1
    data = records[0].to_json()
    assert data["dataset_id"] == "lld_mmri"
    assert data["source_format"] == "nifti"
    assert data["dicom_original"] is False
    assert data["nifti_original"] is True
    assert data["annotation_path"] == "case_001_mask.nii.gz"
    assert data["has_segmentation"] is True


def test_nifti_corrupted_file_fails(tmp_path):
    (tmp_path / "bad.nii.gz").write_text("not a nifti", encoding="utf-8")

    with pytest.raises(PipelineError, match="NIfTI"):
        _records("lld_mmri.yaml", tmp_path)


def test_chaos_config_is_negative_control_not_absolute_normal():
    config = load_dataset_config(Path("configs/datasets/chaos_mri.yaml"))

    assert config.rag_class == "negative"
    assert config.label == "controle_anatomico_sem_patologia_macroscopica_documentada"
    assert any("não representa normalidade clínica absoluta" in item for item in config.limitations)
    assert any("nunca" in item.lower() and "normal absoluto" in item.lower() for item in config.warnings)


def test_registry_cli_writes_jsonl(tmp_path, capsys):
    _image(tmp_path / "case_001.nii.gz", 2)
    out = tmp_path / "registry.jsonl"

    code = ingest_main([
        "--config", "configs/datasets/lld_mmri.yaml",
        "--root", str(tmp_path),
        "--out", str(out),
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert "[OK] 1 registros" in captured.out
    rows = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert rows[0]["schema"] == "argos-dataset-registry-v1"


def test_write_jsonl_is_machine_readable(tmp_path):
    _image(tmp_path / "case_001.nii.gz", 2)
    records = _records("lld_mmri.yaml", tmp_path)
    out = tmp_path / "registry.jsonl"

    write_jsonl(records, out)

    rows = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert rows[0]["case_id"].startswith("lld_mmri-")
    assert rows[0]["review_status"] == "pending_review"
