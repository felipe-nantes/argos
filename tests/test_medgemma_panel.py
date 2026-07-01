import json

import numpy as np
from PIL import Image
import pytest

from dtwin.core import (
    PipelineError,
    array_from,
    array_to_image,
    load_profile,
    read_image,
    save_image,
    sha256_of,
)
from dtwin.medgemma_client import load_screening_config, model_trace
from dtwin.medgemma_panel import generate_liver_panel


def _generate(synthetic_case, tmp_path, **overrides):
    config = load_screening_config("configs/medgemma_4b.yaml")
    args = {
        "volume_path": synthetic_case.volume,
        "liver_mask_path": synthetic_case.mask_organ,
        "case_manifest_path": synthetic_case.manifest,
        "organ_profile": load_profile("profiles/figado.yaml"),
        "screening_config": config,
        "output_dir": tmp_path / "medgemma",
        "model_trace": model_trace(config),
    }
    args.update(overrides)
    return generate_liver_panel(**args)


def test_panel_generates_11_views_without_phi_metadata_or_lesion(synthetic_case, tmp_path):
    result = _generate(synthetic_case, tmp_path)
    assert result.panel_count == 11
    assert len(result.axial_indices) == 9
    with Image.open(result.panel_path) as image:
        assert image.size == (1280, 960)
        assert image.mode == "RGB"
        assert image.info == {}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["panel_count"] == 11
    assert manifest["lesion_pre_marked"] is False
    assert manifest["requires_human_review"] is True
    assert manifest["visible_phi_review_required"] is True
    assert manifest["visible_phi_confirmed"] is False
    assert manifest["png_metadata_keys"] == []
    assert manifest["panel_sha256"] == sha256_of(result.panel_path)
    assert manifest["input_volume_sha256"] == sha256_of(synthetic_case.volume)
    assert manifest["input_liver_mask_sha256"] == sha256_of(synthetic_case.mask_organ)
    assert all("mask_lesion" not in note for note in manifest["notes"])


def test_panel_rejects_volume_that_does_not_match_case_manifest(synthetic_case, tmp_path):
    manifest = synthetic_case.read_manifest()
    manifest["volume_sha256"] = "0" * 64
    synthetic_case.write_manifest(manifest)
    with pytest.raises(PipelineError, match="hash do volume"):
        _generate(synthetic_case, tmp_path)


def test_panel_fails_if_volume_does_not_exist(synthetic_case, tmp_path):
    with pytest.raises(PipelineError, match="Volume de RM não encontrado"):
        _generate(synthetic_case, tmp_path, volume_path=tmp_path / "missing.nii.gz")


def test_panel_fails_if_mask_does_not_exist(synthetic_case, tmp_path):
    with pytest.raises(PipelineError, match="Máscara do fígado não encontrado"):
        _generate(synthetic_case, tmp_path, liver_mask_path=tmp_path / "missing.nii.gz")


def test_panel_fails_on_incompatible_geometry(synthetic_case, tmp_path):
    mask = read_image(synthetic_case.mask_organ)
    mask.SetSpacing((9.0, 9.0, 9.0))
    bad = tmp_path / "bad_geometry.nii.gz"
    save_image(mask, bad)
    with pytest.raises(PipelineError, match="geometria incompatível"):
        _generate(synthetic_case, tmp_path, liver_mask_path=bad)


def test_panel_fails_on_empty_mask(synthetic_case, tmp_path):
    ref = read_image(synthetic_case.volume)
    empty = np.zeros_like(array_from(ref), dtype=np.uint8)
    path = tmp_path / "empty.nii.gz"
    save_image(array_to_image(empty, ref, np.uint8), path)
    with pytest.raises(PipelineError, match="Máscara do fígado vazia"):
        _generate(synthetic_case, tmp_path, liver_mask_path=path)


def test_panel_fails_with_too_few_liver_slices(synthetic_case, tmp_path):
    ref = read_image(synthetic_case.volume)
    mask = np.zeros_like(array_from(ref), dtype=np.uint8)
    mask[10:15, 10:25, 10:25] = 1
    path = tmp_path / "five_slices.nii.gz"
    save_image(array_to_image(mask, ref, np.uint8), path)
    with pytest.raises(PipelineError, match="fatias axiais"):
        _generate(synthetic_case, tmp_path, liver_mask_path=path)


def test_panel_rejects_non_anonymous_case(synthetic_case, tmp_path):
    manifest = synthetic_case.read_manifest()
    manifest["case_id"] = "patient-name"
    synthetic_case.write_manifest(manifest)
    with pytest.raises(PipelineError, match="identificador anônimo"):
        _generate(synthetic_case, tmp_path)
