import json

import numpy as np
from PIL import Image
import pytest
import SimpleITK as sitk

from dtwin.core import Case, PipelineError, load_profile, now_utc, save_image
from dtwin.medgemma_client import load_screening_config, model_trace
from dtwin.medgemma_panel_multiphase import generate_liver_panel_multiphase
from tests.conftest import make_sphere_mask, make_geo_image

CONFIG = "configs/medgemma_local_4b_multiphase.yaml"


@pytest.fixture
def multiphase_case(tmp_path):
    """Synthetic co-registered phases + liver mask + anonymized manifest.

    A focal 'lesion' region gets a different intensity per phase, so the RGB
    fusion has genuinely different channels (no lesion mask is ever written to
    a path MedGemma can read — it only shapes the phase pixels)."""
    shape = (44, 44, 44)
    center = (22, 22, 22)
    organ = make_sphere_mask(shape, center, 14)
    focal = make_sphere_mask(shape, center, 5).astype(bool)
    ref = make_geo_image(np.zeros(shape, np.float32))

    case = Case(tmp_path)
    rng = np.random.default_rng(0)
    tex = rng.normal(0.0, 8.0, shape).astype(np.float32)  # parenchyma texture (real MR is never flat)
    phase_paths = {}
    for name, base, focal_delta in [("art", 100.0, 220.0), ("pv", 140.0, 150.0), ("del", 150.0, 60.0)]:
        arr = organ.astype(np.float32) * base + 10.0 + tex * organ
        arr[focal] = focal_delta + tex[focal]
        p = tmp_path / f"phase_{name}.nii.gz"
        save_image(make_geo_image(arr), p)
        phase_paths[name] = p
    save_image(sitk.Cast(make_geo_image(organ.astype(np.float32)), sitk.sitkUInt8), case.mask_organ)
    case.write_manifest(
        {
            "case_id": "anon-multi000000",
            "policy": "anonymize",
            "modality": "MR",
            "regulatory_state": "PESQUISA",
            "created_utc": now_utc(),
        }
    )
    return case, phase_paths


def _gen(case, phase_paths, tmp_path, **overrides):
    config = load_screening_config(CONFIG)
    args = dict(
        phase_paths=phase_paths,
        liver_mask_path=case.mask_organ,
        case_manifest_path=case.manifest,
        organ_profile=load_profile("profiles/figado.yaml"),
        screening_config=config,
        output_dir=tmp_path / "mp",
        model_trace=model_trace(config),
    )
    args.update(overrides)
    return generate_liver_panel_multiphase(**args)


def test_multiphase_config_validates_and_is_fusion_mode():
    config = load_screening_config(CONFIG)
    assert config["panel"]["mode"] == "multiphase_fusion"
    assert config["panel"]["fusion"]["channel_map"] == {"red": "art", "green": "pv", "blue": "del"}
    assert config["medgemma"]["max_output_tokens"] == 1536


def test_single_phase_config_still_validates():
    # regression: existing grayscale config must keep loading unchanged.
    config = load_screening_config("configs/medgemma_4b.yaml")
    assert config["panel"].get("mode", "single_grayscale") in ("single_grayscale", None)


def test_multiphase_generates_11_rgb_views_without_phi_or_lesion(multiphase_case, tmp_path):
    case, phase_paths = multiphase_case
    result = _gen(case, phase_paths, tmp_path)
    assert result.panel_count == 11
    assert len(result.axial_indices) == 9
    with Image.open(result.panel_path) as image:
        assert image.mode == "RGB"
        assert image.info == {}
        arr = np.asarray(image)
    # fusion actually fused: some pixels have channels that differ (not grayscale).
    assert int((arr[..., 0] != arr[..., 1]).sum()) > 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["lesion_pre_marked"] is False
    assert manifest["requires_human_review"] is True
    assert manifest["input_type"] == "mri_multiphase_rgb_fusion_liver_crop"
    assert manifest["fusion_channel_map"] == {"red": "art", "green": "pv", "blue": "del"}
    assert manifest["phases_used"] == ["art", "del", "pv"]
    assert manifest["png_metadata_keys"] == []
    assert manifest["crop_bounds_zyx"] is not None
    assert any("No lesion mask was read" in n for n in manifest["notes"])


def test_multiphase_fails_when_a_required_phase_is_missing(multiphase_case, tmp_path):
    case, phase_paths = multiphase_case
    partial = {k: v for k, v in phase_paths.items() if k != "del"}
    with pytest.raises(PipelineError, match="ausentes"):
        _gen(case, partial, tmp_path)


def test_multiphase_fails_on_incompatible_phase_geometry(multiphase_case, tmp_path):
    case, phase_paths = multiphase_case
    bad_img = make_geo_image(np.ones((44, 44, 44), np.float32), spacing=(9.0, 9.0, 9.0))
    bad = tmp_path / "phase_art_bad.nii.gz"
    save_image(bad_img, bad)
    broken = dict(phase_paths, art=bad)
    with pytest.raises(PipelineError, match="geometria incompatível"):
        _gen(case, broken, tmp_path)


def test_config_rejects_multiphase_without_channel_map(tmp_path):
    text = (tmp_path / "bad.yaml")
    base = (
        "medgemma_screening:\n"
        "  enabled: true\n  organ: liver\n  modality: MRI\n  regulatory_mode: RESEARCH\n"
        "  lesion_pre_marked: false\n"
        "  panel:\n    mode: multiphase_fusion\n    include_coronal: true\n    include_sagittal: true\n"
        "  privacy:\n    remove_png_metadata: true\n"
        "  report:\n    allowed_states: [POSITIVA, NEGATIVA, INCONCLUSIVA]\n"
        "    allowed_confidence: [baixa, moderada, alta]\n    requires_human_review: true\n"
        "    disclaimer: \"pesquisa; não é diagnóstico; revisão\"\n"
        "  medgemma:\n    provider: http_json_v1\n    model_family: MedGemma\n"
        "    model_version: X\n    model_parameter_scale: '4B'\n    timeout_seconds: 120\n"
        "    minimum_timeout_seconds: 120\n    max_retries: 0\n    execution_mode: local\n"
        "    device: cuda\n    max_input_bytes: 1000\n    max_output_tokens: 10\n"
        "    endpoint_url: 'http://127.0.0.1:8001/generate'\n"
    )
    text.write_text(base, encoding="utf-8")
    with pytest.raises(PipelineError, match="channel_map"):
        load_screening_config(text)
