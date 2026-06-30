# tests/test_gates_extra.py
"""Characterization tests for the pipeline's abort gates and branches that the
core suite did not yet exercise. These cover stage 3 (segmentation, via an
injected fake TotalSegmentator — no GPU/torch), normalization variants, the
lesion-import gates, refino safety gates, and the privacy-policy gate.

If any of these FAIL, an existing safety behavior regressed — investigate the
code, not the test.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dtwin import stages
from dtwin.core import (
    Case,
    PipelineError,
    array_from,
    array_to_image,
    read_image,
    save_image,
)
from dtwin.stages import _make_case_id, stage2_normalize, stage3_segment_organ
from .conftest import make_geo_image, make_sphere_mask

ORGAN_PROFILE = {"segmentacao_orgao": {"rotulo_alvo": "liver", "motor_task": "total_mr"}}


# --------------------------------------------------------------------------- #
# Stage 3 — segmentação automática (fake TotalSegmentator, sem GPU/torch)
# --------------------------------------------------------------------------- #
def _install_fake_totalseg(monkeypatch, writer):
    pkg = types.ModuleType("totalsegmentator")
    api = types.ModuleType("totalsegmentator.python_api")
    api.totalsegmentator = writer
    pkg.python_api = api
    monkeypatch.setitem(sys.modules, "totalsegmentator", pkg)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", api)


def test_stage3_missing_package_aborts(synthetic_case, monkeypatch):
    # ensure no fake is present: stage3 must abort, never fabricate a mask
    monkeypatch.setitem(sys.modules, "totalsegmentator", None)
    with pytest.raises(PipelineError, match="TotalSegmentator"):
        stage3_segment_organ(synthetic_case, ORGAN_PROFILE, device="cpu", fast=True)


def test_stage3_success_writes_organ_mask(synthetic_case, monkeypatch):
    def writer(**kw):
        out = Path(kw["output"])
        vol = read_image(Path(kw["input"]))
        arr = array_from(vol)
        organ = make_sphere_mask(arr.shape, tuple(s // 2 for s in arr.shape), max(arr.shape) // 4)
        save_image(array_to_image(organ, vol, np.uint8), out / (kw["roi_subset"][0] + ".nii.gz"))

    _install_fake_totalseg(monkeypatch, writer)
    synthetic_case.mask_organ.unlink()  # prove stage3 (re)creates it
    stage3_segment_organ(synthetic_case, ORGAN_PROFILE, device="cpu", fast=True)
    assert synthetic_case.mask_organ.exists()
    assert int(array_from(read_image(synthetic_case.mask_organ)).sum()) > 0


def test_stage3_missing_output_aborts(synthetic_case, monkeypatch):
    _install_fake_totalseg(monkeypatch, lambda **kw: None)  # writes nothing
    with pytest.raises(PipelineError, match="não encontrada"):
        stage3_segment_organ(synthetic_case, ORGAN_PROFILE, device="cpu", fast=True)


def test_stage3_empty_segmentation_aborts(synthetic_case, monkeypatch):
    def writer(**kw):
        out = Path(kw["output"])
        vol = read_image(Path(kw["input"]))
        zeros = np.zeros(array_from(vol).shape, dtype=np.uint8)
        save_image(array_to_image(zeros, vol, np.uint8), out / (kw["roi_subset"][0] + ".nii.gz"))

    _install_fake_totalseg(monkeypatch, writer)
    with pytest.raises(PipelineError, match="não encontrou"):
        stage3_segment_organ(synthetic_case, ORGAN_PROFILE, device="cpu", fast=True)


def test_stage3_segmentator_raises_aborts(synthetic_case, monkeypatch):
    def writer(**kw):
        raise RuntimeError("CUDA out of memory")

    _install_fake_totalseg(monkeypatch, writer)
    with pytest.raises(PipelineError, match="Falha na segmentação"):
        stage3_segment_organ(synthetic_case, ORGAN_PROFILE, device="cpu", fast=True)


# --------------------------------------------------------------------------- #
# Stage 2 — normalização minmax (o sucesso só era testado p/ zscore)
# --------------------------------------------------------------------------- #
def test_minmax_normalization_maps_to_unit_range(tmp_path):
    arr = np.random.default_rng(2).normal(40, 12, size=(16, 16, 16)).astype(np.float32)
    case = Case(tmp_path)
    save_image(make_geo_image(arr), case.volume)
    stage2_normalize(case, {"normalizacao": "minmax"})
    out = array_from(read_image(case.volume_zscore))
    assert abs(float(out.min())) < 1e-6
    assert abs(float(out.max()) - 1.0) < 1e-6


def test_minmax_no_contrast_aborts(tmp_path):
    arr = np.full((8, 8, 8), 3.0, np.float32)
    case = Case(tmp_path)
    save_image(make_geo_image(arr), case.volume)
    with pytest.raises(PipelineError):
        stage2_normalize(case, {"normalizacao": "minmax"})


# --------------------------------------------------------------------------- #
# Stage 4b — gates de importação da lesão
# --------------------------------------------------------------------------- #
def test_lesion_size_mismatch_aborts(synthetic_case):
    # overwrite lesion with a differently-sized mask
    small = make_sphere_mask((30, 30, 30), (15, 15, 15), 4)
    save_image(make_geo_image(small), synthetic_case.mask_lesion)
    with pytest.raises(PipelineError, match="tamanho diferente"):
        stages.stage4b_import_lesion(synthetic_case, ORGAN_PROFILE, no_lesion=False)


def test_lesion_outside_organ_warns_but_completes(synthetic_case, tmp_path, caplog):
    # lesion fully disjoint from organ: warning, not abort
    shape = (40, 40, 40)
    lesion = make_sphere_mask(shape, (4, 4, 4), 2)
    ref = read_image(synthetic_case.mask_organ)
    save_image(array_to_image(lesion, ref, np.uint8), synthetic_case.mask_lesion)
    profile = {**ORGAN_PROFILE, "id": "figado", "flywheel": {"dir": str(tmp_path / "fly")}}
    import logging
    with caplog.at_level(logging.WARNING, logger="dtwin"):
        stages.stage4b_import_lesion(synthetic_case, profile, no_lesion=False)
    assert any("não sobrepõe" in r.message for r in caplog.records)


def test_lesion_missing_without_flag_aborts(synthetic_case):
    synthetic_case.mask_lesion.unlink()
    with pytest.raises(PipelineError, match="ausente"):
        stages.stage4b_import_lesion(synthetic_case, ORGAN_PROFILE, no_lesion=False)


def test_organ_missing_aborts(tmp_path):
    case = Case(tmp_path)  # nothing on disk
    with pytest.raises(PipelineError, match="órgão ausente"):
        stages.stage4b_import_lesion(case, ORGAN_PROFILE, no_lesion=True)


# --------------------------------------------------------------------------- #
# Stage 5 — refino nunca pode zerar uma máscara que tinha conteúdo
# --------------------------------------------------------------------------- #
def _case_with_masks(tmp_path):
    shape = (40, 40, 40)
    ref = make_geo_image(np.zeros(shape, np.float32))
    case = Case(tmp_path)
    save_image(ref, case.volume)
    save_image(array_to_image(make_sphere_mask(shape, (20, 20, 20), 12), ref, np.uint8), case.mask_organ)
    save_image(array_to_image(make_sphere_mask(shape, (20, 20, 20), 4), ref, np.uint8), case.mask_lesion)
    return case


def test_refino_zeroing_organ_aborts(tmp_path):
    case = _case_with_masks(tmp_path)
    profile = {"refino": {"orgao": {"min_volume_voxels": 10**9}}}
    with pytest.raises(PipelineError, match="zerou a máscara do órgão"):
        stages.stage5_refine(case, profile)


def test_refino_zeroing_lesion_aborts(tmp_path):
    case = _case_with_masks(tmp_path)
    profile = {"refino": {"lesao": {"min_volume_voxels": 10**9}}}
    with pytest.raises(PipelineError, match="zerou a máscara da lesão"):
        stages.stage5_refine(case, profile)


# --------------------------------------------------------------------------- #
# Stage 6 — malha de órgão vazia aborta
# --------------------------------------------------------------------------- #
def test_empty_organ_mesh_aborts(tmp_path):
    shape = (10, 10, 10)
    ref = make_geo_image(np.zeros(shape, np.float32))
    case = Case(tmp_path)
    save_image(array_to_image(np.zeros(shape, np.uint8), ref, np.uint8), case.mask_organ_clean)
    with pytest.raises(PipelineError, match="Malha do órgão vazia"):
        stages.stage6_mesh(case, {})


# --------------------------------------------------------------------------- #
# Política de privacidade — pseudonimização é gate reservado, não simulação
# --------------------------------------------------------------------------- #
def test_pseudonymize_policy_aborts():
    with pytest.raises(PipelineError, match="Pseudonimização"):
        _make_case_id("pseudonymize", None)


def test_unknown_policy_aborts():
    with pytest.raises(PipelineError, match="desconhecida"):
        _make_case_id("bogus", None)


def test_anonymize_policy_returns_anon_id():
    cid = _make_case_id("anonymize", None)
    assert cid.startswith("anon-") and len(cid) > 5


# --------------------------------------------------------------------------- #
# Stage 1 / 4a — gates de ingestão e de handoff
# --------------------------------------------------------------------------- #
def test_stage1_missing_dicom_dir_aborts(tmp_path):
    case = Case(tmp_path)
    with pytest.raises(PipelineError, match="DICOM inexistente"):
        stages.stage1_ingest(case, {"modalidade": ["MR"]}, tmp_path / "ghost", "anonymize")


def test_stage1_no_readable_dicom_aborts(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "notdicom.txt").write_text("garbage", encoding="utf-8")
    case = Case(tmp_path / "case")
    # pydicom(force=True) lê o lixo como dataset vazio; o gate real dispara
    # adiante, quando o GDCM não encontra nenhuma série montável.
    with pytest.raises(PipelineError, match="Nenhuma série DICOM"):
        stages.stage1_ingest(case, {"modalidade": ["MR"]}, src, "anonymize")


def test_stage4a_incomplete_aborts(tmp_path):
    case = Case(tmp_path)  # no volume / no organ mask
    with pytest.raises(PipelineError, match="incompletos"):
        stages.stage4a_prepare_lesion(case, {})
