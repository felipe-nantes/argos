# tests/test_stage1_ingest.py
"""Hardening de stage1_ingest: lê uma série DICOM real (escrita com SimpleITK),
verifica preservação de geometria e os campos de anonimização do manifesto.
A suíte de prepare já cobre o fluxo com stage3 stubado; aqui o foco é só a
ingestão + des-identificação (estágio 1), com asserts de geometria.
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from dtwin import stages
from dtwin.core import Case, array_from, read_image

PROFILE = {"id": "figado", "modalidade": ["MR"], "segmentacao_orgao": {"rotulo_alvo": "liver"}}


def _write_dicom_series(folder: Path, spacing=(1.0, 1.0, 2.0), n_slices=6, modality="MR"):
    folder.mkdir(parents=True, exist_ok=True)
    vol = (np.random.default_rng(0).random((n_slices, 24, 24)) * 200).astype(np.int16)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing(tuple(float(s) for s in spacing))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125.1." + "".join(
        np.random.default_rng(1).integers(0, 9, 20).astype(str)
    )
    for i in range(n_slices):
        sl = img[:, :, i]
        sl.SetMetaData("0008|0060", modality)
        sl.SetMetaData("0020|000e", uid)
        sl.SetMetaData("0020|0013", str(i))
        # Geometria por fatia: sem isto, o leitor de série não reconstrói o
        # espaçamento em z (cai para 1.0). Posição em z = i * spacing_z.
        sl.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")          # ImageOrientationPatient
        sl.SetMetaData("0020|0032", f"0\\0\\{i * spacing[2]}")   # ImagePositionPatient
        writer.SetFileName(str(folder / f"slice_{i:03d}.dcm"))
        writer.Execute(sl)
    return spacing, (24, 24, n_slices)


def test_ingest_preserves_geometry_and_anonymizes(tmp_path):
    spacing, size_xyz = _write_dicom_series(tmp_path / "dcm")
    case = Case(tmp_path / "case")
    stages.stage1_ingest(case, PROFILE, tmp_path / "dcm", "anonymize")

    assert case.volume.exists()
    vol = read_image(case.volume)
    # geometria preservada (spacing xyz e tamanho)
    np.testing.assert_allclose(vol.GetSpacing(), spacing, atol=1e-6)
    assert vol.GetSize() == size_xyz
    assert array_from(vol).shape == (size_xyz[2], size_xyz[1], size_xyz[0])  # (z,y,x)

    # anonimização: NIfTI não carrega cabeçalho DICOM; manifesto registra metadados
    m = case.read_manifest()
    assert m["case_id"].startswith("anon-")
    assert m["modality"] == "MR"
    assert m["policy"] == "anonymize"
    assert m["regulatory_state"] == "PESQUISA"
    assert m["size_xyz"] == list(size_xyz)
    assert "volume_sha256" in m and len(m["volume_sha256"]) == 64


def test_ingest_rejects_too_few_slices(tmp_path):
    _write_dicom_series(tmp_path / "dcm", n_slices=2)
    case = Case(tmp_path / "case")
    from dtwin.core import PipelineError
    import pytest

    with pytest.raises(PipelineError, match="poucas fatias"):
        stages.stage1_ingest(case, PROFILE, tmp_path / "dcm", "anonymize")
