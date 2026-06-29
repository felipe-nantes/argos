# tests/test_engine_prepare.py
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pytest

from dtwin import stages
from dtwin.engine import Engine
from dtwin.core import read_image, array_from, array_to_image, save_image, PipelineError


def _write_dicom_series(folder: Path, modality: str = "MR", n_slices: int = 6):
    folder.mkdir(parents=True, exist_ok=True)
    vol = (np.random.default_rng(0).random((n_slices, 32, 32)) * 200).astype(np.int16)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 2.0))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1." + "".join(
        np.random.default_rng(1).integers(0, 9, 20).astype(str)
    )
    for i in range(n_slices):
        sl = img[:, :, i]
        sl.SetMetaData("0008|0060", modality)            # Modality
        sl.SetMetaData("0020|000e", series_uid)          # SeriesInstanceUID
        sl.SetMetaData("0020|0013", str(i))              # InstanceNumber
        writer.SetFileName(str(folder / f"slice_{i:03d}.dcm"))
        writer.Execute(sl)


def _fake_segment(case, profile, device, fast):
    """Stand-in for stage3: write a synthetic organ mask matching the volume."""
    vol = read_image(case.volume)
    arr = array_from(vol)
    organ = np.zeros_like(arr, dtype=np.uint8)
    organ[1:-1, 8:24, 8:24] = 1
    save_image(array_to_image(organ, vol, np.uint8), case.mask_organ)


def test_prepare_runs_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "stage3_segment_organ", _fake_segment)
    dicom = tmp_path / "dcm"
    _write_dicom_series(dicom, modality="MR")
    engine = Engine(Path("profiles/figado.yaml"))
    case = engine.prepare(str(dicom), str(tmp_path / "case"), device="cpu", fast=True)
    assert case.volume.exists()
    assert case.volume_zscore.exists()
    assert case.mask_organ.exists()
    assert case.manifest.exists()


def test_prepare_rejects_wrong_modality(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "stage3_segment_organ", _fake_segment)
    dicom = tmp_path / "ct"
    _write_dicom_series(dicom, modality="CT")
    engine = Engine(Path("profiles/figado.yaml"))
    with pytest.raises(PipelineError):
        engine.prepare(str(dicom), str(tmp_path / "case"), device="cpu", fast=True)
