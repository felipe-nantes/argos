#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera um caso sintético em disco para exercitar o pipeline sem GPU/Slicer/DICOM.

Uso:
  python tools/make_synthetic_case.py --out casos/sintetico
  python tools/make_synthetic_case.py --out casos/sintetico --dicom
Depois:
  python digital_twin.py finalize casos/sintetico --profile profiles/figado.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from dtwin.core import Case, array_to_image, save_image, now_utc


def _sphere(shape, center, radius):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    return (((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius).astype(np.uint8)


def _geo(arr, spacing=(1.5, 1.0, 1.0), origin=(10.0, -5.0, 3.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in origin))
    return img


def write_dicom_series(folder: Path, volume_zyx, modality="MR"):
    folder.mkdir(parents=True, exist_ok=True)
    img = _geo(volume_zyx.astype(np.int16))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125.1." + "".join(
        np.random.default_rng(1).integers(0, 9, 20).astype(str)
    )
    spacing_z = img.GetSpacing()[2]
    for i in range(volume_zyx.shape[0]):
        sl = img[:, :, i]
        sl.SetMetaData("0008|0060", modality)
        sl.SetMetaData("0020|000e", uid)
        sl.SetMetaData("0020|0013", str(i))
        # Geometria por fatia: sem isto o leitor de série não reconstrói o
        # espaçamento em z. Posição em z = i * spacing_z.
        sl.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")               # ImageOrientationPatient
        sl.SetMetaData("0020|0032", f"0\\0\\{i * spacing_z}")         # ImagePositionPatient
        writer.SetFileName(str(folder / f"slice_{i:03d}.dcm"))
        writer.Execute(sl)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gera um caso sintético do Digital Twin.")
    ap.add_argument("--out", required=True, help="Pasta do caso a criar.")
    ap.add_argument("--dicom", action="store_true", help="Também gera uma série DICOM sintética.")
    args = ap.parse_args(argv)

    shape, center = (40, 40, 40), (20, 20, 20)
    organ = _sphere(shape, center, 12)
    lesion = _sphere(shape, center, 4)
    volume = (organ.astype(np.float32) * 100.0) + 10.0

    case = Case(Path(args.out))
    ref = _geo(volume)
    save_image(ref, case.volume)
    save_image(array_to_image(organ, ref, np.uint8), case.mask_organ)
    save_image(array_to_image(lesion, ref, np.uint8), case.mask_lesion)
    case.write_manifest(
        {
            "case_id": "anon-synthetic00",
            "policy": "anonymize",
            "modality": "MR",
            "regulatory_state": "PESQUISA",
            "size_xyz": [shape[2], shape[1], shape[0]],
            "software": "make_synthetic_case (teste/demonstração — dados fictícios)",
            "created_utc": now_utc(),
        }
    )
    print(f"[OK] Caso sintético em {case.root}")
    if args.dicom:
        dpath = case.root / "dicom_src"
        write_dicom_series(dpath, (volume).astype(np.int16))
        print(f"[OK] DICOM sintético em {dpath}")
    print("Rode: python digital_twin.py finalize", case.root, "--profile profiles/figado.yaml")


if __name__ == "__main__":
    main()
