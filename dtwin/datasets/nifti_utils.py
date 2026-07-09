"""Utilitários NIfTI seguros para o registry de datasets."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib

from dtwin.core import PipelineError


def is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def discover_nifti_files(root: Path, *, exclude_globs: tuple[str, ...] = ()) -> list[Path]:
    root = Path(root)
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_nifti_path(path):
            continue
        relative = path.relative_to(root).as_posix()
        if any(path.match(pattern) or relative == pattern or Path(relative).match(pattern) for pattern in exclude_globs):
            continue
        validate_nifti(path)
        found.append(path)
    if not found:
        raise PipelineError(f"Nenhum NIfTI válido encontrado em {root}.")
    return found


def validate_nifti(path: Path) -> None:
    try:
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"NIfTI inválido ou corrompido: {path}") from exc
    if len(shape) < 3 or any(value <= 0 for value in shape[:3]):
        raise PipelineError(f"NIfTI deve possuir volume 3D válido: {path}")
