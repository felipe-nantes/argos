#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Montagem 2D segura para triagem visual hepática com MedGemma.

Este módulo consome somente o volume de RM des-identificado, a máscara do
fígado e manifests/configs versionados. Ele nunca lê uma máscara de lesão.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from .core import PipelineError, array_from, now_utc, read_image, sha256_of
from .medgemma_volumetric import panel_strategy, render_volumetric_panel_set


PANEL_FILENAME = "medgemma_liver_screening_panel.png"
PANEL_MANIFEST_FILENAME = "medgemma_liver_screening_manifest.json"


@dataclass(frozen=True)
class PanelResult:
    panel_path: Path
    manifest_path: Path
    panel_count: int
    axial_indices: tuple[int, ...]
    coronal_index: int
    sagittal_index: int
    panel_paths: tuple[Path, ...] = ()


def _geometry_compatible(volume, mask, atol: float = 1e-5) -> bool:
    return (
        volume.GetSize() == mask.GetSize()
        and np.allclose(volume.GetSpacing(), mask.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(volume.GetOrigin(), mask.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(volume.GetDirection(), mask.GetDirection(), atol=atol, rtol=0)
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PipelineError(f"{label} não encontrado: {path}")


def _validate_case_manifest(path: Path) -> dict[str, Any]:
    _require_file(path, "Manifesto do caso")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Manifesto do caso inválido ({path}): {exc}") from exc
    case_id = str(manifest.get("case_id", ""))
    if not case_id.startswith("anon-"):
        raise PipelineError(
            "Caso sem identificador anônimo. A análise MedGemma exige case_id 'anon-*'."
        )
    if manifest.get("policy") != "anonymize":
        raise PipelineError("A análise MedGemma exige policy=anonymize.")
    if manifest.get("regulatory_state") != "PESQUISA":
        raise PipelineError("A análise MedGemma só está habilitada em modo PESQUISA.")
    if str(manifest.get("modality", "")).upper() not in {"MR", "MRI"}:
        raise PipelineError("A análise MedGemma deste fluxo exige modalidade RM (MR/MRI).")
    return manifest


def _select_uniform_indices(indices: np.ndarray, count: int) -> tuple[int, ...]:
    if indices.size < count:
        raise PipelineError(
            f"Fígado aparece em apenas {indices.size} fatias axiais; "
            f"são necessárias pelo menos {count}."
        )
    positions = np.linspace(0, indices.size - 1, count)
    chosen = tuple(int(indices[int(round(p))]) for p in positions)
    if len(set(chosen)) != count:
        raise PipelineError("Não foi possível selecionar fatias axiais distintas suficientes.")
    return chosen


def _window_limits(volume: np.ndarray, mask: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise PipelineError("Volume sem intensidades finitas.")
    # Usa o volume completo para manter contexto e funcionar quando o parênquima
    # sintético é uniforme. Percentis robustos evitam que outliers dominem a RM.
    lo, hi = np.percentile(finite, [low, high]).astype(float)
    if hi - lo < 1e-6:
        liver_values = volume[mask & np.isfinite(volume)]
        if liver_values.size:
            lo = float(min(finite.min(), liver_values.min()))
            hi = float(max(finite.max(), liver_values.max()))
    if hi - lo < 1e-6:
        raise PipelineError("Volume sem contraste suficiente para gerar a montagem.")
    return lo, hi


def _render_tile(
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    label: str,
    tile_size: int,
    lo: float,
    hi: float,
    row_spacing: float,
    col_spacing: float,
    contour_width: int,
    contour_color: tuple[int, int, int],
) -> Image.Image:
    gray = np.clip((image_2d.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.repeat((gray * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    boundary = find_boundaries(mask_2d.astype(bool), mode="inner")
    if contour_width > 1:
        boundary = binary_dilation(boundary, iterations=contour_width - 1)
    # Contorno apenas: o sinal dentro do parênquima permanece integralmente visível.
    rgb[boundary] = np.asarray(contour_color, dtype=np.uint8)

    source = Image.fromarray(np.flipud(rgb), mode="RGB")
    physical_w = max(1.0, source.width * float(col_spacing))
    physical_h = max(1.0, source.height * float(row_spacing))
    usable = tile_size - 28
    scale = min(usable / physical_w, usable / physical_h)
    target = (
        max(1, int(round(physical_w * scale))),
        max(1, int(round(physical_h * scale))),
    )
    source = source.resize(target, Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (tile_size, tile_size), (10, 14, 20))
    tile.paste(source, ((tile_size - target[0]) // 2, 24 + (usable - target[1]) // 2))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 6), label, fill=(235, 240, 246))
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(52, 61, 72))
    return tile


def generate_liver_panel(
    *,
    volume_path: Path,
    liver_mask_path: Path,
    case_manifest_path: Path,
    organ_profile: dict[str, Any],
    screening_config: dict[str, Any],
    output_dir: Path,
    model_trace: dict[str, Any],
    visible_phi_confirmed: bool = False,
) -> PanelResult:
    """Valida entradas e gera 9 axiais + 1 coronal + 1 sagital.

    Nenhum caminho ou argumento para máscara de lesão existe por design.
    """
    volume_path, liver_mask_path = Path(volume_path), Path(liver_mask_path)
    case_manifest_path, output_dir = Path(case_manifest_path), Path(output_dir)
    _require_file(volume_path, "Volume de RM")
    _require_file(liver_mask_path, "Máscara do fígado")
    case_manifest = _validate_case_manifest(case_manifest_path)
    volume_sha256 = sha256_of(volume_path)
    recorded_volume_sha256 = case_manifest.get("volume_sha256")
    if recorded_volume_sha256 and recorded_volume_sha256 != volume_sha256:
        raise PipelineError(
            "O hash do volume não corresponde ao manifest do caso. "
            "Não misture artefatos de casos ou execuções diferentes."
        )
    liver_mask_sha256 = sha256_of(liver_mask_path)

    expected_organ = str(screening_config.get("organ", ""))
    profile_organ = str(organ_profile.get("segmentacao_orgao", {}).get("rotulo_alvo", ""))
    if not expected_organ or profile_organ != expected_organ:
        raise PipelineError(
            f"Perfil/config incompatíveis para MedGemma: {profile_organ!r} != {expected_organ!r}."
        )

    volume_img = read_image(volume_path)
    mask_img = read_image(liver_mask_path)
    if not _geometry_compatible(volume_img, mask_img):
        raise PipelineError("Volume e máscara do fígado têm geometria incompatível.")
    if volume_img.GetDimension() != 3:
        raise PipelineError("A montagem MedGemma exige um volume 3D.")

    volume = array_from(volume_img).astype(np.float32)
    mask = array_from(mask_img) > 0
    mask_voxels = int(mask.sum())
    panel_cfg = screening_config.get("panel", {})
    validation_cfg = screening_config.get("validation", {})
    min_voxels = int(validation_cfg.get("min_liver_voxels", 300))
    max_fraction = float(validation_cfg.get("max_liver_fraction", 0.8))
    if mask_voxels == 0:
        raise PipelineError("Máscara do fígado vazia.")
    if mask_voxels < min_voxels:
        raise PipelineError(
            f"Máscara do fígado implausivelmente pequena ({mask_voxels} < {min_voxels} voxels)."
        )
    fraction = mask_voxels / float(mask.size)
    if fraction > max_fraction:
        raise PipelineError(
            f"Máscara do fígado implausivelmente grande ({fraction:.1%} do volume)."
        )

    axial_present = np.flatnonzero(mask.any(axis=(1, 2)))
    centroid_zyx = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
    zc, yc, xc = (int(x) for x in centroid_zyx)

    low = float(panel_cfg.get("window_percentile_low", 1.0))
    high = float(panel_cfg.get("window_percentile_high", 99.0))
    if not 0 <= low < high <= 100:
        raise PipelineError("Percentis da janela de intensidade são inválidos.")
    lo, hi = _window_limits(volume, mask, low, high)
    tile_size = int(panel_cfg.get("tile_size", 320))
    if tile_size < 128:
        raise PipelineError("panel.tile_size deve ser >= 128.")
    contour_width = int(panel_cfg.get("contour_width", 2))
    color_raw = panel_cfg.get("contour_color_rgb", [255, 196, 0])
    if not isinstance(color_raw, list) or len(color_raw) != 3:
        raise PipelineError("panel.contour_color_rgb deve conter três inteiros RGB.")
    contour_color = tuple(int(np.clip(x, 0, 255)) for x in color_raw)

    sx, sy, sz = (float(x) for x in volume_img.GetSpacing())
    strategy = panel_strategy(panel_cfg)
    if strategy == "volumetric_blocks":
        panel_set = render_volumetric_panel_set(
            mask=mask,
            output_dir=output_dir,
            tile_size=tile_size,
            axial_tiles_per_panel=int(panel_cfg.get("axial_tiles_per_panel", 9)),
            index_offset_zyx=(0, 0, 0),
            render_axial=lambda z, label: _render_tile(
                volume[z], mask[z], label, tile_size, lo, hi, sy, sx,
                contour_width, contour_color,
            ),
            render_coronal=lambda y, label: _render_tile(
                volume[:, y, :], mask[:, y, :], label, tile_size, lo, hi, sz, sx,
                contour_width, contour_color,
            ),
            render_sagittal=lambda x, label: _render_tile(
                volume[:, :, x], mask[:, :, x], label, tile_size, lo, hi, sz, sy,
                contour_width, contour_color,
            ),
            notice_text=(
                "MODO PESQUISA\n\nCobertura volumetrica.\nHipotese visual apenas.\n"
                "NAO e diagnostico.\nNAO e laudo medico.\n\nRevisao humana obrigatoria."
            ),
            max_image_pixels=int(screening_config["medgemma"].get("max_image_pixels", 4_000_000)),
            max_input_bytes=int(screening_config["medgemma"]["max_input_bytes"]),
        )
        first_path = panel_set.panel_paths[0]
        manifest = {
            "schema_version": "dtwin-medgemma-panel-set-v2",
            "case_id": case_manifest["case_id"], "organ": expected_organ,
            "modality": "MRI", "regulatory_mode": "RESEARCH",
            "input_type": "mri_with_liver_contour", "lesion_pre_marked": False,
            "panel_strategy": strategy,
            "panel_image": first_path.name,
            "panel_sha256": sha256_of(first_path),
            "panel_count": 11,
            "panel_image_count": len(panel_set.panel_paths),
            "total_tile_count": sum(len(p["tiles"]) for p in panel_set.panels),
            "panels": list(panel_set.panels),
            "input_volume_sha256": volume_sha256,
            "input_liver_mask_sha256": liver_mask_sha256,
            "views": {
                "axial_indices_zyx": list(panel_set.axial_indices),
                "coronal_centroid_y": yc, "sagittal_centroid_x": xc,
            },
            "coverage": panel_set.coverage,
            "liver_mask_voxels": mask_voxels,
            "png_metadata_keys": list(panel_set.png_metadata_keys),
            "phi_metadata_removed": True,
            "visible_phi_review_required": True,
            "visible_phi_confirmed": bool(visible_phi_confirmed),
            "created_at": now_utc(), "requires_human_review": True,
            **model_trace,
            "notes": [
                "No lesion mask was read, provided, or rendered.",
                "All liver-bearing axial slices are represented exactly once.",
                "No patient identifiers were added to the panels or PNG metadata.",
                "Burned-in pixel PHI cannot be ruled out automatically; visual confirmation is required before inference.",
                "Analysis is for research and education only.",
            ],
        }
        manifest_path = output_dir / PANEL_MANIFEST_FILENAME
        temp_path = output_dir / f".{PANEL_MANIFEST_FILENAME}.tmp"
        temp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(manifest_path)
        return PanelResult(
            panel_path=first_path, panel_paths=panel_set.panel_paths,
            manifest_path=manifest_path, panel_count=11,
            axial_indices=panel_set.axial_indices,
            coronal_index=yc, sagittal_index=xc,
        )

    axial_count = int(panel_cfg.get("axial_slices", 9))
    if axial_count != 9:
        raise PipelineError("O contrato atual exige exatamente 9 fatias axiais (grade 3x3).")
    axial_indices = _select_uniform_indices(axial_present, axial_count)
    tiles: list[Image.Image] = []
    for number, z in enumerate(axial_indices, start=1):
        tiles.append(
            _render_tile(
                volume[z], mask[z], f"AXIAL {number}/9", tile_size, lo, hi,
                sy, sx, contour_width, contour_color,
            )
        )
    tiles.append(
        _render_tile(
            volume[:, yc, :], mask[:, yc, :], "CORONAL (CENTROIDE)",
            tile_size, lo, hi, sz, sx, contour_width, contour_color,
        )
    )
    tiles.append(
        _render_tile(
            volume[:, :, xc], mask[:, :, xc], "SAGITAL (CENTROIDE)",
            tile_size, lo, hi, sz, sy, contour_width, contour_color,
        )
    )

    canvas = Image.new("RGB", (tile_size * 4, tile_size * 3), (10, 14, 20))
    for i, tile in enumerate(tiles[:9]):
        canvas.paste(tile, ((i % 3) * tile_size, (i // 3) * tile_size))
    canvas.paste(tiles[9], (3 * tile_size, 0))
    canvas.paste(tiles[10], (3 * tile_size, tile_size))
    notice = Image.new("RGB", (tile_size, tile_size), (18, 24, 32))
    ndraw = ImageDraw.Draw(notice)
    ndraw.multiline_text(
        (14, 18),
        "MODO PESQUISA\n\nHipotese visual apenas.\nNAO e diagnostico.\nNAO e laudo medico.\n\nRevisao humana obrigatoria.",
        fill=(235, 240, 246),
        spacing=6,
    )
    canvas.paste(notice, (3 * tile_size, 2 * tile_size))

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / PANEL_FILENAME
    canvas.save(panel_path, format="PNG", optimize=True)
    with Image.open(panel_path) as exported:
        metadata_keys = sorted(exported.info.keys())
    if metadata_keys:
        raise PipelineError(
            f"PNG exportado contém metadados inesperados: {metadata_keys}. Abortando."
        )
    panel_sha256 = sha256_of(panel_path)

    manifest = {
        "case_id": case_manifest["case_id"],
        "organ": expected_organ,
        "modality": "MRI",
        "regulatory_mode": "RESEARCH",
        "input_type": "mri_with_liver_contour",
        "lesion_pre_marked": False,
        "panel_image": panel_path.name,
        "panel_sha256": panel_sha256,
        "input_volume_sha256": volume_sha256,
        "input_liver_mask_sha256": liver_mask_sha256,
        "panel_count": 11,
        "views": {
            "axial_indices_zyx": list(axial_indices),
            "coronal_centroid_y": yc,
            "sagittal_centroid_x": xc,
        },
        "liver_mask_voxels": mask_voxels,
        "png_metadata_keys": metadata_keys,
        "phi_metadata_removed": True,
        "visible_phi_review_required": True,
        "visible_phi_confirmed": bool(visible_phi_confirmed),
        "created_at": now_utc(),
        "requires_human_review": True,
        **model_trace,
        "notes": [
            "No lesion mask was read, provided, or rendered.",
            "No patient identifiers were added to the panel or PNG metadata.",
            "Burned-in pixel PHI cannot be ruled out automatically; visual confirmation is required before inference.",
            "Analysis is for research and education only.",
        ],
    }
    manifest_path = output_dir / PANEL_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return PanelResult(
        panel_path=panel_path,
        manifest_path=manifest_path,
        panel_count=11,
        axial_indices=axial_indices,
        coronal_index=yc,
        sagittal_index=xc,
        panel_paths=(panel_path,),
    )
