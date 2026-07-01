#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Montagem 2D multifásica (fusão RGB) para triagem visual hepática MedGemma.

Objetivo: aumentar a acurácia do MedGemma dando a ele o sinal que define o HCC —
a DINÂMICA de realce entre fases (realce arterial + washout tardio) — já
pré-computada nos pixels como cor, em vez de exigir raciocínio entre fatias em
escala de cinza.

Diferenças em relação ao painel de fase única (dtwin/medgemma_panel.py):
  - várias fases de RM co-registradas fundidas em canais RGB (ex.: R=arterial,
    G=portal-venoso, B=tardio);
  - recorte no bounding-box do FÍGADO (usa só a máscara do órgão) para ampliar a
    resolução efetiva sobre o parênquima;
  - janela de intensidade calculada DENTRO do fígado, por fase.

Invariantes de segurança preservados (idênticos ao fluxo de fase única):
  - Cenário A: NUNCA lê, recebe ou renderiza uma máscara de lesão. Não existe
    caminho para isso por design.
  - Sem PHI: PNG sem metadados textuais; revisão humana de PHI queimada continua
    obrigatória antes da inferência (gate no orquestrador).
  - Rastreabilidade do modelo e revisão humana obrigatória no manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from .core import (
    PipelineError,
    array_from,
    array_to_image,
    now_utc,
    read_image,
    save_image,
    sha256_of,
)
from .medgemma_panel import (
    PANEL_FILENAME,
    PANEL_MANIFEST_FILENAME,
    PanelResult,
    _geometry_compatible,
    _require_file,
    _select_uniform_indices,
    _validate_case_manifest,
)

RGB_CHANNELS = ("red", "green", "blue")


def derive_texture_channels(
    volume_path: Path, liver_mask_path: Path, out_dir: Path, panel_cfg: Mapping[str, Any]
) -> dict[str, Path]:
    """Deriva 3 canais de UM único volume de RM (fase única) para a fusão RGB:

    - ``signal``: a própria RM (intensidade);
    - ``clahe``:  realce local de contraste (CLAHE) — levanta lesões sutis;
    - ``hetero``: heterogeneidade local (desvio-padrão numa janela) — lesões focais
      costumam ser mais heterogêneas que o parênquima liso ao redor.

    Assim, mesmo sem fases dinâmicas, a lesão pode "acender" em cor na montagem.
    Nenhuma máscara de lesão é usada (Cenário A). Retorna os caminhos dos .nii.gz.
    """
    from scipy.ndimage import uniform_filter
    from skimage.exposure import equalize_adapthist

    volume_path, liver_mask_path, out_dir = Path(volume_path), Path(liver_mask_path), Path(out_dir)
    _require_file(volume_path, "Volume de RM")
    _require_file(liver_mask_path, "Máscara do fígado")
    vol_img = read_image(volume_path)
    if vol_img.GetDimension() != 3:
        raise PipelineError("A textura-fusão exige um volume 3D.")
    mask_img = read_image(liver_mask_path)
    if not _geometry_compatible(vol_img, mask_img):
        raise PipelineError("Volume e máscara do fígado têm geometria incompatível.")

    vol = array_from(vol_img).astype(np.float32)
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        raise PipelineError("Volume sem intensidades finitas.")
    lo, hi = np.percentile(finite, [1.0, 99.0]).astype(float)
    norm = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    clahe = equalize_adapthist(norm, clip_limit=float(panel_cfg.get("clahe_clip", 0.01))).astype(np.float32)
    k = int(panel_cfg.get("texture_kernel", 5))
    mean = uniform_filter(norm, size=k)
    mean_sq = uniform_filter(norm * norm, size=k)
    hetero = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None)).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, arr in (("signal", vol), ("clahe", clahe), ("hetero", hetero)):
        path = out_dir / f"{name}.nii.gz"
        save_image(array_to_image(arr, vol_img, np.float32), path)
        paths[name] = path
    return paths


def _resolve_channel_map(panel_cfg: Mapping[str, Any]) -> dict[str, str]:
    fusion = panel_cfg.get("fusion", {})
    channel_map = fusion.get("channel_map", {})
    if not isinstance(channel_map, dict) or set(channel_map) != set(RGB_CHANNELS):
        raise PipelineError(
            "panel.fusion.channel_map deve mapear exatamente red/green/blue -> nome de fase."
        )
    resolved = {c: str(channel_map[c]) for c in RGB_CHANNELS}
    if any(not v for v in resolved.values()):
        raise PipelineError("panel.fusion.channel_map tem fase vazia.")
    return resolved


def _crop_bounds(mask: np.ndarray, margin_frac: float) -> tuple[slice, slice, slice]:
    bounds = []
    for axis in range(3):
        present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        if present.size == 0:
            raise PipelineError("Máscara do fígado vazia; recorte impossível.")
        lo, hi = int(present.min()), int(present.max()) + 1
        margin = int(round((hi - lo) * float(margin_frac)))
        lo = max(0, lo - margin)
        hi = min(mask.shape[axis], hi + margin)
        bounds.append(slice(lo, hi))
    return tuple(bounds)  # type: ignore[return-value]


def _phase_window(arr: np.ndarray, liver: np.ndarray, low: float, high: float, scope: str) -> tuple[float, float]:
    if scope == "liver":
        vals = arr[liver & np.isfinite(arr)]
    else:
        vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        raise PipelineError("Fase sem intensidades finitas no escopo da janela.")
    lo, hi = np.percentile(vals, [low, high]).astype(float)
    if hi - lo < 1e-6:
        lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-6:
        raise PipelineError("Fase sem contraste suficiente para gerar a janela.")
    return lo, hi


def _render_color_tile(
    rgb01: np.ndarray,
    mask_2d: np.ndarray,
    label: str,
    tile_size: int,
    row_spacing: float,
    col_spacing: float,
    contour_width: int,
    contour_color: tuple[int, int, int],
    background_dim: float,
) -> Image.Image:
    rgb = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    if background_dim < 1.0:
        outside = ~mask_2d.astype(bool)
        rgb[outside] = (rgb[outside].astype(np.float32) * float(background_dim)).astype(np.uint8)
    boundary = find_boundaries(mask_2d.astype(bool), mode="inner")
    if contour_width > 1:
        boundary = binary_dilation(boundary, iterations=contour_width - 1)
    rgb[boundary] = np.asarray(contour_color, dtype=np.uint8)

    source = Image.fromarray(np.flipud(rgb), mode="RGB")
    physical_w = max(1.0, source.width * float(col_spacing))
    physical_h = max(1.0, source.height * float(row_spacing))
    usable = tile_size - 28
    scale = min(usable / physical_w, usable / physical_h)
    target = (max(1, int(round(physical_w * scale))), max(1, int(round(physical_h * scale))))
    source = source.resize(target, Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (tile_size, tile_size), (10, 14, 20))
    tile.paste(source, ((tile_size - target[0]) // 2, 24 + (usable - target[1]) // 2))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 6), label, fill=(235, 240, 246))
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(52, 61, 72))
    return tile


def generate_liver_panel_multiphase(
    *,
    phase_paths: Mapping[str, Path],
    liver_mask_path: Path,
    case_manifest_path: Path,
    organ_profile: dict[str, Any],
    screening_config: dict[str, Any],
    output_dir: Path,
    model_trace: dict[str, Any],
    visible_phi_confirmed: bool = False,
) -> PanelResult:
    """Valida entradas e gera 9 axiais + 1 coronal + 1 sagital em fusão RGB.

    Nenhum caminho ou argumento para máscara de lesão existe por design.
    """
    liver_mask_path = Path(liver_mask_path)
    case_manifest_path, output_dir = Path(case_manifest_path), Path(output_dir)
    panel_cfg = screening_config.get("panel", {})
    channel_map = _resolve_channel_map(panel_cfg)

    required_phases = sorted(set(channel_map.values()))
    phase_paths = {name: Path(p) for name, p in phase_paths.items()}
    missing_phases = [name for name in required_phases if name not in phase_paths]
    if missing_phases:
        raise PipelineError(
            f"Fases exigidas pelo channel_map ausentes na entrada: {missing_phases}."
        )
    for name in required_phases:
        _require_file(phase_paths[name], f"Fase de RM '{name}'")
    _require_file(liver_mask_path, "Máscara do fígado")

    case_manifest = _validate_case_manifest(case_manifest_path)

    mask_img = read_image(liver_mask_path)
    if mask_img.GetDimension() != 3:
        raise PipelineError("A montagem MedGemma exige um volume 3D.")
    phase_imgs: dict[str, Any] = {}
    for name in required_phases:
        img = read_image(phase_paths[name])
        if not _geometry_compatible(img, mask_img):
            raise PipelineError(
                f"Fase '{name}' e máscara do fígado têm geometria incompatível "
                "(as fases precisam estar co-registradas na mesma grade)."
            )
        phase_imgs[name] = img

    expected_organ = str(screening_config.get("organ", ""))
    profile_organ = str(organ_profile.get("segmentacao_orgao", {}).get("rotulo_alvo", ""))
    if not expected_organ or profile_organ != expected_organ:
        raise PipelineError(
            f"Perfil/config incompatíveis para MedGemma: {profile_organ!r} != {expected_organ!r}."
        )

    mask_full = array_from(mask_img) > 0
    mask_voxels = int(mask_full.sum())
    validation_cfg = screening_config.get("validation", {})
    min_voxels = int(validation_cfg.get("min_liver_voxels", 300))
    max_fraction = float(validation_cfg.get("max_liver_fraction", 0.8))
    if mask_voxels == 0:
        raise PipelineError("Máscara do fígado vazia.")
    if mask_voxels < min_voxels:
        raise PipelineError(
            f"Máscara do fígado implausivelmente pequena ({mask_voxels} < {min_voxels} voxels)."
        )
    if mask_voxels / float(mask_full.size) > max_fraction:
        raise PipelineError("Máscara do fígado implausivelmente grande.")

    # Recorte no fígado (usa só a máscara do órgão — nenhuma pista de lesão).
    if bool(panel_cfg.get("crop_to_liver", True)):
        margin_frac = float(panel_cfg.get("crop_margin_frac", 0.08))
        zc_s, yc_s, xc_s = _crop_bounds(mask_full, margin_frac)
    else:
        zc_s = yc_s = xc_s = slice(None)
    mask = mask_full[zc_s, yc_s, xc_s]

    low = float(panel_cfg.get("window_percentile_low", 2.0))
    high = float(panel_cfg.get("window_percentile_high", 98.0))
    if not 0 <= low < high <= 100:
        raise PipelineError("Percentis da janela de intensidade são inválidos.")
    scope = str(panel_cfg.get("window_scope", "liver"))
    if scope not in {"liver", "volume"}:
        raise PipelineError("panel.window_scope deve ser 'liver' ou 'volume'.")

    # Normaliza cada fase (recortada) para [0,1] com janela dentro do fígado.
    norm_phase: dict[str, np.ndarray] = {}
    phase_sha256: dict[str, str] = {}
    for name in required_phases:
        arr = array_from(phase_imgs[name]).astype(np.float32)[zc_s, yc_s, xc_s]
        lo, hi = _phase_window(arr, mask, low, high, scope)
        norm_phase[name] = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        phase_sha256[name] = sha256_of(phase_paths[name])

    axial_count = int(panel_cfg.get("axial_slices", 9))
    if axial_count != 9:
        raise PipelineError("O contrato atual exige exatamente 9 fatias axiais (grade 3x3).")
    axial_present = np.flatnonzero(mask.any(axis=(1, 2)))
    axial_indices = _select_uniform_indices(axial_present, axial_count)
    centroid_zyx = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
    _zc, yc, xc = (int(v) for v in centroid_zyx)

    tile_size = int(panel_cfg.get("tile_size", 384))
    if tile_size < 128:
        raise PipelineError("panel.tile_size deve ser >= 128.")
    contour_width = int(panel_cfg.get("contour_width", 1))
    color_raw = panel_cfg.get("contour_color_rgb", [255, 255, 255])
    if not isinstance(color_raw, list) or len(color_raw) != 3:
        raise PipelineError("panel.contour_color_rgb deve conter três inteiros RGB.")
    contour_color = tuple(int(np.clip(v, 0, 255)) for v in color_raw)
    background_dim = float(panel_cfg.get("background_dim", 1.0))
    if not 0.0 <= background_dim <= 1.0:
        raise PipelineError("panel.background_dim deve estar em [0, 1].")

    sx, sy, sz = (float(v) for v in mask_img.GetSpacing())
    r, g, b = (channel_map[c] for c in RGB_CHANNELS)

    def fuse(sl) -> np.ndarray:
        return np.stack([norm_phase[r][sl], norm_phase[g][sl], norm_phase[b][sl]], axis=-1)

    tiles: list[Image.Image] = []
    for number, z in enumerate(axial_indices, start=1):
        tiles.append(
            _render_color_tile(
                fuse(np.s_[z]),
                mask[z], f"AXIAL {number}/9", tile_size, sy, sx,
                contour_width, contour_color, background_dim,
            )
        )
    tiles.append(
        _render_color_tile(
            fuse(np.s_[:, yc, :]), mask[:, yc, :], "CORONAL (CENTROIDE)",
            tile_size, sz, sx, contour_width, contour_color, background_dim,
        )
    )
    tiles.append(
        _render_color_tile(
            fuse(np.s_[:, :, xc]), mask[:, :, xc], "SAGITAL (CENTROIDE)",
            tile_size, sz, sy, contour_width, contour_color, background_dim,
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
        "MODO PESQUISA\n\nFusao multifasica RGB:\n"
        f"R={r}  G={g}  B={b}\n\n"
        "Hipotese visual apenas.\nNAO e diagnostico.\n\nRevisao humana obrigatoria.",
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
    liver_mask_sha256 = sha256_of(liver_mask_path)
    representative_phase = channel_map["red"]

    manifest = {
        "case_id": case_manifest["case_id"],
        "organ": expected_organ,
        "modality": "MRI",
        "regulatory_mode": "RESEARCH",
        "input_type": "mri_multiphase_rgb_fusion_liver_crop",
        "lesion_pre_marked": False,
        "panel_image": panel_path.name,
        "panel_sha256": panel_sha256,
        # input_volume_sha256 mantém o envelope do relatório compatível (fase R).
        "input_volume_sha256": phase_sha256[representative_phase],
        "input_phase_sha256": phase_sha256,
        "input_liver_mask_sha256": liver_mask_sha256,
        "fusion_channel_map": channel_map,
        "phases_used": required_phases,
        "crop_to_liver": bool(panel_cfg.get("crop_to_liver", True)),
        "crop_bounds_zyx": [[zc_s.start, zc_s.stop], [yc_s.start, yc_s.stop], [xc_s.start, xc_s.stop]]
        if zc_s.start is not None
        else None,
        "panel_count": 11,
        "views": {
            "axial_indices_zyx_cropped": list(axial_indices),
            "coronal_centroid_y_cropped": yc,
            "sagittal_centroid_x_cropped": xc,
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
            "Phases fused into RGB channels; lesion conspicuity comes from enhancement dynamics.",
            "Liver-only crop and windowing use the organ mask only, never a lesion mask.",
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
    )
