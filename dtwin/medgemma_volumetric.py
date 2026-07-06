"""Planejamento e composição auditável de painéis com cobertura hepática integral."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .core import PipelineError, sha256_of


@dataclass(frozen=True)
class VolumetricPanelSet:
    panel_paths: tuple[Path, ...]
    panels: tuple[dict, ...]
    axial_indices: tuple[int, ...]
    coverage: dict
    png_metadata_keys: tuple[str, ...]


def panel_strategy(panel_cfg: dict) -> str:
    strategy = str(panel_cfg.get("strategy", "uniform_9"))
    if strategy not in {"uniform_9", "volumetric_blocks"}:
        raise PipelineError(
            "panel.strategy deve ser 'uniform_9' ou 'volumetric_blocks'."
        )
    return strategy


def estimate_panel_count(mask: np.ndarray, axial_tiles_per_panel: int = 9) -> int:
    if axial_tiles_per_panel <= 0:
        raise PipelineError("panel.axial_tiles_per_panel deve ser positivo.")
    present = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=(1, 2)))
    if present.size == 0:
        raise PipelineError("Máscara do fígado vazia.")
    return int((present.size + axial_tiles_per_panel - 1) // axial_tiles_per_panel)


def effective_screening_timeout(
    mask: np.ndarray, config: dict, configured_timeout: int
) -> tuple[int, int]:
    panel_cfg = config.get("panel", {})
    count = (
        estimate_panel_count(mask, int(panel_cfg.get("axial_tiles_per_panel", 9)))
        if panel_strategy(panel_cfg) == "volumetric_blocks" else 1
    )
    med = config["medgemma"]
    calculated = 60 + count * (
        int(med["timeout_seconds"]) * (int(med.get("max_retries", 0)) + 1) + 30
    )
    return max(int(configured_timeout), calculated), count


def _relative(index: int, first: int, last: int) -> float:
    if first == last:
        return 50.0
    return round(100.0 * (index - first) / (last - first), 4)


def _empty_tile(tile_size: int) -> Image.Image:
    tile = Image.new("RGB", (tile_size, tile_size), (18, 24, 32))
    draw = ImageDraw.Draw(tile)
    draw.text((12, 12), "SEM CORTE — FIM DA COBERTURA", fill=(150, 160, 170))
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(52, 61, 72))
    return tile


def _notice_tile(tile_size: int, text: str) -> Image.Image:
    tile = Image.new("RGB", (tile_size, tile_size), (18, 24, 32))
    ImageDraw.Draw(tile).multiline_text(
        (14, 18), text, fill=(235, 240, 246), spacing=6
    )
    return tile


def render_volumetric_panel_set(
    *,
    mask: np.ndarray,
    output_dir: Path,
    tile_size: int,
    axial_tiles_per_panel: int,
    index_offset_zyx: tuple[int, int, int],
    render_axial: Callable[[int, str], Image.Image],
    render_coronal: Callable[[int, str], Image.Image],
    render_sagittal: Callable[[int, str], Image.Image],
    notice_text: str,
    max_image_pixels: int,
    max_input_bytes: int,
) -> VolumetricPanelSet:
    """Renderiza todos os planos axiais com fígado e prova a união voxel a voxel."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise PipelineError("Cobertura volumétrica exige máscara hepática 3D não vazia.")
    if axial_tiles_per_panel != 9:
        raise PipelineError("A grade 4x3 exige axial_tiles_per_panel=9.")

    local_axial = tuple(int(v) for v in np.flatnonzero(mask.any(axis=(1, 2))))
    zoff, yoff, xoff = index_offset_zyx
    axial_indices = tuple(z + zoff for z in local_axial)
    flattened = list(axial_indices)
    duplicates = sorted({v for v in flattened if flattened.count(v) > 1})
    expected = list(axial_indices)
    missing = sorted(set(expected) - set(flattened))

    coords = np.argwhere(mask)
    _zc, yc, xc = np.rint(coords.mean(axis=0)).astype(int)
    bounds = {
        "axial": (int(coords[:, 0].min()) + zoff, int(coords[:, 0].max()) + zoff),
        "coronal": (int(coords[:, 1].min()) + yoff, int(coords[:, 1].max()) + yoff),
        "sagittal": (int(coords[:, 2].min()) + xoff, int(coords[:, 2].max()) + xoff),
    }
    total_voxels = int(mask.sum())
    covered = np.zeros_like(mask, dtype=bool)
    panel_total = estimate_panel_count(mask, axial_tiles_per_panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    panel_records: list[dict] = []
    metadata_keys_all: set[str] = set()

    for panel_idx in range(panel_total):
        start = panel_idx * axial_tiles_per_panel
        block_local = local_axial[start : start + axial_tiles_per_panel]
        block_global = axial_indices[start : start + axial_tiles_per_panel]
        canvas = Image.new("RGB", (tile_size * 4, tile_size * 3), (10, 14, 20))
        tile_records: list[dict] = []
        rendered: list[Image.Image] = []
        try:
            for tile_idx in range(axial_tiles_per_panel):
                if tile_idx < len(block_local):
                    zlocal = block_local[tile_idx]
                    zglobal = block_global[tile_idx]
                    voxels = int(mask[zlocal].sum())
                    rel = _relative(zglobal, *bounds["axial"])
                    percent = round(100.0 * voxels / total_voxels, 6)
                    label = f"AXIAL z={zglobal} | rel={rel:.1f}% | vol={percent:.3f}%"
                    tile = render_axial(zlocal, label)
                    covered[zlocal] |= mask[zlocal]
                    tile_records.append({
                        "tile_number": tile_idx + 1,
                        "orientation": "axial",
                        "index": zglobal,
                        "relative_position_percent": rel,
                        "liver_voxels_in_plane": voxels,
                        "liver_volume_percent": percent,
                        "counts_toward_coverage": True,
                    })
                else:
                    tile = _empty_tile(tile_size)
                rendered.append(tile)
                canvas.paste(tile, ((tile_idx % 3) * tile_size, (tile_idx // 3) * tile_size))

            for orientation, local_index, global_index, render, position in (
                ("coronal", int(yc), int(yc) + yoff, render_coronal, (3 * tile_size, 0)),
                ("sagittal", int(xc), int(xc) + xoff, render_sagittal, (3 * tile_size, tile_size)),
            ):
                if orientation == "coronal":
                    voxels = int(mask[:, local_index, :].sum())
                else:
                    voxels = int(mask[:, :, local_index].sum())
                rel = _relative(global_index, *bounds[orientation])
                percent = round(100.0 * voxels / total_voxels, 6)
                label = f"{orientation.upper()} i={global_index} | rel={rel:.1f}% | vol={percent:.3f}%"
                tile = render(local_index, label)
                rendered.append(tile)
                canvas.paste(tile, position)
                tile_records.append({
                    "tile_number": 10 if orientation == "coronal" else 11,
                    "orientation": orientation,
                    "index": global_index,
                    "relative_position_percent": rel,
                    "liver_voxels_in_plane": voxels,
                    "liver_volume_percent": percent,
                    "counts_toward_coverage": False,
                })
            notice = _notice_tile(tile_size, notice_text)
            rendered.append(notice)
            canvas.paste(notice, (3 * tile_size, 2 * tile_size))

            filename = (
                f"medgemma_liver_screening_panel_{panel_idx + 1:03d}"
                f"_of_{panel_total:03d}.png"
            )
            path = output_dir / filename
            if canvas.width * canvas.height > int(max_image_pixels):
                raise PipelineError(f"Painel {filename} excede max_image_pixels.")
            canvas.save(path, format="PNG", optimize=True)
            if path.stat().st_size > int(max_input_bytes):
                raise PipelineError(f"Painel {filename} excede max_input_bytes.")
            with Image.open(path) as exported:
                keys = sorted(exported.info.keys())
            if keys:
                raise PipelineError(
                    f"PNG exportado contém metadados inesperados: {keys}. Abortando."
                )
            metadata_keys_all.update(keys)
            digest = sha256_of(path)
            paths.append(path)
            panel_records.append({
                "panel_number": panel_idx + 1,
                "panel_total": panel_total,
                "image": filename,
                "sha256": digest,
                "axial_indices": list(block_global),
                "axial_interval": [block_global[0], block_global[-1]],
                "tiles": tile_records,
            })
        finally:
            canvas.close()
            for tile in rendered:
                tile.close()

    covered_voxels = int((covered & mask).sum())
    gate_passed = (
        covered_voxels == total_voxels and not missing and not duplicates
    )
    coverage = {
        "expected_axial_indices": expected,
        "first_liver_slice": expected[0],
        "last_liver_slice": expected[-1],
        "missing_axial_indices": missing,
        "duplicate_axial_indices": duplicates,
        "total_liver_voxels": total_voxels,
        "covered_liver_voxels": covered_voxels,
        "coverage_percent": round(100.0 * covered_voxels / total_voxels, 6),
        "gate_passed": gate_passed,
        "gate_rule": "covered_liver_voxels == total_liver_voxels",
    }
    if not gate_passed:
        raise PipelineError(f"Gate de cobertura volumétrica falhou: {coverage}")
    return VolumetricPanelSet(
        panel_paths=tuple(paths), panels=tuple(panel_records),
        axial_indices=axial_indices, coverage=coverage,
        png_metadata_keys=tuple(sorted(metadata_keys_all)),
    )
