"""Testes do núcleo de cobertura volumétrica (dtwin/medgemma_volumetric.py).

Cobrem as garantias do Cenário B sem depender de torch, MedGemma ou GPU: só
PIL/NumPy. Renderers são stubs — a prova de cobertura vem da máscara, não do
conteúdo dos tiles.
"""
import numpy as np
import pytest
from PIL import Image

from dtwin.core import PipelineError
from dtwin.medgemma_volumetric import (
    effective_screening_timeout,
    estimate_panel_count,
    panel_strategy,
    render_volumetric_panel_set,
)

TILE = 160  # tiles pequenos mantêm os testes rápidos


def _mask_with_axial_slices(z_indices, shape_yx=(20, 20), depth=None):
    """Máscara 3D com fígado (bloco 10x10 = 100 voxels) exatamente nos z dados."""
    z_indices = sorted(int(z) for z in z_indices)
    depth = depth if depth is not None else (z_indices[-1] + 3)
    mask = np.zeros((depth, *shape_yx), dtype=bool)
    for z in z_indices:
        mask[z, 5:15, 5:15] = True
    return mask


def _stub_renderer(tile_size):
    def render(_index, _label):
        return Image.new("RGB", (tile_size, tile_size), (30, 30, 30))
    return render


def _render_set(mask, output_dir, *, tile_size=TILE, offset=(0, 0, 0),
                max_image_pixels=4_000_000, max_input_bytes=10_485_760):
    render = _stub_renderer(tile_size)
    return render_volumetric_panel_set(
        mask=mask, output_dir=output_dir, tile_size=tile_size,
        axial_tiles_per_panel=9, index_offset_zyx=offset,
        render_axial=render, render_coronal=render, render_sagittal=render,
        notice_text="MODO PESQUISA", max_image_pixels=max_image_pixels,
        max_input_bytes=max_input_bytes,
    )


def _axial_indices(panels):
    return sorted(
        t["index"] for p in panels for t in p["tiles"] if t["orientation"] == "axial"
    )


# --------------------------------------------------------------------------- #
# Planejamento: contagem de painéis e timeout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_slices,expected", [(1, 1), (5, 1), (9, 1), (10, 2), (36, 4), (37, 5)])
def test_panel_count_scales_without_upper_limit(n_slices, expected):
    mask = _mask_with_axial_slices(range(2, 2 + n_slices))
    assert estimate_panel_count(mask) == expected


def test_panel_strategy_validation():
    assert panel_strategy({}) == "uniform_9"
    assert panel_strategy({"strategy": "volumetric_blocks"}) == "volumetric_blocks"
    with pytest.raises(PipelineError):
        panel_strategy({"strategy": "inexistente"})


def test_effective_timeout_scales_with_panels():
    mask = _mask_with_axial_slices(range(0, 20))  # 20 cortes -> 3 painéis
    config = {
        "panel": {"strategy": "volumetric_blocks", "axial_tiles_per_panel": 9},
        "medgemma": {"timeout_seconds": 100, "max_retries": 1},
    }
    timeout, count = effective_screening_timeout(mask, config, configured_timeout=50)
    assert count == 3
    # 60 + 3 * (100 * (1 + 1) + 30) = 60 + 3 * 230 = 750
    assert timeout == 750


def test_effective_timeout_uniform_is_single_panel_and_honours_floor():
    mask = _mask_with_axial_slices(range(0, 20))
    config = {"panel": {"strategy": "uniform_9"},
              "medgemma": {"timeout_seconds": 100, "max_retries": 0}}
    timeout, count = effective_screening_timeout(mask, config, configured_timeout=600)
    assert count == 1
    assert timeout == 600  # o piso configurado vence 60 + 1*(100+30) = 190


# --------------------------------------------------------------------------- #
# Cobertura: 100% exato, cada corte uma vez, gaps, offset
# --------------------------------------------------------------------------- #
def test_full_coverage_each_axial_slice_exactly_once(tmp_path):
    z = list(range(3, 3 + 25))  # 25 cortes -> 3 painéis
    mask = _mask_with_axial_slices(z)
    result = _render_set(mask, tmp_path)
    cov = result.coverage
    assert cov["gate_passed"] is True
    assert cov["covered_liver_voxels"] == cov["total_liver_voxels"] == int(mask.sum())
    assert cov["coverage_percent"] == 100.0
    assert cov["expected_axial_indices"] == z
    assert cov["first_liver_slice"] == z[0]
    assert cov["last_liver_slice"] == z[-1]
    assert cov["missing_axial_indices"] == []
    assert cov["duplicate_axial_indices"] == []
    assert _axial_indices(result.panels) == z  # cada corte exatamente uma vez


def test_non_contiguous_axial_gaps_are_all_covered(tmp_path):
    z = [2, 3, 4, 10, 11, 20]  # intervalos axiais vazios entre os grupos
    mask = _mask_with_axial_slices(z, depth=25)
    result = _render_set(mask, tmp_path)
    assert result.coverage["gate_passed"] is True
    assert result.coverage["expected_axial_indices"] == sorted(z)
    assert _axial_indices(result.panels) == sorted(z)


def test_index_offset_is_applied_to_global_indices(tmp_path):
    mask = _mask_with_axial_slices([1, 2, 3], depth=6)
    result = _render_set(mask, tmp_path, offset=(100, 50, 30))
    assert result.coverage["first_liver_slice"] == 101
    assert result.coverage["last_liver_slice"] == 103
    assert _axial_indices(result.panels) == [101, 102, 103]


def test_single_slice_mask_is_allowed(tmp_path):
    result = _render_set(_mask_with_axial_slices([5], depth=10), tmp_path)
    assert len(result.panel_paths) == 1
    assert result.coverage["gate_passed"] is True


def test_empty_mask_aborts(tmp_path):
    with pytest.raises(PipelineError):
        _render_set(np.zeros((10, 20, 20), dtype=bool), tmp_path)


# --------------------------------------------------------------------------- #
# Nomes determinísticos, metadados e PHI
# --------------------------------------------------------------------------- #
def test_deterministic_filenames_and_order(tmp_path):
    mask = _mask_with_axial_slices(range(0, 10))  # 10 -> 2 painéis
    result = _render_set(mask, tmp_path)
    assert [p.name for p in result.panel_paths] == [
        "medgemma_liver_screening_panel_001_of_002.png",
        "medgemma_liver_screening_panel_002_of_002.png",
    ]
    assert all(p.is_file() for p in result.panel_paths)
    for record in result.panels:
        assert len(record["sha256"]) == 64


def test_tile_metadata_covers_three_orientations(tmp_path):
    result = _render_set(_mask_with_axial_slices(range(0, 5)), tmp_path)
    tiles = result.panels[0]["tiles"]
    orientations = {t["orientation"] for t in tiles}
    assert {"axial", "coronal", "sagittal"} <= orientations
    required = {
        "tile_number", "orientation", "index", "relative_position_percent",
        "liver_voxels_in_plane", "liver_volume_percent", "counts_toward_coverage",
    }
    for t in tiles:
        assert required <= set(t)
        assert t["counts_toward_coverage"] is (t["orientation"] == "axial")


def test_last_panel_empty_axial_slots_do_not_count_as_coverage(tmp_path):
    mask = _mask_with_axial_slices(range(0, 10))  # 10 cortes -> painel 2 tem 1 axial real
    result = _render_set(mask, tmp_path)
    last = result.panels[-1]
    axial_tiles = [t for t in last["tiles"] if t["orientation"] == "axial"]
    assert len(axial_tiles) == 1  # tiles vazios não geram registro nem contam cobertura


def test_no_phi_metadata_in_exported_png(tmp_path):
    result = _render_set(_mask_with_axial_slices(range(0, 5)), tmp_path)
    assert result.png_metadata_keys == ()


# --------------------------------------------------------------------------- #
# Limites por painel
# --------------------------------------------------------------------------- #
def test_gate_aborts_when_panel_exceeds_pixel_limit(tmp_path):
    with pytest.raises(PipelineError, match="max_image_pixels"):
        _render_set(_mask_with_axial_slices(range(0, 5)), tmp_path, max_image_pixels=100)


def test_gate_aborts_when_panel_exceeds_byte_limit(tmp_path):
    with pytest.raises(PipelineError, match="max_input_bytes"):
        _render_set(_mask_with_axial_slices(range(0, 5)), tmp_path, max_input_bytes=10)
