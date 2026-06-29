# tests/test_stages_units.py
import numpy as np

from dtwin.stages import _refine_mask, _mesh_from_mask
from dtwin.core import save_image, array_to_image
from .conftest import make_sphere_mask, make_geo_image


def test_refine_removes_small_objects():
    m = np.zeros((20, 20, 20), np.uint8)
    m[2:6, 2:6, 2:6] = 1            # big blob (64 voxels)
    m[15, 15, 15] = 1              # speck (1 voxel)
    out = _refine_mask(m, opening=False, radius=1, min_voxels=10)
    assert out[15, 15, 15] == 0
    assert out[3, 3, 3] == 1


def test_refine_does_not_zero_solid_mask():
    m = make_sphere_mask((30, 30, 30), (15, 15, 15), 8)
    out = _refine_mask(m, opening=True, radius=2, min_voxels=50)
    assert out.sum() > 0


def test_mesh_from_full_mask_is_nonempty(tmp_path):
    m = make_sphere_mask((30, 30, 30), (15, 15, 15), 8)
    ref = make_geo_image(m)
    path = tmp_path / "mask.nii.gz"
    save_image(array_to_image(m, ref, np.uint8), path)
    mesh = _mesh_from_mask(path, level=0.5, smooth_iter=0, feature_angle=60.0)
    assert mesh is not None
    assert mesh.n_points > 0 and mesh.n_cells > 0
    # vertices in physical coords: origin x=10, so min x near 10, not 0
    assert mesh.points[:, 0].min() >= 9.0


def test_mesh_from_empty_mask_is_none(tmp_path):
    m = np.zeros((10, 10, 10), np.uint8)
    ref = make_geo_image(m)
    path = tmp_path / "empty.nii.gz"
    save_image(array_to_image(m, ref, np.uint8), path)
    assert _mesh_from_mask(path, 0.5, 0, 60.0) is None
