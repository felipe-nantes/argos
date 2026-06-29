# tests/test_core_geometry.py
import numpy as np
import SimpleITK as sitk

from dtwin.core import world_vertices_from_index
from .conftest import make_geo_image


def test_identity_direction_maps_index_to_physical():
    # spacing xyz=(1.5,1.0,1.0), origin xyz=(10,-5,3), identity direction.
    ref = make_geo_image(np.zeros((4, 4, 4), np.uint8))
    # one vertex at index (z,y,x) = (2, 1, 3)
    verts_zyx = np.array([[2.0, 1.0, 3.0]])
    out = world_vertices_from_index(verts_zyx, ref)
    # physical = origin + spacing * (x, y, z) = (10+1.5*3, -5+1.0*1, 3+1.0*2)
    np.testing.assert_allclose(out[0], [14.5, -4.0, 5.0], atol=1e-9)


def test_oblique_direction_is_applied():
    arr = np.zeros((4, 4, 4), np.uint8)
    ref = sitk.GetImageFromArray(arr)
    ref.SetSpacing((2.0, 2.0, 2.0))
    ref.SetOrigin((0.0, 0.0, 0.0))
    # 90-deg rotation in-plane: x->y, y->-x (column-major direction cosines)
    ref.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    verts_zyx = np.array([[0.0, 0.0, 1.0]])  # index x=1
    out = world_vertices_from_index(verts_zyx, ref)
    # idx_xyz*spacing = (2,0,0); direction@that = (0*2, 1*2, 0) = (0,2,0)
    np.testing.assert_allclose(out[0], [0.0, 2.0, 0.0], atol=1e-9)
