# tests/conftest.py
import json
import numpy as np
import SimpleITK as sitk
import pytest

from dtwin.core import Case, array_to_image, save_image, now_utc


def make_sphere_mask(shape, center, radius):
    """Binary sphere in (z, y, x) array order."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return (d2 <= radius * radius).astype(np.uint8)


def make_geo_image(arr, spacing=(1.5, 1.0, 1.0), origin=(10.0, -5.0, 3.0)):
    """Wrap a (z, y, x) array as a sitk image with non-trivial geometry.

    spacing/origin are in SimpleITK (x, y, z) order.
    """
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in origin))
    return img


@pytest.fixture
def synthetic_case(tmp_path):
    shape = (40, 40, 40)
    center = (20, 20, 20)
    organ = make_sphere_mask(shape, center, 12)
    lesion = make_sphere_mask(shape, center, 4)
    volume_arr = (organ.astype(np.float32) * 100.0) + np.float32(10.0)

    case = Case(tmp_path)
    ref = make_geo_image(volume_arr)
    save_image(ref, case.volume)
    save_image(array_to_image(organ, ref, np.uint8), case.mask_organ)
    save_image(array_to_image(lesion, ref, np.uint8), case.mask_lesion)
    case.write_manifest(
        {
            "case_id": "anon-test000000",
            "policy": "anonymize",
            "modality": "MR",
            "regulatory_state": "PESQUISA",
            "size_xyz": [shape[2], shape[1], shape[0]],
            "created_utc": now_utc(),
        }
    )
    return case
