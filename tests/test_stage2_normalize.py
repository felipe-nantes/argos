# tests/test_stage2_normalize.py
import numpy as np
import pytest

from dtwin import stages
from dtwin.core import Case, save_image, array_to_image, read_image, array_from, PipelineError
from .conftest import make_geo_image


def _case_with_volume(tmp_path, arr):
    case = Case(tmp_path)
    ref = make_geo_image(arr)
    save_image(ref, case.volume)
    return case


def test_zscore_produces_zero_mean_unit_std(tmp_path):
    arr = np.random.default_rng(0).normal(50, 10, size=(20, 20, 20)).astype(np.float32)
    case = _case_with_volume(tmp_path, arr)
    stages.stage2_normalize(case, {"normalizacao": "zscore"})
    out = array_from(read_image(case.volume_zscore))
    assert abs(float(out.mean())) < 1e-3
    assert abs(float(out.std()) - 1.0) < 1e-2


def test_constant_volume_aborts(tmp_path):
    arr = np.full((10, 10, 10), 7.0, np.float32)
    case = _case_with_volume(tmp_path, arr)
    with pytest.raises(PipelineError):
        stages.stage2_normalize(case, {"normalizacao": "zscore"})


def test_unknown_method_aborts(tmp_path):
    arr = np.random.default_rng(1).normal(0, 1, size=(10, 10, 10)).astype(np.float32)
    case = _case_with_volume(tmp_path, arr)
    with pytest.raises(PipelineError):
        stages.stage2_normalize(case, {"normalizacao": "bogus"})
