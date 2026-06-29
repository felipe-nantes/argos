# Digital Twin Cirúrgico — Produção-Ready Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the existing Digital Twin pipeline from "code exists" to "fully functional and verifiable" — packaging, an automated test suite, a synthetic fixture + smoke test, a static Three.js viewer, and a `doctor` preflight — while staying in Research mode.

**Architecture:** The deterministic, organ-agnostic engine (`dtwin/`) stays untouched in contract — domain rules remain in YAML profiles. New work wraps it: PEP 621 packaging, a `pytest` suite that exercises every stage except the GPU-bound segmentation (stubbed/monkeypatched), a generator that writes a synthetic case to disk so `finalize` runs end-to-end with no GPU/Slicer/DICOM, a zero-build HTML viewer that reads `viewer_manifest.json` + STLs, and a small `doctor` CLI subcommand.

**Tech Stack:** Python 3.13 (venv), SimpleITK, numpy, scipy, scikit-image, pyvista, pydicom, PyYAML, nibabel (core); TotalSegmentator (optional `[seg]` extra, GPU box only); pytest (dev); Three.js + STLLoader + OrbitControls via CDN (viewer).

## Global Constraints

- `requires-python = ">=3.10,<3.14"`. Local venv uses Python **3.13** (`py -3.13`).
- TotalSegmentator/torch are **not** installed locally — only via the `[seg]` extra on the GPU machine. No test may import `totalsegmentator` at module load; stage 3 is always monkeypatched in tests.
- Package version is `0.1.0` and must match `dtwin.__version__`.
- Domain rules live in YAML profiles, never in engine code. No task changes that contract.
- Output STL paths in `viewer_manifest.json` stay **relative** (filenames only) — the viewer depends on this.
- Coordinate system is **LPS** throughout.
- Project stays in **Research mode**: every user-facing surface (viewer, docs) carries the "NÃO destinado a decisão clínica" disclaimer.
- Use `pytest` as the test runner. Commit after every task with the shown message.

---

### Task 1: Packaging + repo hygiene

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Modify: `requirements.txt` (add pointer comment to pyproject as source of truth)

**Interfaces:**
- Consumes: nothing.
- Produces: installable package `digital-twin-cirurgico`; console script `digital-twin` → `digital_twin:main`; extras `seg` and `dev`. Later tasks rely on `pip install -e .[dev]` working and on `pytest` being available.

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/

# Pipeline artifacts (patient/output data never committed)
casos/
flywheel/
*.nii
*.nii.gz
*.stl
*.vtp

# OS / editor
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "digital-twin-cirurgico"
version = "0.1.0"
description = "Pipeline órgão-agnóstico: DICOM (RM) -> modelo 3D (órgão + lesão). MVP Nível 1, modo Pesquisa."
readme = "README.md"
requires-python = ">=3.10,<3.14"
license = { text = "Proprietary" }
authors = [{ name = "UEM · GETS · HU" }]
dependencies = [
    "SimpleITK>=2.3",
    "nibabel>=5.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-image>=0.22",
    "pyvista>=0.43",
    "pydicom>=2.4",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
seg = ["TotalSegmentator>=2.4"]
dev = ["pytest>=8"]

[project.scripts]
digital-twin = "digital_twin:main"

[tool.setuptools]
py-modules = ["digital_twin"]
packages = ["dtwin"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Add pointer comment at top of `requirements.txt`**

Insert these two lines immediately after the existing header block (after line 3, the `===` line), keeping the rest of the file as-is:

```
# FONTE DA VERDADE dos pins: pyproject.toml. Este arquivo é um atalho de
# conveniência (pip install -r requirements.txt) e deve espelhar aquele.
```

- [ ] **Step 4: Create venv and install**

Run:
```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e .[dev]
```
Expected: install completes; no torch/TotalSegmentator pulled. If a core wheel fails to build on 3.13, stop and report — do not fall back to 3.14.

- [ ] **Step 5: Verify import + console script**

Run:
```bash
.venv/Scripts/python.exe -c "import dtwin; print(dtwin.__version__)"
.venv/Scripts/python.exe -m pytest --version
```
Expected: prints `0.1.0` then a pytest version.

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt
git commit -m "build: add pyproject packaging, gitignore, requirements pointer"
```

---

### Task 2: Test helpers — synthetic image builders

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `dtwin.core` (`save_image`, `array_to_image`), SimpleITK, numpy.
- Produces: pytest fixtures and builders used by all later test tasks:
  - `make_sphere_mask(shape, center, radius) -> np.ndarray` (uint8, 0/1), array order (z, y, x).
  - `make_geo_image(arr, spacing=(1.5,1.0,1.0), origin=(10.0,-5.0,3.0)) -> sitk.Image` — wraps an array with non-trivial geometry (spacing is SimpleITK xyz order).
  - fixture `synthetic_case(tmp_path) -> dtwin.core.Case` — writes `volume.nii.gz`, `mask_organ.nii.gz`, `mask_lesion.nii.gz`, and a minimal `manifest.json` into `tmp_path`, returns the `Case`. Organ is a radius-12 sphere centered in a 40³ volume; lesion is a radius-4 sphere at the same center (fully inside organ).

- [ ] **Step 1: Write the helpers and fixture**

```python
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
```

- [ ] **Step 2: Sanity-run the fixture via a throwaway test**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: `no tests ran` (0 tests collected) with no import/collection errors. Confirms `conftest.py` imports cleanly.

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add synthetic image builders and case fixture"
```

---

### Task 3: Test `core.world_vertices_from_index` (geometry)

**Files:**
- Create: `tests/test_core_geometry.py`

**Interfaces:**
- Consumes: `dtwin.core.world_vertices_from_index(verts_zyx, ref)`, `make_geo_image`.
- Produces: regression coverage for the axis-order + LPS transform.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it passes (code already exists)**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_core_geometry.py -v
```
Expected: 2 passed. (This is a characterization test of existing correct code; if it fails, the geometry has a real bug — investigate before editing the test.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_core_geometry.py
git commit -m "test: cover world_vertices_from_index identity + oblique"
```

---

### Task 4: Test `core.load_profile` gates

**Files:**
- Create: `tests/test_core_profile.py`

**Interfaces:**
- Consumes: `dtwin.core.load_profile(path)`, `dtwin.core.PipelineError`.
- Produces: coverage of profile validation gates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_profile.py
import pytest

from dtwin.core import load_profile, PipelineError


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_profile_loads(tmp_path):
    prof = _write(
        tmp_path / "ok.yaml",
        "id: figado\nmodalidade: [MRI]\n"
        "segmentacao_orgao:\n  rotulo_alvo: liver\n",
    )
    data = load_profile(prof)
    assert data["id"] == "figado"
    assert data["segmentacao_orgao"]["rotulo_alvo"] == "liver"


def test_missing_file_aborts(tmp_path):
    with pytest.raises(PipelineError):
        load_profile(tmp_path / "nope.yaml")


def test_missing_required_key_aborts(tmp_path):
    prof = _write(tmp_path / "bad.yaml", "id: figado\nmodalidade: [MRI]\n")
    with pytest.raises(PipelineError):
        load_profile(prof)


def test_missing_rotulo_alvo_aborts(tmp_path):
    prof = _write(
        tmp_path / "bad2.yaml",
        "id: figado\nmodalidade: [MRI]\nsegmentacao_orgao:\n  motor: x\n",
    )
    with pytest.raises(PipelineError):
        load_profile(prof)


def test_real_figado_profile_loads():
    data = load_profile(__import__("pathlib").Path("profiles/figado.yaml"))
    assert data["id"] == "figado"
```

- [ ] **Step 2: Run to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_core_profile.py -v
```
Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_core_profile.py
git commit -m "test: cover load_profile validation gates"
```

---

### Task 5: Test `stages._refine_mask` and `_mesh_from_mask`

**Files:**
- Create: `tests/test_stages_units.py`

**Interfaces:**
- Consumes: `dtwin.stages._refine_mask(mask_zyx, opening, radius, min_voxels)`, `dtwin.stages._mesh_from_mask(mask_path, level, smooth_iter, feature_angle)`, helpers from conftest.
- Produces: coverage of refine + mesh helpers.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_stages_units.py -v
```
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stages_units.py
git commit -m "test: cover _refine_mask and _mesh_from_mask"
```

---

### Task 6: Test `stage2_normalize` gates

**Files:**
- Create: `tests/test_stage2_normalize.py`

**Interfaces:**
- Consumes: `dtwin.stages.stage2_normalize(case, profile)`, `dtwin.core` readers, conftest helpers.
- Produces: coverage of normalization gates + output.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_stage2_normalize.py -v
```
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stage2_normalize.py
git commit -m "test: cover stage2_normalize gates and output"
```

---

### Task 7: Test `engine.finalize` end-to-end on synthetic case (smoke)

**Files:**
- Create: `tests/test_engine_finalize.py`

**Interfaces:**
- Consumes: `dtwin.engine.Engine(profile_path).finalize(case_dir, no_lesion=False)`, `synthetic_case` fixture.
- Produces: the core smoke proof — stages 4b–7 produce valid STLs + manifest.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_finalize.py
import json
from pathlib import Path

from dtwin.engine import Engine


def test_finalize_produces_stls_and_manifest(synthetic_case):
    engine = Engine(Path("profiles/figado.yaml"))
    case = engine.finalize(str(synthetic_case.root), no_lesion=False)

    organ_stl = case.outputs / "figado_orgao.stl"
    lesion_stl = case.outputs / "figado_lesao.stl"
    manifest = case.outputs / "viewer_manifest.json"
    assert organ_stl.exists() and organ_stl.stat().st_size > 0
    assert lesion_stl.exists() and lesion_stl.stat().st_size > 0
    assert manifest.exists()

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["organ"] == "figado"
    assert data["coordinate_system"] == "LPS"
    roles = {m["role"]: m for m in data["meshes"]}
    assert set(roles) == {"orgao", "lesao"}
    # STL refs are relative filenames only (viewer depends on this)
    for m in data["meshes"]:
        assert "/" not in m["stl"] and "\\" not in m["stl"]
        assert (case.outputs / m["stl"]).exists()


def test_finalize_no_lesion_flag(synthetic_case):
    # remove the lesion mask, finalize with --no-lesion
    synthetic_case.mask_lesion.unlink()
    engine = Engine(Path("profiles/figado.yaml"))
    case = engine.finalize(str(synthetic_case.root), no_lesion=True)
    data = json.loads((case.outputs / "viewer_manifest.json").read_text(encoding="utf-8"))
    roles = {m["role"] for m in data["meshes"]}
    assert "orgao" in roles
    assert "lesao" not in roles
```

- [ ] **Step 2: Run to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_engine_finalize.py -v
```
Expected: 2 passed. If `flywheel/` is created by the archive step, it is gitignored — fine.

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_finalize.py
git commit -m "test: end-to-end finalize smoke on synthetic case"
```

---

### Task 8: Test `engine.prepare` with stage 3 monkeypatched + modality gate

**Files:**
- Create: `tests/test_engine_prepare.py`

**Interfaces:**
- Consumes: `dtwin.engine.Engine(...).prepare(dicom_dir, case_dir, ...)`, `dtwin.stages`, pydicom, SimpleITK.
- Produces: coverage of stages 1, 2, 4a without torch; modality gate.
- **Note for implementer:** stage 3 (`stage3_segment_organ`) imports TotalSegmentator and needs a GPU. Monkeypatch it to write a synthetic `mask_organ` so `prepare` completes. Build a tiny synthetic DICOM series with SimpleITK (writes valid DICOM with `Modality` tag) — simpler than hand-rolling pydicom datasets.

- [ ] **Step 1: Write a DICOM-writing helper in this test file**

```python
# tests/test_engine_prepare.py
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pytest

from dtwin import stages
from dtwin.engine import Engine
from dtwin.core import read_image, array_from, array_to_image, save_image, PipelineError


def _write_dicom_series(folder: Path, modality: str = "MR", n_slices: int = 6):
    folder.mkdir(parents=True, exist_ok=True)
    vol = (np.random.default_rng(0).random((n_slices, 32, 32)) * 200).astype(np.int16)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 2.0))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1." + "".join(
        np.random.default_rng(1).integers(0, 9, 20).astype(str)
    )
    for i in range(n_slices):
        sl = img[:, :, i]
        sl.SetMetaData("0008|0060", modality)            # Modality
        sl.SetMetaData("0020|000e", series_uid)          # SeriesInstanceUID
        sl.SetMetaData("0020|0013", str(i))              # InstanceNumber
        writer.SetFileName(str(folder / f"slice_{i:03d}.dcm"))
        writer.Execute(sl)
```

- [ ] **Step 2: Write the tests using a stage-3 monkeypatch**

Append to the same file:

```python
def _fake_segment(case, profile, device, fast):
    """Stand-in for stage3: write a synthetic organ mask matching the volume."""
    vol = read_image(case.volume)
    arr = array_from(vol)
    organ = np.zeros_like(arr, dtype=np.uint8)
    organ[1:-1, 8:24, 8:24] = 1
    save_image(array_to_image(organ, vol, np.uint8), case.mask_organ)


def test_prepare_runs_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "stage3_segment_organ", _fake_segment)
    dicom = tmp_path / "dcm"
    _write_dicom_series(dicom, modality="MR")
    engine = Engine(Path("profiles/figado.yaml"))
    case = engine.prepare(str(dicom), str(tmp_path / "case"), device="cpu", fast=True)
    assert case.volume.exists()
    assert case.volume_zscore.exists()
    assert case.mask_organ.exists()
    assert case.manifest.exists()


def test_prepare_rejects_wrong_modality(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "stage3_segment_organ", _fake_segment)
    dicom = tmp_path / "ct"
    _write_dicom_series(dicom, modality="CT")
    engine = Engine(Path("profiles/figado.yaml"))
    with pytest.raises(PipelineError):
        engine.prepare(str(dicom), str(tmp_path / "case"), device="cpu", fast=True)
```

- [ ] **Step 3: Run to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_engine_prepare.py -v
```
Expected: 2 passed. If SimpleITK cannot read the written series back (GDCM round-trip), the modality assertion is the key check — if the read fails, adjust the helper to also set `0008|0016` (SOPClassUID) to `1.2.840.10008.5.1.4.1.1.4` (MR Image Storage) and `0008|0018` (SOPInstanceUID) per slice, then re-run. Do not weaken the test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_prepare.py
git commit -m "test: prepare path with stubbed segmentation + modality gate"
```

---

### Task 9: Synthetic case generator tool

**Files:**
- Create: `tools/__init__.py` (empty)
- Create: `tools/make_synthetic_case.py`

**Interfaces:**
- Consumes: `dtwin.core` (`Case`, `save_image`, `array_to_image`, `now_utc`), SimpleITK, numpy.
- Produces: a CLI that writes a runnable synthetic case to a target dir so a human can run `finalize` (and, with `--dicom`, a synthetic DICOM dir for `prepare`). Reuses the same geometry as the test fixture but is standalone (no pytest dependency).

- [ ] **Step 1: Write the generator**

```python
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
    for i in range(volume_zyx.shape[0]):
        sl = img[:, :, i]
        sl.SetMetaData("0008|0060", modality)
        sl.SetMetaData("0020|000e", uid)
        sl.SetMetaData("0020|0013", str(i))
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
```

- [ ] **Step 2: Run the generator + finalize end-to-end**

Run:
```bash
.venv/Scripts/python.exe tools/make_synthetic_case.py --out casos/sintetico
.venv/Scripts/python.exe digital_twin.py finalize casos/sintetico --profile profiles/figado.yaml
```
Expected: prints `[OK] finalize concluído`; `casos/sintetico/outputs/` contains `figado_orgao.stl`, `figado_lesao.stl`, `viewer_manifest.json`.

- [ ] **Step 3: Commit**

```bash
git add tools/__init__.py tools/make_synthetic_case.py
git commit -m "feat: synthetic case generator for GPU-free pipeline runs"
```

---

### Task 10: `doctor` preflight subcommand

**Files:**
- Modify: `digital_twin.py` (add `doctor` subparser + handler)
- Create: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `argparse`, importlib, the existing `build_parser()` / `main()` in `digital_twin.py`.
- Produces: `digital-twin doctor` — checks core deps + reports TotalSegmentator availability and torch device, returns 0 always (it is a report, not a gate). Adds `run_doctor() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_doctor.py
import digital_twin


def test_doctor_runs_and_returns_zero(capsys):
    rc = digital_twin.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TotalSegmentator" in out
    assert "SimpleITK" in out
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_doctor.py -v
```
Expected: FAIL — `doctor` is not a valid subcommand yet (SystemExit from argparse).

- [ ] **Step 3: Add the `doctor` subparser and handler**

In `digital_twin.py`, add to `build_parser()` (after the `finalize` parser `f`, before `return ap`):

```python
    sub.add_parser("doctor", help="Checa dependências e ambiente (não roda segmentação).")
```

Add this function above `main()`:

```python
def run_doctor() -> int:
    import importlib

    core = ["SimpleITK", "numpy", "scipy", "skimage", "pyvista", "pydicom", "yaml", "nibabel"]
    print("Dependências do núcleo:")
    all_ok = True
    for mod in core:
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {mod}")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"  [FALTA] {mod}: {e}")

    print("Segmentação automática (opcional, requer extra [seg]):")
    try:
        importlib.import_module("totalsegmentator")
        print("  [OK]   TotalSegmentator importável")
    except Exception:  # noqa: BLE001
        print("  [FALTA] TotalSegmentator — instale com: pip install .[seg]")

    try:
        torch = importlib.import_module("torch")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [INFO] torch device: {dev}")
    except Exception:  # noqa: BLE001
        print("  [INFO] torch ausente (esperado se ainda não instalou [seg]).")

    print("\nNúcleo completo." if all_ok else "\nNúcleo INCOMPLETO — rode: pip install -e .")
    return 0
```

In `main()`, handle the new command. Change the dispatch block so `doctor` is handled before constructing the `Engine` (doctor must not require a profile). Replace the body of the `try:` in `main()` with:

```python
        if args.cmd == "doctor":
            return run_doctor()
        engine = Engine(Path(args.profile))
        if args.cmd == "prepare":
            case = engine.prepare(
                args.dicom_dir,
                args.case_dir,
                policy=args.policy,
                device=args.device,
                fast=args.fast,
            )
            print(f"\n[OK] 'prepare' concluído para {case.root}.")
            print("Marque a lesão no 3D Slicer (instruções acima) e rode 'finalize'.")
        else:
            case = engine.finalize(args.case_dir, no_lesion=args.no_lesion)
            print(f"\n[OK] 'finalize' concluído. Saídas em: {case.outputs}")
            print("STL(s) e viewer_manifest.json prontos para o visualizador web.")
        return 0
```

Note: `doctor` has no `--profile`, so `args.profile` is not accessed on that path.

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_doctor.py -v
.venv/Scripts/python.exe digital_twin.py doctor
```
Expected: test passes; `doctor` prints core deps OK and TotalSegmentator missing.

- [ ] **Step 5: Commit**

```bash
git add digital_twin.py tests/test_doctor.py
git commit -m "feat: add doctor preflight subcommand"
```

---

### Task 11: Static Three.js viewer

**Files:**
- Create: `viewer/index.html`
- Create: `viewer/app.js`
- Create: `viewer/README.md`

**Interfaces:**
- Consumes: a case `outputs/` folder containing `viewer_manifest.json` + the STLs it references (relative filenames).
- Produces: a zero-build web viewer. Loads via drag-drop of the `outputs/` folder OR `?case=<relative-path>` when served by `python -m http.server`. Uses Three.js r160 + STLLoader + OrbitControls from a pinned CDN (esm.sh).

- [ ] **Step 1: Write `viewer/index.html`**

```html
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Digital Twin — Visualizador (modo Pesquisa)</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3; }
    #banner { background: #7a1320; color: #fff; padding: 8px 14px; font-size: 13px; text-align: center; }
    #wrap { display: flex; height: calc(100vh - 35px); }
    #canvas-holder { flex: 1; position: relative; }
    canvas { display: block; }
    #panel { width: 280px; padding: 16px; background: #161b22; border-left: 1px solid #30363d; overflow-y: auto; }
    #panel h1 { font-size: 15px; margin: 0 0 12px; }
    .row { margin: 10px 0; font-size: 13px; }
    label { display: flex; align-items: center; gap: 8px; }
    #drop { border: 2px dashed #30363d; border-radius: 8px; padding: 18px; text-align: center; font-size: 13px; color: #8b949e; cursor: pointer; }
    #drop.hover { border-color: #58a6ff; color: #58a6ff; }
    .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
    #meta { font-size: 12px; color: #8b949e; margin-top: 14px; white-space: pre-wrap; }
    input[type=range] { width: 100%; }
  </style>
</head>
<body>
  <div id="banner">⚠️ Modo Pesquisa — NÃO destinado a decisão clínica. Coordenadas LPS.</div>
  <div id="wrap">
    <div id="canvas-holder"></div>
    <div id="panel">
      <h1>Digital Twin — Visualizador</h1>
      <div id="drop">Arraste a pasta <b>outputs/</b> aqui<br/>(ou os arquivos: manifest + STLs)</div>
      <div id="controls"></div>
      <div id="meta"></div>
    </div>
  </div>
  <script type="module" src="./app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `viewer/app.js`**

```javascript
import * as THREE from "https://esm.sh/three@0.160.0";
import { STLLoader } from "https://esm.sh/three@0.160.0/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const holder = document.getElementById("canvas-holder");
const controlsDiv = document.getElementById("controls");
const metaDiv = document.getElementById("meta");
const drop = document.getElementById("drop");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
camera.position.set(120, 120, 120);
const renderer = new THREE.WebGLRenderer({ antialias: true });
holder.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(1, 1, 1);
scene.add(dir);
const orbit = new OrbitControls(camera, renderer.domElement);
const group = new THREE.Group();
scene.add(group);

function resize() {
  const w = holder.clientWidth, h = holder.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();
(function loop() { requestAnimationFrame(loop); orbit.update(); renderer.render(scene, camera); })();

const loader = new STLLoader();
const meshes = {};

function clearScene() {
  for (const k of Object.keys(meshes)) { group.remove(meshes[k]); delete meshes[k]; }
  controlsDiv.innerHTML = "";
}

function addMesh(role, geometry, colorHex) {
  geometry.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colorHex), transparent: true, opacity: role === "lesao" ? 1.0 : 0.5,
    roughness: 0.7, metalness: 0.0,
  });
  const mesh = new THREE.Mesh(geometry, mat);
  meshes[role] = mesh;
  group.add(mesh);
}

function frameScene() {
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  group.position.sub(center);
  camera.position.set(size, size, size);
  camera.near = size / 100; camera.far = size * 10; camera.updateProjectionMatrix();
  orbit.target.set(0, 0, 0); orbit.update();
}

function buildControls(items) {
  for (const it of items) {
    const row = document.createElement("div"); row.className = "row";
    const label = document.createElement("label");
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = true;
    cb.onchange = () => { if (meshes[it.role]) meshes[it.role].visible = cb.checked; };
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = it.color;
    label.append(cb, sw, document.createTextNode(" " + it.role));
    row.appendChild(label);
    const op = document.createElement("input");
    op.type = "range"; op.min = "0"; op.max = "1"; op.step = "0.05";
    op.value = it.role === "lesao" ? "1" : "0.5";
    op.oninput = () => { if (meshes[it.role]) meshes[it.role].material.opacity = parseFloat(op.value); };
    row.appendChild(op);
    controlsDiv.appendChild(row);
  }
}

// fileMap: role -> ArrayBuffer ; manifest object
function render(manifest, fileMap) {
  clearScene();
  for (const it of manifest.meshes) {
    const buf = fileMap[it.stl];
    if (!buf) { console.warn("STL ausente:", it.stl); continue; }
    addMesh(it.role, loader.parse(buf), it.color);
  }
  frameScene();
  buildControls(manifest.meshes);
  metaDiv.textContent =
    `caso: ${manifest.case_id}\norgão: ${manifest.organ}\n` +
    `coordenadas: ${manifest.coordinate_system}\nestado: ${manifest.regulatory_state}\n` +
    `${manifest.disclaimer || ""}`;
}

// --- Drag & drop of the outputs/ folder (or its files) ---
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hover"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
drop.addEventListener("drop", async (e) => {
  e.preventDefault(); drop.classList.remove("hover");
  const files = [...e.dataTransfer.files];
  const byName = {};
  let manifest = null;
  for (const f of files) {
    const buf = await f.arrayBuffer();
    if (f.name.endsWith(".json")) manifest = JSON.parse(new TextDecoder().decode(buf));
    else byName[f.name] = buf;
  }
  if (!manifest) { alert("Inclua o viewer_manifest.json no que foi arrastado."); return; }
  render(manifest, byName);
});

// --- Optional ?case=<path> when served over http ---
const params = new URLSearchParams(location.search);
const casePath = params.get("case");
if (casePath) {
  (async () => {
    const base = casePath.replace(/\/$/, "");
    const manifest = await (await fetch(`${base}/viewer_manifest.json`)).json();
    const fileMap = {};
    for (const it of manifest.meshes) {
      fileMap[it.stl] = await (await fetch(`${base}/${it.stl}`)).arrayBuffer();
    }
    render(manifest, fileMap);
  })().catch((err) => { console.error(err); alert("Falha ao carregar via ?case: " + err.message); });
}
```

- [ ] **Step 3: Write `viewer/README.md`**

```markdown
# Visualizador (modo Pesquisa)

Visualizador 3D estático (Three.js, sem build) para os STLs gerados pelo pipeline.
**NÃO destinado a decisão clínica.** Coordenadas LPS.

## Uso rápido (drag & drop)

1. Abra `viewer/index.html` no navegador (duplo clique funciona).
2. Arraste para a área indicada o conteúdo da pasta `outputs/` de um caso
   (o `viewer_manifest.json` **e** os arquivos `.stl`).

## Uso servido (carregamento automático via ?case=)

Por restrição do navegador, `fetch` só funciona via http. Sirva a raiz do projeto:

```bash
python -m http.server 8000
```

Depois abra (ajuste o caminho do caso):

```
http://localhost:8000/viewer/index.html?case=../casos/sintetico/outputs
```

Controles: orbitar (arrastar), zoom (scroll), alternar visibilidade e opacidade de
órgão/lesão no painel à direita.
```

- [ ] **Step 4: Manual verification**

Run:
```bash
.venv/Scripts/python.exe -m http.server 8000
```
Open `http://localhost:8000/viewer/index.html?case=../casos/sintetico/outputs` and confirm the organ (translucent) + lesion render and orbit. Then stop the server.
Expected: both meshes visible; panel shows case metadata + disclaimer. (If `?case` path differs, use drag-drop of `casos/sintetico/outputs/` instead.)

- [ ] **Step 5: Commit**

```bash
git add viewer/index.html viewer/app.js viewer/README.md
git commit -m "feat: static Three.js viewer for organ + lesion STLs"
```

---

### Task 12: RUNNING guide + README wiring + full test pass

**Files:**
- Create: `docs/RUNNING.md`
- Modify: `README.md` (add links to viewer, synthetic case, doctor, RUNNING)

**Interfaces:**
- Consumes: everything built above.
- Produces: operator documentation for the real GPU-box path and a green full test run.

- [ ] **Step 1: Write `docs/RUNNING.md`**

```markdown
# Executando o pipeline

Dois ambientes: (a) **desenvolvimento/teste** nesta máquina, sem GPU; (b)
**execução real** na máquina com GPU, onde roda a segmentação automática.

## (a) Dev/teste — sem GPU/Slicer/DICOM

```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m pytest          # suíte verde
.venv/Scripts/python.exe digital_twin.py doctor

# caso sintético ponta a ponta (estágios 4b–7)
.venv/Scripts/python.exe tools/make_synthetic_case.py --out casos/sintetico
.venv/Scripts/python.exe digital_twin.py finalize casos/sintetico --profile profiles/figado.yaml
```

Abra o resultado no visualizador: ver `viewer/README.md`.

## (b) Execução real (máquina com GPU)

```bash
pip install -e .[seg]        # traz TotalSegmentator + torch (grande)
digital-twin doctor          # confirme "torch device: cuda"
```

**Fase 1 — prepare** (estágios 1–4a):

```bash
digital-twin prepare /caminho/serie_dicom \
    --case-dir casos/paciente001 --profile profiles/figado.yaml
# CPU: adicione --device cpu --fast (lento)
```

**Etapa manual — 3D Slicer:** abra `casos/paciente001/volume.nii.gz` e
`mask_organ.nii.gz`, revise o órgão, marque a lesão e salve EXATAMENTE em
`casos/paciente001/mask_lesion.nii.gz` (instruções exatas são impressas ao fim do
`prepare`).

**Fase 2 — finalize** (estágios 4b–7):

```bash
digital-twin finalize casos/paciente001 --profile profiles/figado.yaml
# Sem lesão (escolha explícita): adicione --no-lesion
```

Saídas em `casos/paciente001/outputs/`: `figado_orgao.stl`, `figado_lesao.stl`,
`viewer_manifest.json`.

## Troubleshooting

- `TotalSegmentator não está instalado` → `pip install -e .[seg]`.
- `Saída de segmentação esperada não encontrada` / classe inválida → confira nomes:
  `totalseg_info --classes -ta total_mr`.
- `Modalidade do exame (...) não bate` → use o perfil correto; `figado.yaml` espera MRI.
- Wheels falhando na instalação → confirme Python **3.13** (`py -3.13`); 3.14 ainda
  não tem wheels de torch/SimpleITK.
```

- [ ] **Step 2: Add a "Funcional / Ferramentas" section to `README.md`**

Insert this block immediately before the final `### Fora do escopo do MVP` section:

```markdown
### Ferramentas de produção

- **Testes:** `.venv/Scripts/python.exe -m pytest` (não requer GPU/torch).
- **Preflight:** `digital-twin doctor` — checa dependências e device.
- **Caso sintético:** `python tools/make_synthetic_case.py --out casos/sintetico`
  gera um caso fictício para rodar `finalize` sem GPU/Slicer/DICOM.
- **Visualizador web:** `viewer/index.html` (Three.js, sem build) — ver `viewer/README.md`.
- **Guia de execução completo:** `docs/RUNNING.md`.

```

- [ ] **Step 3: Run the full suite**

Run:
```bash
.venv/Scripts/python.exe -m pytest -v
```
Expected: all tests from Tasks 3–10 pass (green), no torch import anywhere.

- [ ] **Step 4: Commit**

```bash
git add docs/RUNNING.md README.md
git commit -m "docs: add RUNNING guide and wire README to new tooling"
```

---

## Self-Review

**Spec coverage:**
- A. Packaging + repo hygiene → Task 1 (pyproject, gitignore, requirements pointer, venv 3.13). ✓
- B. Tests → Tasks 3 (geometry), 4 (profile), 5 (refine/mesh), 6 (normalize), 7 (finalize smoke), 8 (prepare + modality gate). ✓ — every unit named in spec §B has a task.
- C. Synthetic fixture + smoke → Task 2 (fixture) + Task 7 (smoke) + Task 9 (standalone generator, incl. DICOM dir). ✓
- D. Real-data execution → Task 10 (`doctor`) + Task 12 (`docs/RUNNING.md`). ✓
- E. Viewer → Task 11. ✓
- F. Robustness pass → relative-path assertion in Task 7; footprint/mesh.save guarded by Tasks 5/7 acting as regression tests; CDN pinned in Task 11. ✓
- Success criteria 1–6 → Tasks 1, 12 (pytest green), 9, 11, 10, 12 respectively. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full content; commands have expected output. ✓

**Type/name consistency:** `Case` properties (`volume`, `mask_organ`, `mask_lesion`, `volume_zscore`, `outputs`, `manifest`) used as defined in `dtwin/core.py`. Helper names (`make_sphere_mask`, `make_geo_image`, `synthetic_case`, `_fake_segment`) consistent across Tasks 2, 3, 5, 8. `run_doctor`/`doctor` consistent in Task 10. Viewer `render(manifest, fileMap)` / `meshes[role]` consistent within Task 11. Manifest fields (`meshes`, `role`, `stl`, `color`, `case_id`, `organ`, `coordinate_system`, `regulatory_state`, `disclaimer`) match `stage7_export_publish` output in `dtwin/stages.py`. ✓
