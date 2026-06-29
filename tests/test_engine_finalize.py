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


def test_refinalize_no_lesion_drops_prior_lesion(synthetic_case):
    """Re-running finalize with --no-lesion after a lesion run must not keep the
    stale lesion mesh/STL. Finalize has to be idempotent against prior artifacts."""
    engine = Engine(Path("profiles/figado.yaml"))
    # first pass: real lesion present
    case = engine.finalize(str(synthetic_case.root), no_lesion=False)
    assert (case.outputs / "figado_lesao.stl").exists()

    # operator decides there is no lesion: drop the mask, re-finalize
    synthetic_case.mask_lesion.unlink()
    case = engine.finalize(str(synthetic_case.root), no_lesion=True)

    data = json.loads((case.outputs / "viewer_manifest.json").read_text(encoding="utf-8"))
    roles = {m["role"] for m in data["meshes"]}
    assert "lesao" not in roles, "stale lesion survived a --no-lesion re-finalize"
    assert not (case.outputs / "figado_lesao.stl").exists()
    assert not case.mesh_lesion.exists()
