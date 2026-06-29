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
