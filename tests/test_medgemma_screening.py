import json

import pytest

from dtwin.core import PipelineError
from dtwin.medgemma_screening import main, run_screening


class TestOnlyClient:
    """Injeção direta da suíte; não é exposta pela CLI nem por variável de ambiente."""

    def generate(self, panel_path, prompt):
        assert panel_path.name == "medgemma_liver_screening_panel.png"
        assert "INCONCLUSIVA" in prompt
        return {
            "resultado_hipotese": "INCONCLUSIVA",
            "resumo_do_achado": "Entrada sintética sem interpretação clínica.",
            "localizacao_aproximada": "Não aplicável ao teste sintético.",
            "sinais_visuais_observados": [],
            "confianca": "baixa",
            "limitacoes_da_analise": ["Saída artificial exclusiva da suíte de testes."],
            "necessidade_de_revisao_humana": True,
        }


def _args(synthetic_case, tmp_path):
    return {
        "volume_path": synthetic_case.volume,
        "liver_mask_path": synthetic_case.mask_organ,
        "profile_path": "profiles/figado.yaml",
        "medgemma_config_path": "configs/medgemma_4b.yaml",
        "output_dir": tmp_path / "screening",
    }


def test_panel_only_never_creates_fake_report(synthetic_case, tmp_path):
    result = run_screening(**_args(synthetic_case, tmp_path), panel_only=True)
    assert result["status"] == "panel_ready"
    assert result["report_path"] is None
    assert not (tmp_path / "screening" / "medgemma_report.json").exists()


def test_full_flow_persists_traceable_pending_review_report(synthetic_case, tmp_path):
    result = run_screening(
        **_args(synthetic_case, tmp_path),
        visible_phi_confirmed=True,
        client=TestOnlyClient(),
    )
    report = json.loads((tmp_path / "screening" / "medgemma_report.json").read_text("utf-8"))
    assert result["status"] == "pending_review"
    assert report["model_version"] == "MedGemma 1.5 4B Instruction-Tuned"
    assert report["model_parameter_scale"] == "4B"
    assert report["lesion_pre_marked"] is False
    assert report["requires_human_review"] is True
    assert report["report"]["necessidade_de_revisao_humana"] is True
    assert len(report["input_panel_sha256"]) == 64
    assert len(report["input_volume_sha256"]) == 64
    assert len(report["input_liver_mask_sha256"]) == 64
    assert len(report["screening_config_sha256"]) == 64


def test_full_flow_requires_visible_phi_confirmation(synthetic_case, tmp_path):
    with pytest.raises(PipelineError, match="Confirmação visual"):
        run_screening(**_args(synthetic_case, tmp_path), client=TestOnlyClient())
    assert not (tmp_path / "screening" / "medgemma_report.json").exists()


def test_real_client_path_aborts_without_backend(synthetic_case, tmp_path):
    with pytest.raises(PipelineError, match="backend not configured"):
        run_screening(
            **_args(synthetic_case, tmp_path),
            visible_phi_confirmed=True,
        )
    assert not (tmp_path / "screening" / "medgemma_report.json").exists()


def test_cli_case_dir_resolves_safe_case_paths(synthetic_case):
    rc = main(["--case-dir", str(synthetic_case.root), "--panel-only"])
    assert rc == 0
    assert (
        synthetic_case.root
        / "outputs"
        / "medgemma"
        / "medgemma_liver_screening_panel.png"
    ).exists()
