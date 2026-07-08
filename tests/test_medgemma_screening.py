import json
from types import SimpleNamespace

import pytest
import yaml

from dtwin.core import PipelineError, sha256_of
from dtwin.medgemma_client import load_screening_config
from dtwin.medgemma_screening import (
    _aggregate_panel_reports,
    _authoritative_panels,
    main,
    run_screening,
)
from dtwin.rag.index import build_bm25_index


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


class RagAwareTestClient(TestOnlyClient):
    def __init__(self):
        self.prompts = []

    def generate(self, panel_path, prompt):
        self.prompts.append(prompt)
        assert "CONTEXTO RAG TEXTUAL DE APOIO" in prompt
        assert "não substitui a análise da imagem" in prompt
        return super().generate(panel_path, prompt)


def _args(synthetic_case, tmp_path):
    return {
        "volume_path": synthetic_case.volume,
        "liver_mask_path": synthetic_case.mask_organ,
        "profile_path": "profiles/figado.yaml",
        "medgemma_config_path": "configs/medgemma_4b.yaml",
        "output_dir": tmp_path / "screening",
    }


def _write_rag_fixture_repo(tmp_path):
    repo = tmp_path / "rag_repo"
    corpus = repo / "rag" / "corpus" / "test"
    chunks_dir = corpus / "chunks"
    chunks_dir.mkdir(parents=True)
    chunk = {
        "chunk_id": "hcc::c1",
        "doc_id": "hcc",
        "title": "HCC MRI",
        "section": "Criteria",
        "text": "HCC can show arterial phase hyperenhancement, washout and enhancing capsule.",
        "sha256": "sha-hcc",
        "categories": ["hcc", "li_rads"],
        "priority": "core",
        "url": "https://example.org/hcc",
        "pmcid": "PMC_HCC",
        "doi": "10.0000/hcc",
    }
    (chunks_dir / "hcc__c1.json").write_text(json.dumps(chunk), encoding="utf-8")
    (corpus / "manifest.json").write_text(
        json.dumps({
            "schema": "argos-rag-corpus-v1",
            "corpus_version": "screening_rag_test_v1",
            "chunks": [{
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "path": "chunks/hcc__c1.json",
                "sha256": chunk["sha256"],
            }],
        }),
        encoding="utf-8",
    )
    build_bm25_index(corpus_dir=corpus, out_dir=repo / "rag" / "index" / "test")
    eval_path = repo / "docs" / "rag" / "retrieval_eval_v1.yaml"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(
        yaml.safe_dump({
            "schema": "argos-rag-retrieval-eval-v1",
            "queries": [{
                "id": "hcc_aphe_washout_capsule",
                "query": "arterial washout capsule HCC",
                "expected_categories": ["hcc"],
            }],
        }, sort_keys=False),
        encoding="utf-8",
    )
    configs = repo / "configs"
    configs.mkdir()
    config = load_screening_config("configs/medgemma_4b.yaml")
    config["rag"] = {
        "enabled": True,
        "index_path": "rag/index/test/bm25_index.json",
        "retrieval_eval": "docs/rag/retrieval_eval_v1.yaml",
        "top_k": 1,
        "max_sources": 2,
        "max_chunk_chars": 160,
        "min_score": 0.0,
        "query_ids": ["hcc_aphe_washout_capsule"],
    }
    config_path = configs / "medgemma_rag_test.yaml"
    config_path.write_text(
        yaml.safe_dump({"medgemma_screening": config}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


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
    assert report["durations_seconds"]["panel_generation"] >= 0
    assert report["durations_seconds"]["screening_total"] >= 0


def test_full_flow_with_rag_persists_context_and_prompt_audit(synthetic_case, tmp_path):
    config_path = _write_rag_fixture_repo(tmp_path)
    client = RagAwareTestClient()
    args = _args(synthetic_case, tmp_path)
    args["medgemma_config_path"] = config_path
    result = run_screening(
        **args,
        visible_phi_confirmed=True,
        client=client,
    )
    out = tmp_path / "screening"
    report = json.loads((out / "medgemma_report.json").read_text("utf-8"))
    rag_context = json.loads((out / "rag_context.json").read_text("utf-8"))
    assert result["status"] == "pending_review"
    assert len(client.prompts) == 1
    assert report["rag"]["enabled"] is True
    assert report["rag"]["context_file"] == "rag_context.json"
    assert report["rag"]["context_sha256"] == rag_context["context_sha256"]
    assert report["rag"]["source_count"] == 1
    assert report["prompt_audit"]["enabled"] is True
    assert len(report["prompt_audit"]["final_prompt_sha256"]) == 64
    assert report["panel_reports"][0]["rag_context_sha256"] == rag_context["context_sha256"]
    assert len(report["panel_reports"][0]["prompt_sha256"]) == 64
    assert report["durations_seconds"]["rag_retrieval"] >= 0


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


# ======================================================================= #
# Cenário B — cobertura volumétrica (agregação, inferência por painel, gate)
# ======================================================================= #
VOLUMETRIC_CONFIG = "configs/medgemma_local_4b_volumetric.yaml"


def _panel_entry(number, total, state, confidence):
    return {
        "panel_number": number, "panel_total": total,
        "report": {
            "resultado_hipotese": state, "confianca": confidence,
            "resumo_do_achado": f"resumo {number}",
            "localizacao_aproximada": f"loc {number}",
            "sinais_visuais_observados": [f"sinal {number}"],
            "limitacoes_da_analise": [f"limite {number}"],
            "necessidade_de_revisao_humana": True,
        },
    }


def test_aggregation_any_positive_wins_with_its_confidence():
    agg = _aggregate_panel_reports([
        _panel_entry(1, 3, "NEGATIVA", "alta"),
        _panel_entry(2, 3, "POSITIVA", "moderada"),
        _panel_entry(3, 3, "INCONCLUSIVA", "baixa"),
    ])
    assert agg["resultado_hipotese"] == "POSITIVA"
    assert agg["confianca"] == "moderada"  # menor confiança entre os que determinaram
    assert agg["necessidade_de_revisao_humana"] is True


def test_aggregation_inconclusive_when_no_positive():
    agg = _aggregate_panel_reports([
        _panel_entry(1, 2, "NEGATIVA", "alta"),
        _panel_entry(2, 2, "INCONCLUSIVA", "moderada"),
    ])
    assert agg["resultado_hipotese"] == "INCONCLUSIVA"
    assert agg["confianca"] == "moderada"


def test_aggregation_negative_only_when_all_negative_preserves_per_panel():
    agg = _aggregate_panel_reports([
        _panel_entry(1, 2, "NEGATIVA", "alta"),
        _panel_entry(2, 2, "NEGATIVA", "baixa"),
    ])
    assert agg["resultado_hipotese"] == "NEGATIVA"
    assert agg["confianca"] == "baixa"  # menor confiança entre os negativos
    assert any("Painel 1/2" in s for s in agg["sinais_visuais_observados"])
    assert any("Painel 2/2" in s for s in agg["limitacoes_da_analise"])


class _CountingVolumetricClient:
    """Cliente de teste: uma resposta por painel, contando as chamadas."""

    def __init__(self, states):
        self.states = list(states)
        self.calls = []

    def generate(self, panel_path, prompt):
        idx = len(self.calls)
        self.calls.append(panel_path.name)
        assert "avaliação parcial" in prompt  # o prompt por painel é parcial
        state = self.states[idx] if idx < len(self.states) else self.states[-1]
        return {
            "resultado_hipotese": state, "confianca": "baixa",
            "resumo_do_achado": "entrada sintética", "localizacao_aproximada": "N/A",
            "sinais_visuais_observados": [], "limitacoes_da_analise": ["teste"],
            "necessidade_de_revisao_humana": True,
        }


def _volumetric_args(synthetic_case, tmp_path):
    return {
        "volume_path": synthetic_case.volume,
        "liver_mask_path": synthetic_case.mask_organ,
        "profile_path": "profiles/figado.yaml",
        "medgemma_config_path": VOLUMETRIC_CONFIG,
        "output_dir": tmp_path / "screening",
    }


def test_volumetric_calls_model_once_per_panel_and_aggregates(synthetic_case, tmp_path):
    # Esfera r=12 centrada em z=20 -> fígado em 25 cortes -> ceil(25/9) = 3 painéis.
    client = _CountingVolumetricClient(["NEGATIVA", "NEGATIVA", "NEGATIVA"])
    result = run_screening(
        **_volumetric_args(synthetic_case, tmp_path),
        visible_phi_confirmed=True, client=client,
    )
    assert result["panel_strategy"] == "volumetric_blocks"
    assert len(client.calls) == 3
    out = tmp_path / "screening"
    report = json.loads((out / "medgemma_report.json").read_text("utf-8"))
    assert report["report"]["resultado_hipotese"] == "NEGATIVA"
    assert len(report["panel_reports"]) == 3
    assert report["coverage"]["gate_passed"] is True
    assert report["coverage"]["covered_liver_voxels"] == report["coverage"]["total_liver_voxels"]
    assert "aggregation_rule" in report
    assert len(report["input_panels"]) == 3
    assert (out / "medgemma_panel_reports.json").exists()


def test_volumetric_single_positive_panel_makes_case_positive(synthetic_case, tmp_path):
    client = _CountingVolumetricClient(["NEGATIVA", "POSITIVA", "NEGATIVA"])
    run_screening(
        **_volumetric_args(synthetic_case, tmp_path),
        visible_phi_confirmed=True, client=client,
    )
    report = json.loads((tmp_path / "screening" / "medgemma_report.json").read_text("utf-8"))
    assert report["report"]["resultado_hipotese"] == "POSITIVA"


class _FailingPanelClient:
    def __init__(self, fail_at):
        self.fail_at = fail_at
        self.calls = 0

    def generate(self, panel_path, prompt):
        self.calls += 1
        if self.calls == self.fail_at:
            raise PipelineError("Backend MedGemma inacessível durante o painel.")
        return {
            "resultado_hipotese": "NEGATIVA", "confianca": "baixa",
            "resumo_do_achado": "ok", "localizacao_aproximada": "N/A",
            "sinais_visuais_observados": [], "limitacoes_da_analise": ["teste"],
            "necessidade_de_revisao_humana": True,
        }


def test_volumetric_technical_failure_in_middle_panel_fails_whole_case(synthetic_case, tmp_path):
    with pytest.raises(PipelineError):
        run_screening(
            **_volumetric_args(synthetic_case, tmp_path),
            visible_phi_confirmed=True, client=_FailingPanelClient(fail_at=2),
        )
    # cobertura parcial nunca vira relatório final
    assert not (tmp_path / "screening" / "medgemma_report.json").exists()


def test_authoritative_panels_rejects_hash_mismatch(tmp_path):
    (tmp_path / "p1.png").write_bytes(b"conteudo-real")
    manifest = {
        "panel_strategy": "volumetric_blocks", "panel_sha256": "x",
        "coverage": {"gate_passed": True, "covered_liver_voxels": 5, "total_liver_voxels": 5},
        "panels": [{"panel_number": 1, "panel_total": 1, "image": "p1.png", "sha256": "0" * 64}],
    }
    panel = SimpleNamespace(manifest_path=tmp_path / "manifest.json", panel_path=tmp_path / "p1.png")
    with pytest.raises(PipelineError, match="hash inconsistente|ausente"):
        _authoritative_panels(panel, manifest)


def test_authoritative_panels_rejects_failed_coverage_gate(tmp_path):
    (tmp_path / "p1.png").write_bytes(b"conteudo-real")
    manifest = {
        "panel_strategy": "volumetric_blocks", "panel_sha256": "x",
        "coverage": {"gate_passed": False, "covered_liver_voxels": 4, "total_liver_voxels": 5},
        "panels": [{
            "panel_number": 1, "panel_total": 1, "image": "p1.png",
            "sha256": sha256_of(tmp_path / "p1.png"),
        }],
    }
    panel = SimpleNamespace(manifest_path=tmp_path / "manifest.json", panel_path=tmp_path / "p1.png")
    with pytest.raises(PipelineError, match="cobertura|reprovad"):
        _authoritative_panels(panel, manifest)


def test_baseline_uniform_9_still_produces_single_legacy_panel(synthetic_case, tmp_path):
    """Regressão: a estratégia baseline (uniform_9) permanece reproduzível."""
    result = run_screening(**_args(synthetic_case, tmp_path), panel_only=True)
    assert result["panel_strategy"] == "uniform_9"
    assert len(result["panel_paths"]) == 1
    assert result["panel_paths"][0].endswith("medgemma_liver_screening_panel.png")
