import json

import pytest

from dtwin.core import PipelineError
from dtwin.medgemma_client import (
    HTTPJSONMedGemmaClient,
    build_medgemma_prompt,
    load_screening_config,
    model_trace,
    validate_medgemma_report,
)


def valid_report(state="INCONCLUSIVA"):
    return {
        "resultado_hipotese": state,
        "resumo_do_achado": "Hipótese visual limitada à montagem fornecida.",
        "localizacao_aproximada": "Não determinável nesta montagem.",
        "sinais_visuais_observados": [],
        "confianca": "baixa",
        "limitacoes_da_analise": ["Montagem 2D não representa todo o volume."],
        "necessidade_de_revisao_humana": True,
    }


def test_default_config_uses_medgemma_15_4b():
    config = load_screening_config("configs/medgemma_4b.yaml")
    trace = model_trace(config)
    assert trace["model_family"] == "MedGemma"
    assert trace["model_version"] == "MedGemma 1.5 4B Instruction-Tuned"
    assert trace["model_parameter_scale"] == "4B"
    assert trace["model_id"] == "google/medgemma-1.5-4b-it"


def test_config_switches_to_27b_without_code_change():
    config = load_screening_config("configs/medgemma_27b.yaml")
    trace = model_trace(config)
    assert trace["model_version"] == "MedGemma 1.5 27B"
    assert trace["model_parameter_scale"] == "27B"
    assert trace["model_id"] is None
    assert config["medgemma"]["timeout_seconds"] == 300


def test_environment_can_override_model_and_endpoint():
    config = load_screening_config(
        "configs/medgemma_4b.yaml",
        environ={
            "MEDGEMMA_MODEL_VERSION": "Configured Test Model",
            "MEDGEMMA_MODEL_PARAMETER_SCALE": "TEST",
            "MEDGEMMA_ENDPOINT_URL": "http://127.0.0.1:9999/generate",
        },
    )
    assert config["medgemma"]["model_version"] == "Configured Test Model"
    assert config["medgemma"]["endpoint_url"].endswith(":9999/generate")


def test_invalid_numeric_environment_override_is_pipeline_error():
    with pytest.raises(PipelineError, match="MEDGEMMA_TIMEOUT_SECONDS"):
        load_screening_config(
            "configs/medgemma_4b.yaml",
            environ={"MEDGEMMA_TIMEOUT_SECONDS": "not-a-number"},
        )


def test_prompt_contains_research_states_and_human_review():
    prompt = build_medgemma_prompt(load_screening_config("configs/medgemma_4b.yaml"))
    for text in (
        "modo de pesquisa", "POSITIVA", "NEGATIVA", "INCONCLUSIVA",
        "não é diagnóstico", "não é laudo médico", "Revisão humana obrigatória",
    ):
        assert text.lower() in prompt.lower()


@pytest.mark.parametrize("state", ["POSITIVA", "NEGATIVA", "INCONCLUSIVA"])
def test_report_accepts_only_the_three_defined_states(state):
    config = load_screening_config("configs/medgemma_4b.yaml")
    assert validate_medgemma_report(valid_report(state), config["report"])[
        "resultado_hipotese"
    ] == state


def test_report_rejects_unknown_state():
    config = load_screening_config("configs/medgemma_4b.yaml")
    with pytest.raises(PipelineError, match="Estado MedGemma inválido"):
        validate_medgemma_report(valid_report("PROVAVEL"), config["report"])


def test_parser_recovers_english_and_cased_state():
    config = load_screening_config("configs/medgemma_4b.yaml")
    assert validate_medgemma_report(valid_report("NEGATIVE"), config["report"])["resultado_hipotese"] == "NEGATIVA"
    assert validate_medgemma_report(valid_report("Positiva"), config["report"])["resultado_hipotese"] == "POSITIVA"
    assert validate_medgemma_report(valid_report("INCONCLUSIVE"), config["report"])["resultado_hipotese"] == "INCONCLUSIVA"


def test_parser_recovers_confidence_locale_and_case():
    config = load_screening_config("configs/medgemma_4b.yaml")
    report = valid_report("NEGATIVA")
    report["confianca"] = "Low"
    assert validate_medgemma_report(report, config["report"])["confianca"] == "baixa"
    report["confianca"] = "MODERATE"
    assert validate_medgemma_report(report, config["report"])["confianca"] == "moderada"


def test_parser_coerces_string_list_field_and_drops_extras():
    config = load_screening_config("configs/medgemma_4b.yaml")
    report = valid_report("NEGATIVA")
    report["limitacoes_da_analise"] = "Montagem 2D limita a avaliação."  # veio como string
    report["necessidade_de_revisao_humana"] = "true"  # veio como string
    report["comentario_extra"] = "ruído que deve ser descartado"
    out = validate_medgemma_report(report, config["report"])
    assert out["limitacoes_da_analise"] == ["Montagem 2D limita a avaliação."]
    assert out["necessidade_de_revisao_humana"] is True
    assert "comentario_extra" not in out


def test_parser_still_blocks_diagnosis_after_normalization():
    # normalizar idioma/caixa NÃO pode contrabandear uma conclusão diagnóstica
    config = load_screening_config("configs/medgemma_4b.yaml")
    report = valid_report("POSITIVE")
    report["resumo_do_achado"] = "Cancer confirmed no lobo direito."
    with pytest.raises(PipelineError, match="diagnóstico definitivo"):
        validate_medgemma_report(report, config["report"])


def test_parser_still_rejects_untranslatable_state():
    config = load_screening_config("configs/medgemma_4b.yaml")
    with pytest.raises(PipelineError, match="Estado MedGemma inválido"):
        validate_medgemma_report(valid_report("MAYBE"), config["report"])


def test_report_always_requires_human_review():
    config = load_screening_config("configs/medgemma_4b.yaml")
    report = valid_report()
    report["necessidade_de_revisao_humana"] = False
    with pytest.raises(PipelineError, match="sempre true"):
        validate_medgemma_report(report, config["report"])


def test_report_rejects_definitive_diagnosis():
    config = load_screening_config("configs/medgemma_4b.yaml")
    report = valid_report("POSITIVA")
    report["resumo_do_achado"] = "O paciente tem câncer confirmado."
    with pytest.raises(PipelineError, match="diagnóstico definitivo"):
        validate_medgemma_report(report, config["report"])


def test_report_extracts_single_fenced_json_after_model_reasoning():
    config = load_screening_config("configs/medgemma_4b.yaml")
    raw = (
        "<unused94>thought\ninternal reasoning that must not be persisted"
        "<unused95>```json\n"
        + json.dumps(valid_report("NEGATIVA"), ensure_ascii=False)
        + "\n```"
    )
    parsed = validate_medgemma_report(raw, config["report"])
    assert parsed["resultado_hipotese"] == "NEGATIVA"


def test_backend_not_configured_aborts_clearly(tmp_path):
    config = load_screening_config("configs/medgemma_4b.yaml")
    client = HTTPJSONMedGemmaClient(config)
    with pytest.raises(PipelineError, match="backend not configured"):
        client.generate(tmp_path / "not-needed.png", "prompt")


def test_unavailable_27b_aborts_clearly(tmp_path):
    config = load_screening_config(
        "configs/medgemma_27b.yaml",
        environ={"MEDGEMMA_BACKEND_CONFIGURED": "true"},
    )
    client = HTTPJSONMedGemmaClient(config)
    with pytest.raises(PipelineError, match="Modelo configurado não está disponível"):
        client.generate(tmp_path / "not-needed.png", "prompt")


class _Context:
    def __init__(self, body=b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_http_adapter_sends_contract_and_validates_echoed_model(tmp_path, monkeypatch):
    config = load_screening_config(
        "configs/medgemma_4b.yaml",
        environ={
            "MEDGEMMA_BACKEND_CONFIGURED": "true",
            "MEDGEMMA_MODEL_AVAILABLE": "true",
        },
    )
    panel = tmp_path / "panel.png"
    panel.write_bytes(b"test-png-bytes")
    captured = {}

    monkeypatch.setattr(
        "dtwin.medgemma_client.socket.create_connection",
        lambda *_args, **_kwargs: _Context(),
    )

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        body = json.dumps(
            {
                "model_id": config["medgemma"]["model_id"],
                "model_version": config["medgemma"]["model_version"],
                "report": valid_report("NEGATIVA"),
            }
        ).encode("utf-8")
        return _Context(body)

    monkeypatch.setattr("dtwin.medgemma_client.urlopen", fake_urlopen)
    report = HTTPJSONMedGemmaClient(config).generate(panel, "research prompt")
    assert captured["contract"] == "dtwin-medgemma-v1"
    assert captured["model_id"] == "google/medgemma-1.5-4b-it"
    assert captured["image"]["mime_type"] == "image/png"
    assert report["resultado_hipotese"] == "NEGATIVA"


def test_http_adapter_discards_response_from_wrong_model(tmp_path, monkeypatch):
    config = load_screening_config(
        "configs/medgemma_4b.yaml",
        environ={
            "MEDGEMMA_BACKEND_CONFIGURED": "true",
            "MEDGEMMA_MODEL_AVAILABLE": "true",
        },
    )
    panel = tmp_path / "panel.png"
    panel.write_bytes(b"test-png-bytes")
    monkeypatch.setattr(
        "dtwin.medgemma_client.socket.create_connection",
        lambda *_args, **_kwargs: _Context(),
    )
    monkeypatch.setattr(
        "dtwin.medgemma_client.urlopen",
        lambda *_args, **_kwargs: _Context(
            json.dumps(
                {
                    "model_id": "wrong/model",
                    "model_version": "Wrong model",
                    "report": valid_report(),
                }
            ).encode("utf-8")
        ),
    )
    with pytest.raises(PipelineError, match="não confirmou exatamente"):
        HTTPJSONMedGemmaClient(config).generate(panel, "research prompt")
