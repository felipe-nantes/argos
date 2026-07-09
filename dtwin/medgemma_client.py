#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuração, prompt, adaptador HTTP e gates de resposta do MedGemma."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

from .core import PipelineError


REQUIRED_REPORT_FIELDS = {
    "resultado_hipotese",
    "resumo_do_achado",
    "localizacao_aproximada",
    "sinais_visuais_observados",
    "confianca",
    "limitacoes_da_analise",
    "necessidade_de_revisao_humana",
}
OPTIONAL_REPORT_V2_FIELDS = {
    "alvo_da_triagem",
    "ha_lesao_focal_suspeita",
    "ha_variante_anatomica_benigna",
    "ha_pseudolesao_ou_artefato",
    "tipo_alteracao_nao_alvo",
    "justificativa_da_separacao",
}
REPORT_V2_TARGET = "lesao_focal_hepatica_suspeita"
NON_TARGET_ALTERATION_TYPES = {
    "none",
    "vascular_variant",
    "perfusion_pseudolesion",
    "artifact",
    "focal_fat",
    "cystic_benign",
    "other",
}

# Sinônimos aceitos para canonicalizar saída SEMANTICAMENTE válida ao vocabulário
# exigido (idioma/caixa). Não recupera valores fora desta lista — isso não seria
# canonicalização, seria adivinhação.
_STATE_SYNONYMS = {
    "POSITIVA": "POSITIVA", "POSITIVE": "POSITIVA", "POSITIVO": "POSITIVA",
    "NEGATIVA": "NEGATIVA", "NEGATIVE": "NEGATIVA", "NEGATIVO": "NEGATIVA",
    "INCONCLUSIVA": "INCONCLUSIVA", "INCONCLUSIVE": "INCONCLUSIVA",
    "INCONCLUSIVO": "INCONCLUSIVA", "INDETERMINADA": "INCONCLUSIVA",
    "INDETERMINADO": "INCONCLUSIVA", "INDETERMINATE": "INCONCLUSIVA",
}
_CONFIDENCE_SYNONYMS = {
    "BAIXA": "baixa", "LOW": "baixa",
    "MODERADA": "moderada", "MODERATE": "moderada", "MEDIA": "moderada",
    "MÉDIA": "moderada", "MEDIUM": "moderada",
    "ALTA": "alta", "HIGH": "alta",
}


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "sim"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "nao", "não"}:
        return False
    raise PipelineError(f"Configuração booleana inválida em {label}: {value!r}")


def _set_env_override(target: dict[str, Any], key: str, value: str) -> None:
    integer_keys = {
        "timeout_seconds",
        "max_retries",
        "response_validation_max_retries",
        "max_input_bytes",
        "max_output_tokens",
        "max_prompt_chars",
        "max_image_pixels",
    }
    boolean_keys = {
        "enabled",
        "backend_configured",
        "model_available",
        "allow_remote",
        "local_files_only",
    }
    if key in integer_keys:
        target[key] = int(value)
    elif key in boolean_keys:
        target[key] = _bool(value, key)
    else:
        target[key] = value


def load_screening_config(
    path: Path | str, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"Configuração MedGemma não encontrada: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Configuração MedGemma inválida ({path}): {exc}") from exc
    inherited: dict[str, Any] = {}
    if raw.get("extends"):
        parent = (path.parent / str(raw["extends"])).resolve()
        if parent.parent != path.parent.resolve() or parent == path.resolve():
            raise PipelineError("Configuração MedGemma extends deve apontar para arquivo no mesmo diretório.")
        inherited = load_screening_config(parent, environ={})
    if not isinstance(raw.get("medgemma_screening"), dict):
        raise PipelineError("Configuração sem bloco 'medgemma_screening'.")
    config = copy.deepcopy(inherited)
    _deep_merge(config, raw["medgemma_screening"])
    med = config.get("medgemma")
    if not isinstance(med, dict):
        raise PipelineError("Configuração sem bloco 'medgemma_screening.medgemma'.")
    med.setdefault("response_validation_max_retries", 1)

    env = os.environ if environ is None else environ
    overrides = {
        "MEDGEMMA_ENABLED": "enabled",
        "MEDGEMMA_PROVIDER": "provider",
        "MEDGEMMA_MODEL_ID": "model_id",
        "MEDGEMMA_MODEL_VERSION": "model_version",
        "MEDGEMMA_MODEL_PARAMETER_SCALE": "model_parameter_scale",
        "MEDGEMMA_ENDPOINT_URL": "endpoint_url",
        "MEDGEMMA_TIMEOUT_SECONDS": "timeout_seconds",
        "MEDGEMMA_MAX_RETRIES": "max_retries",
        "MEDGEMMA_RESPONSE_VALIDATION_MAX_RETRIES": "response_validation_max_retries",
        "MEDGEMMA_MAX_INPUT_BYTES": "max_input_bytes",
        "MEDGEMMA_MAX_OUTPUT_TOKENS": "max_output_tokens",
        "MEDGEMMA_MAX_PROMPT_CHARS": "max_prompt_chars",
        "MEDGEMMA_MAX_IMAGE_PIXELS": "max_image_pixels",
        "MEDGEMMA_EXECUTION_MODE": "execution_mode",
        "MEDGEMMA_DEVICE": "device",
        "MEDGEMMA_QUANTIZATION": "quantization",
        "MEDGEMMA_BACKEND_CONFIGURED": "backend_configured",
        "MEDGEMMA_MODEL_AVAILABLE": "model_available",
        "MEDGEMMA_ALLOW_REMOTE": "allow_remote",
        "MEDGEMMA_LOCAL_FILES_ONLY": "local_files_only",
    }
    for env_name, key in overrides.items():
        if env_name in env:
            try:
                _set_env_override(med, key, env[env_name])
            except (TypeError, ValueError) as exc:
                raise PipelineError(
                    f"Variável de ambiente {env_name} inválida: {env[env_name]!r}"
                ) from exc
    _validate_config(config)
    return config


def _deep_merge(target: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("enabled") is not True:
        raise PipelineError("medgemma_screening.enabled deve ser true.")
    if str(config.get("regulatory_mode", "")) != "RESEARCH":
        raise PipelineError("MedGemma screening deve permanecer em regulatory_mode=RESEARCH.")
    if config.get("lesion_pre_marked") is not False:
        raise PipelineError("Este fluxo exige lesion_pre_marked=false.")
    report = config.get("report", {})
    allowed_states = report.get("allowed_states", [])
    if set(allowed_states) != {"POSITIVA", "NEGATIVA", "INCONCLUSIVA"}:
        raise PipelineError("report.allowed_states deve conter exatamente os três estados permitidos.")
    if set(report.get("allowed_confidence", [])) != {"baixa", "moderada", "alta"}:
        raise PipelineError(
            "report.allowed_confidence deve conter exatamente baixa, moderada e alta."
        )
    if report.get("requires_human_review") is not True:
        raise PipelineError("report.requires_human_review deve ser true.")
    disclaimer = str(report.get("disclaimer", ""))
    disclaimer_required = ("pesquisa", "não é diagnóstico", "revisão")
    if any(fragment not in disclaimer.lower() for fragment in disclaimer_required):
        raise PipelineError(
            "report.disclaimer deve declarar modo Pesquisa, não diagnóstico e revisão humana."
        )
    panel = config.get("panel", {})
    strategy = str(panel.get("strategy", "uniform_9"))
    if strategy not in {"uniform_9", "volumetric_blocks"}:
        raise PipelineError(
            "panel.strategy deve ser 'uniform_9' ou 'volumetric_blocks'."
        )
    if strategy == "volumetric_blocks" and int(
        panel.get("axial_tiles_per_panel", 9)
    ) != 9:
        raise PipelineError(
            "panel.axial_tiles_per_panel deve ser 9 para preservar a grade 4x3."
        )
    if panel.get("include_coronal") is not True or panel.get("include_sagittal") is not True:
        raise PipelineError("O painel exige vistas coronal e sagital.")
    mode = str(panel.get("mode", "single_grayscale"))
    if mode == "single_grayscale":
        if panel.get("overlay_mode") != "contour":
            raise PipelineError("Somente overlay_mode=contour é permitido neste fluxo.")
        if panel.get("preserve_grayscale_signal") is not True:
            raise PipelineError("panel.preserve_grayscale_signal deve ser true.")
    elif mode in ("multiphase_fusion", "texture_fusion"):
        # Fusão RGB (multifásica ou textura de um único volume): a fidelidade em
        # cinza é abandonada de propósito (imagem derivada, assim rotulada). A
        # revisão humana do painel e a confirmação de PHI seguem obrigatórias.
        fusion = panel.get("fusion", {})
        channel_map = fusion.get("channel_map", {})
        if not isinstance(channel_map, dict) or set(channel_map) != {"red", "green", "blue"}:
            raise PipelineError(
                "panel.fusion.channel_map deve mapear exatamente red/green/blue -> fase."
            )
        if any(not str(channel_map[c]).strip() for c in ("red", "green", "blue")):
            raise PipelineError("panel.fusion.channel_map tem fase vazia.")
    else:
        raise PipelineError(f"panel.mode desconhecido: {mode!r}.")
    privacy = config.get("privacy", {})
    if privacy.get("remove_png_metadata") is not True:
        raise PipelineError("privacy.remove_png_metadata deve ser true.")
    rag = config.get("rag", {})
    if rag:
        if not isinstance(rag, dict):
            raise PipelineError("rag deve ser um bloco de configuração.")
        if rag.get("enabled") is True:
            for key in ("index_path", "retrieval_eval"):
                value = str(rag.get(key, "")).strip()
                path = Path(value)
                if not value:
                    raise PipelineError(f"rag.{key} é obrigatório quando rag.enabled=true.")
                if path.is_absolute() or ".." in path.parts:
                    raise PipelineError(f"rag.{key} deve ser caminho relativo seguro dentro do repositório.")
            for key in ("top_k", "max_sources", "max_chunk_chars"):
                if int(rag.get(key, 0)) <= 0:
                    raise PipelineError(f"rag.{key} deve ser positivo quando rag.enabled=true.")
            if float(rag.get("min_score", 0.0)) < 0:
                raise PipelineError("rag.min_score não pode ser negativo.")

    med = config["medgemma"]
    required = (
        "provider", "model_family", "model_version", "model_parameter_scale",
        "timeout_seconds", "max_retries", "execution_mode", "device",
        "max_input_bytes", "max_output_tokens", "minimum_timeout_seconds",
    )
    missing = [key for key in required if key not in med]
    if missing:
        raise PipelineError(f"Configuração MedGemma sem campos obrigatórios: {missing}")
    if med.get("model_family") != "MedGemma":
        raise PipelineError("model_family deve ser MedGemma.")
    timeout = int(med["timeout_seconds"])
    if timeout < int(med["minimum_timeout_seconds"]):
        raise PipelineError(
            f"Timeout de {timeout}s é insuficiente para {med['model_parameter_scale']} "
            f"(mínimo configurado: {med['minimum_timeout_seconds']}s)."
        )
    if int(med["max_retries"]) < 0:
        raise PipelineError("max_retries não pode ser negativo.")
    if int(med.get("response_validation_max_retries", 1)) < 0:
        raise PipelineError("response_validation_max_retries não pode ser negativo.")
    if int(med["max_input_bytes"]) <= 0 or int(med["max_output_tokens"]) <= 0:
        raise PipelineError("Limites de entrada/saída MedGemma devem ser positivos.")
    if int(med.get("max_prompt_chars", 12000)) <= 0:
        raise PipelineError("medgemma.max_prompt_chars deve ser positivo.")
    if int(med.get("max_image_pixels", 4_000_000)) <= 0:
        raise PipelineError("medgemma.max_image_pixels deve ser positivo.")
    if med.get("execution_mode") not in {"local", "remote"}:
        raise PipelineError("medgemma.execution_mode deve ser local ou remote.")
    if not str(med.get("endpoint_url", "")).strip():
        raise PipelineError("medgemma.endpoint_url é obrigatório.")


def model_trace(config: dict[str, Any]) -> dict[str, Any]:
    med = config["medgemma"]
    return {
        "model_family": med["model_family"],
        "model_version": med["model_version"],
        "model_parameter_scale": med["model_parameter_scale"],
        "model_id": med.get("model_id"),
        "execution_mode": med["execution_mode"],
        "device": med["device"],
        "quantization": med.get("quantization"),
        "requires_human_review": True,
    }


def effective_config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_medgemma_prompt(config: dict[str, Any]) -> str:
    prompt = str(config.get("prompt", {}).get("template", "")).strip()
    required_fragments = (
        "POSITIVA", "NEGATIVA", "INCONCLUSIVA", "modo de pesquisa",
        "revisão humana obrigatória", "não é diagnóstico", "não é laudo médico",
    )
    missing = [fragment for fragment in required_fragments if fragment.lower() not in prompt.lower()]
    if missing:
        raise PipelineError(f"Prompt MedGemma sem salvaguardas obrigatórias: {missing}")
    max_chars = int(config["medgemma"].get("max_prompt_chars", 12000))
    if len(prompt) > max_chars:
        raise PipelineError(
            f"Prompt MedGemma excede max_prompt_chars ({len(prompt)} > {max_chars})."
        )
    return prompt


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_error(value: Any, limit: int = 500) -> str:
    text = str(value).replace("\n", " ").strip()
    return text[:limit]


def _validation_retry_prompt(original_prompt: str, error: str, max_chars: int) -> str:
    suffix = f"""

CORREÇÃO OBRIGATÓRIA DE FORMATO:
A resposta anterior foi rejeitada pelo validador interno por este erro de schema: {_short_error(error, 350)}

Refaça a análise do mesmo painel e responda novamente. Retorne somente um objeto JSON válido, sem Markdown,
sem texto fora do JSON, sem copiar opções separadas por "|", sem usar placeholders e sem diagnóstico definitivo.

Use exatamente estes valores permitidos:
- resultado_hipotese: "POSITIVA", "NEGATIVA" ou "INCONCLUSIVA"
- confianca: "baixa", "moderada" ou "alta"
- necessidade_de_revisao_humana: true
- Se usar campos v2 opcionais:
  - alvo_da_triagem: "lesao_focal_hepatica_suspeita"
  - resultado_hipotese=POSITIVA exige ha_lesao_focal_suspeita=true
  - variante anatômica benigna ou pseudolesão isolada não pode tornar o caso POSITIVA
  - tipo_alteracao_nao_alvo: "none", "vascular_variant", "perfusion_pseudolesion", "artifact", "focal_fat", "cystic_benign" ou "other"

Mantenha modo de pesquisa, não é diagnóstico, não é laudo médico e revisão humana obrigatória.
"""
    if len(original_prompt) + len(suffix) <= max_chars:
        return original_prompt + suffix

    compact = f"""
Você está analisando uma montagem de RM abdominal em modo de pesquisa. A região do fígado está segmentada por contorno.
A resposta anterior foi rejeitada por erro de schema: {_short_error(error, 350)}

Baseie-se apenas no painel enviado. Não é diagnóstico. Não é laudo médico. Revisão humana obrigatória.
Não recomende tratamento, cirurgia, biópsia ou medicação.

Retorne somente um objeto JSON válido, sem Markdown e sem texto adicional:
- resultado_hipotese deve ser uma única string escolhida entre POSITIVA, NEGATIVA, INCONCLUSIVA.
- resumo_do_achado deve ser string.
- localizacao_aproximada deve ser string.
- sinais_visuais_observados deve ser lista de strings.
- confianca deve ser uma única string escolhida entre baixa, moderada, alta.
- limitacoes_da_analise deve ser lista de strings.
- necessidade_de_revisao_humana deve ser true.
- Se usar campos v2, POSITIVA exige ha_lesao_focal_suspeita=true.
- Variante anatômica benigna/pseudolesão isolada não pode ser POSITIVA.
"""
    if len(compact) > max_chars:
        raise PipelineError(
            f"Prompt de retry MedGemma excede max_prompt_chars ({len(compact)} > {max_chars})."
        )
    return compact.strip()


def _parse_json_report(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise PipelineError("Resposta MedGemma não contém objeto JSON de relatório.")
    text = value.strip()
    candidates: list[dict[str, Any]] = []

    # MedGemma pode emitir tokens de raciocínio antes da resposta e envolver o
    # objeto final em ```json```. O raciocínio nunca é persistido: extraímos
    # somente objetos JSON completos e o schema clínico é validado em seguida.
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    chunks = fenced or [text]
    decoder = json.JSONDecoder()
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                candidates.append(parsed)
                continue
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", chunk):
            try:
                parsed, _end = decoder.raw_decode(chunk[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)

    unique = {
        json.dumps(candidate, sort_keys=True, ensure_ascii=False): candidate
        for candidate in candidates
    }
    if not unique:
        raise PipelineError("Resposta MedGemma não contém objeto JSON válido.")
    if len(unique) != 1:
        raise PipelineError("Resposta MedGemma contém múltiplos objetos JSON ambíguos.")
    return next(iter(unique.values()))


def _all_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_all_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_all_text(v) for v in value)
    return ""


def _canonicalize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Canonicaliza saída semanticamente válida ao vocabulário exigido.

    Tolerante a idioma/caixa/tipo: ``"NEGATIVE"`` -> ``NEGATIVA``, confiança
    ``"low"`` -> ``baixa``, um campo de lista vindo como string -> lista de uma
    string. NÃO inventa conteúdo, NÃO afrouxa o gate de segurança e NÃO recupera
    estados/valores não reconhecidos (esses continuam sendo rejeitados adiante).
    """
    report = dict(report)
    state = report.get("resultado_hipotese")
    if isinstance(state, str):
        report["resultado_hipotese"] = _STATE_SYNONYMS.get(state.strip().upper(), state.strip().upper())
    conf = report.get("confianca")
    if isinstance(conf, str):
        report["confianca"] = _CONFIDENCE_SYNONYMS.get(conf.strip().upper(), conf.strip().lower())
    nrh = report.get("necessidade_de_revisao_humana")
    if isinstance(nrh, str):
        try:
            report["necessidade_de_revisao_humana"] = _bool(nrh, "necessidade_de_revisao_humana")
        except PipelineError:
            pass
    for key in ("resumo_do_achado", "localizacao_aproximada"):
        if isinstance(report.get(key), (int, float, bool)):
            report[key] = str(report[key])
    for key in ("sinais_visuais_observados", "limitacoes_da_analise"):
        v = report.get(key)
        if isinstance(v, str):
            report[key] = [v.strip()] if v.strip() else []
        elif isinstance(v, list):
            report[key] = [x if isinstance(x, str) else str(x) for x in v]
    for key in (
        "ha_lesao_focal_suspeita",
        "ha_variante_anatomica_benigna",
        "ha_pseudolesao_ou_artefato",
    ):
        value = report.get(key)
        if isinstance(value, str):
            try:
                report[key] = _bool(value, key)
            except PipelineError:
                pass
    for key in ("alvo_da_triagem", "tipo_alteracao_nao_alvo", "justificativa_da_separacao"):
        if isinstance(report.get(key), (int, float, bool)):
            report[key] = str(report[key])
    if isinstance(report.get("tipo_alteracao_nao_alvo"), str):
        report["tipo_alteracao_nao_alvo"] = report["tipo_alteracao_nao_alvo"].strip().lower()
    if isinstance(report.get("alvo_da_triagem"), str):
        report["alvo_da_triagem"] = report["alvo_da_triagem"].strip().lower()
    return report


def _validate_optional_report_v2(report: dict[str, Any]) -> dict[str, Any]:
    present = OPTIONAL_REPORT_V2_FIELDS.intersection(report)
    if not present:
        return {}
    v2 = {key: report[key] for key in OPTIONAL_REPORT_V2_FIELDS if key in report}
    if "alvo_da_triagem" in v2 and v2["alvo_da_triagem"] != REPORT_V2_TARGET:
        raise PipelineError(
            f"alvo_da_triagem inválido: {v2['alvo_da_triagem']!r}; esperado {REPORT_V2_TARGET!r}."
        )
    for key in (
        "ha_lesao_focal_suspeita",
        "ha_variante_anatomica_benigna",
        "ha_pseudolesao_ou_artefato",
    ):
        if key in v2 and not isinstance(v2[key], bool):
            raise PipelineError(f"Campo v2 {key} deve ser booleano.")
    if "tipo_alteracao_nao_alvo" in v2:
        value = v2["tipo_alteracao_nao_alvo"]
        if value not in NON_TARGET_ALTERATION_TYPES:
            raise PipelineError(f"tipo_alteracao_nao_alvo inválido: {value!r}")
    if "justificativa_da_separacao" in v2:
        value = v2["justificativa_da_separacao"]
        if not isinstance(value, str) or not value.strip():
            raise PipelineError("justificativa_da_separacao deve ser string não vazia.")

    state = report["resultado_hipotese"]
    lesion_flag = v2.get("ha_lesao_focal_suspeita")
    benign_flag = v2.get("ha_variante_anatomica_benigna") is True
    pseudo_flag = v2.get("ha_pseudolesao_ou_artefato") is True
    if state == "POSITIVA" and lesion_flag is False:
        raise PipelineError(
            "Inconsistência v2: resultado_hipotese=POSITIVA exige ha_lesao_focal_suspeita=true."
        )
    if state == "POSITIVA" and lesion_flag is not True and (benign_flag or pseudo_flag):
        raise PipelineError(
            "Inconsistência v2: variante anatômica/pseudolesão isolada não pode tornar o caso POSITIVA."
        )
    if state == "POSITIVA" and lesion_flag is None and present:
        raise PipelineError(
            "Inconsistência v2: relatório POSITIVA com campos v2 deve declarar ha_lesao_focal_suspeita=true."
        )
    return v2


def validate_medgemma_report(
    raw_report: Any, report_config: dict[str, Any]
) -> dict[str, Any]:
    report = _canonicalize_report(_parse_json_report(raw_report))
    missing = REQUIRED_REPORT_FIELDS - set(report)
    if missing:
        raise PipelineError(
            f"Campos obrigatórios do relatório MedGemma ausentes: {sorted(missing)}."
        )
    report = {
        key: report[key]
        for key in (*REQUIRED_REPORT_FIELDS, *OPTIONAL_REPORT_V2_FIELDS)
        if key in report
    }  # descarta chaves extras não autorizadas
    state = report["resultado_hipotese"]
    if state not in report_config["allowed_states"]:
        raise PipelineError(f"Estado MedGemma inválido: {state!r}")
    confidence = report["confianca"]
    if confidence not in report_config["allowed_confidence"]:
        raise PipelineError(f"Confiança MedGemma inválida: {confidence!r}")
    if report["necessidade_de_revisao_humana"] is not True:
        raise PipelineError("necessidade_de_revisao_humana deve ser sempre true.")
    for key in ("resumo_do_achado", "localizacao_aproximada"):
        if not isinstance(report[key], str) or not report[key].strip():
            raise PipelineError(f"Campo {key} deve ser string não vazia.")
    for key in ("sinais_visuais_observados", "limitacoes_da_analise"):
        if not isinstance(report[key], list) or not all(isinstance(x, str) for x in report[key]):
            raise PipelineError(f"Campo {key} deve ser uma lista de strings.")
    report.update(_validate_optional_report_v2(report))

    text = _all_text(report).lower()
    forbidden = (
        r"\b(?:paciente\s+(?:tem|não tem)|patient\s+(?:has|does not have))\b",
        r"\b(?:câncer|cancer|tumor|neoplasia)\s+(?:confirmad[oa]|descartad[oa]|confirmed|ruled out)\b",
        r"\bdiagnóstico\s+(?:definitivo|confirmado|de)\b",
        r"\bdefinitive diagnosis\b",
        r"\b(?:recomendo|prescrever|prescrevo)\b",
        r"\b(?:recommend|recommended)\s+(?:treatment|surgery|biopsy|medication)\b",
        r"\bdeve\s+(?:iniciar|realizar|operar|ser submetid[oa])\b",
    )
    for pattern in forbidden:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise PipelineError(
                "Resposta MedGemma contém diagnóstico definitivo ou sugestão de conduta clínica."
            )
    return report


class HTTPJSONMedGemmaClient:
    """Adaptador para um gateway HTTP local com contrato dtwin-medgemma-v1.

    O pipeline conhece apenas esta interface. Backends com API diferente devem
    ganhar outro adaptador, sem alterar montagem, gates ou persistência.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.med = config["medgemma"]
        self.last_timings: dict[str, float] = {}
        self.last_response_audit: dict[str, Any] = {}

    def _ensure_ready(self) -> None:
        if not _bool(self.med.get("enabled", False), "medgemma.enabled"):
            raise PipelineError("MedGemma está desabilitado na configuração. Abortando análise.")
        if self.med.get("provider") != "http_json_v1":
            raise PipelineError(
                f"Provider MedGemma não suportado: {self.med.get('provider')!r}. "
                "Implemente um adaptador específico para esse backend."
            )
        if not self.med.get("model_id"):
            raise PipelineError(
                f"Modelo configurado não está disponível: {self.med['model_version']} "
                "(model_id ausente). Abortando análise."
            )
        if not _bool(self.med.get("backend_configured", False), "backend_configured"):
            raise PipelineError("MedGemma backend not configured. Aborting analysis.")
        if not _bool(self.med.get("model_available", False), "model_available"):
            raise PipelineError(
                f"Modelo configurado não está disponível: {self.med['model_version']}. Abortando análise."
            )
        endpoint = str(self.med.get("endpoint_url", ""))
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise PipelineError(f"endpoint_url MedGemma inválido: {endpoint!r}")
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if self.med.get("execution_mode") == "local" and parsed.hostname not in local_hosts:
            raise PipelineError("execution_mode=local exige endpoint loopback (localhost/127.0.0.1/::1).")
        if self.med.get("execution_mode") == "remote" and not _bool(
            self.med.get("allow_remote", False), "allow_remote"
        ):
            raise PipelineError("Execução remota bloqueada: allow_remote=false.")

    def check_ready(self) -> dict[str, Any]:
        """Valida configuração, conectividade e identidade do backend sem inferir."""
        self._ensure_ready()
        parsed = urlparse(str(self.med["endpoint_url"]))
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection(
                (str(parsed.hostname), int(port)), timeout=min(10, int(self.med["timeout_seconds"]))
            ):
                pass
        except OSError as exc:
            raise PipelineError(
                f"Backend MedGemma inacessível em {parsed.hostname}:{port}: {exc}"
            ) from exc
        health_url = self.med.get("healthcheck_url")
        if health_url:
            try:
                request = Request(str(health_url), headers={"Accept": "application/json"}, method="GET")
                with urlopen(request, timeout=min(15, int(self.med["timeout_seconds"]))) as response:
                    health = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                raise PipelineError(f"Health check MedGemma falhou: {exc}") from exc
            if not isinstance(health, dict) or health.get("status") != "ready":
                detail = health.get("load_error") if isinstance(health, dict) else "resposta inválida"
                raise PipelineError(f"Backend MedGemma não está pronto: {detail}")
            if (
                health.get("model_id") != self.med["model_id"]
                or health.get("model_version") != self.med["model_version"]
            ):
                raise PipelineError("Health check não confirmou exatamente o modelo configurado.")
            if health.get("contract") not in {None, "dtwin-medgemma-v1"}:
                raise PipelineError("Health check retornou contrato MedGemma incompatível.")
            return health
        return {
            "status": "reachable",
            "model_id": self.med["model_id"],
            "model_version": self.med["model_version"],
        }

    def _post_generate(self, panel_bytes: bytes, prompt: str) -> dict[str, Any]:
        payload = {
            "contract": "dtwin-medgemma-v1",
            "model_id": self.med["model_id"],
            "model_version": self.med["model_version"],
            "prompt": prompt,
            "image": {
                "mime_type": "image/png",
                "base64": base64.b64encode(panel_bytes).decode("ascii"),
            },
            "generation": {"max_output_tokens": int(self.med["max_output_tokens"])},
        }
        request = Request(
            str(self.med["endpoint_url"]),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        attempts = int(self.med["max_retries"]) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=int(self.med["timeout_seconds"])) as response:
                    body = response.read()
                decoded = json.loads(body.decode("utf-8"))
                break
            except HTTPError as exc:
                detail = exc.read(500).decode("utf-8", errors="replace")
                raise PipelineError(
                    f"Backend MedGemma retornou HTTP {exc.code}: {detail}"
                ) from exc
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 5))
        else:
            raise PipelineError(
                f"Falha ao chamar backend MedGemma após {attempts} tentativa(s): {last_error}"
            ) from last_error
        if not isinstance(decoded, dict):
            raise PipelineError("Resposta do backend MedGemma deve ser um objeto JSON.")
        return decoded

    def generate(self, panel_path: Path, prompt: str) -> dict[str, Any]:
        total_started = time.monotonic()
        self._ensure_ready()
        panel_path = Path(panel_path)
        if not panel_path.is_file():
            raise PipelineError(f"Painel MedGemma não encontrado: {panel_path}")
        panel_bytes = panel_path.read_bytes()
        if len(panel_bytes) > int(self.med["max_input_bytes"]):
            raise PipelineError(
                f"Painel excede max_input_bytes ({len(panel_bytes)} > {self.med['max_input_bytes']})."
            )
        ready_started = time.monotonic()
        self.check_ready()
        self.last_timings = {"backend_readiness": round(time.monotonic() - ready_started, 4)}
        self.last_response_audit = {
            "schema": "argos-medgemma-response-validation-v1",
            "max_validation_retries": int(self.med.get("response_validation_max_retries", 1)),
            "raw_response_persisted": False,
            "attempts": [],
        }

        validation_attempts = int(self.med.get("response_validation_max_retries", 1)) + 1
        max_prompt_chars = int(self.med.get("max_prompt_chars", 12000))
        current_prompt = prompt
        total_inference_seconds = 0.0
        total_validation_seconds = 0.0
        last_validation_error: PipelineError | None = None
        for validation_attempt in range(1, validation_attempts + 1):
            inference_started = time.monotonic()
            decoded = self._post_generate(panel_bytes, current_prompt)
            total_inference_seconds += time.monotonic() - inference_started
            audit_entry = {
                "attempt": validation_attempt,
                "prompt_sha256": _sha256_text(current_prompt),
                "repair_prompt": validation_attempt > 1,
            }
            response_model_id = decoded.get("model_id")
            response_version = decoded.get("model_version")
            if response_model_id != self.med["model_id"] or response_version != self.med["model_version"]:
                audit_entry.update(status="rejected", error="model_identity_mismatch")
                self.last_response_audit["attempts"].append(audit_entry)
                raise PipelineError(
                    "Backend não confirmou exatamente o modelo configurado; relatório descartado."
                )
            raw_report = decoded.get("report", decoded.get("output"))
            validation_started = time.monotonic()
            try:
                validated = validate_medgemma_report(raw_report, self.config["report"])
            except PipelineError as exc:
                total_validation_seconds += time.monotonic() - validation_started
                last_validation_error = exc
                audit_entry.update(
                    status="invalid",
                    error_type=type(exc).__name__,
                    error_message=_short_error(exc),
                )
                self.last_response_audit["attempts"].append(audit_entry)
                if validation_attempt >= validation_attempts:
                    self.last_response_audit["repair_attempted"] = validation_attempts > 1
                    self.last_response_audit["repaired"] = False
                    self.last_timings["medgemma_inference"] = round(total_inference_seconds, 4)
                    self.last_timings["response_validation"] = round(total_validation_seconds, 4)
                    self.last_timings["response_validation_attempts"] = validation_attempt
                    self.last_timings["response_repair_used"] = False
                    self.last_timings["client_total"] = round(time.monotonic() - total_started, 4)
                    raise PipelineError(
                        f"Resposta MedGemma inválida após {validation_attempts} tentativa(s): {exc}"
                    ) from exc
                current_prompt = _validation_retry_prompt(prompt, str(exc), max_prompt_chars)
                continue

            total_validation_seconds += time.monotonic() - validation_started
            audit_entry["status"] = "accepted"
            self.last_response_audit["attempts"].append(audit_entry)
            self.last_response_audit["repair_attempted"] = validation_attempt > 1 or last_validation_error is not None
            self.last_response_audit["repaired"] = validation_attempt > 1
            self.last_timings["medgemma_inference"] = round(total_inference_seconds, 4)
            self.last_timings["response_validation"] = round(total_validation_seconds, 4)
            self.last_timings["response_validation_attempts"] = validation_attempt
            self.last_timings["response_repair_used"] = validation_attempt > 1
            self.last_timings["client_total"] = round(time.monotonic() - total_started, 4)
            return validated

        raise PipelineError(
            f"Resposta MedGemma inválida após {validation_attempts} tentativa(s): {last_validation_error}"
        )


def create_medgemma_client(config: dict[str, Any]) -> HTTPJSONMedGemmaClient:
    provider = config["medgemma"].get("provider")
    if provider == "http_json_v1":
        return HTTPJSONMedGemmaClient(config)
    raise PipelineError(
        f"Provider MedGemma sem adaptador registrado: {provider!r}. Abortando análise."
    )
