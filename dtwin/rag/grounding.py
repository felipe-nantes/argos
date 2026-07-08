"""Montagem do bloco textual RAG com salvaguardas para o prompt MedGemma."""
from __future__ import annotations

import hashlib
from typing import Any

from dtwin.core import PipelineError


def build_rag_prompt_addendum(context: dict[str, Any]) -> str:
    if context.get("enabled") is not True:
        return ""
    sources = context.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PipelineError("Contexto RAG habilitado sem fontes.")
    lines = [
        "CONTEXTO RAG TEXTUAL DE APOIO (não visual):",
        "Use as fontes abaixo apenas como lembrete de critérios radiológicos gerais.",
        "Não use o RAG para criar, confirmar ou descartar achado visual que não esteja nas imagens.",
        "Se o texto recuperado conflitar com as imagens ou se o painel for insuficiente, priorize as imagens e declare limitação/INCONCLUSIVA.",
        "Ao mencionar critério textual, referencie mentalmente as fontes [S#], mas mantenha a saída no schema JSON solicitado.",
        "",
        "Fontes recuperadas:",
    ]
    for source in sources:
        lines.extend([
            f"[{source['source_id']}] {source.get('title', '')} — seção: {source.get('section', '')}",
            f"Categorias: {', '.join(source.get('categories', []))}",
            f"Trecho: {source.get('text', '')}",
            "",
        ])
    lines.extend([
        "Regra de segurança RAG:",
        "- O resultado_hipotese continua sendo visual e baseado somente no painel enviado.",
        "- O RAG pode apoiar consistência, diferenciais e limitações, mas não substitui a análise da imagem.",
        "- Não emita diagnóstico definitivo, laudo médico ou recomendação de conduta.",
    ])
    return "\n".join(lines).strip()


def append_rag_to_prompt(
    base_prompt: str,
    context: dict[str, Any],
    *,
    max_prompt_chars: int,
) -> tuple[str, dict[str, Any]]:
    """Anexa o bloco RAG ao prompt e retorna prompt + auditoria de hashes."""

    if context.get("enabled") is not True:
        return base_prompt, {
            "enabled": False,
            "base_prompt_sha256": sha256_text(base_prompt),
            "final_prompt_sha256": sha256_text(base_prompt),
            "prompt_chars": len(base_prompt),
        }
    addendum = build_rag_prompt_addendum(context)
    final_prompt = f"{base_prompt}\n\n{addendum}".strip()
    if len(final_prompt) > max_prompt_chars:
        raise PipelineError(
            f"Prompt MedGemma com RAG excede max_prompt_chars ({len(final_prompt)} > {max_prompt_chars})."
        )
    return final_prompt, {
        "enabled": True,
        "base_prompt_sha256": sha256_text(base_prompt),
        "rag_addendum_sha256": sha256_text(addendum),
        "final_prompt_sha256": sha256_text(final_prompt),
        "base_prompt_chars": len(base_prompt),
        "rag_addendum_chars": len(addendum),
        "prompt_chars": len(final_prompt),
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

