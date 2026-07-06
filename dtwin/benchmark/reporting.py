"""Persistência atômica e relatório humano das execuções de benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import BenchmarkCaseResult


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value * 100:.1f}%"
    return str(value)


def build_summary(run_manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    primary = metrics["primary"]
    secondary = metrics["decisions_only"]
    gate = metrics["gate"]
    rows = [
        ("Sensibilidade", primary.get("sensitivity")),
        ("Especificidade", primary.get("specificity")),
        ("Acurácia", primary.get("accuracy")),
        ("Precisão penalizada", primary.get("precision")),
        ("F1 penalizado", primary.get("f1_score")),
        ("Cobertura", primary.get("coverage_rate")),
    ]
    lines = [
        "# Benchmark MedGemma — ARGOS",
        "",
        f"- run_id: `{run_manifest.get('run_id')}`",
        f"- criado em: `{run_manifest.get('created_at')}`",
        f"- commit: `{run_manifest.get('code_commit')}`",
        f"- dirty tree: `{'sim' if run_manifest.get('git_dirty') else 'não'}`",
        f"- modelo: `{run_manifest.get('model_id')}` ({run_manifest.get('model_parameter_scale')})",
        f"- estratégia: `{run_manifest.get('experimental_strategy')}`",
        f"- casos: {primary.get('total_cases')} ({primary.get('positive_cases')} positivos / {primary.get('negative_cases')} negativos)",
        f"- inconclusivos: {primary.get('inconclusive_count')}",
        f"- falhas: {primary.get('failure_count')}",
        f"- timeouts: {primary.get('timeout_count')}",
        f"- inválidos: {primary.get('invalid_response_count')}",
        "",
        "## Métricas primárias — all-cases",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        *[f"| {name} | {_fmt(value)} |" for name, value in rows],
        "",
        "## Métricas secundárias — decisions-only",
        "",
        f"> {secondary['warning']}",
        "",
        f"- Acurácia: {_fmt(secondary.get('accuracy'))}",
        f"- Sensibilidade: {_fmt(secondary.get('sensitivity'))}",
        f"- Especificidade: {_fmt(secondary.get('specificity'))}",
        "",
        "## Gate de 75%",
        "",
        f"**{'PASS' if gate.get('passed') else 'FAIL'}** — sensibilidade e especificidade devem atingir os limites simultaneamente.",
        "",
        "## Limitações",
        "",
        "- Uso exclusivo em Pesquisa; não é diagnóstico nem laudo médico.",
        "- Máscaras de lesão são permitidas somente na avaliação posterior.",
        "- O resultado depende dos labels públicos e da qualidade da importação.",
        "- A matriz binária é uma pontuação conservadora; consulte também a matriz categórica.",
        "",
    ]
    return "\n".join(lines)


def write_run_outputs(
    run_dir: Path,
    run_manifest: dict[str, Any],
    case_results: Iterable[BenchmarkCaseResult | dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, str]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        item.to_dict()
        if isinstance(item, BenchmarkCaseResult)
        else BenchmarkCaseResult.from_mapping(item).to_dict()
        for item in case_results
    ]
    _json(run_dir / "run_manifest.json", run_manifest)
    _atomic_text(
        run_dir / "cases.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cases),
    )
    _json(run_dir / "metrics_primary.json", metrics["primary"])
    _json(run_dir / "metrics_decisions_only.json", metrics["decisions_only"])
    _json(
        run_dir / "confusion_matrices.json",
        {
            "penalized_binary_scoring_matrix": metrics["primary"]["penalized_binary_scoring_matrix"],
            "categorical_confusion_matrix": metrics["primary"]["categorical_confusion_matrix"],
            "decisions_only": metrics["decisions_only"]["confusion_matrix"],
        },
    )
    _atomic_text(run_dir / "summary.md", build_summary(run_manifest, metrics))
    return {
        name: str(run_dir / name)
        for name in (
            "run_manifest.json", "cases.jsonl", "metrics_primary.json",
            "metrics_decisions_only.json", "confusion_matrices.json", "summary.md",
        )
    }
