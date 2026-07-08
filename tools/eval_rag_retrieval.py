#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Avalia a recuperação lexical do RAG ARGOS antes da integração ao MedGemma."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from dtwin.core import PipelineError, now_utc, sha256_of
from dtwin.rag.index import load_bm25_index, search_bm25, tokenize

EVAL_SCHEMA = "argos-rag-retrieval-eval-v1"
REPORT_SCHEMA = "argos-rag-retrieval-report-v1"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler YAML de avaliação RAG {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError("Avaliação RAG precisa ser um mapa YAML.")
    return data


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _load_chunk_texts(index: dict[str, Any]) -> dict[str, str]:
    corpus_dir_value = index.get("corpus_dir")
    if not corpus_dir_value:
        return {}
    corpus_dir = Path(str(corpus_dir_value))
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    texts: dict[str, str] = {}
    for entry in manifest.get("chunks", []):
        chunk_id = entry.get("chunk_id")
        rel = entry.get("path")
        if not chunk_id or not rel:
            continue
        path = (corpus_dir / rel).resolve()
        try:
            path.relative_to(corpus_dir.resolve())
        except ValueError:
            continue
        if not path.is_file():
            continue
        try:
            chunk = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if chunk.get("chunk_id") == chunk_id:
            texts[str(chunk_id)] = str(chunk.get("text", ""))
    return texts


def validate_eval_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema") != EVAL_SCHEMA:
        raise PipelineError(f"Schema de avaliação RAG inválido: {spec.get('schema')}")
    queries = spec.get("queries")
    if not isinstance(queries, list) or not queries:
        raise PipelineError("Avaliação RAG precisa conter lista não vazia em queries.")
    default_top_k = int(spec.get("default_top_k", 5))
    if default_top_k <= 0:
        raise PipelineError("default_top_k deve ser positivo.")
    seen: set[str] = set()
    for item in queries:
        if not isinstance(item, dict):
            raise PipelineError("Cada consulta RAG precisa ser um mapa.")
        query_id = item.get("id")
        query = item.get("query")
        expected_categories = item.get("expected_categories")
        if not query_id or not query:
            raise PipelineError("Consulta RAG sem id/query.")
        if query_id in seen:
            raise PipelineError(f"Consulta RAG duplicada: {query_id}")
        seen.add(str(query_id))
        if not tokenize(str(query)):
            raise PipelineError(f"Consulta RAG sem termos indexáveis: {query_id}")
        if not isinstance(expected_categories, list) or not expected_categories:
            raise PipelineError(f"Consulta RAG sem expected_categories: {query_id}")
        top_k = int(item.get("top_k", default_top_k))
        if top_k <= 0:
            raise PipelineError(f"Consulta RAG {query_id} com top_k inválido.")


def _rank_of_first_category_hit(results: list[dict[str, Any]], expected_categories: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if expected_categories & set(result.get("categories", [])):
            return rank
    return None


def _rank_of_first_doc_hit(results: list[dict[str, Any]], expected_doc_ids: set[str]) -> int | None:
    if not expected_doc_ids:
        return None
    for rank, result in enumerate(results, start=1):
        if result.get("doc_id") in expected_doc_ids:
            return rank
    return None


def _expected_term_hits(results: list[dict[str, Any]], expected_terms: list[str], chunk_texts: dict[str, str]) -> list[str]:
    if not expected_terms:
        return []
    haystack_parts: list[str] = []
    for result in results:
        haystack_parts.extend([
            str(result.get("title", "")),
            str(result.get("section", "")),
            str(chunk_texts.get(str(result.get("chunk_id")), "")),
        ])
    haystack_tokens = set(tokenize(" ".join(haystack_parts)))
    return sorted({term.lower() for term in expected_terms if term.lower() in haystack_tokens})


def evaluate_retrieval(
    *,
    eval_path: Path,
    index_path: Path,
    top_k_override: int | None = None,
) -> dict[str, Any]:
    spec = _read_yaml(eval_path)
    validate_eval_spec(spec)
    index = load_bm25_index(index_path)
    chunk_texts = _load_chunk_texts(index)

    default_top_k = int(spec.get("default_top_k", 5))
    rows: list[dict[str, Any]] = []
    reciprocal_category_ranks: list[float] = []
    reciprocal_doc_ranks: list[float] = []

    for item in spec["queries"]:
        query_id = str(item["id"])
        top_k = int(top_k_override or item.get("top_k", default_top_k))
        expected_categories = {str(cat) for cat in item["expected_categories"]}
        expected_doc_ids = {str(doc_id) for doc_id in item.get("expected_doc_ids", [])}
        require_expected_doc_hit = bool(item.get("require_expected_doc_hit", False))
        expected_terms = [str(term) for term in item.get("expected_terms", [])]
        results = search_bm25(index, str(item["query"]), top_k=top_k)
        found_categories = sorted({cat for result in results for cat in result.get("categories", [])})
        covered_expected_categories = sorted(expected_categories & set(found_categories))
        category_rank = _rank_of_first_category_hit(results, expected_categories)
        doc_rank = _rank_of_first_doc_hit(results, expected_doc_ids)
        term_hits = _expected_term_hits(results, expected_terms, chunk_texts)
        category_hit = category_rank is not None
        doc_hit = not expected_doc_ids or doc_rank is not None
        terms_hit = not expected_terms or bool(term_hits)
        passed = bool(results) and category_hit and terms_hit and (doc_hit or not require_expected_doc_hit)
        if category_rank is not None:
            reciprocal_category_ranks.append(1.0 / category_rank)
        if doc_rank is not None:
            reciprocal_doc_ranks.append(1.0 / doc_rank)
        rows.append(
            {
                "id": query_id,
                "intent": item.get("intent"),
                "query": item["query"],
                "top_k": top_k,
                "passed": passed,
                "result_count": len(results),
                "empty_result": len(results) == 0,
                "expected_categories": sorted(expected_categories),
                "covered_expected_categories": covered_expected_categories,
                "category_hit": category_hit,
                "category_rank": category_rank,
                "expected_doc_ids": sorted(expected_doc_ids),
                "require_expected_doc_hit": require_expected_doc_hit,
                "doc_hit": doc_hit,
                "doc_rank": doc_rank,
                "expected_terms": expected_terms,
                "term_hits": term_hits,
                "terms_hit": terms_hit,
                "top_results": [
                    {
                        "rank": rank,
                        "score": float(result["score"]),
                        "chunk_id": result["chunk_id"],
                        "doc_id": result["doc_id"],
                        "title": result["title"],
                        "section": result["section"],
                        "categories": result.get("categories", []),
                        "url": result.get("url"),
                        "sha256": result.get("sha256"),
                    }
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )

    query_count = len(rows)
    passed_count = sum(1 for row in rows if row["passed"])
    category_hit_count = sum(1 for row in rows if row["category_hit"])
    doc_hit_count = sum(1 for row in rows if row["doc_hit"])
    terms_hit_count = sum(1 for row in rows if row["terms_hit"])
    empty_count = sum(1 for row in rows if row["empty_result"])
    pass_rate = passed_count / query_count
    category_hit_rate = category_hit_count / query_count
    doc_hit_rate = doc_hit_count / query_count
    terms_hit_rate = terms_hit_count / query_count
    acceptance = spec.get("acceptance", {}) or {}
    min_pass_rate = float(acceptance.get("min_pass_rate", 0.0))
    min_category_hit_rate = float(acceptance.get("min_category_hit_rate", 0.0))
    max_empty_results = int(acceptance.get("max_empty_results", query_count))
    require_all_queries_pass = bool(acceptance.get("require_all_queries_pass", False))
    accepted = (
        pass_rate >= min_pass_rate
        and category_hit_rate >= min_category_hit_rate
        and empty_count <= max_empty_results
        and (not require_all_queries_pass or passed_count == query_count)
    )

    return {
        "schema": REPORT_SCHEMA,
        "created_utc": now_utc(),
        "eval_file": str(eval_path),
        "eval_sha256": sha256_of(eval_path),
        "index_file": str(index_path),
        "index_sha256": sha256_of(index_path),
        "dataset_version": spec.get("dataset_version"),
        "corpus_version": spec.get("corpus_version"),
        "index_corpus_version": index.get("corpus_version"),
        "retriever": "bm25",
        "chunk_text_available": bool(chunk_texts),
        "acceptance": {
            "min_pass_rate": min_pass_rate,
            "min_category_hit_rate": min_category_hit_rate,
            "max_empty_results": max_empty_results,
            "require_all_queries_pass": require_all_queries_pass,
            "accepted": accepted,
        },
        "summary": {
            "query_count": query_count,
            "passed_count": passed_count,
            "failed_count": query_count - passed_count,
            "pass_rate": pass_rate,
            "category_hit_count": category_hit_count,
            "category_hit_rate": category_hit_rate,
            "doc_hit_count": doc_hit_count,
            "doc_hit_rate": doc_hit_rate,
            "terms_hit_count": terms_hit_count,
            "terms_hit_rate": terms_hit_rate,
            "empty_result_count": empty_count,
            "category_mrr": sum(reciprocal_category_ranks) / query_count,
            "doc_mrr": sum(reciprocal_doc_ranks) / query_count,
        },
        "queries": rows,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    acceptance = report["acceptance"]
    lines = [
        "# ARGOS RAG retrieval evaluation",
        "",
        f"- Dataset: `{report.get('dataset_version')}`",
        f"- Corpus esperado: `{report.get('corpus_version')}`",
        f"- Corpus do índice: `{report.get('index_corpus_version')}`",
        f"- Retriever: `{report.get('retriever')}`",
        f"- Aceito: `{acceptance['accepted']}`",
        f"- Pass rate: `{summary['pass_rate']:.3f}` ({summary['passed_count']}/{summary['query_count']})",
        f"- Category hit rate: `{summary['category_hit_rate']:.3f}`",
        f"- Doc hit rate: `{summary['doc_hit_rate']:.3f}`",
        f"- Terms hit rate: `{summary['terms_hit_rate']:.3f}`",
        f"- Empty results: `{summary['empty_result_count']}`",
        "",
        "| Query | Pass | Cat@k | Doc@k | Termos | Top resultado |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["queries"]:
        top = row["top_results"][0] if row["top_results"] else {}
        lines.append(
            "| {id} | {passed} | {cat} | {doc} | {terms} | {top_doc} / {top_section} |".format(
                id=row["id"],
                passed="✅" if row["passed"] else "❌",
                cat=row["category_rank"] or "-",
                doc=row["doc_rank"] or "-",
                terms=", ".join(row["term_hits"]) or "-",
                top_doc=top.get("doc_id", "-"),
                top_section=str(top.get("section", "-")).replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia a recuperação BM25 do RAG ARGOS.")
    parser.add_argument("--eval", default="docs/rag/retrieval_eval_v1.yaml", help="YAML de perguntas-semente.")
    parser.add_argument("--index", default="rag/index/liver_mri_rag_v1/bm25_index.json", help="Índice BM25 JSON.")
    parser.add_argument("--out", default="artifacts/rag_eval/liver_mri_rag_v1", help="Pasta de relatório.")
    parser.add_argument("--top-k", type=int, default=None, help="Sobrescreve top_k de todas as consultas.")
    parser.add_argument("--no-fail", action="store_true", help="Sempre retorna zero mesmo se critérios falharem.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    try:
        if args.top_k is not None and args.top_k <= 0:
            raise PipelineError("--top-k deve ser positivo.")
        report = evaluate_retrieval(
            eval_path=Path(args.eval),
            index_path=Path(args.index),
            top_k_override=args.top_k,
        )
        _write_json_atomic(out_dir / "retrieval_eval_report.json", report)
        _write_text_atomic(out_dir / "retrieval_eval_report.md", render_markdown_report(report))
    except PipelineError as exc:
        print(f"[ABORTADO] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "complete",
        "accepted": report["acceptance"]["accepted"],
        "query_count": report["summary"]["query_count"],
        "passed_count": report["summary"]["passed_count"],
        "pass_rate": report["summary"]["pass_rate"],
        "category_hit_rate": report["summary"]["category_hit_rate"],
        "empty_result_count": report["summary"]["empty_result_count"],
        "out": str(out_dir),
    }, indent=2, ensure_ascii=False))
    if args.no_fail:
        return 0
    return 0 if report["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
