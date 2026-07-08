"""Retriever RAG local para apoiar o prompt MedGemma sem alterar a decisão visual."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from dtwin.core import PipelineError, now_utc, sha256_of
from dtwin.rag.index import load_bm25_index, search_bm25


RAG_CONTEXT_SCHEMA = "argos-rag-context-v1"
DEFAULT_QUERY_IDS = [
    "hcc_aphe_washout_capsule",
    "hemangioma_t2_fill_in",
    "fnh_central_scar_hepatobiliary",
    "hepatic_adenoma_subtypes",
    "metastases_rim_diffusion",
    "cholangiocarcinoma_delayed_enhancement",
    "dwi_adc_benign_malignant",
    "dynamic_contrast_phases",
    "arterial_phase_motion_artifact",
    "pseudolesion_perfusion_pitfall",
]


def _safe_repo_path(value: str | Path, *, repo_root: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise PipelineError(f"{label} deve ser caminho relativo versionado, não absoluto: {path}")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise PipelineError(f"{label} aponta para fora do repositório: {path}") from exc
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler YAML RAG {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError(f"YAML RAG deve ser objeto: {path}")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler JSON RAG {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError(f"JSON RAG deve ser objeto: {path}")
    return data


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _load_chunk_texts(index: dict[str, Any]) -> dict[str, str]:
    corpus_dir = Path(str(index.get("corpus_dir") or ""))
    manifest_path = corpus_dir / "manifest.json"
    if not corpus_dir.is_dir() or not manifest_path.is_file():
        raise PipelineError("RAG habilitado, mas corpus_dir do índice não está acessível.")
    manifest = _read_json(manifest_path)
    texts: dict[str, str] = {}
    for entry in manifest.get("chunks", []):
        chunk_id = entry.get("chunk_id")
        rel = entry.get("path")
        if not chunk_id or not rel:
            continue
        path = (corpus_dir / rel).resolve()
        try:
            path.relative_to(corpus_dir.resolve())
        except ValueError as exc:
            raise PipelineError(f"Chunk RAG fora do corpus_dir durante retrieval: {path}") from exc
        if not path.is_file():
            raise PipelineError(f"Chunk RAG ausente durante retrieval: {path}")
        chunk = _read_json(path)
        if chunk.get("chunk_id") != chunk_id:
            raise PipelineError(f"Chunk RAG não corresponde ao manifesto: {chunk_id}")
        texts[str(chunk_id)] = str(chunk.get("text", ""))
    return texts


def _load_eval_queries(path: Path, query_ids: list[str]) -> list[dict[str, Any]]:
    spec = _read_yaml(path)
    if spec.get("schema") != "argos-rag-retrieval-eval-v1":
        raise PipelineError(f"Schema de retrieval_eval RAG inválido: {spec.get('schema')}")
    by_id = {str(item.get("id")): item for item in spec.get("queries", []) if isinstance(item, dict)}
    selected_ids = query_ids or DEFAULT_QUERY_IDS
    missing = [query_id for query_id in selected_ids if query_id not in by_id]
    if missing:
        raise PipelineError(f"Queries RAG ausentes em retrieval_eval: {missing}")
    return [by_id[query_id] for query_id in selected_ids]


def _clip_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def build_rag_context(
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Recupera contexto textual auditável para o prompt MedGemma.

    Falha fechado quando `rag.enabled=true`: índice/corpus/queries ausentes abortam
    antes da inferência, para evitar benchmark misturando casos com e sem RAG.
    """

    rag_cfg = config.get("rag") or {}
    if not isinstance(rag_cfg, dict):
        raise PipelineError("Bloco rag da configuração deve ser um objeto.")
    if rag_cfg.get("enabled") is not True:
        return {"schema": RAG_CONTEXT_SCHEMA, "enabled": False, "created_utc": now_utc()}

    index_path = _safe_repo_path(rag_cfg.get("index_path", ""), repo_root=repo_root, label="rag.index_path")
    eval_path = _safe_repo_path(
        rag_cfg.get("retrieval_eval", ""), repo_root=repo_root, label="rag.retrieval_eval"
    )
    if not index_path.is_file():
        raise PipelineError(f"RAG habilitado, mas índice não encontrado: {index_path}")
    if not eval_path.is_file():
        raise PipelineError(f"RAG habilitado, mas retrieval_eval não encontrado: {eval_path}")

    top_k = int(rag_cfg.get("top_k", 2))
    max_sources = int(rag_cfg.get("max_sources", 12))
    max_chunk_chars = int(rag_cfg.get("max_chunk_chars", 700))
    min_score = float(rag_cfg.get("min_score", 0.0))
    if top_k <= 0 or max_sources <= 0 or max_chunk_chars <= 0:
        raise PipelineError("rag.top_k, rag.max_sources e rag.max_chunk_chars devem ser positivos.")
    query_ids = [str(item) for item in rag_cfg.get("query_ids", [])]
    queries = _load_eval_queries(eval_path, query_ids)
    index = load_bm25_index(index_path)
    chunk_texts = _load_chunk_texts(index)

    sources_by_chunk: dict[str, dict[str, Any]] = {}
    query_audit: list[dict[str, Any]] = []
    for query in queries:
        results = search_bm25(index, str(query["query"]), top_k=top_k)
        kept_results = []
        for result in results:
            if float(result.get("score", 0.0)) < min_score:
                continue
            chunk_id = str(result["chunk_id"])
            kept_results.append({
                "chunk_id": chunk_id,
                "doc_id": result["doc_id"],
                "score": round(float(result["score"]), 6),
            })
            if chunk_id not in sources_by_chunk and len(sources_by_chunk) < max_sources:
                source_number = len(sources_by_chunk) + 1
                sources_by_chunk[chunk_id] = {
                    "source_id": f"S{source_number}",
                    "chunk_id": chunk_id,
                    "doc_id": result["doc_id"],
                    "title": result["title"],
                    "section": result["section"],
                    "categories": result.get("categories", []),
                    "score": round(float(result["score"]), 6),
                    "url": result.get("url"),
                    "sha256": result.get("sha256"),
                    "text": _clip_text(chunk_texts.get(chunk_id, ""), max_chunk_chars),
                }
        query_audit.append({
            "id": query["id"],
            "query": query["query"],
            "intent": query.get("intent"),
            "top_k": top_k,
            "result_count": len(results),
            "kept_results": kept_results,
        })

    sources = list(sources_by_chunk.values())
    if not sources:
        raise PipelineError("RAG habilitado, mas nenhuma fonte foi recuperada.")
    context_payload = {
        "queries": query_audit,
        "sources": sources,
    }
    return {
        "schema": RAG_CONTEXT_SCHEMA,
        "enabled": True,
        "created_utc": now_utc(),
        "retriever": "bm25",
        "corpus_version": index.get("corpus_version"),
        "index_path": str(Path(str(rag_cfg["index_path"]))),
        "index_sha256": sha256_of(index_path),
        "retrieval_eval": str(Path(str(rag_cfg["retrieval_eval"]))),
        "retrieval_eval_sha256": sha256_of(eval_path),
        "top_k": top_k,
        "max_sources": max_sources,
        "max_chunk_chars": max_chunk_chars,
        "min_score": min_score,
        "source_count": len(sources),
        "query_count": len(query_audit),
        "queries": query_audit,
        "sources": sources,
        "context_sha256": _sha256_json(context_payload),
    }


def persist_rag_context(path: Path, context: dict[str, Any]) -> None:
    _write_json_atomic(path, context)


def _sha256_json(data: Any) -> str:
    import hashlib

    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

