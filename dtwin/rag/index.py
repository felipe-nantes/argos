"""Índice BM25 local e auditável para a F0 do RAG ARGOS.

Este módulo cria um índice lexical puro sobre os chunks já normalizados. Ele é o
primeiro degrau antes de embeddings/MedCPT: barato, determinístico e suficiente
para validar se o corpus está consultável.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dtwin.core import PipelineError, now_utc, sha256_of

SCHEMA_VERSION = "argos-rag-bm25-index-v1"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", re.UNICODE)


@dataclass(frozen=True)
class IndexedChunk:
    ordinal: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    sha256: str
    categories: tuple[str, ...]
    priority: str
    url: str
    pmcid: str | None = None
    doi: str | None = None


def tokenize(text: str) -> list[str]:
    """Tokenização lexical simples e estável para BM25."""

    return [token.lower() for token in _TOKEN_RE.findall(text or "") if token.strip()]


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler JSON RAG {path}: {exc}") from exc


def _load_corpus_manifest(corpus_dir: Path) -> dict[str, Any]:
    path = corpus_dir / "manifest.json"
    if not path.is_file():
        raise PipelineError(f"Manifesto do corpus RAG ausente: {path}")
    data = _read_json(path)
    if data.get("schema") != "argos-rag-corpus-v1":
        raise PipelineError(f"Schema de corpus RAG inválido: {data.get('schema')}")
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise PipelineError("Manifesto do corpus RAG não contém chunks.")
    return data


def _load_chunks(corpus_dir: Path, corpus_manifest: dict[str, Any]) -> list[IndexedChunk]:
    chunks: list[IndexedChunk] = []
    seen: set[str] = set()
    for ordinal, entry in enumerate(corpus_manifest["chunks"]):
        rel = entry.get("path")
        chunk_id = entry.get("chunk_id")
        if not rel or not chunk_id:
            raise PipelineError("Entrada de chunk RAG sem path/chunk_id.")
        path = (corpus_dir / rel).resolve()
        try:
            path.relative_to(corpus_dir.resolve())
        except ValueError as exc:
            raise PipelineError(f"Chunk RAG fora do corpus_dir: {path}") from exc
        if not path.is_file():
            raise PipelineError(f"Arquivo de chunk RAG ausente: {path}")
        chunk = _read_json(path)
        if chunk.get("chunk_id") != chunk_id:
            raise PipelineError(f"Chunk RAG {path} não corresponde ao manifesto.")
        if chunk_id in seen:
            raise PipelineError(f"Chunk RAG duplicado no índice: {chunk_id}")
        seen.add(chunk_id)
        if chunk.get("sha256") != entry.get("sha256"):
            raise PipelineError(f"Hash de chunk RAG diverge do manifesto: {chunk_id}")
        required = ("doc_id", "title", "section", "text", "sha256", "categories", "priority", "url")
        missing = [key for key in required if chunk.get(key) in (None, "", [])]
        if missing:
            raise PipelineError(f"Chunk RAG {chunk_id} sem campos para indexação: {missing}")
        chunks.append(
            IndexedChunk(
                ordinal=ordinal,
                chunk_id=str(chunk["chunk_id"]),
                doc_id=str(chunk["doc_id"]),
                title=str(chunk["title"]),
                section=str(chunk["section"]),
                text=str(chunk["text"]),
                sha256=str(chunk["sha256"]),
                categories=tuple(str(item) for item in chunk["categories"]),
                priority=str(chunk["priority"]),
                url=str(chunk["url"]),
                pmcid=chunk.get("pmcid"),
                doi=chunk.get("doi"),
            )
        )
    return chunks


def _bm25_idf(total_docs: int, document_frequency: int) -> float:
    return math.log(1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))


def build_bm25_index(
    *,
    corpus_dir: Path,
    out_dir: Path,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> dict[str, Any]:
    corpus_dir = corpus_dir.resolve()
    out_dir = out_dir.resolve()
    if k1 <= 0:
        raise PipelineError("BM25 k1 deve ser positivo.")
    if not 0 <= b <= 1:
        raise PipelineError("BM25 b deve estar entre 0 e 1.")

    corpus_manifest = _load_corpus_manifest(corpus_dir)
    chunks = _load_chunks(corpus_dir, corpus_manifest)
    if not chunks:
        raise PipelineError("Nenhum chunk RAG para indexar.")

    term_doc_freq: dict[str, int] = defaultdict(int)
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_lengths: list[int] = []
    docs: list[dict[str, Any]] = []

    for chunk in chunks:
        terms = tokenize(chunk.text)
        if not terms:
            raise PipelineError(f"Chunk RAG sem termos indexáveis: {chunk.chunk_id}")
        counts = Counter(terms)
        doc_lengths.append(len(terms))
        for term, tf in sorted(counts.items()):
            postings[term].append((chunk.ordinal, int(tf)))
        for term in counts:
            term_doc_freq[term] += 1
        docs.append(
            {
                "ordinal": chunk.ordinal,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.section,
                "sha256": chunk.sha256,
                "categories": list(chunk.categories),
                "priority": chunk.priority,
                "url": chunk.url,
                "pmcid": chunk.pmcid,
                "doi": chunk.doi,
                "token_count": len(terms),
            }
        )

    total_docs = len(chunks)
    avg_doc_len = sum(doc_lengths) / total_docs
    index_terms = {
        term: {
            "df": int(term_doc_freq[term]),
            "idf": _bm25_idf(total_docs, term_doc_freq[term]),
            "postings": [[doc_ord, tf] for doc_ord, tf in postings[term]],
        }
        for term in sorted(postings)
    }

    index = {
        "schema": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "corpus_version": corpus_manifest.get("corpus_version"),
        "corpus_dir": str(corpus_dir),
        "corpus_manifest_sha256": sha256_of(corpus_dir / "manifest.json"),
        "chunk_count": total_docs,
        "vocabulary_size": len(index_terms),
        "avg_doc_len": avg_doc_len,
        "k1": float(k1),
        "b": float(b),
        "documents": docs,
        "terms": index_terms,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "bm25_index.json"
    _write_json_atomic(index_path, index)
    manifest = {
        "schema": "argos-rag-index-manifest-v1",
        "created_utc": now_utc(),
        "corpus_version": corpus_manifest.get("corpus_version"),
        "index_type": "bm25",
        "index_file": "bm25_index.json",
        "index_sha256": sha256_of(index_path),
        "corpus_dir": str(corpus_dir),
        "corpus_manifest_sha256": index["corpus_manifest_sha256"],
        "chunk_count": total_docs,
        "vocabulary_size": len(index_terms),
        "k1": float(k1),
        "b": float(b),
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def load_bm25_index(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if data.get("schema") != SCHEMA_VERSION:
        raise PipelineError(f"Schema de índice BM25 RAG inválido: {data.get('schema')}")
    if not data.get("documents") or not data.get("terms"):
        raise PipelineError("Índice BM25 RAG incompleto.")
    return data


def search_bm25(
    index: dict[str, Any],
    query: str,
    *,
    top_k: int = 5,
    categories: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Busca BM25 local para smoke tests e avaliação lexical inicial."""

    if top_k <= 0:
        raise PipelineError("top_k da busca BM25 deve ser positivo.")
    terms = tokenize(query)
    if not terms:
        raise PipelineError("Consulta BM25 vazia.")
    docs = index["documents"]
    avg_doc_len = float(index["avg_doc_len"])
    k1 = float(index["k1"])
    b = float(index["b"])
    allowed_categories = set(categories or [])
    scores: dict[int, float] = defaultdict(float)

    for term in terms:
        term_info = index["terms"].get(term)
        if not term_info:
            continue
        idf = float(term_info["idf"])
        for doc_ord, tf in term_info["postings"]:
            doc = docs[int(doc_ord)]
            if allowed_categories and not (allowed_categories & set(doc.get("categories", []))):
                continue
            doc_len = float(doc["token_count"])
            denom = float(tf) + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            scores[int(doc_ord)] += idf * ((float(tf) * (k1 + 1.0)) / denom)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], docs[item[0]]["chunk_id"]))
    results: list[dict[str, Any]] = []
    for doc_ord, score in ranked[:top_k]:
        doc = dict(docs[doc_ord])
        doc["score"] = score
        results.append(doc)
    return results
