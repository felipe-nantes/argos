import json
from pathlib import Path

import pytest
import yaml

from dtwin.core import PipelineError
from dtwin.rag.grounding import append_rag_to_prompt, build_rag_prompt_addendum
from dtwin.rag.index import build_bm25_index
from dtwin.rag.retriever import build_rag_context, persist_rag_context


def _chunk(chunk_id, text, *, categories):
    doc_id = chunk_id.split("::")[0]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": f"Title {doc_id}",
        "section": "Findings",
        "text": text,
        "sha256": f"sha-{chunk_id}",
        "categories": categories,
        "priority": "core",
        "url": f"https://example.org/{doc_id}",
        "pmcid": f"PMC_{doc_id}",
        "doi": f"10.0000/{doc_id}",
    }


def _write_corpus(repo_root: Path, chunks: list[dict]) -> Path:
    corpus_dir = repo_root / "rag" / "corpus" / "test"
    (corpus_dir / "chunks").mkdir(parents=True)
    manifest_chunks = []
    for chunk in chunks:
        rel = f"chunks/{chunk['chunk_id'].replace('::', '__')}.json"
        (corpus_dir / rel).write_text(json.dumps(chunk), encoding="utf-8")
        manifest_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "path": rel,
            "sha256": chunk["sha256"],
        })
    (corpus_dir / "manifest.json").write_text(
        json.dumps({
            "schema": "argos-rag-corpus-v1",
            "corpus_version": "sidecar_test_v1",
            "chunks": manifest_chunks,
        }),
        encoding="utf-8",
    )
    return corpus_dir


def _write_eval(repo_root: Path) -> Path:
    path = repo_root / "docs" / "rag" / "retrieval_eval_v1.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump({
            "schema": "argos-rag-retrieval-eval-v1",
            "queries": [
                {
                    "id": "hcc_aphe_washout_capsule",
                    "query": "arterial washout capsule HCC",
                    "intent": "HCC criteria",
                    "expected_categories": ["hcc"],
                },
                {
                    "id": "hemangioma_t2_fill_in",
                    "query": "hemangioma T2 peripheral nodular enhancement",
                    "intent": "Hemangioma criteria",
                    "expected_categories": ["hemangioma"],
                },
            ],
        }, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _build_repo_with_index(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    corpus_dir = _write_corpus(repo_root, [
        _chunk("hcc::c1", "HCC arterial phase hyperenhancement with washout and capsule.", categories=["hcc", "li_rads"]),
        _chunk("hem::c1", "Hemangioma has very high T2 signal and peripheral nodular enhancement.", categories=["hemangioma"]),
    ])
    build_bm25_index(corpus_dir=corpus_dir, out_dir=repo_root / "rag" / "index" / "test")
    _write_eval(repo_root)
    return repo_root


def _rag_config():
    return {
        "rag": {
            "enabled": True,
            "index_path": "rag/index/test/bm25_index.json",
            "retrieval_eval": "docs/rag/retrieval_eval_v1.yaml",
            "top_k": 1,
            "max_sources": 4,
            "max_chunk_chars": 120,
            "min_score": 0.0,
            "query_ids": ["hcc_aphe_washout_capsule", "hemangioma_t2_fill_in"],
        }
    }


def test_build_rag_context_recovers_sources_with_audit(tmp_path):
    repo_root = _build_repo_with_index(tmp_path)

    context = build_rag_context(config=_rag_config(), repo_root=repo_root)

    assert context["enabled"] is True
    assert context["retriever"] == "bm25"
    assert context["source_count"] == 2
    assert context["query_count"] == 2
    assert len(context["index_sha256"]) == 64
    assert len(context["context_sha256"]) == 64
    assert [source["source_id"] for source in context["sources"]] == ["S1", "S2"]
    assert all(len(source["text"]) <= 121 for source in context["sources"])


def test_build_rag_context_disabled_is_explicit(tmp_path):
    context = build_rag_context(config={"rag": {"enabled": False}}, repo_root=tmp_path)
    assert context["enabled"] is False
    assert context["schema"] == "argos-rag-context-v1"


def test_build_rag_context_fails_closed_when_index_is_missing(tmp_path):
    repo_root = _build_repo_with_index(tmp_path)
    config = _rag_config()
    config["rag"]["index_path"] = "rag/index/missing/bm25_index.json"
    with pytest.raises(PipelineError, match="índice não encontrado"):
        build_rag_context(config=config, repo_root=repo_root)


def test_build_rag_context_rejects_path_traversal(tmp_path):
    repo_root = _build_repo_with_index(tmp_path)
    config = _rag_config()
    config["rag"]["index_path"] = "../outside.json"
    with pytest.raises(PipelineError, match="fora do repositório"):
        build_rag_context(config=config, repo_root=repo_root)


def test_rag_prompt_addendum_contains_safety_rules(tmp_path):
    repo_root = _build_repo_with_index(tmp_path)
    context = build_rag_context(config=_rag_config(), repo_root=repo_root)

    addendum = build_rag_prompt_addendum(context)

    assert "CONTEXTO RAG TEXTUAL DE APOIO" in addendum
    assert "[S1]" in addendum
    assert "não substitui a análise da imagem" in addendum
    assert "não visual" in addendum.lower()


def test_append_rag_to_prompt_hashes_and_limits(tmp_path):
    repo_root = _build_repo_with_index(tmp_path)
    context = build_rag_context(config=_rag_config(), repo_root=repo_root)
    base = "modo de pesquisa POSITIVA NEGATIVA INCONCLUSIVA revisão humana obrigatória não é diagnóstico não é laudo médico"

    final, audit = append_rag_to_prompt(base, context, max_prompt_chars=6000)

    assert final.startswith(base)
    assert "CONTEXTO RAG TEXTUAL" in final
    assert audit["enabled"] is True
    assert len(audit["final_prompt_sha256"]) == 64
    assert audit["prompt_chars"] == len(final)


def test_append_rag_to_prompt_fails_when_prompt_exceeds_limit(tmp_path):
    repo_root = _build_repo_with_index(tmp_path)
    context = build_rag_context(config=_rag_config(), repo_root=repo_root)
    with pytest.raises(PipelineError, match="excede max_prompt_chars"):
        append_rag_to_prompt("base", context, max_prompt_chars=50)


def test_persist_rag_context_writes_json_atomically(tmp_path):
    path = tmp_path / "rag_context.json"
    persist_rag_context(path, {"enabled": True, "sources": []})
    assert json.loads(path.read_text("utf-8"))["enabled"] is True

