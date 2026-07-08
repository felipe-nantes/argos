import json
from pathlib import Path

import pytest
import yaml

from dtwin.core import PipelineError
from dtwin.rag.index import build_bm25_index
from tools.eval_rag_retrieval import (
    evaluate_retrieval,
    main as eval_main,
    render_markdown_report,
    validate_eval_spec,
)


def _chunk(chunk_id, text, *, categories, doc_id=None, title="Synthetic source", section="Findings"):
    doc = doc_id or chunk_id.split("::")[0]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc,
        "title": title,
        "section": section,
        "text": text,
        "sha256": f"sha-{chunk_id}",
        "categories": categories,
        "priority": "core",
        "url": f"https://example.org/{doc}",
        "pmcid": f"PMC_{doc}",
        "doi": f"10.0000/{doc}",
    }


def _write_corpus(tmp_path: Path, chunks: list[dict]) -> Path:
    corpus_dir = tmp_path / "corpus"
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
            "corpus_version": "eval_test_v1",
            "chunks": manifest_chunks,
        }),
        encoding="utf-8",
    )
    return corpus_dir


def _write_eval(tmp_path: Path, queries: list[dict], *, acceptance=None) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump({
            "schema": "argos-rag-retrieval-eval-v1",
            "dataset_version": "eval_test",
            "corpus_version": "eval_test_v1",
            "default_top_k": 3,
            "acceptance": acceptance or {
                "min_pass_rate": 1.0,
                "min_category_hit_rate": 1.0,
                "max_empty_results": 0,
            },
            "queries": queries,
        }, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _build_index(tmp_path: Path, chunks: list[dict]) -> Path:
    corpus_dir = _write_corpus(tmp_path, chunks)
    index_dir = tmp_path / "index"
    build_bm25_index(corpus_dir=corpus_dir, out_dir=index_dir)
    return index_dir / "bm25_index.json"


def test_retrieval_eval_passes_with_category_doc_and_term_hits(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk(
            "hcc::c1",
            "HCC arterial phase hyperenhancement washout enhancing capsule LI-RADS.",
            categories=["hcc", "li_rads"],
            title="HCC MRI",
        )
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "arterial washout capsule HCC",
        "expected_categories": ["hcc"],
        "expected_doc_ids": ["hcc"],
        "expected_terms": ["washout"],
    }])

    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path)

    assert report["acceptance"]["accepted"] is True
    assert report["summary"]["passed_count"] == 1
    assert report["queries"][0]["category_rank"] == 1
    assert report["queries"][0]["doc_rank"] == 1


def test_retrieval_eval_fails_when_expected_category_is_absent(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("hem::c1", "hemangioma very high T2 signal", categories=["hemangioma"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "hemangioma T2",
        "expected_categories": ["hcc"],
    }])

    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path)

    assert report["acceptance"]["accepted"] is False
    assert report["queries"][0]["category_hit"] is False


def test_retrieval_eval_fails_when_expected_doc_is_absent(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("doc_a::c1", "arterial washout capsule", categories=["hcc"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "arterial washout",
        "expected_categories": ["hcc"],
        "expected_doc_ids": ["doc_b"],
        "require_expected_doc_hit": True,
    }])

    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path)

    assert report["queries"][0]["category_hit"] is True
    assert report["queries"][0]["doc_hit"] is False
    assert report["acceptance"]["accepted"] is False


def test_retrieval_eval_treats_expected_doc_as_auxiliary_by_default(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("doc_a::c1", "arterial washout capsule", categories=["hcc"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "arterial washout",
        "expected_categories": ["hcc"],
        "expected_doc_ids": ["doc_b"],
    }])

    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path)

    assert report["queries"][0]["doc_hit"] is False
    assert report["queries"][0]["require_expected_doc_hit"] is False
    assert report["acceptance"]["accepted"] is True


def test_retrieval_eval_fails_when_expected_terms_are_absent(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("hcc::c1", "arterial washout capsule", categories=["hcc"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "arterial washout",
        "expected_categories": ["hcc"],
        "expected_terms": ["hepatobiliary"],
    }])

    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path)

    assert report["queries"][0]["terms_hit"] is False
    assert report["acceptance"]["accepted"] is False


def test_retrieval_eval_records_empty_results(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("hcc::c1", "arterial washout capsule", categories=["hcc"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "unknown",
        "query": "xylophone unrelated",
        "expected_categories": ["hcc"],
    }])

    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path)

    assert report["summary"]["empty_result_count"] == 1
    assert report["queries"][0]["empty_result"] is True


def test_retrieval_eval_accepts_top_k_override(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("doc_a::c1", "arterial washout", categories=["hcc"]),
        _chunk("doc_b::c1", "arterial washout", categories=["hcc"]),
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "arterial",
        "expected_categories": ["hcc"],
    }])

    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path, top_k_override=1)

    assert report["queries"][0]["top_k"] == 1
    assert len(report["queries"][0]["top_results"]) == 1


def test_render_markdown_report_contains_summary_and_rows(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("hcc::c1", "arterial washout capsule", categories=["hcc"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "arterial washout",
        "expected_categories": ["hcc"],
    }])
    report = evaluate_retrieval(eval_path=eval_path, index_path=index_path)

    markdown = render_markdown_report(report)

    assert "# ARGOS RAG retrieval evaluation" in markdown
    assert "hcc" in markdown
    assert "Pass rate" in markdown


def test_eval_cli_writes_json_and_markdown_reports(tmp_path, capsys):
    index_path = _build_index(tmp_path, [
        _chunk("hcc::c1", "arterial washout capsule", categories=["hcc"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "arterial washout",
        "expected_categories": ["hcc"],
    }])
    out_dir = tmp_path / "report"

    code = eval_main(["--eval", str(eval_path), "--index", str(index_path), "--out", str(out_dir)])
    captured = capsys.readouterr()

    assert code == 0
    assert json.loads(captured.out)["accepted"] is True
    assert (out_dir / "retrieval_eval_report.json").is_file()
    assert (out_dir / "retrieval_eval_report.md").is_file()


def test_eval_cli_returns_nonzero_when_acceptance_fails(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("hem::c1", "hemangioma T2", categories=["hemangioma"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "hemangioma T2",
        "expected_categories": ["hcc"],
    }])

    assert eval_main(["--eval", str(eval_path), "--index", str(index_path), "--out", str(tmp_path / "out")]) == 1


def test_eval_cli_no_fail_returns_zero_when_acceptance_fails(tmp_path):
    index_path = _build_index(tmp_path, [
        _chunk("hem::c1", "hemangioma T2", categories=["hemangioma"])
    ])
    eval_path = _write_eval(tmp_path, [{
        "id": "hcc",
        "query": "hemangioma T2",
        "expected_categories": ["hcc"],
    }])

    assert eval_main([
        "--eval", str(eval_path),
        "--index", str(index_path),
        "--out", str(tmp_path / "out"),
        "--no-fail",
    ]) == 0


def test_eval_cli_rejects_invalid_top_k(tmp_path):
    assert eval_main(["--top-k", "0"]) == 1


def test_validate_eval_spec_rejects_wrong_schema():
    with pytest.raises(PipelineError, match="Schema"):
        validate_eval_spec({"schema": "wrong", "queries": []})


def test_validate_eval_spec_rejects_duplicate_query_ids():
    spec = {
        "schema": "argos-rag-retrieval-eval-v1",
        "default_top_k": 5,
        "queries": [
            {"id": "dup", "query": "arterial", "expected_categories": ["hcc"]},
            {"id": "dup", "query": "washout", "expected_categories": ["hcc"]},
        ],
    }
    with pytest.raises(PipelineError, match="duplicada"):
        validate_eval_spec(spec)


def test_validate_eval_spec_rejects_query_without_categories():
    spec = {
        "schema": "argos-rag-retrieval-eval-v1",
        "queries": [{"id": "q", "query": "arterial"}],
    }
    with pytest.raises(PipelineError, match="expected_categories"):
        validate_eval_spec(spec)


def test_validate_eval_spec_rejects_unindexable_query():
    spec = {
        "schema": "argos-rag-retrieval-eval-v1",
        "queries": [{"id": "q", "query": "---", "expected_categories": ["hcc"]}],
    }
    with pytest.raises(PipelineError, match="sem termos"):
        validate_eval_spec(spec)


def test_real_retrieval_eval_references_existing_manifest_categories_and_docs():
    corpus_manifest = yaml.safe_load(Path("docs/rag/corpus_manifest_v1.yaml").read_text("utf-8"))
    eval_spec = yaml.safe_load(Path("docs/rag/retrieval_eval_v1.yaml").read_text("utf-8"))
    categories = {cat for article in corpus_manifest["articles"] for cat in article["categories"]}
    doc_ids = {article["id"] for article in corpus_manifest["articles"]}

    missing_categories = {
        cat for query in eval_spec["queries"] for cat in query.get("expected_categories", []) if cat not in categories
    }
    missing_doc_ids = {
        doc_id for query in eval_spec["queries"] for doc_id in query.get("expected_doc_ids", []) if doc_id not in doc_ids
    }

    assert missing_categories == set()
    assert missing_doc_ids == set()
