import json
from pathlib import Path

import pytest

from dtwin.core import PipelineError
from dtwin.rag.index import build_bm25_index, load_bm25_index, search_bm25, tokenize
from tools.build_rag_index import main as build_index_main


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str | None = None,
    title: str = "Synthetic liver MRI source",
    section: str = "Findings",
    categories: list[str] | None = None,
    priority: str = "core",
    sha256: str | None = None,
) -> dict:
    doc = doc_id or chunk_id.split("::")[0]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc,
        "title": title,
        "section": section,
        "text": text,
        "sha256": sha256 or f"sha-{chunk_id}",
        "categories": categories or ["hcc", "liver_mri"],
        "priority": priority,
        "url": f"https://example.org/{doc}",
        "pmcid": f"PMC_{doc}",
        "doi": f"10.0000/{doc}",
    }


def _write_corpus(tmp_path: Path, chunks: list[dict], *, manifest_extra: dict | None = None) -> Path:
    corpus_dir = tmp_path / "corpus"
    chunks_dir = corpus_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    manifest_chunks = []
    for chunk in chunks:
        rel = f"chunks/{chunk['chunk_id'].replace('::', '__')}.json"
        (corpus_dir / rel).write_text(json.dumps(chunk), encoding="utf-8")
        manifest_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "path": rel,
                "sha256": chunk["sha256"],
            }
        )
    manifest = {
        "schema": "argos-rag-corpus-v1",
        "corpus_version": "hardening_v1",
        "chunks": manifest_chunks,
        "chunk_count": len(manifest_chunks),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus_dir


def _build_index(tmp_path: Path, chunks: list[dict]) -> dict:
    corpus_dir = _write_corpus(tmp_path, chunks)
    index_dir = tmp_path / "index"
    build_bm25_index(corpus_dir=corpus_dir, out_dir=index_dir)
    return load_bm25_index(index_dir / "bm25_index.json")


def test_rag_hardening_01_tokenize_empty_text_returns_empty_list():
    assert tokenize("") == []


def test_rag_hardening_02_tokenize_keeps_hyphenated_terms_lowercase():
    assert tokenize("T2-Hyperintense LI-RADS") == ["t2-hyperintense", "li-rads"]


def test_rag_hardening_03_tokenize_unicode_accents_are_stable():
    assert tokenize("difusão hepática cápsula") == ["difusão", "hepática", "cápsula"]


def test_rag_hardening_04_build_rejects_non_positive_k1(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    with pytest.raises(PipelineError, match="k1"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index", k1=0)


def test_rag_hardening_05_build_rejects_negative_b(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    with pytest.raises(PipelineError, match="entre 0 e 1"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index", b=-0.1)


def test_rag_hardening_06_build_rejects_b_above_one(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    with pytest.raises(PipelineError, match="entre 0 e 1"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index", b=1.1)


def test_rag_hardening_07_build_rejects_missing_corpus_manifest(tmp_path):
    with pytest.raises(PipelineError, match="Manifesto"):
        build_bm25_index(corpus_dir=tmp_path / "missing", out_dir=tmp_path / "index")


def test_rag_hardening_08_build_rejects_invalid_corpus_schema(tmp_path):
    corpus_dir = _write_corpus(
        tmp_path,
        [_chunk("doc::c1", "arterial washout")],
        manifest_extra={"schema": "wrong-schema"},
    )
    with pytest.raises(PipelineError, match="Schema"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_09_build_rejects_empty_manifest_chunks(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"schema": "argos-rag-corpus-v1", "chunks": []}),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="chunks"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_10_build_rejects_manifest_entry_without_path(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    manifest["chunks"][0].pop("path")
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PipelineError, match="path/chunk_id"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_11_build_rejects_manifest_entry_without_chunk_id(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    manifest["chunks"][0].pop("chunk_id")
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PipelineError, match="path/chunk_id"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_12_build_rejects_chunk_path_traversal(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    manifest["chunks"][0]["path"] = "../outside.json"
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PipelineError, match="fora do corpus_dir"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_13_build_rejects_chunk_id_mismatch(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    chunk_path = corpus_dir / "chunks" / "doc__c1.json"
    chunk = json.loads(chunk_path.read_text("utf-8"))
    chunk["chunk_id"] = "doc::different"
    chunk_path.write_text(json.dumps(chunk), encoding="utf-8")
    with pytest.raises(PipelineError, match="não corresponde"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_14_build_rejects_duplicate_chunk_ids(tmp_path):
    corpus_dir = _write_corpus(
        tmp_path,
        [
            _chunk("doc::c1", "arterial washout"),
            _chunk("doc::c2", "capsule threshold growth"),
        ],
    )
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    second_path = corpus_dir / manifest["chunks"][1]["path"]
    second = json.loads(second_path.read_text("utf-8"))
    second["chunk_id"] = manifest["chunks"][0]["chunk_id"]
    second_path.write_text(json.dumps(second), encoding="utf-8")
    manifest["chunks"][1]["chunk_id"] = manifest["chunks"][0]["chunk_id"]
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PipelineError, match="duplicado"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


@pytest.mark.parametrize(
    "field",
    ["doc_id", "title", "section", "text", "sha256", "categories", "priority", "url"],
)
def test_rag_hardening_15_to_22_build_rejects_missing_required_chunk_fields(tmp_path, field):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout")])
    chunk_path = corpus_dir / "chunks" / "doc__c1.json"
    chunk = json.loads(chunk_path.read_text("utf-8"))
    chunk[field] = [] if field == "categories" else ""
    chunk_path.write_text(json.dumps(chunk), encoding="utf-8")
    if field == "sha256":
        manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
        manifest["chunks"][0]["sha256"] = ""
        (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PipelineError):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_23_build_rejects_corpus_with_only_unindexable_chunks(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::noise", "-\n-\n-")])
    with pytest.raises(PipelineError, match="Nenhum chunk"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_rag_hardening_24_index_records_skipped_chunk_counts(tmp_path):
    index = _build_index(
        tmp_path,
        [
            _chunk("doc::noise", "-\n-\n-", section="Share"),
            _chunk("doc::valid", "arterial phase hyperenhancement washout capsule"),
        ],
    )
    assert index["chunk_count"] == 2
    assert index["indexed_chunk_count"] == 1
    assert index["skipped_chunk_count"] == 1
    assert index["skipped_chunks"][0]["reason"] == "no_indexable_terms"


def test_rag_hardening_25_search_rejects_non_positive_top_k(tmp_path):
    index = _build_index(tmp_path, [_chunk("doc::c1", "arterial washout")])
    with pytest.raises(PipelineError, match="top_k"):
        search_bm25(index, "arterial", top_k=0)


def test_rag_hardening_26_search_rejects_empty_query(tmp_path):
    index = _build_index(tmp_path, [_chunk("doc::c1", "arterial washout")])
    with pytest.raises(PipelineError, match="Consulta"):
        search_bm25(index, "---")


def test_rag_hardening_27_search_unknown_terms_returns_empty_list(tmp_path):
    index = _build_index(tmp_path, [_chunk("doc::c1", "arterial washout")])
    assert search_bm25(index, "xylophone unrelated", top_k=5) == []


def test_rag_hardening_28_search_category_filter_can_remove_all_results(tmp_path):
    index = _build_index(tmp_path, [_chunk("doc::c1", "arterial washout", categories=["hcc"])])
    assert search_bm25(index, "arterial washout", categories=["hemangioma"]) == []


def test_rag_hardening_29_search_category_filter_accepts_any_overlap(tmp_path):
    index = _build_index(
        tmp_path,
        [
            _chunk("hcc::c1", "arterial washout capsule", categories=["hcc", "li_rads"]),
            _chunk("hem::c1", "peripheral nodular fill-in", categories=["hemangioma"]),
        ],
    )
    results = search_bm25(index, "arterial washout peripheral", categories=["li_rads", "other"])
    assert {result["doc_id"] for result in results} == {"hcc"}


def test_rag_hardening_30_search_is_case_insensitive(tmp_path):
    index = _build_index(tmp_path, [_chunk("doc::c1", "Arterial Phase Hyperenhancement")])
    lower = search_bm25(index, "arterial hyperenhancement", top_k=1)
    upper = search_bm25(index, "ARTERIAL HYPERENHANCEMENT", top_k=1)
    assert lower[0]["chunk_id"] == upper[0]["chunk_id"]
    assert lower[0]["score"] == pytest.approx(upper[0]["score"])


def test_rag_hardening_31_search_tie_breaks_by_chunk_id(tmp_path):
    index = _build_index(
        tmp_path,
        [
            _chunk("doc_b::c1", "arterial washout"),
            _chunk("doc_a::c1", "arterial washout"),
        ],
    )
    results = search_bm25(index, "arterial washout", top_k=2)
    assert [result["chunk_id"] for result in results] == ["doc_a::c1", "doc_b::c1"]


def test_rag_hardening_32_search_honors_top_k_limit(tmp_path):
    index = _build_index(
        tmp_path,
        [
            _chunk("doc1::c1", "arterial washout capsule"),
            _chunk("doc2::c1", "arterial washout capsule"),
            _chunk("doc3::c1", "arterial washout capsule"),
        ],
    )
    assert len(search_bm25(index, "arterial washout", top_k=2)) == 2


def test_rag_hardening_33_load_rejects_invalid_schema(tmp_path):
    path = tmp_path / "bad_index.json"
    path.write_text(json.dumps({"schema": "wrong", "documents": [{}], "terms": {"x": {}}}), encoding="utf-8")
    with pytest.raises(PipelineError, match="Schema"):
        load_bm25_index(path)


def test_rag_hardening_34_load_rejects_missing_documents(tmp_path):
    path = tmp_path / "bad_index.json"
    path.write_text(json.dumps({"schema": "argos-rag-bm25-index-v1", "terms": {"x": {}}}), encoding="utf-8")
    with pytest.raises(PipelineError, match="incompleto"):
        load_bm25_index(path)


def test_rag_hardening_35_load_rejects_missing_terms(tmp_path):
    path = tmp_path / "bad_index.json"
    path.write_text(json.dumps({"schema": "argos-rag-bm25-index-v1", "documents": [{}]}), encoding="utf-8")
    with pytest.raises(PipelineError, match="incompleto"):
        load_bm25_index(path)


def test_rag_hardening_36_cli_returns_nonzero_for_missing_corpus(tmp_path, capsys):
    code = build_index_main(["--corpus", str(tmp_path / "missing"), "--out", str(tmp_path / "index")])
    captured = capsys.readouterr()
    assert code == 1
    assert "[ABORTADO]" in captured.err


def test_rag_hardening_37_manifest_and_index_sha_are_stable_length(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [_chunk("doc::c1", "arterial washout capsule")])
    manifest = build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")
    assert len(manifest["index_sha256"]) == 64
    assert len(manifest["corpus_manifest_sha256"]) == 64

