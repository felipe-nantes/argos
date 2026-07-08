import json

import pytest
import yaml

from dtwin.core import PipelineError
from dtwin.rag.index import build_bm25_index, load_bm25_index, search_bm25, tokenize
from tools.build_rag_corpus import build_corpus
from tools.build_rag_index import main as build_index_main


def _corpus_manifest(tmp_path):
    hcc = tmp_path / "hcc.md"
    hemangioma = tmp_path / "hemangioma.md"
    hcc.write_text(
        """# LI-RADS HCC

Arterial phase hyperenhancement, washout, enhancing capsule, threshold growth,
and ancillary MRI features are used for HCC risk stratification.
""",
        encoding="utf-8",
    )
    hemangioma.write_text(
        """# Hemangioma

Hepatic hemangioma often has very high T2 signal, peripheral nodular
discontinuous enhancement, and progressive centripetal fill-in.
""",
        encoding="utf-8",
    )
    manifest = {
        "corpus_version": "test_index_v1",
        "articles": [
            {
                "id": "doc_hcc",
                "pmcid": "PMC_HCC",
                "pmid": "1",
                "doi": "10.0000/hcc",
                "url": "https://example.org/hcc",
                "title": "LI-RADS HCC",
                "journal": "Test",
                "year": 2026,
                "priority": "core",
                "categories": ["hcc", "li_rads"],
                "license_status": "approved_by_felipe_for_research_corpus_v1",
                "source_path": hcc.name,
            },
            {
                "id": "doc_hemangioma",
                "pmcid": "PMC_HEM",
                "pmid": "2",
                "doi": "10.0000/hem",
                "url": "https://example.org/hem",
                "title": "Hemangioma MRI",
                "journal": "Test",
                "year": 2026,
                "priority": "core",
                "categories": ["hemangioma", "benign_liver_lesions"],
                "license_status": "approved_by_felipe_for_research_corpus_v1",
                "source_path": hemangioma.name,
            },
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _build_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    build_corpus(
        manifest_path=_corpus_manifest(tmp_path),
        out_dir=corpus_dir,
        max_tokens=80,
        overlap_tokens=5,
        no_download=True,
    )
    return corpus_dir


def test_tokenize_is_lowercase_and_stable():
    assert tokenize("APHE washout/capsule, T2-hyperintense") == [
        "aphe", "washout", "capsule", "t2-hyperintense"
    ]


def test_build_bm25_index_and_search(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    index_dir = tmp_path / "index"
    manifest = build_bm25_index(corpus_dir=corpus_dir, out_dir=index_dir)
    assert manifest["schema"] == "argos-rag-index-manifest-v1"
    assert manifest["index_type"] == "bm25"
    assert manifest["chunk_count"] == 2
    assert manifest["vocabulary_size"] > 10
    assert len(manifest["index_sha256"]) == 64

    index = load_bm25_index(index_dir / "bm25_index.json")
    hcc = search_bm25(index, "arterial washout capsule LI-RADS HCC", top_k=1)
    assert hcc
    assert hcc[0]["doc_id"] == "doc_hcc"
    hem = search_bm25(index, "very high T2 peripheral nodular fill-in hemangioma", top_k=1)
    assert hem[0]["doc_id"] == "doc_hemangioma"


def test_search_bm25_supports_category_filter(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    index_dir = tmp_path / "index"
    build_bm25_index(corpus_dir=corpus_dir, out_dir=index_dir)
    index = load_bm25_index(index_dir / "bm25_index.json")
    results = search_bm25(
        index,
        "arterial enhancement hemangioma",
        top_k=5,
        categories=["hemangioma"],
    )
    assert results
    assert {result["doc_id"] for result in results} == {"doc_hemangioma"}


def test_build_rag_index_cli_completes(tmp_path, capsys):
    corpus_dir = _build_corpus(tmp_path)
    index_dir = tmp_path / "cli_index"
    code = build_index_main(["--corpus", str(corpus_dir), "--out", str(index_dir)])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "complete"
    assert payload["index_type"] == "bm25"
    assert (index_dir / "manifest.json").is_file()
    assert (index_dir / "bm25_index.json").is_file()


def test_build_bm25_index_aborts_when_chunk_file_missing(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    missing = corpus_dir / manifest["chunks"][0]["path"]
    missing.unlink()
    with pytest.raises(PipelineError, match="Arquivo de chunk"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")


def test_build_bm25_index_aborts_when_chunk_hash_differs(tmp_path):
    corpus_dir = _build_corpus(tmp_path)
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    chunk_path = corpus_dir / manifest["chunks"][0]["path"]
    chunk = json.loads(chunk_path.read_text("utf-8"))
    chunk["sha256"] = "0" * 64
    chunk_path.write_text(json.dumps(chunk), encoding="utf-8")
    with pytest.raises(PipelineError, match="Hash de chunk"):
        build_bm25_index(corpus_dir=corpus_dir, out_dir=tmp_path / "index")
