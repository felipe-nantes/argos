import pytest

from dtwin.core import PipelineError
from dtwin.rag.chunking import CorpusChunk, CorpusDocument, chunk_document, rough_token_count, validate_chunks


def _metadata():
    return {
        "source_id": "PMC_TEST",
        "title": "Test liver MRI source",
        "url": "https://example.org/article",
        "pmcid": "PMC_TEST",
        "pmid": "123",
        "doi": "10.0000/test",
        "journal": "Test Journal",
        "year": 2026,
        "categories": ["general_focal_liver_lesions_mri", "dwi_adc"],
        "priority": "core",
        "license_status": "approved_by_felipe_for_research_corpus_v1",
    }


def test_chunk_document_preserves_metadata_and_token_ceiling():
    text = """# Hemangioma

Hepatic hemangioma is a benign vascular lesion with very high T2 signal.
Peripheral nodular discontinuous enhancement with progressive centripetal fill-in
is a classic dynamic MRI pattern.

# DWI

Diffusion weighted imaging may help detect focal liver lesions, but ADC values
must be interpreted with T2 shine-through, lesion type, acquisition parameters,
and image quality in mind.
"""
    chunks = chunk_document(
        CorpusDocument("doc-test", text, _metadata()),
        max_tokens=35,
        overlap_tokens=5,
    )
    assert len(chunks) >= 2
    assert {chunk.doc_id for chunk in chunks} == {"doc-test"}
    assert {chunk.metadata["source_id"] for chunk in chunks} == {"PMC_TEST"}
    assert all(chunk.token_count <= 35 for chunk in chunks)
    assert all(chunk.sha256 for chunk in chunks)
    validate_chunks(chunks, max_tokens=35)


def test_chunk_document_is_deterministic():
    doc = CorpusDocument("doc-test", "# Title\n\n" + ("liver MRI lesion " * 80), _metadata())
    first = [chunk.to_dict() for chunk in chunk_document(doc, max_tokens=30, overlap_tokens=5)]
    second = [chunk.to_dict() for chunk in chunk_document(doc, max_tokens=30, overlap_tokens=5)]
    assert first == second


def test_chunk_document_rejects_missing_required_metadata():
    metadata = _metadata()
    metadata.pop("url")
    with pytest.raises(PipelineError, match="metadados obrigatórios"):
        chunk_document(CorpusDocument("doc-test", "Texto suficiente " * 20, metadata))


def test_rough_token_count_counts_words_and_punctuation():
    assert rough_token_count("APHE + washout/capsule.") >= 5


def test_chunk_document_rejects_invalid_overlap():
    with pytest.raises(PipelineError, match="overlap_tokens"):
        chunk_document(
            CorpusDocument("doc-test", "Texto suficiente " * 20, _metadata()),
            max_tokens=20,
            overlap_tokens=20,
        )


def test_long_unpunctuated_section_is_hard_windowed_under_limit():
    doc = CorpusDocument("doc-test", "# Long\n\n" + ("gadoxetic " * 125), _metadata())
    chunks = chunk_document(doc, max_tokens=25, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 25 for chunk in chunks)


def test_validate_chunks_rejects_tampered_hash():
    chunk = chunk_document(
        CorpusDocument("doc-test", "# Title\n\nHCC arterial washout capsule " * 10, _metadata()),
        max_tokens=40,
        overlap_tokens=5,
    )[0]
    tampered = CorpusChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        section=chunk.section,
        text=chunk.text,
        token_count=chunk.token_count,
        metadata=chunk.metadata,
        sha256="0" * 64,
    )
    with pytest.raises(PipelineError, match="hash inconsistente"):
        validate_chunks([tampered], max_tokens=40)
