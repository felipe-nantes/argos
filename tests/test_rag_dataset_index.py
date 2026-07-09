"""Testes do gerador de documentos RAG derivados do dataset registry."""
from __future__ import annotations

import json

import pytest

from dtwin.core import PipelineError
from dtwin.rag import dataset_index
from dtwin.datasets.schema import REGISTRY_SCHEMA


def _record(**overrides):
    base = {
        "schema": REGISTRY_SCHEMA,
        "case_id": "anon-001",
        "dataset_id": "chaos_mri",
        "dataset_name": "CHAOS MRI",
        "rag_class": "negative",
        "label": "controle_anatomico",
        "negative_subtype": "normal",
        "positive_subtype": None,
        "phenotype_tags": [],
        "modality": "MR",
        "source_format": "dicom",
        "sequence_or_phase": "unknown",
        "has_segmentation": False,
        "research_only": True,
        "clinical_use_allowed": False,
        "limitations": [],
    }
    base.update(overrides)
    return base


def test_chaos_document_never_claims_absolute_normality():
    doc = dataset_index.build_document(_record())
    assert doc["schema"] == dataset_index.RAG_DOCUMENT_SCHEMA
    assert doc["doc_id"] == "chaos_mri__anon-001"
    assert "não representa normalidade clínica absoluta" in doc["text"].lower()


def test_benign_anatomic_variant_is_flagged_as_non_positive():
    doc = dataset_index.build_document(
        _record(
            dataset_id="chaos_mri",
            negative_subtype="benign_anatomic_variant",
            phenotype_tags=["prominent_hepatic_vein", "vascular_structure"],
        )
    )
    assert "não deve ser contado como positivo patológico" in doc["text"].lower()
    assert "prominent_hepatic_vein" in doc["text"]
    assert doc["metadata"]["negative_subtype"] == "benign_anatomic_variant"


def test_lld_mmri_positive_document_marks_not_dicom_original():
    doc = dataset_index.build_document(
        _record(
            dataset_id="lld_mmri",
            dataset_name="LLD-MMRI",
            rag_class="positive",
            negative_subtype=None,
            positive_subtype="focal_lesion_suspicious",
            source_format="nifti",
        )
    )
    assert "não é dicom original" in doc["text"].lower()
    assert doc["rag_class"] == "positive"


def test_hcc_datasets_are_positive_target_examples():
    for dataset_id in ("liverhccseg", "tcga_lihc_mr"):
        doc = dataset_index.build_document(
            _record(
                dataset_id=dataset_id,
                rag_class="positive",
                negative_subtype=None,
                positive_subtype="hcc_suspicious",
            )
        )
        assert "hcc" in doc["text"].lower()


def test_document_metadata_excludes_raw_paths_and_uids():
    doc = dataset_index.build_document(_record(raw_path="data/raw/x", annotation_path="data/ann/y"))
    assert "raw_path" not in doc["metadata"]
    assert "annotation_path" not in doc["metadata"]


def test_iter_records_rejects_leaked_uid(tmp_path):
    leaked = _record()
    leaked["metadata"] = {"SeriesInstanceUID": "1.2.3"}
    manifest = tmp_path / "leak.jsonl"
    manifest.write_text(json.dumps(leaked) + "\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="metadado bruto proibido"):
        list(dataset_index.iter_registry_records([manifest]))


def test_cli_writes_documents_jsonl(tmp_path):
    manifest = tmp_path / "chaos.jsonl"
    manifest.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    out = tmp_path / "rag" / "dataset_documents.jsonl"
    code = dataset_index.main(["--manifests", str(manifest), "--out", str(out)])
    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["doc_id"] == "chaos_mri__anon-001"
    assert loaded["metadata"]["research_only"] is True


def test_build_documents_rejects_duplicate_doc_id():
    records = [_record(), _record()]
    with pytest.raises(PipelineError, match="doc_id duplicado"):
        dataset_index.build_documents(records)
