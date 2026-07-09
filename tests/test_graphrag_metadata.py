import json
from pathlib import Path

import pytest

from dtwin.core import PipelineError
from dtwin.graphrag.config import load_graphrag_config
from dtwin.graphrag.context import build_metadata_graphrag_context
from dtwin.graphrag.ingest_registry import ingest_records, iter_registry_records
from dtwin.graphrag.schema import MIMIC_RELATIONS, registry_record_to_graph_params, validate_registry_record


class FakeGraphStore:
    def __init__(self):
        self.schema_ready = False
        self.records = []
        self.mimics = [{"phenotype_tag": tag, "finding": finding} for tag, finding in MIMIC_RELATIONS]

    def ensure_schema(self):
        self.schema_ready = True

    def upsert_registry_record(self, record):
        self.records.append(registry_record_to_graph_params(record))

    def query_context(self, *, negative_subtype, phenotype_tag, target, limit=10):
        rows = []
        for record in self.records:
            if negative_subtype and record.get("negative_subtype") != negative_subtype:
                continue
            if phenotype_tag and phenotype_tag not in record.get("phenotype_tags", []):
                continue
            rows.append({
                "case_id": record["case_id"],
                "dataset_id": record["dataset_id"],
                "negative_subtype": record.get("negative_subtype"),
                "phenotype_tags": record.get("phenotype_tags", []),
                "review_status": record.get("review_status"),
            })
        mimics = [
            item for item in self.mimics
            if (not phenotype_tag or item["phenotype_tag"] == phenotype_tag)
            and (not target or item["finding"] == target)
        ]
        limitations = sorted({item for record in self.records for item in record.get("limitations", [])})
        return {"retrieved_cases": rows[:limit], "mimic_context": mimics, "limitations": limitations}


def _registry_record(**extra):
    record = {
        "schema": "argos-dataset-registry-v1",
        "case_id": "anon-001",
        "series_id": "hash-series",
        "dataset_id": "chaos_mri",
        "dataset_name": "CHAOS MRI",
        "rag_class": "negative",
        "label": "controle_anatomico_sem_patologia_macroscopica_documentada",
        "negative_subtype": "benign_anatomic_variant",
        "positive_subtype": None,
        "phenotype_tags": ["prominent_hepatic_vein", "vascular_structure"],
        "modality": "MR",
        "source_format": "dicom",
        "dicom_original": True,
        "nifti_original": False,
        "derived_from": None,
        "sequence_or_phase": "unknown",
        "body_region": "abdomen_liver",
        "raw_path": "private/path/not-for-graph",
        "annotation_path": None,
        "has_segmentation": False,
        "source_url": "https://example.invalid",
        "clinical_use_allowed": False,
        "research_only": True,
        "review_status": "reviewed",
        "limitations": ["Não representa normalidade clínica absoluta."],
        "warnings": [],
        "metadata": {"series_uid_sha256_prefix": "abc123", "raw_path": "must_be_removed"},
    }
    record.update(extra)
    return record


def test_ingest_records_creates_schema_and_upserts_records():
    store = FakeGraphStore()
    count = ingest_records(store, [_registry_record(), _registry_record(case_id="anon-002")])

    assert count == 2
    assert store.schema_ready is True
    assert len(store.records) == 2
    assert store.records[0]["case_id"] == "anon-001"


def test_metadata_context_recovers_hard_negative_and_mimic_relation():
    store = FakeGraphStore()
    ingest_records(store, [_registry_record()])

    context = build_metadata_graphrag_context(
        store,
        negative_subtype="benign_anatomic_variant",
        phenotype_tag="prominent_hepatic_vein",
        target="focal_liver_lesion",
    )

    assert context["query_mode"] == "metadata_graphrag"
    assert context["research_only"] is True
    assert context["clinical_use_allowed"] is False
    assert context["retrieved_cases"][0]["case_id"] == "anon-001"
    assert context["mimic_context"] == [{
        "phenotype_tag": "prominent_hepatic_vein",
        "finding": "focal_liver_lesion",
    }]


def test_graph_params_do_not_persist_raw_path_or_uid():
    params = registry_record_to_graph_params(_registry_record())
    encoded = json.dumps(params, ensure_ascii=False)

    assert "private/path/not-for-graph" not in encoded
    assert "raw_path" not in params["metadata"]
    assert "SeriesInstanceUID" not in encoded


def test_registry_record_rejects_raw_uid_key():
    record = _registry_record(metadata={"SeriesInstanceUID": "1.2.3"})

    with pytest.raises(PipelineError, match="proibido"):
        validate_registry_record(record)


def test_iter_registry_records_validates_jsonl(tmp_path):
    manifest = tmp_path / "registry.jsonl"
    manifest.write_text(json.dumps(_registry_record(), ensure_ascii=False) + "\n", encoding="utf-8")

    records = list(iter_registry_records([manifest]))

    assert records[0]["case_id"] == "anon-001"


def test_load_graphrag_config_uses_password_env(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    config = load_graphrag_config(Path("configs/graphrag_neo4j.yaml"))

    assert config.neo4j.uri == "bolt://localhost:7687"
    assert config.neo4j.password == "secret"
    assert config.research_only is True
    assert config.clinical_use_allowed is False


def test_load_graphrag_config_fails_without_password_env(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    config = load_graphrag_config(Path("configs/graphrag_neo4j.yaml"))

    with pytest.raises(PipelineError, match="NEO4J_PASSWORD"):
        _ = config.neo4j.password
