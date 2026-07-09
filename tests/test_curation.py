"""Testes da curadoria de negativos difíceis (hard negatives)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dtwin.core import PipelineError
from dtwin.datasets import curation


def _review(**overrides):
    base = {
        "case_id": "anon-001",
        "current_label": "NEGATIVE",
        "recommended_label": "NEGATIVE",
        "recommended_negative_subtype": "benign_anatomic_variant",
        "phenotype_tags": ["prominent_hepatic_vein"],
        "reviewer": "human",
        "review_status": "reviewed",
        "notes": "Veia calibrosa sem massa focal.",
    }
    base.update(overrides)
    return base


def test_reviewed_variant_becomes_protected_negative_label():
    record = curation.parse_curation_record(_review(), "ref")
    metadata = record.to_protected_label_metadata()
    assert metadata["label"] == "NEGATIVE"
    assert metadata["negative_subtype"] == "benign_anatomic_variant"
    assert metadata["target_condition"] == "focal_liver_lesion_suspicion"
    assert metadata["label_basis"] == "human_review"
    assert metadata["phenotype_tags"] == ["prominent_hepatic_vein"]


def test_negative_subtype_requires_negative_recommended_label():
    with pytest.raises(PipelineError, match="recommended_label=NEGATIVE"):
        curation.parse_curation_record(
            _review(recommended_label="POSITIVE"), "ref"
        )


def test_invalid_negative_subtype_rejected():
    with pytest.raises(PipelineError, match="recommended_negative_subtype"):
        curation.parse_curation_record(
            _review(recommended_negative_subtype="not_a_subtype"), "ref"
        )


def test_negative_and_positive_subtype_mutually_exclusive():
    with pytest.raises(PipelineError, match="mutuamente exclusivos"):
        curation.parse_curation_record(
            _review(
                recommended_negative_subtype="benign_anatomic_variant",
                recommended_positive_subtype="hcc_suspicious",
            ),
            "ref",
        )


def test_invalid_phenotype_tag_rejected():
    with pytest.raises(PipelineError, match="phenotype_tags"):
        curation.parse_curation_record(
            _review(phenotype_tags=["not_a_tag"]), "ref"
        )


def test_invalid_review_status_rejected():
    with pytest.raises(PipelineError, match="review_status"):
        curation.parse_curation_record(_review(review_status="approved"), "ref")


def test_reclassification_to_positive_is_summarized(tmp_path):
    manifest = tmp_path / "review.jsonl"
    rows = [
        _review(),
        _review(
            case_id="anon-002",
            recommended_label="POSITIVE",
            recommended_negative_subtype=None,
            recommended_positive_subtype="focal_lesion_suspicious",
            phenotype_tags=["arterial_hyperenhancement"],
        ),
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    records = curation.load_curation_manifest(manifest)
    summary = curation.summarize(records)
    assert summary["total"] == 2
    assert summary["reviewed"] == 2
    assert summary["reclassified"] == 1
    assert summary["positive_subtypes"] == {"focal_lesion_suspicious": 1}


def test_build_protected_labels_only_includes_reviewed():
    records = [
        curation.parse_curation_record(_review(), "a"),
        curation.parse_curation_record(
            _review(case_id="anon-002", review_status="needs_second_opinion"), "b"
        ),
    ]
    labels = curation.build_protected_labels(records)
    assert [label["case_id"] for label in labels] == ["anon-001"]


def test_duplicate_case_id_rejected(tmp_path):
    manifest = tmp_path / "review.jsonl"
    manifest.write_text("\n".join(json.dumps(_review()) for _ in range(2)) + "\n", "utf-8")
    with pytest.raises(PipelineError, match="case_id duplicado"):
        curation.load_curation_manifest(manifest)


def test_cli_validates_versioned_template_and_emits_labels(tmp_path):
    template = Path("configs/curation/negative_hard_cases_review.template.jsonl")
    out = tmp_path / "protected_labels.jsonl"
    code = curation.main(["--review", str(template), "--out", str(out)])
    assert code == 0
    lines = out.read_text("utf-8").splitlines()
    # Só os casos "reviewed" do template viram rótulo protegido (2 de 3).
    assert len(lines) == 2
    assert all(json.loads(line)["label_basis"] == "human_review" for line in lines)
