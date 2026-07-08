import json
from pathlib import Path

import pytest
import yaml

from dtwin.core import PipelineError
from tools.build_rag_corpus import SourcePayload, build_corpus, extract_markdown, main


def _html(title="MRI of focal liver lesions"):
    return f"""<!doctype html>
<html>
  <head><title>{title}</title><style>.x{{color:red}}</style></head>
  <body>
    <article>
      <h1>{title}</h1>
      <p>Focal liver lesions are characterized on MRI using T1, T2, DWI, ADC,
      arterial phase, portal venous phase, delayed phase, and hepatobiliary phase
      appearances.</p>
      <h2>Hemangioma</h2>
      <p>Hemangioma often demonstrates very high T2 signal and peripheral nodular
      discontinuous enhancement with progressive centripetal fill-in.</p>
      <h2>References</h2>
      <p>This boilerplate should not be retained in normalized output.</p>
    </article>
  </body>
</html>"""


def _manifest(tmp_path: Path, source: Path, missing_source: bool = False) -> Path:
    manifest = {
        "corpus_version": "test_liver_mri_v1",
        "articles": [
            {
                "id": "argos_rag_test_001",
                "pmcid": "PMC_TEST_001",
                "pmid": "1",
                "doi": "10.0000/test.1",
                "url": "https://example.org/one",
                "title": "MRI of focal liver lesions",
                "journal": "Test Journal",
                "year": 2026,
                "priority": "core",
                "categories": ["general_focal_liver_lesions_mri", "hemangioma"],
                "license_status": "approved_by_felipe_for_research_corpus_v1",
                "source_path": "missing.html" if missing_source else source.name,
            }
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _multi_manifest(tmp_path: Path) -> Path:
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    first.write_text(_html("Hemangioma MRI"), encoding="utf-8")
    second.write_text(
        """# LI-RADS HCC

Arterial phase hyperenhancement, nonperipheral washout, enhancing capsule,
threshold growth, and ancillary MRI features support HCC risk stratification.

# Pseudolesions

Perfusion-related pseudolesions and focal fat sparing can mimic focal liver
lesions and should be considered when enhancement is transient or geographic.
""",
        encoding="utf-8",
    )
    manifest = {
        "corpus_version": "test_liver_mri_v1",
        "articles": [
            {
                "id": "argos_rag_test_001",
                "pmcid": "PMC_TEST_001",
                "pmid": "1",
                "doi": "10.0000/test.1",
                "url": "https://example.org/one",
                "title": "Hemangioma MRI",
                "journal": "Test Journal",
                "year": 2026,
                "priority": "core",
                "categories": ["hemangioma", "benign_liver_lesions"],
                "license_status": "approved_by_felipe_for_research_corpus_v1",
                "source_path": first.name,
            },
            {
                "id": "argos_rag_test_002",
                "pmcid": "PMC_TEST_002",
                "pmid": "2",
                "doi": "10.0000/test.2",
                "url": "https://example.org/two",
                "title": "LI-RADS and pseudolesions",
                "journal": "Test Journal",
                "year": 2026,
                "priority": "core",
                "categories": ["li_rads", "hcc", "pseudolesions"],
                "license_status": "approved_by_felipe_for_research_corpus_v1",
                "source_path": second.name,
            },
        ],
    }
    path = tmp_path / "multi_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_extract_markdown_removes_script_style_and_reference_tail():
    markdown = extract_markdown(SourcePayload(_html().encode("utf-8"), "text/html", "local"))
    assert "# MRI of focal liver lesions" in markdown
    assert "Hemangioma" in markdown
    assert "color:red" not in markdown
    assert "boilerplate should not be retained" not in markdown


def test_build_corpus_from_local_html_generates_auditable_outputs(tmp_path):
    source = tmp_path / "article.html"
    source.write_text(_html(), encoding="utf-8")
    manifest = _manifest(tmp_path, source)

    out = tmp_path / "corpus"
    result = build_corpus(
        manifest_path=manifest,
        out_dir=out,
        max_tokens=40,
        overlap_tokens=5,
        no_download=True,
    )

    assert result["schema"] == "argos-rag-corpus-v1"
    assert result["article_count"] == 1
    assert result["chunk_count"] >= 2
    assert (out / "manifest.json").is_file()
    assert (out / "chunks_manifest.json").is_file()
    assert (out / "normalized" / "argos_rag_test_001.md").is_file()
    assert (out / "sources" / "argos_rag_test_001.html").is_file()

    chunks_manifest = json.loads((out / "chunks_manifest.json").read_text("utf-8"))
    first = chunks_manifest["chunks"][0]
    assert first["doc_id"] == "argos_rag_test_001"
    assert first["token_count"] <= 40
    assert "hemangioma" in result["category_counts"]


def test_build_corpus_no_download_requires_local_source(tmp_path):
    source = tmp_path / "article.html"
    manifest = _manifest(tmp_path, source, missing_source=True)
    with pytest.raises(PipelineError, match="Fonte local"):
        build_corpus(manifest_path=manifest, out_dir=tmp_path / "out", no_download=True)


def test_build_corpus_multiple_articles_preserves_categories_and_hashes(tmp_path):
    manifest = _multi_manifest(tmp_path)
    out = tmp_path / "corpus"
    result = build_corpus(
        manifest_path=manifest,
        out_dir=out,
        max_tokens=45,
        overlap_tokens=5,
        no_download=True,
    )
    assert result["article_count"] == 2
    assert result["chunk_count"] >= 3
    assert result["category_counts"]["hemangioma"] >= 1
    assert result["category_counts"]["li_rads"] >= 1
    assert result["category_counts"]["pseudolesions"] >= 1
    assert all(len(article["raw_sha256"]) == 64 for article in result["articles"])
    assert all(len(chunk["sha256"]) == 64 for chunk in result["chunks"])


def test_build_rag_corpus_cli_completes_with_local_sources(tmp_path, capsys):
    manifest = _multi_manifest(tmp_path)
    out = tmp_path / "cli_corpus"
    code = main([
        "--manifest", str(manifest),
        "--out", str(out),
        "--max-tokens", "45",
        "--overlap-tokens", "5",
        "--no-download",
    ])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "complete"
    assert payload["article_count"] == 2
    assert (out / "manifest.json").is_file()


def test_real_corpus_manifest_v1_has_expected_coverage_and_approval():
    data = yaml.safe_load(Path("docs/rag/corpus_manifest_v1.yaml").read_text("utf-8"))
    articles = data["articles"]
    assert data["corpus_version"] == "liver_mri_rag_v1"
    assert len(articles) == 41
    assert {article["license_status"] for article in articles} == {
        "approved_by_felipe_for_research_corpus_v1"
    }
    ids = [article["id"] for article in articles]
    assert len(ids) == len(set(ids))
    categories = {category for article in articles for category in article["categories"]}
    for expected in (
        "general_focal_liver_lesions_mri",
        "hcc",
        "li_rads",
        "dwi_adc",
        "gadoxetic_acid",
        "hemangioma",
        "fnh",
        "hepatic_adenoma",
        "metastases",
        "cholangiocarcinoma",
        "cystic_liver_lesions",
        "pseudolesions",
        "couinaud_segmental_anatomy",
        "liver_mri_protocol",
        "inflammatory_mimickers",
    ):
        assert expected in categories
