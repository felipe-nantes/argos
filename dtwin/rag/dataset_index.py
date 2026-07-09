"""Gera documentos RAG textuais seguros a partir do dataset registry.

Cada linha de um manifesto ``.jsonl`` do registry vira um documento textual
enxuto (``doc_id``/``case_id``/``rag_class``/``title``/``text``/``metadata``)
para apoiar prompt, revisão e auditoria. O texto por classe segue as regras
metodológicas do plano de patologia alvo — em especial, CHAOS nunca é descrito
como normalidade clínica absoluta e variantes anatômicas benignas são
explicitamente marcadas como negativas que podem mimetizar lesão focal.

Nenhum dado bruto sensível é emitido: UIDs DICOM e identificadores de paciente
são recusados, e o documento carrega apenas metadados de rótulo já sanitizados.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from dtwin.core import PipelineError
from dtwin.datasets.schema import REGISTRY_SCHEMA

RAG_DOCUMENT_SCHEMA = "argos-rag-dataset-document-v1"

# Chaves de metadado bruto que jamais podem vazar para um documento RAG.
_FORBIDDEN_KEYS = {
    "seriesinstanceuid",
    "studyinstanceuid",
    "sopinstanceuid",
    "patientname",
    "patientid",
    "patientbirthdate",
}

# Metadados de rótulo seguros preservados em cada documento.
_SAFE_METADATA_KEYS = (
    "dataset_id",
    "rag_class",
    "label",
    "negative_subtype",
    "positive_subtype",
    "phenotype_tags",
    "modality",
    "source_format",
    "sequence_or_phase",
    "has_segmentation",
    "research_only",
    "clinical_use_allowed",
)


def _forbidden_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
            if normalized in _FORBIDDEN_KEYS:
                found.append(str(key))
            found.extend(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_forbidden_keys(child))
    return found


def _validate_record(record: dict[str, Any], ref: str) -> None:
    if record.get("schema") != REGISTRY_SCHEMA:
        raise PipelineError(f"Registro com schema inválido em {ref}.")
    if record.get("clinical_use_allowed") is not False or record.get("research_only") is not True:
        raise PipelineError(f"Documento RAG exige research_only sem uso clínico em {ref}.")
    leaked = _forbidden_keys(record)
    if leaked:
        raise PipelineError(f"Registro contém metadado bruto proibido em {ref}: {leaked}")


def iter_registry_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise PipelineError(f"Falha ao ler manifesto registry: {path}") from exc
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            ref = f"{path}:{line_no}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"JSONL inválido em {ref}") from exc
            if not isinstance(record, dict):
                raise PipelineError(f"Registro registry deve ser objeto em {ref}")
            _validate_record(record, ref)
            yield record


def _base_text(record: dict[str, Any]) -> str:
    """Texto metodológico obrigatório por classe/subtipo do registry."""
    dataset_id = str(record.get("dataset_id") or "")
    negative_subtype = record.get("negative_subtype")
    positive_subtype = record.get("positive_subtype")

    # Hard negative anatômico tem prioridade: nunca pode virar positivo patológico.
    if negative_subtype == "benign_anatomic_variant":
        return (
            "Caso negativo para patologia alvo, com variante anatômica benigna documentada. "
            "Pode mimetizar lesão focal, mas não deve ser contado como positivo patológico."
        )
    if negative_subtype in {"pseudolesion_or_artifact", "poor_quality_non_diagnostic"}:
        return (
            "Caso negativo para patologia alvo, associado a pseudolesão/artefato ou qualidade "
            "insuficiente. Não representa lesão focal hepática suspeita."
        )
    if dataset_id == "chaos_mri":
        return (
            "Controle anatômico negativo de RM abdominal. Não representa normalidade clínica "
            "absoluta. Usar como comparação anatômica e basal."
        )
    if dataset_id == "lld_mmri":
        return (
            "Caso positivo de lesão hepática focal em NIfTI. Não é DICOM original. "
            "Usar como exemplo positivo amplo."
        )
    if dataset_id in {"liverhccseg", "tcga_lihc_mr"}:
        return (
            "Caso positivo de HCC em RM. DICOM original quando disponível. "
            "Usar como exemplo positivo para patologia alvo."
        )
    if positive_subtype:
        return (
            "Caso positivo de patologia hepática alvo suspeita. "
            "Usar como exemplo positivo para triagem experimental."
        )
    return (
        "Caso negativo para patologia hepática alvo. Não representa normalidade clínica absoluta."
    )


def _document_text(record: dict[str, Any]) -> str:
    parts = [_base_text(record)]
    tags = [str(tag) for tag in (record.get("phenotype_tags") or []) if str(tag).strip()]
    if tags:
        parts.append("Fenótipos observados: " + ", ".join(tags) + ".")
    limitations = [str(item) for item in (record.get("limitations") or []) if str(item).strip()]
    if limitations:
        parts.append("Limitações: " + "; ".join(limitations) + ".")
    return " ".join(parts)


def _safe_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _SAFE_METADATA_KEYS:
        if key not in record:
            continue
        value = record[key]
        metadata[key] = list(value) if isinstance(value, list) else value
    return metadata


def build_document(record: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(record.get("dataset_id") or "unknown")
    case_id = str(record.get("case_id") or "unknown")
    dataset_name = str(record.get("dataset_name") or dataset_id)
    rag_class = str(record.get("rag_class") or "")
    return {
        "schema": RAG_DOCUMENT_SCHEMA,
        "doc_id": f"{dataset_id}__{case_id}",
        "case_id": case_id,
        "dataset_id": dataset_id,
        "rag_class": rag_class,
        "title": f"{dataset_name} — {case_id} ({rag_class})",
        "text": _document_text(record),
        "metadata": _safe_metadata(record),
    }


def build_documents(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        document = build_document(record)
        if document["doc_id"] in seen:
            raise PipelineError(f"doc_id duplicado no índice RAG: {document['doc_id']}")
        seen.add(document["doc_id"])
        documents.append(document)
    return documents


def write_documents(documents: list[dict[str, Any]], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    temp.replace(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera documentos RAG textuais seguros a partir do dataset registry."
    )
    parser.add_argument("--manifests", nargs="+", required=True, help="Um ou mais JSONL do dataset registry.")
    parser.add_argument("--out", required=True, help="Arquivo JSONL de saída com os documentos RAG.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        documents = build_documents(iter_registry_records(Path(item) for item in args.manifests))
        write_documents(documents, Path(args.out))
    except PipelineError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2
    print(f"[OK] {len(documents)} documentos RAG gerados em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
