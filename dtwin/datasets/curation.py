"""Curadoria operacional de negativos difíceis (hard negatives).

Transforma a revisão humana de falsos positivos num conjunto de rótulos
protegidos reutilizáveis pelo benchmark. O fluxo é:

    erro atual (falso positivo) → hard negative documentado → rótulo protegido
    → melhor métrica estratificada → GraphRAG mais forte

O arquivo de trabalho (`data/curation/negative_hard_cases_review.jsonl`) fica
fora do Git — `/data/` é ignorado. Um template versionado sem dados sensíveis
vive em `configs/curation/`. Esta ferramenta valida a revisão contra o
vocabulário fechado da taxonomia e emite metadados de rótulo prontos para o
manifesto de labels do benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dtwin.benchmark.models import (
    NEGATIVE_SUBTYPES,
    PHENOTYPE_TAGS,
    POSITIVE_SUBTYPES,
    GroundTruthLabel,
)
from dtwin.core import PipelineError

CURATION_SCHEMA = "argos-hard-negative-curation-v1"
TARGET_CONDITION = "focal_liver_lesion_suspicion"
REVIEW_STATES = {"pending_review", "reviewed", "needs_second_opinion"}
LABEL_BASIS = "human_review"


@dataclass(frozen=True)
class CurationRecord:
    case_id: str
    current_label: GroundTruthLabel
    reviewer: str
    review_status: str
    recommended_label: GroundTruthLabel
    recommended_negative_subtype: str | None
    recommended_positive_subtype: str | None
    phenotype_tags: tuple[str, ...]
    notes: str | None

    def to_protected_label_metadata(self) -> dict[str, Any]:
        """Metadados protegidos prontos para o manifesto de labels do benchmark."""
        return {
            "case_id": self.case_id,
            "label": self.recommended_label.value.upper(),
            "target_condition": TARGET_CONDITION,
            "negative_subtype": self.recommended_negative_subtype,
            "positive_subtype": self.recommended_positive_subtype,
            "phenotype_tags": list(self.phenotype_tags),
            "label_basis": LABEL_BASIS,
            "review_status": self.review_status,
        }


def _token(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def _label(value: str | None, key: str, ref: str) -> GroundTruthLabel:
    if not value:
        raise PipelineError(f"{key} ausente em {ref}.")
    try:
        return GroundTruthLabel(value)
    except ValueError as exc:
        raise PipelineError(f"{key} inválido em {ref}: {value!r}") from exc


def parse_curation_record(item: dict[str, Any], ref: str) -> CurationRecord:
    case_id = str(item.get("case_id") or "").strip()
    if not case_id:
        raise PipelineError(f"case_id ausente em {ref}.")

    current_label = _label(_token(item, "current_label"), "current_label", ref)
    # recommended_label opcional: por padrão mantém o current_label.
    recommended_label = (
        _label(_token(item, "recommended_label"), "recommended_label", ref)
        if item.get("recommended_label")
        else current_label
    )

    reviewer = str(item.get("reviewer") or "").strip()
    if not reviewer:
        raise PipelineError(f"reviewer ausente em {ref}.")
    review_status = _token(item, "review_status") or "pending_review"
    if review_status not in REVIEW_STATES:
        raise PipelineError(f"review_status inválido em {ref}: {review_status!r}")

    negative_subtype = _token(item, "recommended_negative_subtype")
    positive_subtype = _token(item, "recommended_positive_subtype")
    if negative_subtype and positive_subtype:
        raise PipelineError(f"Subtipos negativo e positivo são mutuamente exclusivos em {ref}.")
    if negative_subtype is not None and negative_subtype not in NEGATIVE_SUBTYPES:
        raise PipelineError(f"recommended_negative_subtype inválido em {ref}: {negative_subtype!r}")
    if positive_subtype is not None and positive_subtype not in POSITIVE_SUBTYPES:
        raise PipelineError(f"recommended_positive_subtype inválido em {ref}: {positive_subtype!r}")
    if negative_subtype and recommended_label is not GroundTruthLabel.NEGATIVE:
        raise PipelineError(f"recommended_negative_subtype exige recommended_label=NEGATIVE em {ref}.")
    if positive_subtype and recommended_label is not GroundTruthLabel.POSITIVE:
        raise PipelineError(f"recommended_positive_subtype exige recommended_label=POSITIVE em {ref}.")

    raw_tags = item.get("phenotype_tags") or []
    if not isinstance(raw_tags, list):
        raise PipelineError(f"phenotype_tags deve ser lista em {ref}.")
    tags = tuple(str(tag).strip().lower() for tag in raw_tags if str(tag).strip())
    invalid = [tag for tag in tags if tag not in PHENOTYPE_TAGS]
    if invalid:
        raise PipelineError(f"phenotype_tags inválidas em {ref}: {invalid}")

    notes = item.get("notes")
    notes = str(notes).strip() if notes not in (None, "") else None

    return CurationRecord(
        case_id=case_id,
        current_label=current_label,
        reviewer=reviewer,
        review_status=review_status,
        recommended_label=recommended_label,
        recommended_negative_subtype=negative_subtype,
        recommended_positive_subtype=positive_subtype,
        phenotype_tags=tags,
        notes=notes,
    )


def load_curation_manifest(path: Path) -> list[CurationRecord]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PipelineError(f"Falha ao ler manifesto de curadoria: {path}") from exc
    records: list[CurationRecord] = []
    seen: set[str] = set()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        ref = f"{path}:{line_no}"
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"JSONL inválido em {ref}") from exc
        if not isinstance(item, dict):
            raise PipelineError(f"Registro de curadoria deve ser objeto em {ref}")
        record = parse_curation_record(item, ref)
        if record.case_id in seen:
            raise PipelineError(f"case_id duplicado na curadoria: {record.case_id}")
        seen.add(record.case_id)
        records.append(record)
    return records


def summarize(records: Iterable[CurationRecord]) -> dict[str, Any]:
    records = list(records)
    reviewed = [r for r in records if r.review_status == "reviewed"]
    negatives: dict[str, int] = {}
    positives: dict[str, int] = {}
    reclassified = 0
    for record in records:
        if record.recommended_negative_subtype:
            negatives[record.recommended_negative_subtype] = (
                negatives.get(record.recommended_negative_subtype, 0) + 1
            )
        if record.recommended_positive_subtype:
            positives[record.recommended_positive_subtype] = (
                positives.get(record.recommended_positive_subtype, 0) + 1
            )
        if record.recommended_label is not record.current_label:
            reclassified += 1
    return {
        "total": len(records),
        "reviewed": len(reviewed),
        "reclassified": reclassified,
        "negative_subtypes": dict(sorted(negatives.items())),
        "positive_subtypes": dict(sorted(positives.items())),
    }


def build_protected_labels(records: Iterable[CurationRecord]) -> list[dict[str, Any]]:
    """Apenas casos já revisados viram rótulo protegido aplicável."""
    return [
        record.to_protected_label_metadata()
        for record in records
        if record.review_status == "reviewed"
    ]


def _write_jsonl(rows: list[dict[str, Any]], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida a curadoria de negativos difíceis e emite rótulos protegidos."
    )
    parser.add_argument("--review", required=True, help="JSONL de revisão dos hard negatives.")
    parser.add_argument(
        "--out",
        help="Se informado, escreve os rótulos protegidos (apenas casos reviewed).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_curation_manifest(Path(args.review))
        summary = summarize(records)
        if args.out:
            _write_jsonl(build_protected_labels(records), Path(args.out))
    except PipelineError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2
    print(
        "[OK] curadoria: {total} casos, {reviewed} revisados, {reclassified} reclassificados".format(
            **summary
        )
    )
    if summary["negative_subtypes"]:
        print(f"      negativos por subtipo: {summary['negative_subtypes']}")
    if summary["positive_subtypes"]:
        print(f"      positivos por subtipo: {summary['positive_subtypes']}")
    if args.out:
        print(f"      rótulos protegidos escritos em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
