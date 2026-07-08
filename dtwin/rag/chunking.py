"""Chunking auditável para o corpus RAG do ARGOS.

Objetivo da F0:
- preservar proveniência por chunk;
- respeitar teto de tokens compatível com MedCPT (~480 tokens úteis);
- falhar fechado quando metadados obrigatórios estiverem ausentes;
- produzir IDs e hashes determinísticos.

O tokenizador aqui é deliberadamente simples e sem dependências externas. Ele não
substitui o tokenizador real do embedding, mas é estável e suficiente para impedir
chunks gigantes antes da indexação.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from dtwin.core import PipelineError

DEFAULT_MAX_TOKENS = 480
DEFAULT_OVERLAP_TOKENS = 40
REQUIRED_DOCUMENT_METADATA = (
    "source_id",
    "title",
    "url",
    "categories",
    "priority",
    "license_status",
)
REQUIRED_CHUNK_METADATA = (
    "doc_id",
    "chunk_id",
    "source_id",
    "title",
    "url",
    "section",
    "categories",
    "priority",
    "license_status",
    "token_count",
    "sha256",
)

_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True)
class CorpusDocument:
    """Documento normalizado pronto para chunking."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusChunk:
    """Chunk com proveniência completa."""

    chunk_id: str
    doc_id: str
    section: str
    text: str
    token_count: int
    metadata: dict[str, Any]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        data = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "text": self.text,
            "token_count": self.token_count,
            "sha256": self.sha256,
        }
        data.update(self.metadata)
        return data


def normalize_text(text: str) -> str:
    """Normaliza whitespace sem destruir headings Markdown."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    blank = False
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if not blank:
                lines.append("")
            blank = True
            continue
        lines.append(line)
        blank = False
    return "\n".join(lines).strip()


def rough_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_document(document: CorpusDocument) -> None:
    if not document.doc_id or not re.match(r"^[A-Za-z0-9_.:-]+$", document.doc_id):
        raise PipelineError("Documento RAG sem doc_id válido.")
    if not normalize_text(document.text):
        raise PipelineError(f"Documento RAG vazio: {document.doc_id}")
    missing = [key for key in REQUIRED_DOCUMENT_METADATA if not document.metadata.get(key)]
    if missing:
        raise PipelineError(f"Documento RAG {document.doc_id} sem metadados obrigatórios: {missing}")
    categories = document.metadata.get("categories")
    if not isinstance(categories, list) or not categories:
        raise PipelineError(f"Documento RAG {document.doc_id}: categories deve ser lista não vazia.")


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Divide Markdown em seções por headings.

    Se não houver heading, cria uma seção única "Document".
    """

    text = normalize_text(text)
    sections: list[tuple[str, list[str]]] = []
    current_title = "Document"
    current_lines: list[str] = []
    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    if not sections:
        sections.append(("Document", [text]))
    return [(title, normalize_text("\n".join(lines))) for title, lines in sections if normalize_text("\n".join(lines))]


def _paragraphs(section_text: str) -> list[str]:
    paragraphs = [normalize_text(part) for part in re.split(r"\n\s*\n", section_text)]
    return [part for part in paragraphs if part]


def _sentence_units(text: str) -> list[str]:
    units = [normalize_text(part) for part in _SENTENCE_RE.split(text)]
    return [part for part in units if part]


def _hard_token_windows(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    toks = _tokens(text)
    if len(toks) <= max_tokens:
        return [text]
    chunks: list[str] = []
    step = max(max_tokens - overlap_tokens, 1)
    for start in range(0, len(toks), step):
        window = toks[start : start + max_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_tokens >= len(toks):
            break
    return chunks


def _split_units_to_chunks(units: Iterable[str], *, max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(normalize_text("\n\n".join(current)))
            current = []
            current_tokens = 0

    for unit in units:
        unit = normalize_text(unit)
        if not unit:
            continue
        unit_tokens = rough_token_count(unit)
        if unit_tokens > max_tokens:
            flush()
            chunks.extend(_hard_token_windows(unit, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
            continue
        if current and current_tokens + unit_tokens > max_tokens:
            flush()
        current.append(unit)
        current_tokens += unit_tokens
    flush()
    return chunks


def split_section_to_chunks(section_text: str, *, max_tokens: int, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list[str]:
    """Quebra uma seção respeitando parágrafos; cai para sentenças/janelas se necessário."""

    section_text = normalize_text(section_text)
    if rough_token_count(section_text) <= max_tokens:
        return [section_text]
    paragraphs = _paragraphs(section_text)
    chunks = _split_units_to_chunks(paragraphs, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    refined: list[str] = []
    for chunk in chunks:
        if rough_token_count(chunk) <= max_tokens:
            refined.append(chunk)
        else:
            refined.extend(
                _split_units_to_chunks(
                    _sentence_units(chunk), max_tokens=max_tokens, overlap_tokens=overlap_tokens
                )
            )
    final: list[str] = []
    for chunk in refined:
        if rough_token_count(chunk) <= max_tokens:
            final.append(chunk)
        else:
            final.extend(_hard_token_windows(chunk, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return [normalize_text(chunk) for chunk in final if normalize_text(chunk)]


def _chunk_hash_payload(*, doc_id: str, section: str, text: str, metadata: dict[str, Any]) -> str:
    categories = ",".join(str(item) for item in metadata.get("categories", []))
    return "\n".join(
        [
            f"doc_id:{doc_id}",
            f"source_id:{metadata.get('source_id')}",
            f"title:{metadata.get('title')}",
            f"url:{metadata.get('url')}",
            f"section:{section}",
            f"categories:{categories}",
            "",
            text,
        ]
    )


def chunk_document(
    document: CorpusDocument,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[CorpusChunk]:
    """Gera chunks determinísticos para um documento normalizado."""

    if max_tokens <= 0:
        raise PipelineError("max_tokens do RAG deve ser positivo.")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise PipelineError("overlap_tokens do RAG deve ser >= 0 e menor que max_tokens.")
    _validate_document(document)

    chunks: list[CorpusChunk] = []
    ordinal = 0
    for section_title, section_text in split_markdown_sections(document.text):
        for piece in split_section_to_chunks(
            section_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens
        ):
            ordinal += 1
            token_count = rough_token_count(piece)
            if token_count > max_tokens:
                raise PipelineError(
                    f"Chunk RAG excedeu teto de tokens ({token_count}>{max_tokens}): {document.doc_id}"
                )
            chunk_id = f"{document.doc_id}::chunk_{ordinal:04d}"
            metadata = {
                "source_id": document.metadata["source_id"],
                "title": document.metadata["title"],
                "url": document.metadata["url"],
                "pmcid": document.metadata.get("pmcid"),
                "pmid": document.metadata.get("pmid"),
                "doi": document.metadata.get("doi"),
                "journal": document.metadata.get("journal"),
                "year": document.metadata.get("year"),
                "categories": list(document.metadata["categories"]),
                "priority": document.metadata["priority"],
                "license_status": document.metadata["license_status"],
            }
            sha256 = _sha256_text(
                _chunk_hash_payload(
                    doc_id=document.doc_id, section=section_title, text=piece, metadata=metadata
                )
            )
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    section=section_title,
                    text=piece,
                    token_count=token_count,
                    metadata=metadata,
                    sha256=sha256,
                )
            )
    validate_chunks(chunks, max_tokens=max_tokens)
    return chunks


def validate_chunks(chunks: Iterable[CorpusChunk], *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    materialized = list(chunks)
    if not materialized:
        raise PipelineError("Corpus RAG não gerou chunks.")
    seen: set[str] = set()
    for chunk in materialized:
        data = chunk.to_dict()
        missing = [key for key in REQUIRED_CHUNK_METADATA if data.get(key) in (None, "", [])]
        if missing:
            raise PipelineError(f"Chunk RAG {chunk.chunk_id} sem metadados obrigatórios: {missing}")
        if chunk.chunk_id in seen:
            raise PipelineError(f"Chunk RAG duplicado: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        if chunk.token_count != rough_token_count(chunk.text):
            raise PipelineError(f"Chunk RAG {chunk.chunk_id} com token_count inconsistente.")
        if chunk.token_count > max_tokens:
            raise PipelineError(f"Chunk RAG {chunk.chunk_id} excede max_tokens.")
        expected_sha = _sha256_text(
            _chunk_hash_payload(
                doc_id=chunk.doc_id, section=chunk.section, text=chunk.text, metadata=chunk.metadata
            )
        )
        if chunk.sha256 != expected_sha:
            raise PipelineError(f"Chunk RAG {chunk.chunk_id} com hash inconsistente.")
