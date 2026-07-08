#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constrói o corpus normalizado/chunkado do RAG ARGOS.

F0 do RAG:
  1. ler docs/rag/corpus_manifest_v1.yaml;
  2. baixar ou carregar fontes aprovadas;
  3. extrair texto em Markdown simples;
  4. gerar chunks com metadados e hashes;
  5. emitir manifestos auditáveis.

Este script NÃO cria embeddings, NÃO constrói vector store e NÃO chama MedGemma.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import yaml

from dtwin.core import PipelineError, now_utc, sha256_of
from dtwin.rag.chunking import DEFAULT_MAX_TOKENS, CorpusDocument, chunk_document


SCHEMA_VERSION = "argos-rag-corpus-v1"
USER_AGENT = "ARGOS-RAG-CorpusBuilder/0.1 (research; local)"


class _MarkdownExtractor(HTMLParser):
    """Extrator HTML simples e determinístico para artigos/páginas abertas."""

    BLOCK_TAGS = {"p", "div", "section", "article", "br", "tr", "table", "blockquote"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "math"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._href_stack: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self._newline(double=True)
            self.parts.append("#" * min(level, 6) + " ")
        elif tag == "li":
            self._newline()
            self.parts.append("- ")
        elif tag in self.BLOCK_TAGS:
            self._newline()
        elif tag == "a":
            href = dict(attrs).get("href")
            self._href_stack.append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} | self.BLOCK_TAGS:
            self._newline(double=True)
        elif tag == "li":
            self._newline()
        elif tag == "a" and self._href_stack:
            self._href_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            if self.parts and self.parts[-1] and not self.parts[-1].endswith((" ", "\n", "# ", "- ")):
                self.parts.append(" ")
            self.parts.append(text)

    def _newline(self, *, double: bool = False) -> None:
        if not self.parts:
            return
        suffix = "\n\n" if double else "\n"
        joined_tail = "".join(self.parts[-2:])
        if not joined_tail.endswith(suffix):
            self.parts.append(suffix)

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n\s+(\n|$)", r"\n\1", text)
        return text.strip()


@dataclass(frozen=True)
class SourcePayload:
    bytes_data: bytes
    media_type: str
    source_kind: str


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"Manifesto do corpus não encontrado: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler manifesto do corpus {path}: {exc}") from exc
    articles = data.get("articles")
    if not isinstance(articles, list) or not articles:
        raise PipelineError("Manifesto do corpus precisa conter lista não vazia em articles.")
    return data


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    return cleaned.strip("._") or "source"


def _article_doc_id(article: dict[str, Any]) -> str:
    value = str(article.get("id") or article.get("pmcid") or "").strip()
    if not value:
        raise PipelineError("Artigo do corpus sem id/pmcid.")
    return _safe_name(value)


def _required_article_fields(article: dict[str, Any]) -> None:
    required = ("title", "url", "categories", "priority", "license_status")
    missing = [key for key in required if article.get(key) in (None, "", [])]
    if missing:
        ident = article.get("id") or article.get("pmcid") or "<sem id>"
        raise PipelineError(f"Artigo {ident} sem campos obrigatórios: {missing}")
    if not isinstance(article.get("categories"), list):
        raise PipelineError(f"Artigo {article.get('id')} precisa de categories como lista.")


def _source_path_from_article(article: dict[str, Any], manifest_dir: Path) -> Path | None:
    local = article.get("source_path") or article.get("local_path")
    if not local:
        return None
    path = Path(str(local))
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def _read_local_source(path: Path) -> SourcePayload:
    if not path.is_file():
        raise PipelineError(f"Fonte local do corpus não encontrada: {path}")
    data = path.read_bytes()
    suffix = path.suffix.lower()
    media_type = "text/html" if suffix in {".html", ".htm"} else "text/plain"
    return SourcePayload(data, media_type, "local")


def _fetch_url(url: str, *, timeout: int) -> SourcePayload:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return _read_local_source(Path(unquote(parsed.path)))
    if parsed.scheme not in {"http", "https"}:
        candidate = Path(url)
        if candidate.exists():
            return _read_local_source(candidate)
        raise PipelineError(f"URL/esquema de fonte RAG não suportado: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fontes aprovadas no manifesto
            media_type = response.headers.get_content_type() or "application/octet-stream"
            return SourcePayload(response.read(), media_type, "download")
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Falha ao baixar fonte RAG {url}: {exc}") from exc


def _load_or_fetch_source(
    article: dict[str, Any],
    *,
    manifest_dir: Path,
    sources_dir: Path,
    no_download: bool,
    timeout: int,
) -> tuple[SourcePayload, Path]:
    doc_id = _article_doc_id(article)
    cached = sorted(sources_dir.glob(f"{doc_id}.*"))
    if cached:
        path = cached[0]
        return _read_local_source(path), path

    local_path = _source_path_from_article(article, manifest_dir)
    if local_path is not None:
        payload = _read_local_source(local_path)
        suffix = local_path.suffix or (".html" if payload.media_type == "text/html" else ".txt")
        target = sources_dir / f"{doc_id}{suffix}"
        target.write_bytes(payload.bytes_data)
        return payload, target

    if no_download:
        raise PipelineError(f"Fonte RAG ausente em cache e --no-download ativo: {doc_id}")

    payload = _fetch_url(str(article["url"]), timeout=timeout)
    suffix = ".html" if payload.media_type == "text/html" else ".txt"
    target = sources_dir / f"{doc_id}{suffix}"
    target.write_bytes(payload.bytes_data)
    return payload, target


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _strip_boilerplate(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    skip_prefixes = (
        "pubmed disclaimer",
        "copyright",
        "conflict of interest",
        "competing interests",
        "references",
        "acknowledgments",
        "acknowledgements",
    )
    skipping = False
    for line in lines:
        normalized = line.strip().strip("#").strip().lower()
        if normalized in skip_prefixes or any(normalized.startswith(prefix + " ") for prefix in skip_prefixes):
            skipping = True
            continue
        if skipping and line.startswith("#"):
            skipping = False
        if not skipping:
            out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_markdown(payload: SourcePayload) -> str:
    text = _decode_bytes(payload.bytes_data)
    is_html = payload.media_type == "text/html" or bool(re.search(r"<\s*(html|article|body|p|h1)\b", text, re.I))
    if is_html:
        parser = _MarkdownExtractor()
        parser.feed(text)
        markdown = parser.markdown()
    else:
        markdown = text
    markdown = _strip_boilerplate(markdown)
    if len(markdown.strip()) < 100:
        raise PipelineError("Texto extraído de fonte RAG é curto demais para ingestão.")
    return markdown


def _front_matter(article: dict[str, Any]) -> str:
    fields = {
        "id": article.get("id"),
        "pmcid": article.get("pmcid"),
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        "url": article.get("url"),
        "title": article.get("title"),
        "journal": article.get("journal"),
        "year": article.get("year"),
        "priority": article.get("priority"),
        "categories": article.get("categories"),
        "license_status": article.get("license_status"),
    }
    return "---\n" + yaml.safe_dump(fields, allow_unicode=True, sort_keys=False).strip() + "\n---\n\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, data: Any) -> None:
    _write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def _doc_metadata(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(article.get("pmcid") or article["id"]),
        "title": str(article["title"]),
        "url": str(article["url"]),
        "pmcid": article.get("pmcid"),
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        "journal": article.get("journal"),
        "year": article.get("year"),
        "categories": list(article["categories"]),
        "priority": str(article["priority"]),
        "license_status": str(article["license_status"]),
    }


def build_corpus(
    *,
    manifest_path: Path,
    out_dir: Path,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = 40,
    no_download: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    out_dir = out_dir.resolve()
    source_manifest = _read_manifest(manifest_path)
    corpus_version = str(source_manifest.get("corpus_version") or out_dir.name)

    sources_dir = out_dir / "sources"
    normalized_dir = out_dir / "normalized"
    chunks_dir = out_dir / "chunks"
    for directory in (sources_dir, normalized_dir, chunks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    article_results: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    all_chunks = []

    for article in source_manifest["articles"]:
        _required_article_fields(article)
        doc_id = _article_doc_id(article)
        payload, raw_path = _load_or_fetch_source(
            article,
            manifest_dir=manifest_path.parent,
            sources_dir=sources_dir,
            no_download=no_download,
            timeout=timeout,
        )
        markdown_body = extract_markdown(payload)
        normalized_path = normalized_dir / f"{doc_id}.md"
        _write_text_atomic(normalized_path, _front_matter(article) + markdown_body + "\n")

        document = CorpusDocument(doc_id=doc_id, text=markdown_body, metadata=_doc_metadata(article))
        chunks = chunk_document(document, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for chunk in chunks:
            chunk_path = chunks_dir / f"{chunk.chunk_id.replace('::', '__')}.json"
            chunk_dict = chunk.to_dict()
            _write_json_atomic(chunk_path, chunk_dict)
            chunk_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "token_count": chunk.token_count,
                    "sha256": chunk.sha256,
                    "path": str(chunk_path.relative_to(out_dir)),
                    "categories": chunk.metadata["categories"],
                    "priority": chunk.metadata["priority"],
                }
            )
            all_chunks.append(chunk)

        article_results.append(
            {
                "id": doc_id,
                "title": article["title"],
                "url": article["url"],
                "pmcid": article.get("pmcid"),
                "doi": article.get("doi"),
                "priority": article["priority"],
                "categories": article["categories"],
                "raw_path": str(raw_path.relative_to(out_dir)),
                "raw_sha256": sha256_of(raw_path),
                "normalized_path": str(normalized_path.relative_to(out_dir)),
                "normalized_sha256": sha256_of(normalized_path),
                "chunk_count": len(chunks),
                "token_count": sum(chunk.token_count for chunk in chunks),
            }
        )

    category_counts: dict[str, int] = {}
    for chunk in all_chunks:
        for category in chunk.metadata["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1

    build_manifest = {
        "schema": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "corpus_version": corpus_version,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_of(manifest_path),
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "article_count": len(article_results),
        "chunk_count": len(chunk_results),
        "category_counts": dict(sorted(category_counts.items())),
        "articles": article_results,
        "chunks": chunk_results,
    }
    _write_json_atomic(out_dir / "manifest.json", build_manifest)
    _write_json_atomic(out_dir / "chunks_manifest.json", {"chunks": chunk_results})
    return build_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constrói o corpus RAG normalizado/chunkado do ARGOS.")
    parser.add_argument("--manifest", default="docs/rag/corpus_manifest_v1.yaml")
    parser.add_argument("--out", default="rag/corpus/liver_mri_rag_v1")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--no-download", action="store_true", help="Usa apenas fontes já em cache/local_path.")
    parser.add_argument("--clean", action="store_true", help="Remove a pasta de saída antes do build.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    try:
        if args.clean and out_dir.exists():
            resolved = out_dir.resolve()
            cwd = Path.cwd().resolve()
            try:
                resolved.relative_to(cwd)
            except ValueError as exc:
                raise PipelineError(f"--clean recusado fora do workspace: {resolved}") from exc
            shutil.rmtree(resolved)
        manifest = build_corpus(
            manifest_path=Path(args.manifest),
            out_dir=out_dir,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            no_download=args.no_download,
            timeout=args.timeout,
        )
    except PipelineError as exc:
        print(f"[ABORTADO] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "complete",
        "corpus_version": manifest["corpus_version"],
        "article_count": manifest["article_count"],
        "chunk_count": manifest["chunk_count"],
        "out": str(out_dir),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
