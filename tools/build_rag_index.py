#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constrói o índice lexical inicial do RAG ARGOS.

Esta etapa consome o corpus já normalizado/chunkado por `build_rag_corpus.py`.
Ela cria apenas BM25 local e manifesto auditável. Embeddings/MedCPT entram em uma
etapa posterior.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from dtwin.core import PipelineError
from dtwin.rag.index import DEFAULT_B, DEFAULT_K1, build_bm25_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constrói índice BM25 local do corpus RAG ARGOS.")
    parser.add_argument("--corpus", default="rag/corpus/liver_mri_rag_v1")
    parser.add_argument("--out", default="rag/index/liver_mri_rag_v1")
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--clean", action="store_true", help="Remove a pasta de índice antes do build.")
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
        manifest = build_bm25_index(
            corpus_dir=Path(args.corpus),
            out_dir=out_dir,
            k1=args.k1,
            b=args.b,
        )
    except PipelineError as exc:
        print(f"[ABORTADO] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "complete",
        "index_type": manifest["index_type"],
        "corpus_version": manifest["corpus_version"],
        "chunk_count": manifest["chunk_count"],
        "vocabulary_size": manifest["vocabulary_size"],
        "out": str(out_dir),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
