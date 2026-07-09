"""CLI para ingerir manifestos do dataset registry no Neo4j."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from dtwin.core import PipelineError

from .config import load_graphrag_config
from .neo4j_store import Neo4jStore
from .schema import validate_registry_record


def iter_registry_records(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise PipelineError(f"Falha ao ler manifesto registry: {path}") from exc
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"JSONL inválido em {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise PipelineError(f"Registro registry deve ser objeto em {path}:{line_no}")
            validate_registry_record(record)
            yield record


def ingest_records(store, records: Iterable[dict]) -> int:
    store.ensure_schema()
    count = 0
    for record in records:
        store.upsert_registry_record(record)
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingere registry JSONL no GraphRAG Neo4j.")
    parser.add_argument("--config", required=True, help="Config YAML do GraphRAG Neo4j.")
    parser.add_argument("--manifests", nargs="+", required=True, help="Um ou mais JSONL do dataset registry.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_graphrag_config(Path(args.config))
        store = Neo4jStore(config)
        try:
            count = ingest_records(store, iter_registry_records(Path(item) for item in args.manifests))
        finally:
            store.close()
    except PipelineError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2
    print(f"[OK] {count} registros ingeridos no GraphRAG Neo4j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
