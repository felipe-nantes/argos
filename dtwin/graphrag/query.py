"""CLI de consulta do metadata GraphRAG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dtwin.core import PipelineError

from .config import load_graphrag_config
from .context import build_metadata_graphrag_context
from .neo4j_store import Neo4jStore
from .schema import TARGET_FINDING


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consulta contexto metadata GraphRAG no Neo4j.")
    parser.add_argument("--config", required=True, help="Config YAML do GraphRAG Neo4j.")
    parser.add_argument("--negative-subtype", default=None)
    parser.add_argument("--phenotype-tag", default=None)
    parser.add_argument("--target", default=TARGET_FINDING)
    parser.add_argument("--limit", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_graphrag_config(Path(args.config))
        store = Neo4jStore(config)
        try:
            context = build_metadata_graphrag_context(
                store,
                negative_subtype=args.negative_subtype,
                phenotype_tag=args.phenotype_tag,
                target=args.target,
                limit=args.limit,
            )
        finally:
            store.close()
    except PipelineError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(context, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
