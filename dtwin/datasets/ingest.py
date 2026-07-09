"""CLI para gerar manifestos JSONL do registry de datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dtwin.core import PipelineError

from .registry import ingest_dataset_config, load_dataset_config, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingere dataset hepático para registry JSONL seguro.")
    parser.add_argument("--config", required=True, help="Config YAML do dataset.")
    parser.add_argument("--root", required=True, help="Raiz local do dataset bruto.")
    parser.add_argument("--out", required=True, help="Arquivo JSONL de saída.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_dataset_config(Path(args.config))
        records = ingest_dataset_config(config, Path(args.root))
        write_jsonl(records, Path(args.out))
    except PipelineError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2
    print(f"[OK] {len(records)} registros escritos em {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
