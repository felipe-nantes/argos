"""CLI auditável do benchmark MedGemma, compartilhando o núcleo com o webapp."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dtwin.benchmark.hashing import git_state
from dtwin.benchmark.importers import load_dataset_manifest, validate_inference_source
from dtwin.benchmark.runner import load_experiment_config, recalculate_existing_run, run_benchmark
from dtwin.core import PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MedGemma do ARGOS (modo Pesquisa).")
    parser.add_argument("--datasets-manifest", required=True)
    parser.add_argument("--medgemma-config", required=True)
    parser.add_argument("--experiment-config")
    parser.add_argument("--out", default="casos/webapp/benchmarks/runs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-inference-use-existing-reports", action="store_true")
    parser.add_argument("--existing-run", help="run anterior validado para reuso de relatórios")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_dataset_manifest(Path(args.datasets_manifest))
        if args.dataset:
            selected = set(args.dataset)
            cases = [case for case in cases if case.inference.dataset in selected]
        if args.case_id:
            selected_ids = set(args.case_id)
            cases = [case for case in cases if case.inference.case_id in selected_ids]
        if args.limit is not None:
            if args.limit <= 0:
                raise PipelineError("--limit deve ser positivo.")
            cases = cases[: args.limit]
        if not cases:
            raise PipelineError("Nenhum caso selecionado.")
        readiness = [validate_inference_source(case.inference) for case in cases]
        if args.dry_run:
            print(json.dumps({
                "status": "ready", "research_only": True, "cases": readiness,
                "git": git_state(Path.cwd()), "inference_called": False,
            }, indent=2, ensure_ascii=False, default=str))
            return 0
        experiment_path = Path(args.experiment_config) if args.experiment_config else None
        experiment = load_experiment_config(experiment_path)
        common = dict(
            cases=cases, repo=Path.cwd(), out_root=Path(args.out),
            medgemma_config=Path(args.medgemma_config),
            experiment_config_path=experiment_path, experiment=experiment, seed=args.seed,
        )
        if args.skip_inference_use_existing_reports:
            if not args.existing_run:
                raise PipelineError("--skip-inference-use-existing-reports exige --existing-run.")
            run_dir, metrics, _ = recalculate_existing_run(
                existing_run=Path(args.existing_run), **common,
            )
        else:
            run_dir, metrics, _ = run_benchmark(fail_fast=args.fail_fast, **common)
    except PipelineError as exc:
        print(f"[ABORTADO] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "gate": metrics["gate"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
