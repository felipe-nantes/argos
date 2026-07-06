"""Runner compartilhável: inferência isolada, avaliação posterior e relatórios."""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from dtwin.core import PipelineError, sha256_of

from .hashing import git_state
from .importers import DatasetCase, attach_ground_truth, prepare_inference_case, validate_inference_source
from .metrics import compute_benchmark_metrics
from .models import BenchmarkCaseResult, BenchmarkStatus, ModelResult
from .reporting import write_run_outputs


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "current_panel"
    profile_path: Path = Path("profiles/figado.yaml")
    minimum_sensitivity: float = 0.75
    minimum_specificity: float = 0.75
    timeout_seconds: int = 900
    segment_if_missing: bool = False
    device: str = "gpu"
    visible_phi_confirmed: bool = False
    final_evaluation: bool = False


def load_experiment_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Config experimental inválida: {exc}") from exc
    gate = data.get("gate") or {}
    return ExperimentConfig(
        name=str(data.get("name") or "current_panel"),
        profile_path=Path(data.get("profile_path") or "profiles/figado.yaml"),
        minimum_sensitivity=float(gate.get("minimum_sensitivity", 0.75)),
        minimum_specificity=float(gate.get("minimum_specificity", 0.75)),
        timeout_seconds=int(data.get("timeout_seconds", 900)),
        segment_if_missing=bool(data.get("segment_if_missing", False)),
        device=str(data.get("device") or "gpu"),
        visible_phi_confirmed=bool(data.get("visible_phi_confirmed", False)),
        final_evaluation=bool(data.get("final_evaluation", False)),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:48] or "experiment"


def make_run_id(commit: str | None, experiment: str, now: datetime | None = None) -> str:
    instant = now or datetime.now(timezone.utc)
    stamp = instant.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{(commit or 'nogit')[:8]}_{_slug(experiment)}"


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Config MedGemma inválida: {exc}") from exc
    screening = config.get("medgemma_screening") or {}
    model = screening.get("medgemma") or {}
    if not model.get("model_id"):
        raise PipelineError("Config MedGemma sem model_id.")
    return screening, model


def _environment() -> dict[str, Any]:
    cuda = False
    device = "cpu"
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            device = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": cuda,
        "device": device,
    }


def build_run_manifest(
    *,
    run_id: str,
    repo: Path,
    medgemma_config: Path,
    experiment_config_path: Path | None,
    experiment: ExperimentConfig,
    model: dict[str, Any],
    cases: Iterable[DatasetCase],
    started_at: str,
    seed: int = 42,
) -> dict[str, Any]:
    cases = list(cases)
    state = git_state(repo)
    if experiment.final_evaluation and state["git_dirty"]:
        raise PipelineError("Avaliação final exige árvore Git limpa.")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": started_at,
        **state,
        "model_family": "MedGemma",
        "model_id": model.get("model_id"),
        "model_version": model.get("model_version"),
        "model_parameter_scale": model.get("model_parameter_scale"),
        "runtime": model.get("runtime", "transformers"),
        "medgemma_config_path": str(Path(medgemma_config)),
        "medgemma_config_hash": sha256_of(Path(medgemma_config)),
        "experiment_config_path": str(experiment_config_path) if experiment_config_path else None,
        "experiment_config_hash": sha256_of(experiment_config_path) if experiment_config_path else None,
        "experimental_strategy": experiment.name,
        "dataset_names": sorted({case.inference.dataset for case in cases}),
        "num_cases_total": len(cases),
        "num_cases_positive": sum(case.ground_truth.label.value == "positive" for case in cases),
        "num_cases_negative": sum(case.ground_truth.label.value == "negative" for case in cases),
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds_total": None,
        "seed": seed,
        "environment": _environment(),
        "research_only": True,
    }


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return Path(path).name


def classify_screening_failure(message: str) -> BenchmarkStatus:
    normalized = (message or "").lower()
    invalid_markers = (
        "resposta medgemma", "resposta do backend", "objeto json", "json válido",
        "estado medgemma inválido", "confiança medgemma inválida", "schema",
    )
    return (
        BenchmarkStatus.INVALID_RESPONSE
        if any(marker in normalized for marker in invalid_markers)
        else BenchmarkStatus.FAILURE
    )


def run_case(
    case: DatasetCase,
    *,
    run_dir: Path,
    medgemma_config: Path,
    experiment: ExperimentConfig,
    python_executable: str = sys.executable,
) -> BenchmarkCaseResult:
    case_started = time.monotonic()
    import_started = time.monotonic()
    workspace = run_dir / "workspaces" / _slug(f"{case.inference.dataset}-{case.inference.case_id}") / "inference"
    try:
        inference = prepare_inference_case(
            case.inference, workspace, profile_path=experiment.profile_path,
            segment_if_missing=experiment.segment_if_missing, device=experiment.device,
        )
    except Exception as exc:  # noqa: BLE001
        return BenchmarkCaseResult(
            case_id=case.inference.case_id, dataset=case.inference.dataset,
            input_format=case.inference.input_format, truth=case.ground_truth.label,
            status=BenchmarkStatus.FAILURE, error_type=type(exc).__name__,
            error_message=str(exc), durations_seconds={"import": round(time.monotonic() - import_started, 4), "total": round(time.monotonic() - case_started, 4)},
        )
    import_duration = time.monotonic() - import_started
    output = inference.workspace / "outputs" / "medgemma"
    command = [
        python_executable, "-m", "dtwin.medgemma_screening",
        "--volume", str(inference.volume_path),
        "--liver-mask", str(inference.organ_mask_path),
        "--case-manifest", str(inference.manifest_path),
        "--output", str(output),
        "--profile", str(experiment.profile_path),
        "--medgemma-config", str(medgemma_config),
    ]
    if experiment.visible_phi_confirmed:
        command.append("--confirm-no-visible-phi")
    screening_started = time.monotonic()
    status = BenchmarkStatus.FAILURE
    prediction = None
    confidence = None
    error_type = None
    error_message = None
    screening_timings: dict[str, float | None] = {}
    try:
        process = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[2], capture_output=True,
            text=True, timeout=experiment.timeout_seconds, check=False,
        )
        report_path = output / "medgemma_report.json"
        if process.returncode != 0 or not report_path.is_file():
            error_type = "ScreeningFailure"
            error_message = (process.stderr or process.stdout or "Triagem não gerou relatório.")[-2000:]
            status = classify_screening_failure(error_message)
        else:
            try:
                envelope = json.loads(report_path.read_text(encoding="utf-8"))
                report = envelope.get("report") or {}
                screening_timings = dict(envelope.get("durations_seconds") or {})
                prediction = ModelResult(str(report.get("resultado_hipotese") or "").upper())
                confidence = report.get("confianca")
                status = BenchmarkStatus.INCONCLUSIVE if prediction is ModelResult.INCONCLUSIVE else BenchmarkStatus.DECISIVE
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                status = BenchmarkStatus.INVALID_RESPONSE
                error_type, error_message = type(exc).__name__, str(exc)
    except subprocess.TimeoutExpired as exc:
        status = BenchmarkStatus.TIMEOUT
        error_type, error_message = type(exc).__name__, "Tempo limite excedido."
    screening_duration = time.monotonic() - screening_started
    # Somente agora o avaliador toca label/anotações protegidas.
    evaluation_started = time.monotonic()
    evaluation = attach_ground_truth(case, inference)
    report_path = output / "medgemma_report.json"
    panel_candidates = list(output.glob("*.png")) if output.is_dir() else []
    hashes = dict(inference.input_hashes)
    if panel_candidates:
        hashes["panel"] = sha256_of(panel_candidates[0])
    result = BenchmarkCaseResult(
        case_id=inference.case_id, dataset=inference.dataset, input_format=inference.input_format,
        truth=evaluation.label, status=status, prediction=prediction, confidence=confidence,
        input_hashes=hashes, protected_ground_truth_hashes=evaluation.protected_ground_truth_hashes,
        durations_seconds={
            "import": round(import_duration, 4),
            "panel_generation": screening_timings.get("panel_generation"),
            "backend_readiness": screening_timings.get("backend_readiness"),
            "medgemma_inference": screening_timings.get("medgemma_inference"),
            "response_validation": screening_timings.get("response_validation"),
            "screening_total": screening_timings.get("screening_total", round(screening_duration, 4)),
            "evaluation": round(time.monotonic() - evaluation_started, 4),
            "total": round(time.monotonic() - case_started, 4),
        },
        error_type=error_type, error_message=error_message,
        report_path=_relative(report_path if report_path.is_file() else None, run_dir),
        panel_path=_relative(panel_candidates[0] if panel_candidates else None, run_dir),
        extra={},
    )
    return result


def run_benchmark(
    cases: list[DatasetCase],
    *,
    repo: Path,
    out_root: Path,
    medgemma_config: Path,
    experiment_config_path: Path | None,
    experiment: ExperimentConfig,
    fail_fast: bool = False,
    seed: int = 42,
) -> tuple[Path, dict[str, Any], list[BenchmarkCaseResult]]:
    started = datetime.now(timezone.utc)
    _, model = _load_model_config(medgemma_config)
    state = git_state(repo)
    run_id = make_run_id(state.get("code_commit"), experiment.name, started)
    run_dir = Path(out_root).resolve() / run_id
    if run_dir.exists():
        raise PipelineError(f"Run já existe: {run_dir}")
    manifest = build_run_manifest(
        run_id=run_id, repo=repo, medgemma_config=medgemma_config,
        experiment_config_path=experiment_config_path, experiment=experiment,
        model=model, cases=cases, started_at=started.isoformat(), seed=seed,
    )
    run_dir.mkdir(parents=True)
    results: list[BenchmarkCaseResult] = []
    for case in cases:
        result = run_case(
            case, run_dir=run_dir, medgemma_config=medgemma_config,
            experiment=experiment,
        )
        results.append(result)
        if fail_fast and result.status not in {BenchmarkStatus.DECISIVE, BenchmarkStatus.INCONCLUSIVE}:
            break
    metrics = compute_benchmark_metrics(
        results, minimum_sensitivity=experiment.minimum_sensitivity,
        minimum_specificity=experiment.minimum_specificity,
    )
    finished = datetime.now(timezone.utc)
    manifest.update(
        finished_at=finished.isoformat(),
        duration_seconds_total=round((finished - started).total_seconds(), 4),
        num_cases_processed=len(results),
    )
    write_run_outputs(run_dir, manifest, results, metrics)
    return run_dir, metrics, results


def recalculate_existing_run(
    cases: list[DatasetCase],
    *,
    existing_run: Path,
    repo: Path,
    out_root: Path,
    medgemma_config: Path,
    experiment_config_path: Path | None,
    experiment: ExperimentConfig,
    seed: int = 42,
) -> tuple[Path, dict[str, Any], list[BenchmarkCaseResult]]:
    """Recalcula métricas sem inferência, após validar identidade e hashes."""
    existing_run = Path(existing_run).resolve()
    try:
        source_manifest = json.loads((existing_run / "run_manifest.json").read_text("utf-8"))
        source_rows = [
            json.loads(line) for line in (existing_run / "cases.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Run existente inválido: {exc}") from exc
    _, model = _load_model_config(medgemma_config)
    checks = {
        "model_id": model.get("model_id"),
        "model_version": model.get("model_version"),
        "medgemma_config_hash": sha256_of(medgemma_config),
        "experimental_strategy": experiment.name,
    }
    mismatches = [key for key, expected in checks.items() if source_manifest.get(key) != expected]
    if mismatches:
        raise PipelineError(f"Run existente incompatível em: {', '.join(mismatches)}")
    source_by_key = {(str(row.get("dataset") or "unknown"), str(row.get("case_id"))): row for row in source_rows}
    started = datetime.now(timezone.utc)
    state = git_state(repo)
    run_id = make_run_id(state.get("code_commit"), f"{experiment.name}-recalculated", started)
    run_dir = Path(out_root).resolve() / run_id
    manifest = build_run_manifest(
        run_id=run_id, repo=repo, medgemma_config=medgemma_config,
        experiment_config_path=experiment_config_path, experiment=experiment,
        model=model, cases=cases, started_at=started.isoformat(), seed=seed,
    )
    manifest.update(
        source_run_id=source_manifest.get("run_id"),
        source_code_commit=source_manifest.get("code_commit"),
        inference_reused=True,
    )
    run_dir.mkdir(parents=True)
    results: list[BenchmarkCaseResult] = []
    for case in cases:
        key = (case.inference.dataset, case.inference.case_id)
        prior = source_by_key.get(key)
        if prior is None:
            raise PipelineError(f"Caso ausente no run reutilizado: {key[0]}/{key[1]}")
        inference = prepare_inference_case(
            case.inference, run_dir / "workspaces" / _slug(f"{key[0]}-{key[1]}") / "inference",
            profile_path=experiment.profile_path,
            segment_if_missing=experiment.segment_if_missing,
            device=experiment.device,
        )
        if dict(prior.get("input_hashes") or {}) != inference.input_hashes:
            raise PipelineError(f"Hashes divergentes ao reutilizar {key[0]}/{key[1]}")
        if str(prior.get("truth") or prior.get("ground_truth_label") or "").lower() != case.ground_truth.label.value:
            raise PipelineError(f"Label divergente ao reutilizar {key[0]}/{key[1]}")
        result = BenchmarkCaseResult.from_mapping(prior)
        evaluation = attach_ground_truth(case, inference)
        result.input_hashes = inference.input_hashes
        result.protected_ground_truth_hashes = evaluation.protected_ground_truth_hashes
        result.extra.update(source_run_id=source_manifest.get("run_id"), inference_reused=True)
        results.append(result)
    metrics = compute_benchmark_metrics(
        results, minimum_sensitivity=experiment.minimum_sensitivity,
        minimum_specificity=experiment.minimum_specificity,
    )
    finished = datetime.now(timezone.utc)
    manifest.update(
        finished_at=finished.isoformat(),
        duration_seconds_total=round((finished - started).total_seconds(), 4),
        num_cases_processed=len(results),
    )
    write_run_outputs(run_dir, manifest, results, metrics)
    return run_dir, metrics, results
