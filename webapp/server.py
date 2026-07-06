#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Webapp de orquestração (modo Pesquisa/demo).

Fluxo, invisível para o usuário: recebe uma pasta DICOM de RM -> des-identifica ->
segmenta o fígado -> gera a montagem 2D -> chama o MedGemma -> devolve um relatório
simples. É "à prova de falhas": qualquer etapa que falhe produz um cartão gracioso
e honesto ("análise não concluída") — NUNCA um achado clínico fabricado.

Robustez: o pipeline pesado (TotalSegmentator com torch/CUDA, e a triagem MedGemma)
roda em SUBPROCESSOS isolados. Assim, mesmo um crash nativo (segfault, OOM de CUDA)
não derruba o servidor web — o subprocesso retorna erro e o job vira um cartão
gracioso. O servidor permanece sempre responsivo.

Aviso: para a experiência hands-off do demo, a confirmação humana de PHI queimada
nos pixels é auto-assumida. Isto é aceitável apenas em modo Pesquisa/demonstração;
o uso clínico exige a revisão humana real do painel.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import pydicom
import SimpleITK as sitk
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.datastructures import FormData

from dtwin.benchmark.metrics import compute_benchmark_metrics
from dtwin.benchmark.hashing import git_state
from dtwin.benchmark.reporting import write_run_outputs
from dtwin.benchmark.runner import classify_screening_failure
from dtwin.core import sha256_of

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("dtwin.webapp")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STATIC = ROOT / "static"
VIEWER = REPO / "viewer"
WORKSPACE = Path("casos/webapp")
PROFILE = "profiles/figado.yaml"
MEDGEMMA_CONFIG = os.environ.get("WEBAPP_MEDGEMMA_CONFIG", "configs/medgemma_local_4b.yaml")
HEALTH_URL = os.environ.get("WEBAPP_MEDGEMMA_HEALTH", "http://127.0.0.1:8001/health")
MIN_SLICES = 3
PREP_TIMEOUT_GPU = int(os.environ.get("WEBAPP_PREP_TIMEOUT_GPU", "900"))
PREP_TIMEOUT_CPU = int(os.environ.get("WEBAPP_PREP_TIMEOUT_CPU", "2400"))
SCREEN_TIMEOUT = int(os.environ.get("WEBAPP_SCREEN_TIMEOUT", "600"))
MODEL_TIMEOUT = int(os.environ.get("WEBAPP_MODEL_TIMEOUT", "300"))
# O Starlette limita uploads multipart a 1000 arquivos por padrão (proteção
# genérica contra DoS). Um dataset de benchmark real (muitos exames, cada um com
# centenas/milhares de fatias DICOM) estoura isso com facilidade. O servidor só
# escuta em loopback (uso local de pesquisa), então um teto bem mais alto — mas
# ainda explícito, nunca ilimitado — é seguro aqui.
MAX_UPLOAD_FILES = int(os.environ.get("WEBAPP_MAX_UPLOAD_FILES", "50000"))
DISCLAIMER = (
    "Uso em pesquisa. Não destinado à decisão clínica. Não é diagnóstico nem "
    "laudo médico. Revisão médica obrigatória."
)

PY = sys.executable
# A segmentação roda por um launcher a partir de um diretório TEMPORÁRIO (fora do
# OneDrive/repo), senão os workers spawnados do nnU-Net falham no Windows (ver
# webapp/seg_worker.py). Instalamos uma cópia do launcher no %TEMP% na subida.
_SEG_DIR = Path(tempfile.gettempdir()) / "dtwin_webapp"
_SEG_LAUNCHER = _SEG_DIR / "seg_worker.py"


def _install_seg_launcher() -> None:
    try:
        _SEG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "seg_worker.py", _SEG_LAUNCHER)
    except Exception:  # noqa: BLE001
        log.exception("Não foi possível instalar o launcher de segmentação")


_jobs: dict[str, dict] = {}
_benchmarks: dict[str, dict] = {}
_lock = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(**kw)


def _set_benchmark(benchmark_id: str, **kw) -> None:
    with _lock:
        if benchmark_id in _benchmarks:
            _benchmarks[benchmark_id].update(**kw)


def _graceful(motivo: str, detalhe: str = "") -> dict:
    return {
        "status": "nao_concluido",
        "titulo": "Análise não concluída",
        "motivo": motivo,
        "detalhe": detalhe,
        "requires_human_review": True,
        "disclaimer": DISCLAIMER,
    }


def _friendly_text(reason: str) -> str:
    s = (reason or "").lower()
    if any(t in s for t in ("backend not configured", "inacessível", "não está disponível", "não configurado")):
        return "O serviço de análise (MedGemma) não está ativo no momento."
    if "modalidade" in s:
        return "O exame enviado não parece ser uma RM compatível."
    if "poucas fatias" in s or "3d inviável" in s or "fatias axiais" in s:
        return "O exame enviado tem cortes insuficientes para a análise."
    if any(t in s for t in ("segmenta", "fígado", "liver", "totalsegmentator")):
        return "Não foi possível segmentar o fígado neste exame."
    if "diagnóstico definitivo" in s or "conduta" in s:
        return "A resposta do modelo não passou na verificação de segurança."
    if "phi" in s:
        return "O exame não passou na verificação de privacidade."
    return "Não foi possível concluir a análise deste exame."


def _friendly(err: Exception) -> str:  # conveniência p/ testes
    return _friendly_text(str(err))


def _expected_modalities() -> set[str]:
    """Modalidades aceitas pelo perfil ativo (ex.: {'MR','MRI'} para o fígado)."""
    try:
        prof = yaml.safe_load((REPO / PROFILE).read_text("utf-8")) or {}
        return {str(m).upper() for m in (prof.get("modalidade") or [])}
    except Exception:  # noqa: BLE001
        return {"MR", "MRI"}


def _modality_of(names: list[str]) -> str:
    """Lê a Modality (0008,0060) do primeiro arquivo legível da série."""
    for name in names[:5]:
        try:
            ds = pydicom.dcmread(name, stop_before_pixels=True, force=True)
        except Exception:  # noqa: BLE001
            continue
        modality = str(getattr(ds, "Modality", "") or "").upper()
        if modality:
            return modality
    return ""


def _modality_ok(names: list[str], expected: set[str]) -> bool:
    """Aceita se a modalidade bate com o perfil (ou é desconhecida — o gate do
    stage1 decide). REJEITA modalidade conhecida e incompatível (ex.: CT quando o
    perfil é MR), para não escolher a série errada num envio misto CT+MR."""
    modality = _modality_of(names)
    return not modality or not expected or modality in expected


def find_best_series(root: Path) -> tuple[list[str], int]:
    """Acha a maior série DICOM COMPATÍVEL com o perfil, em qualquer estrutura:

    1) enumera TODAS as séries de cada diretório (uma pasta pode ter várias séries);
    2) filtra por modalidade (o perfil do fígado é MR): num envio misto CT+MR,
       ignora as séries CT em vez de escolher a maior e abortar no gate do stage1;
    3) se nenhuma série multi-arquivo servir, tenta um único DICOM **multi-frame**
       (um só `.dcm` que já é o volume 3D inteiro), medindo a profundidade real.

    Retorna (lista_de_arquivos_da_melhor_série, nº_de_cortes)."""
    reader = sitk.ImageSeriesReader()
    expected = _expected_modalities()
    best_files: list[str] = []
    for dirpath, _dirs, _files in os.walk(root):
        try:
            series_ids = list(reader.GetGDCMSeriesIDs(dirpath)) or [""]
        except Exception:  # noqa: BLE001
            series_ids = [""]
        for sid in series_ids:
            try:
                names = (reader.GetGDCMSeriesFileNames(dirpath, sid) if sid
                         else reader.GetGDCMSeriesFileNames(dirpath))
            except Exception:  # noqa: BLE001
                names = []
            if len(names) <= len(best_files):
                continue
            if not _modality_ok(list(names), expected):
                continue
            best_files = list(names)
    if len(best_files) >= MIN_SLICES:
        return best_files, len(best_files)

    # Fallback: DICOM multi-frame (um arquivo = volume inteiro) ou série de 1 arquivo.
    best_file, best_depth = None, 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            try:
                img = sitk.ReadImage(path)
            except Exception:  # noqa: BLE001
                continue
            depth = img.GetSize()[2] if img.GetDimension() >= 3 else 1
            if depth <= best_depth or not _modality_ok([path], expected):
                continue
            best_file, best_depth = path, depth
    if best_file and best_depth >= MIN_SLICES:
        return [best_file], best_depth
    return best_files, len(best_files)


def _run(cmd: list[str], timeout: int, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd or str(REPO), capture_output=True, text=True, timeout=timeout)


def _segment(series_dir: str, case_dir: Path, device: str, timeout: int) -> subprocess.CompletedProcess:
    """Roda a segmentação pelo launcher, a partir do %TEMP% (fora do OneDrive)."""
    prof = str((REPO / PROFILE).resolve())
    return _run([PY, str(_SEG_LAUNCHER), str(REPO), prof, series_dir, str(case_dir), device],
                timeout=timeout, cwd=str(_SEG_DIR))


def _cli_reason(proc: subprocess.CompletedProcess) -> str:
    """Extrai a razão impressa ('[ABORTADO] ...' ou 'PREP_FAIL: ...'); senão o fim do log."""
    for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines():
        if "[ABORTADO]" in line:
            return line.split("[ABORTADO]", 1)[1].strip()
        if "PREP_FAIL:" in line:
            return line.split("PREP_FAIL:", 1)[1].strip()
    tail = (proc.stderr or proc.stdout or "").strip()
    return tail[-300:] if tail else f"código de saída {proc.returncode}"


def _seg_done(case_dir: Path) -> bool:
    """Segmentação concluída = volume + máscara existem (o returncode é ignorado:
    libs nativas podem crashar no shutdown no Windows APÓS gravar os artefatos)."""
    return (case_dir / "mask_organ.nii.gz").is_file() and (case_dir / "volume.nii.gz").is_file()


def _success_result(report: dict) -> dict:
    """Monta o resultado de sucesso para o frontend.

    IMPORTANTE: o envelope do relatório tem sua própria chave 'status'
    ('pending_review'); ela NÃO pode sobrescrever o marcador de conclusão que o
    frontend usa. Por isso 'status'='concluido' é aplicado por ÚLTIMO."""
    return {**report, "status": "concluido"}


def _viewer_result(report: dict, job_id: str, viewer_ready: bool) -> dict:
    """Acrescenta a visualizacao sem alterar o contrato do relatorio MedGemma."""
    result = _success_result(report)
    result.update(
        viewer_ready=bool(viewer_ready),
        viewer_url=(
            f"/viewer/index.html?case=/api/jobs/{job_id}/model&job={job_id}"
            if viewer_ready
            else None
        ),
        approval={"status": "pending"} if viewer_ready else None,
    )
    return result


def _load_report(path: Path) -> dict | None:
    """Relatório válido = sucesso, independentemente do returncode do subprocesso.
    run_screening grava o JSON atomicamente só após validá-lo, então a existência de
    um relatório com 'resultado_hipotese' significa que a triagem concluiu de fato."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    report = data.get("report")
    if not isinstance(report, dict) or not report.get("resultado_hipotese"):
        return None
    return data


def _model_done(case_dir: Path) -> bool:
    """Modelo publicavel = manifesto valido e todos os STLs presentes."""
    manifest_path = case_dir / "outputs" / "viewer_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
        meshes = manifest.get("meshes", [])
        return bool(meshes) and all(
            isinstance(item, dict)
            and Path(str(item.get("stl", ""))).name == item.get("stl")
            and (manifest_path.parent / item["stl"]).is_file()
            for item in meshes
        )
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return False


def _build_model(case_dir: Path) -> tuple[bool, str]:
    """Gera a malha do figado em subprocesso, sem inventar uma lesao."""
    proc = _run(
        [
            PY,
            "digital_twin.py",
            "finalize",
            str(case_dir),
            "--profile",
            PROFILE,
            "--no-lesion",
        ],
        timeout=MODEL_TIMEOUT,
    )
    if _model_done(case_dir):
        return True, ""
    return False, _cli_reason(proc)


def _case_dir_for_job(job_id: str) -> Path:
    if not job_id or any(ch not in "0123456789abcdef" for ch in job_id.lower()):
        raise HTTPException(status_code=404, detail="Job nao encontrado.")
    return (WORKSPACE / job_id / "case").resolve()


class ApprovalPayload(BaseModel):
    status: Literal["approved", "revision_requested"]


def process_job(job_id: str, raw_dir: Path) -> None:
    # case_dir e raw_dir (_upload) são IRMÃOS sob WORKSPACE/job_id; nunca aninhados,
    # senão limpar o case_dir apagaria o DICOM enviado (necessário no fallback CPU).
    case_dir = (WORKSPACE / job_id / "case").resolve()
    try:
        _set(job_id, state="processing", step="ingestao", progress=15)
        best_files, n = find_best_series(raw_dir)
        if not best_files or n < MIN_SLICES:
            _set(job_id, state="done", result=_graceful(
                "Não encontramos uma série DICOM de RM válida no envio.",
                "Envie a pasta de um exame de RM (DICOM) com múltiplos cortes — "
                "ou um único arquivo DICOM multi-frame."))
            return
        # Copia a série escolhida para um diretório limpo: isola de estruturas
        # bagunçadas / múltiplas séries e garante que o prepare veja só esta série.
        series_dir_path = WORKSPACE / job_id / "_series"
        series_dir_path.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(best_files):
            shutil.copyfile(f, series_dir_path / f"{i:05d}_{os.path.basename(f)}")
        series_dir = str(series_dir_path.resolve())

        # --- Fase 1: ingestão + des-identificação + segmentação (launcher isolado) ---
        # Sucesso medido pelos ARTEFATOS, não pelo returncode (crash de shutdown de
        # libs nativas no Windows não invalida um volume+máscara já gravados).
        _set(job_id, step="segmentacao", progress=45)
        prep = _segment(series_dir, case_dir, "gpu", PREP_TIMEOUT_GPU)
        if not _seg_done(case_dir):
            reason = _cli_reason(prep)
            log.warning("Segmentação na GPU falhou (%s); tentando CPU...", reason[:100])
            shutil.rmtree(case_dir, ignore_errors=True)
            _set(job_id, step="segmentacao", progress=55)
            prep = _segment(series_dir, case_dir, "cpu", PREP_TIMEOUT_CPU)
            if not _seg_done(case_dir):
                reason = _cli_reason(prep)
                _set(job_id, state="done", result=_graceful(_friendly_text(reason), reason))
                return

        # --- Fase 2: montagem 2D + MedGemma (subprocesso isolado) ---
        _set(job_id, step="medgemma", progress=80)
        scr = _run([PY, "-m", "dtwin.medgemma_screening", "--case-dir", str(case_dir),
                    "--medgemma-config", MEDGEMMA_CONFIG, "--confirm-no-visible-phi"],
                   timeout=SCREEN_TIMEOUT)
        report = _load_report(case_dir / "outputs" / "medgemma" / "medgemma_report.json")
        if report is None:
            reason = _cli_reason(scr)
            _set(job_id, state="done", result=_graceful(_friendly_text(reason), reason))
            return
        # Fase 3: mascara hepatica -> malha/STL para revisao humana.
        _set(job_id, step="modelo_3d", progress=92)
        viewer_ready, viewer_error = _build_model(case_dir)
        if not viewer_ready:
            log.warning("Job %s: relatorio concluido, mas modelo 3D falhou: %s", job_id, viewer_error)
        _set(
            job_id,
            state="done",
            step="concluido",
            progress=100,
            viewer_error=viewer_error or None,
            approval={"status": "pending"} if viewer_ready else None,
            result=_viewer_result(report, job_id, viewer_ready),
        )
    except subprocess.TimeoutExpired:
        _set(job_id, state="done", result=_graceful(
            "O processamento excedeu o tempo limite.", "timeout"))
    except Exception as exc:  # noqa: BLE001
        log.exception("Job %s: falha inesperada", job_id)
        _set(job_id, state="done", result=_graceful(
            "Ocorreu um erro inesperado no processamento.", type(exc).__name__))
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(WORKSPACE / job_id / "_series", ignore_errors=True)


def calculate_benchmark_metrics(results: list[dict]) -> dict:
    """Adaptador retrocompatível para o núcleo compartilhado do benchmark."""
    metrics = compute_benchmark_metrics(results)
    metrics["scoring_policy"] = "inconclusive_and_failed_count_as_errors"
    return metrics


def _benchmark_model_info() -> dict:
    try:
        config = yaml.safe_load((REPO / MEDGEMMA_CONFIG).read_text("utf-8")) or {}
        model = config.get("medgemma_screening", {}).get("medgemma", {})
        return {
            "model_id": model.get("model_id"),
            "model_version": model.get("model_version"),
            "model_parameter_scale": model.get("model_parameter_scale"),
            "runtime": model.get("runtime", "transformers"),
            "experimental_strategy": str(
                config.get("medgemma_screening", {}).get("panel", {}).get("mode", "single_grayscale")
            ),
            "config": MEDGEMMA_CONFIG,
        }
    except Exception:  # noqa: BLE001
        return {"model_id": None, "model_version": None, "config": MEDGEMMA_CONFIG}


def _run_benchmark_case(
    benchmark_id: str,
    index: int,
    item: dict,
    raw_case_dir: Path,
) -> dict:
    """Executa segmentação + triagem para um exame, sem gerar a malha 3D."""
    benchmark_root = WORKSPACE / "benchmarks" / benchmark_id
    # case_dir PRECISA ser absoluto: a segmentação roda por um launcher com
    # cwd=%TEMP% (workaround do nnU-Net no Windows). Se for relativo, a saída cai
    # sob %TEMP% e _seg_done() — avaliado a partir da raiz do repo — nunca a
    # encontra, marcando TODO exame como falha (e forçando o fallback lento p/ CPU).
    # O fluxo de exame individual (process_job) já resolve por isso; espelhamos aqui.
    case_dir = (benchmark_root / "cases" / f"{index:04d}").resolve()
    series_dir = benchmark_root / "_series" / f"{index:04d}"
    started = time.monotonic()
    base = {
        "case_id": item["id"],
        "dataset": item.get("dataset", "web_upload"),
        "input_format": "DICOM",
        "prediction": None,
        "confidence": None,
        "status": "failed",
        "error": None,
        "input_hashes": {},
        "durations_seconds": {},
    }
    try:
        import_started = time.monotonic()
        best_files, n = find_best_series(raw_case_dir)
        if not best_files or n < MIN_SLICES:
            base["error"] = "Nenhuma série DICOM de RM válida foi encontrada."
            return base

        series_dir.mkdir(parents=True, exist_ok=True)
        for file_index, source in enumerate(best_files):
            shutil.copyfile(source, series_dir / f"{file_index:05d}_{os.path.basename(source)}")
        base["durations_seconds"]["import"] = round(time.monotonic() - import_started, 4)

        preparation_started = time.monotonic()
        prep = _segment(str(series_dir.resolve()), case_dir, "gpu", PREP_TIMEOUT_GPU)
        if not _seg_done(case_dir):
            reason = _cli_reason(prep)
            log.warning("Benchmark %s/%s: GPU falhou (%s); tentando CPU", benchmark_id, item["id"], reason[:100])
            shutil.rmtree(case_dir, ignore_errors=True)
            prep = _segment(str(series_dir.resolve()), case_dir, "cpu", PREP_TIMEOUT_CPU)
            if not _seg_done(case_dir):
                base["error"] = _friendly_text(_cli_reason(prep))
                return base
        base["durations_seconds"]["preparation_and_segmentation"] = round(
            time.monotonic() - preparation_started, 4
        )

        screening_started = time.monotonic()
        screening = _run(
            [
                PY,
                "-m",
                "dtwin.medgemma_screening",
                "--case-dir",
                str(case_dir),
                "--medgemma-config",
                MEDGEMMA_CONFIG,
                "--confirm-no-visible-phi",
            ],
            timeout=SCREEN_TIMEOUT,
        )
        envelope = _load_report(case_dir / "outputs" / "medgemma" / "medgemma_report.json")
        if envelope is None:
            base["error"] = _friendly_text(_cli_reason(screening))
            base["status"] = classify_screening_failure(_cli_reason(screening)).value
            return base

        report = envelope["report"]
        base["durations_seconds"].update(envelope.get("durations_seconds") or {})
        base["durations_seconds"]["screening_subprocess"] = round(
            time.monotonic() - screening_started, 4
        )
        base["input_hashes"] = {
            "volume": envelope.get("input_volume_sha256"),
            "mask_organ": envelope.get("input_liver_mask_sha256"),
            "panel": envelope.get("input_panel_sha256"),
            "screening_config": envelope.get("screening_config_sha256"),
        }
        prediction = str(report.get("resultado_hipotese", "")).upper()
        if prediction not in {"POSITIVA", "NEGATIVA", "INCONCLUSIVA"}:
            base["error"] = "O relatório retornou uma classificação inválida."
            base["status"] = "invalid_response"
            return base
        base.update(
            prediction=prediction,
            confidence=report.get("confianca"),
            status="inconclusive" if prediction == "INCONCLUSIVA" else "decisive",
            report_summary=report.get("resumo_do_achado"),
            report_path=str(
                Path("cases") / f"{index:04d}" / "outputs" / "medgemma" / "medgemma_report.json"
            ),
            panel_path=str(
                Path("cases") / f"{index:04d}" / "outputs" / "medgemma" / str(envelope.get("input_panel") or "")
            ),
        )
        return base
    except subprocess.TimeoutExpired:
        base["error"] = "O processamento excedeu o tempo limite."
        base["status"] = "timeout"
        return base
    except Exception as exc:  # noqa: BLE001
        log.exception("Benchmark %s/%s: falha inesperada", benchmark_id, item["id"])
        base["error"] = f"Falha inesperada: {type(exc).__name__}"
        return base
    finally:
        base["duration_seconds"] = round(time.monotonic() - started, 2)
        base["durations_seconds"]["total"] = round(time.monotonic() - started, 4)
        shutil.rmtree(series_dir, ignore_errors=True)


def _evaluate_benchmark_result(inference_result: dict, label: str) -> dict:
    """Anexa o ground truth somente após a inferência ter encerrado."""
    started = time.monotonic()
    result = dict(inference_result)
    expected = "POSITIVA" if label == "positive" else "NEGATIVA"
    prediction = result.get("prediction")
    result.update(
        truth=label,
        correct=(prediction == expected) if prediction in {"POSITIVA", "NEGATIVA"} else None,
        protected_ground_truth_hashes={"lesion_mask": None, "annotation_manifest": None},
    )
    durations = dict(result.get("durations_seconds") or {})
    durations["evaluation"] = round(time.monotonic() - started, 4)
    result["durations_seconds"] = durations
    return result


def process_benchmark(benchmark_id: str, manifest: dict, raw_dir: Path) -> None:
    benchmark_root = WORKSPACE / "benchmarks" / benchmark_id
    cases = manifest["cases"]
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []
    try:
        _set_benchmark(benchmark_id, state="processing", started_at=started_at)
        for index, item in enumerate(cases, start=1):
            progress = 5 + int(((index - 1) / max(len(cases), 1)) * 90)
            _set_benchmark(
                benchmark_id,
                current_case=item["id"],
                processed=index - 1,
                progress=progress,
            )
            inference_result = _run_benchmark_case(
                benchmark_id,
                index,
                {"id": item["id"], "dataset": manifest["dataset_name"]},
                raw_dir / f"{index:04d}",
            )
            results.append(_evaluate_benchmark_result(inference_result, item["label"]))
            _set_benchmark(benchmark_id, processed=index, progress=5 + int(index / len(cases) * 90))

        completed_at = datetime.now(timezone.utc).isoformat()
        model_info = _benchmark_model_info()
        metrics = calculate_benchmark_metrics(results)
        report = {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "dataset_name": manifest["dataset_name"],
            "dataset_kind": manifest["dataset_kind"],
            "started_at": started_at,
            "completed_at": completed_at,
            "model": model_info,
            "metrics": metrics,
            "cases": results,
            "disclaimer": DISCLAIMER,
        }
        benchmark_root.mkdir(parents=True, exist_ok=True)
        report_path = benchmark_root / "benchmark_report.json"
        temp = benchmark_root / ".benchmark_report.json.tmp"
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(report_path)
        config_path = (REPO / MEDGEMMA_CONFIG).resolve()
        run_manifest = {
            "schema_version": 1,
            "run_id": benchmark_id,
            "created_at": started_at,
            **git_state(REPO),
            "model_family": "MedGemma",
            **model_info,
            "medgemma_config_path": MEDGEMMA_CONFIG,
            "medgemma_config_hash": sha256_of(config_path) if config_path.is_file() else None,
            "dataset_names": [manifest["dataset_name"]],
            "num_cases_total": len(cases),
            "num_cases_positive": sum(item["label"] == "positive" for item in cases),
            "num_cases_negative": sum(item["label"] == "negative" for item in cases),
            "started_at": started_at,
            "finished_at": completed_at,
            "duration_seconds_total": round(
                sum(float(item.get("duration_seconds") or 0) for item in results), 4
            ),
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "research_only": True,
        }
        write_run_outputs(benchmark_root, run_manifest, results, metrics)
        _set_benchmark(
            benchmark_id,
            state="done",
            current_case=None,
            processed=len(cases),
            progress=100,
            report=report,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Benchmark %s: falha inesperada", benchmark_id)
        _set_benchmark(
            benchmark_id,
            state="failed",
            progress=100,
            error=f"Não foi possível concluir o benchmark: {type(exc).__name__}",
        )
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)


def _parse_benchmark_manifest(raw: str, file_count: int) -> dict:
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Manifesto do benchmark inválido.") from exc
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=400, detail="Manifesto do benchmark inválido.")
    dataset_name = str(manifest.get("dataset_name") or "").strip()[:120]
    dataset_kind = manifest.get("dataset_kind")
    cases = manifest.get("cases")
    if not dataset_name:
        raise HTTPException(status_code=400, detail="Informe o nome do dataset.")
    if dataset_kind not in {"positive", "negative", "mixed"}:
        raise HTTPException(status_code=400, detail="Tipo de dataset inválido.")
    if not isinstance(cases, list) or not cases:
        raise HTTPException(status_code=400, detail="Nenhum exame foi identificado no dataset.")

    seen_ids: set[str] = set()
    seen_files: set[int] = set()
    normalized = []
    for case in cases:
        if not isinstance(case, dict):
            raise HTTPException(status_code=400, detail="Definição de exame inválida.")
        case_id = str(case.get("id") or "").strip()[:120]
        label = case.get("label")
        indices = case.get("file_indices")
        if not case_id or case_id in seen_ids:
            raise HTTPException(status_code=400, detail="Os exames precisam de identificadores únicos.")
        if label not in {"positive", "negative"}:
            raise HTTPException(status_code=400, detail=f"Rótulo inválido no exame {case_id}.")
        if dataset_kind in {"positive", "negative"} and label != dataset_kind:
            raise HTTPException(
                status_code=400,
                detail=f"O rótulo do exame {case_id} não corresponde ao tipo do dataset.",
            )
        if not isinstance(indices, list) or not indices:
            raise HTTPException(status_code=400, detail=f"O exame {case_id} não contém arquivos.")
        clean_indices = []
        for value in indices:
            if not isinstance(value, int) or value < 0 or value >= file_count or value in seen_files:
                raise HTTPException(status_code=400, detail="Mapeamento de arquivos do benchmark inválido.")
            seen_files.add(value)
            clean_indices.append(value)
        seen_ids.add(case_id)
        normalized.append({"id": case_id, "label": label, "file_indices": clean_indices})
    if seen_files != set(range(file_count)):
        raise HTTPException(status_code=400, detail="Todos os arquivos devem pertencer a um exame.")
    return {"dataset_name": dataset_name, "dataset_kind": dataset_kind, "cases": normalized}


def _benchmark_csv(report: dict) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow([
        "case_id", "truth", "prediction", "status", "correct", "confidence",
        "duration_seconds", "error",
    ])
    for item in report.get("cases", []):
        writer.writerow([
            item.get("case_id"), item.get("truth"), item.get("prediction"),
            item.get("status"), item.get("correct"), item.get("confidence"),
            item.get("duration_seconds"), item.get("error"),
        ])
    return stream.getvalue()


async def _upload_form(request: Request) -> FormData:
    """Analisa o multipart com o teto de arquivos elevado (MAX_UPLOAD_FILES).

    FastAPI não expõe max_files/max_fields do parser do Starlette através de
    File(...)/Form(...); por isso o form é lido manualmente aqui, nos dois
    endpoints que recebem upload de exames. Sem `async with`: os UploadFile
    precisam continuar abertos até serem lidos no corpo do endpoint; o
    encerramento/limpeza é feito pelo próprio Starlette ao fim da requisição."""
    return await request.form(max_files=MAX_UPLOAD_FILES, max_fields=MAX_UPLOAD_FILES)


app = FastAPI(title="Digital Twin — Triagem MedGemma (demo, modo Pesquisa)")


@app.get("/api/health")
def health() -> dict:
    backend = "desligado"
    try:
        with urlopen(HEALTH_URL, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        backend = "pronto" if data.get("status") == "ready" else "carregando"
    except Exception:  # noqa: BLE001
        backend = "desligado"
    return {"backend": backend}


@app.post("/api/analyze")
async def analyze(request: Request) -> dict:
    form = await _upload_form(request)
    files = [v for v in form.getlist("files") if not isinstance(v, str)]
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")
    relpaths = form.get("relpaths")
    job_id = uuid.uuid4().hex[:12]
    raw_dir = WORKSPACE / job_id / "_upload"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = json.loads(relpaths) if isinstance(relpaths, str) else []
        if not isinstance(paths, list):
            paths = []
    except Exception:  # noqa: BLE001
        paths = []
    for i, uf in enumerate(files):
        rel = (paths[i] if i < len(paths) and paths[i] else uf.filename) or f"file_{i}"
        parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", "..", ".")]
        dest = raw_dir.joinpath(*parts) if parts else raw_dir / f"file_{i}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(await uf.read())
    with _lock:
        _jobs[job_id] = {"state": "queued", "step": "recebendo", "progress": 5, "result": None}
    threading.Thread(target=process_job, args=(job_id, raw_dir), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return {
        "state": job["state"],
        "step": job["step"],
        "progress": job["progress"],
        "result": job["result"] if job["state"] == "done" else None,
        "approval": job.get("approval"),
    }


@app.post("/api/benchmarks")
async def create_benchmark(request: Request) -> dict:
    form = await _upload_form(request)
    files = [v for v in form.getlist("files") if not isinstance(v, str)]
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")
    manifest = form.get("manifest")
    if not isinstance(manifest, str):
        raise HTTPException(status_code=400, detail="Manifesto do benchmark ausente.")
    parsed = _parse_benchmark_manifest(manifest, len(files))
    benchmark_id = uuid.uuid4().hex[:12]
    raw_dir = WORKSPACE / "benchmarks" / benchmark_id / "_upload"
    try:
        for case_index, item in enumerate(parsed["cases"], start=1):
            case_upload = raw_dir / f"{case_index:04d}"
            case_upload.mkdir(parents=True, exist_ok=True)
            for local_index, file_index in enumerate(item["file_indices"]):
                upload = files[file_index]
                original_name = Path(upload.filename or f"file_{file_index}").name
                destination = case_upload / f"{local_index:06d}_{original_name}"
                destination.write_bytes(await upload.read())
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(raw_dir.parent, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Falha ao receber os arquivos do dataset.") from exc

    with _lock:
        _benchmarks[benchmark_id] = {
            "state": "queued",
            "progress": 2,
            "processed": 0,
            "total": len(parsed["cases"]),
            "current_case": None,
            "report": None,
            "error": None,
        }
    threading.Thread(
        target=process_benchmark,
        args=(benchmark_id, parsed, raw_dir),
        daemon=True,
    ).start()
    return {"benchmark_id": benchmark_id, "total_cases": len(parsed["cases"])}


@app.get("/api/benchmarks/{benchmark_id}")
def benchmark_status(benchmark_id: str) -> dict:
    with _lock:
        benchmark = _benchmarks.get(benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark não encontrado.")
    return {
        "state": benchmark["state"],
        "progress": benchmark["progress"],
        "processed": benchmark["processed"],
        "total": benchmark["total"],
        "current_case": benchmark.get("current_case"),
        "report": benchmark.get("report") if benchmark["state"] == "done" else None,
        "error": benchmark.get("error"),
    }


@app.get("/api/benchmarks/{benchmark_id}/report.json")
def benchmark_report_json(benchmark_id: str):
    if not benchmark_id or any(ch not in "0123456789abcdef" for ch in benchmark_id.lower()):
        raise HTTPException(status_code=404, detail="Benchmark não encontrado.")
    path = WORKSPACE / "benchmarks" / benchmark_id / "benchmark_report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Relatório ainda não disponível.")
    return FileResponse(path, media_type="application/json", filename=f"benchmark-{benchmark_id}.json")


@app.get("/api/benchmarks/{benchmark_id}/report.csv")
def benchmark_report_csv(benchmark_id: str):
    if not benchmark_id or any(ch not in "0123456789abcdef" for ch in benchmark_id.lower()):
        raise HTTPException(status_code=404, detail="Benchmark não encontrado.")
    path = WORKSPACE / "benchmarks" / benchmark_id / "benchmark_report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Relatório ainda não disponível.")
    try:
        report = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Relatório inválido.") from exc
    return Response(
        content=_benchmark_csv(report),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="benchmark-{benchmark_id}.csv"'},
    )


@app.get("/api/jobs/{job_id}/model/viewer_manifest.json")
def model_manifest(job_id: str):
    path = _case_dir_for_job(job_id) / "outputs" / "viewer_manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Modelo 3D nao disponivel.")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{job_id}/model/{filename}")
def model_file(job_id: str, filename: str):
    case_dir = _case_dir_for_job(job_id)
    manifest_path = case_dir / "outputs" / "viewer_manifest.json"
    if Path(filename).name != filename or not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo do modelo nao encontrado.")
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
        allowed = {item.get("stl") for item in manifest.get("meshes", []) if isinstance(item, dict)}
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Manifesto do modelo invalido.")
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Arquivo do modelo nao encontrado.")
    path = manifest_path.parent / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo do modelo nao encontrado.")
    return FileResponse(path, media_type="model/stl", filename=filename)


@app.post("/api/jobs/{job_id}/approval")
def approve_model(job_id: str, payload: ApprovalPayload) -> dict:
    case_dir = _case_dir_for_job(job_id)
    if not _model_done(case_dir):
        raise HTTPException(status_code=409, detail="Modelo 3D ainda nao esta disponivel.")
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job nao encontrado.")
    approval = {
        "status": payload.status,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "review_type": "human_visual_review",
    }
    path = case_dir / "outputs" / "approval.json"
    temp = path.with_name(".approval.json.tmp")
    temp.write_text(json.dumps(approval, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    _set(job_id, approval=approval)
    return approval


STATIC.mkdir(parents=True, exist_ok=True)
_install_seg_launcher()
app.mount("/viewer", StaticFiles(directory=str(VIEWER), html=True), name="viewer")
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")


def main() -> int:
    import uvicorn

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("WEBAPP_PORT", "8000")), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
