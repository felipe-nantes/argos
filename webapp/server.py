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

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.request import urlopen

import SimpleITK as sitk
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("dtwin.webapp")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STATIC = ROOT / "static"
WORKSPACE = Path("casos/webapp")
PROFILE = "profiles/figado.yaml"
MEDGEMMA_CONFIG = os.environ.get("WEBAPP_MEDGEMMA_CONFIG", "configs/medgemma_local_4b.yaml")
HEALTH_URL = os.environ.get("WEBAPP_MEDGEMMA_HEALTH", "http://127.0.0.1:8001/health")
MIN_SLICES = 3
PREP_TIMEOUT_GPU = int(os.environ.get("WEBAPP_PREP_TIMEOUT_GPU", "900"))
PREP_TIMEOUT_CPU = int(os.environ.get("WEBAPP_PREP_TIMEOUT_CPU", "2400"))
SCREEN_TIMEOUT = int(os.environ.get("WEBAPP_SCREEN_TIMEOUT", "600"))
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
_lock = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(**kw)


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


def find_best_series(root: Path) -> tuple[list[str], int]:
    """Acha a maior série DICOM em QUALQUER estrutura de pastas, robustamente:

    1) enumera TODAS as séries de cada diretório (uma pasta pode ter várias séries);
    2) se nenhuma série multi-arquivo servir, tenta um único DICOM **multi-frame**
       (um só `.dcm` que já é o volume 3D inteiro), medindo a profundidade real.

    Retorna (lista_de_arquivos_da_melhor_série, nº_de_cortes)."""
    reader = sitk.ImageSeriesReader()
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
            if len(names) > len(best_files):
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
            if depth > best_depth:
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
        _set(job_id, state="done", step="concluido", progress=100,
             result=_success_result(report))
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
async def analyze(files: list[UploadFile] = File(...), relpaths: str = Form(default="[]")) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")
    job_id = uuid.uuid4().hex[:12]
    raw_dir = WORKSPACE / job_id / "_upload"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = json.loads(relpaths)
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
    }


STATIC.mkdir(parents=True, exist_ok=True)
_install_seg_launcher()
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")


def main() -> int:
    import uvicorn

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("WEBAPP_PORT", "8000")), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
