#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtwin.core — Núcleo determinístico e órgão-agnóstico do pipeline.

Este módulo NÃO conhece "fígado". Ele conhece três coisas:
  1. Como ler/gravar imagens preservando a geometria (spacing, origin, direction).
  2. Como carregar e validar um PERFIL de órgão (config, não código).
  3. Como representar o estado de um caso em disco (pipeline resumível).

Trocar de órgão = adicionar um arquivo de perfil em profiles/. O motor não muda.
Referência de projeto: contexto/04_ARQUITETURA.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import yaml

log = logging.getLogger("dtwin")


# --------------------------------------------------------------------------- #
# Erros e gates
# --------------------------------------------------------------------------- #
class PipelineError(RuntimeError):
    """Erro de gate.

    REGRA DE OURO Nº 1 (contexto/05_PIPELINE.md): ao encontrar um problema, o
    pipeline ABORTA com esta exceção. Nunca fabrica dado (ex.: máscara aleatória)
    e nunca 'segue mesmo assim'. Foi exatamente o oposto que o script original
    fazia ao gerar uma máscara aleatória quando a IA não carregava.
    """


# --------------------------------------------------------------------------- #
# Geometria — pareia SEMPRE array com a sua geometria, para o eixo nunca se perder
# --------------------------------------------------------------------------- #
def read_dicom_series(folder: Path) -> sitk.Image:
    """Lê uma série DICOM preservando a geometria correta.

    O ImageSeriesReader já aplica RescaleSlope/Intercept e monta o volume na
    ordem/orientação corretas. NÃO reaplicar slope/intercept (bug do original) e
    NÃO reescrever spacing manualmente.
    """
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(folder))
    if not names:
        raise PipelineError(f"Nenhuma série DICOM encontrada em {folder}")
    reader.SetFileNames(names)
    try:
        return reader.Execute()
    except Exception as e:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler a série DICOM em {folder}: {e}") from e


def read_image(path: Path) -> sitk.Image:
    try:
        return sitk.ReadImage(str(path))
    except Exception as e:  # noqa: BLE001
        raise PipelineError(f"Falha ao ler imagem {path}: {e}") from e


def save_image(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sitk.WriteImage(image, str(path), useCompression=True)
    except Exception as e:  # noqa: BLE001
        raise PipelineError(f"Falha ao gravar imagem {path}: {e}") from e


def array_from(image: sitk.Image) -> np.ndarray:
    """Array em ordem (z, y, x). Use SEMPRE a geometria da MESMA imagem."""
    return sitk.GetArrayFromImage(image)


def array_to_image(arr: np.ndarray, ref: sitk.Image, dtype=None) -> sitk.Image:
    """Constrói uma imagem herdando a geometria da referência (origin/spacing/dir)."""
    if dtype is not None:
        arr = arr.astype(dtype)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref)
    return img


def world_vertices_from_index(verts_zyx: np.ndarray, ref: sitk.Image) -> np.ndarray:
    """Converte vértices em índice de array (z, y, x) para coordenadas físicas LPS.

    Aplica origin + direction + spacing da imagem de referência. SimpleITK
    trabalha em LPS, então o resultado já sai em LPS — o STL fica no sistema
    esperado pelo 3D Slicer e por impressão 3D, SEM o flip manual Scale(-1,-1,1)
    do script original.

    Corrige de uma vez o bug de ordem de eixos do original (que passava spacing
    (x,y,z) a um array (z,y,x)) e ainda trata aquisições oblíquas (direction != I).
    """
    origin = np.asarray(ref.GetOrigin(), dtype=np.float64)            # (x, y, z)
    spacing = np.asarray(ref.GetSpacing(), dtype=np.float64)          # (x, y, z)
    direction = np.asarray(ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    idx_xyz = verts_zyx[:, ::-1].astype(np.float64)                   # (z,y,x)->(x,y,z)
    return (direction @ (idx_xyz * spacing).T).T + origin


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Perfil de órgão (config versionada, NÃO código)
# --------------------------------------------------------------------------- #
REQUIRED_PROFILE_KEYS = ("id", "modalidade", "segmentacao_orgao")


def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PipelineError(f"Perfil não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_PROFILE_KEYS if k not in profile]
    if missing:
        raise PipelineError(f"Perfil {path} sem chaves obrigatórias: {missing}")
    seg = profile["segmentacao_orgao"]
    if "rotulo_alvo" not in seg:
        raise PipelineError(f"Perfil {path}: segmentacao_orgao precisa de 'rotulo_alvo'.")
    return profile


# --------------------------------------------------------------------------- #
# Caso (case dir) — estado em disco; cada estágio lê e grava aqui
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """Aponta para os artefatos de um caso. Torna o pipeline resumível e testável
    estágio a estágio (cada um consome a saída do anterior em disco)."""
    root: Path

    # Imagem e normalização
    @property
    def volume(self) -> Path: return self.root / "volume.nii.gz"
    @property
    def volume_zscore(self) -> Path: return self.root / "volume_zscore.nii.gz"

    # Segmentação
    @property
    def seg_dir(self) -> Path: return self.root / "seg_raw"
    @property
    def mask_organ(self) -> Path: return self.root / "mask_organ.nii.gz"
    @property
    def mask_lesion(self) -> Path: return self.root / "mask_lesion.nii.gz"
    @property
    def mask_organ_clean(self) -> Path: return self.root / "mask_organ_clean.nii.gz"
    @property
    def mask_lesion_clean(self) -> Path: return self.root / "mask_lesion_clean.nii.gz"

    # Malhas intermediárias (PyVista .vtp; o STL é produto do estágio 7)
    @property
    def mesh_organ(self) -> Path: return self.root / "mesh_organ.vtp"
    @property
    def mesh_lesion(self) -> Path: return self.root / "mesh_lesion.vtp"

    # Manifesto e saídas finais
    @property
    def manifest(self) -> Path: return self.root / "manifest.json"
    @property
    def outputs(self) -> Path: return self.root / "outputs"

    def read_manifest(self) -> dict:
        if not self.manifest.exists():
            raise PipelineError(f"Manifesto ausente: {self.manifest}")
        return json.loads(self.manifest.read_text(encoding="utf-8"))

    def write_manifest(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
