#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtwin.engine — Orquestrador determinístico dos estágios.

Lê o perfil (config) e roda os estágios na ordem, em duas fases:
  prepare  : estágios 1–4a  (até a marcação humana da lesão no 3D Slicer)
  finalize : estágios 4b–7  (após a marcação)

Determinístico: mesmo input + mesma config + mesma marcação humana => mesma saída.
Nenhuma aleatoriedade e nenhum fallback que fabrique dado.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import stages
from .core import Case, load_profile

log = logging.getLogger("dtwin")


class Engine:
    """Motor órgão-agnóstico. O comportamento por órgão vem do perfil (config)."""

    def __init__(self, profile_path: Path):
        self.profile = load_profile(Path(profile_path))

    def prepare(
        self,
        dicom_dir,
        case_dir,
        policy: str = "anonymize",
        device: str = "gpu",
        fast: bool = False,
    ) -> Case:
        case = Case(Path(case_dir))
        log.info("== PREPARE (perfil: %s) ==", self.profile["id"])
        stages.stage1_ingest(case, self.profile, dicom_dir, policy)
        stages.stage2_normalize(case, self.profile)
        stages.stage3_segment_organ(case, self.profile, device, fast)
        stages.stage4a_prepare_lesion(case, self.profile)
        return case

    def finalize(self, case_dir, no_lesion: bool = False) -> Case:
        case = Case(Path(case_dir))
        log.info("== FINALIZE (perfil: %s) ==", self.profile["id"])
        stages.stage4b_import_lesion(case, self.profile, no_lesion)
        stages.stage5_refine(case, self.profile)
        stages.stage6_mesh(case, self.profile)
        stages.stage7_export_publish(case, self.profile)
        return case
