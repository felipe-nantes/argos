#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test de ponta a ponta na máquina com GPU, com um exame de RM REAL.

Cobre as provas que o caso sintético e a suíte (que stubam a segmentação) NÃO
conseguem cobrir: a segmentação automática real do órgão (TotalSegmentator) e a
ingestão de um DICOM clínico real. Roda 'prepare' (estágios 1–4a) e, em seguida,
'finalize' (estágios 4b–7), validando CADA artefato e devolvendo exit code 0
(tudo passou) ou 1 (alguma checagem falhou).

A LESÃO normalmente é marcada por um humano no 3D Slicer; este smoke aceita uma
máscara de lesão já pronta via --lesion, ou roda com --no-lesion para validar
mecanicamente os estágios 4b–7 sem lesão. Ele NUNCA fabrica uma lesão.

Uso (na caixa com GPU, na raiz do projeto):
  python tools/smoke_gpu.py --dicom "C:/caminho/serie_dicom"
  python tools/smoke_gpu.py --dicom "C:/serie" --lesion "C:/mask_lesion.nii.gz"
  python tools/smoke_gpu.py --dicom "C:/serie" --no-lesion --device cpu --fast

Pré-requisito: ambiente com o extra [seg] instalado (pip install -e .[seg]).
Confirme antes com:  python digital_twin.py doctor   (espera "torch device: cuda").
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
import time
from pathlib import Path

from dtwin.core import Case, PipelineError, array_from, load_profile, read_image
from dtwin.engine import Engine


class Checker:
    """Acumula resultados de checagem e imprime um relatório linha a linha."""

    def __init__(self) -> None:
        self.total = 0
        self.failed = 0

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.total += 1
        if not ok:
            self.failed += 1
        mark = "[OK]   " if ok else "[FALHA]"
        suffix = f" — {detail}" if detail else ""
        print(f"  {mark} {label}{suffix}")
        return ok


def _stl_triangle_count(path: Path):
    """Nº de triângulos de um STL binário válido, ou None se inválido/corrompido."""
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) < 84:
        return None
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + n * 50:  # 80B header + uint32 + 50B/triângulo
        return None
    return n


def _voxel_sum(path: Path) -> int:
    return int(array_from(read_image(path)).sum())


def validate_prepare(chk: Checker, case: Case) -> None:
    print("\n[PREPARE] Validando artefatos dos estágios 1–4a:")
    chk.check("volume.nii.gz existe", case.volume.exists())
    chk.check("volume_zscore.nii.gz existe (normalização)", case.volume_zscore.exists())
    organ_ok = case.mask_organ.exists()
    chk.check("mask_organ.nii.gz existe", organ_ok)
    if organ_ok:
        vox = _voxel_sum(case.mask_organ)
        chk.check("órgão segmentado é NÃO-vazio", vox > 0, f"{vox} voxels")
    if case.manifest.exists():
        m = case.read_manifest()
        chk.check("manifest tem case_id", bool(m.get("case_id")), str(m.get("case_id")))
        chk.check("manifest registra a modalidade", bool(m.get("modality")), str(m.get("modality")))
        chk.check(
            "estado regulatório = PESQUISA",
            m.get("regulatory_state") == "PESQUISA",
            str(m.get("regulatory_state")),
        )
    else:
        chk.check("manifest.json existe", False)


def validate_finalize(chk: Checker, case: Case, profile: dict, expect_lesion: bool) -> None:
    print("\n[FINALIZE] Validando artefatos dos estágios 4b–7:")
    organ_stl = case.outputs / f"{profile['id']}_orgao.stl"
    n_org = _stl_triangle_count(organ_stl)
    chk.check(
        f"{organ_stl.name}: STL binário válido e não-vazio",
        n_org is not None and n_org > 0,
        f"{n_org} triângulos" if n_org else "ausente/corrompido",
    )

    lesion_stl = case.outputs / f"{profile['id']}_lesao.stl"
    if expect_lesion:
        n_les = _stl_triangle_count(lesion_stl)
        chk.check(
            f"{lesion_stl.name}: STL binário válido e não-vazio",
            n_les is not None and n_les > 0,
            f"{n_les} triângulos" if n_les else "ausente/corrompido",
        )
    else:
        chk.check(
            f"{lesion_stl.name} AUSENTE (--no-lesion)",
            not lesion_stl.exists(),
        )

    mani_path = case.outputs / "viewer_manifest.json"
    if not chk.check("viewer_manifest.json existe", mani_path.exists()):
        return
    import json

    mani = json.loads(mani_path.read_text(encoding="utf-8"))
    chk.check("manifesto: organ bate com o perfil", mani.get("organ") == profile["id"], str(mani.get("organ")))
    chk.check("manifesto: coordenadas = LPS", mani.get("coordinate_system") == "LPS")
    chk.check("manifesto: tem disclaimer de pesquisa", bool(mani.get("disclaimer")))
    roles = {m["role"] for m in mani.get("meshes", [])}
    chk.check("manifesto: contém o órgão", "orgao" in roles)
    chk.check(
        "manifesto: lesão presente conforme esperado",
        ("lesao" in roles) == expect_lesion,
        f"roles={sorted(roles)}",
    )
    for m in mani.get("meshes", []):
        stl = m.get("stl", "")
        chk.check(
            f"manifesto: ref STL relativa ({stl})",
            "/" not in stl and "\\" not in stl and (case.outputs / stl).exists(),
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Smoke test GPU ponta a ponta (prepare+finalize) com exame real."
    )
    ap.add_argument("--dicom", required=True, help="Pasta com a série DICOM de RM real.")
    ap.add_argument("--case-dir", default="casos/smoke_gpu", help="Pasta de trabalho do caso.")
    ap.add_argument("--profile", default="profiles/figado.yaml", help="Perfil do órgão (YAML).")
    ap.add_argument("--device", default="gpu", help="gpu | cpu | gpu:N (TotalSegmentator).")
    ap.add_argument("--fast", action="store_true", help="Modo rápido (recomendado em CPU).")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lesion", help="Máscara de lesão pronta (.nii.gz) para o finalize.")
    grp.add_argument("--no-lesion", action="store_true", help="Finalizar sem lesão.")
    args = ap.parse_args(argv)

    print("=" * 70)
    print(" SMOKE TEST GPU — Digital Twin (exame real, prepare + finalize)")
    print(" MODO PESQUISA — saída NÃO destinada a decisão clínica.")
    print("=" * 70)

    profile = load_profile(Path(args.profile))
    engine = Engine(Path(args.profile))
    case = Case(Path(args.case_dir))
    chk = Checker()

    # ---- Fase 1: prepare (segmentação real — a parte cara) ------------------
    print(f"\n[1/2] prepare: {args.dicom}  ->  {args.case_dir}  (device={args.device})")
    t0 = time.time()
    try:
        engine.prepare(args.dicom, args.case_dir, device=args.device, fast=args.fast)
    except PipelineError as e:
        print(f"\n[ABORTADO no prepare] {e}")
        return 1
    print(f"  prepare concluído em {time.time() - t0:.1f}s")
    validate_prepare(chk, case)

    # ---- Lesão: pronta (--lesion), ou sem lesão (--no-lesion) ---------------
    no_lesion = bool(args.no_lesion)
    if args.lesion:
        src = Path(args.lesion)
        if not src.exists():
            chk.check(f"máscara de lesão fornecida existe ({src})", False)
            return 1
        shutil.copy2(src, case.mask_lesion)
        print(f"\n  lesão fornecida copiada -> {case.mask_lesion}")
    elif not no_lesion:
        # sem --lesion e sem --no-lesion: a lesão teria de ser marcada no Slicer.
        print(
            "\n[INFO] Sem --lesion e sem --no-lesion. O finalize exige a máscara de "
            "lesão marcada no 3D Slicer em:\n"
            f"       {case.mask_lesion}\n"
            "       Marque-a e rode o finalize manualmente, ou re-rode este smoke "
            "com --lesion <arquivo> ou --no-lesion."
        )
        # Ainda assim valida o prepare; reporta e sai conforme as checagens dele.
        return _report(chk)

    # ---- Fase 2: finalize ---------------------------------------------------
    print(f"\n[2/2] finalize: {args.case_dir}  (no_lesion={no_lesion})")
    t1 = time.time()
    try:
        engine.finalize(args.case_dir, no_lesion=no_lesion)
    except PipelineError as e:
        print(f"\n[ABORTADO no finalize] {e}")
        return 1
    print(f"  finalize concluído em {time.time() - t1:.1f}s")
    validate_finalize(chk, case, profile, expect_lesion=not no_lesion)

    return _report(chk)


def _report(chk: Checker) -> int:
    print("\n" + "-" * 70)
    if chk.failed == 0:
        print(f"RESULTADO: {chk.total}/{chk.total} checagens OK. Sistema funcional ponta a ponta.")
        return 0
    print(f"RESULTADO: {chk.failed} de {chk.total} checagens FALHARAM. Investigue acima.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
