#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtwin.stages — Os sete estágios do pipeline, cada um com seu gate.

Filosofia (contexto/05_PIPELINE.md): cada estágio valida a própria entrada e
ABORTA com PipelineError se algo estiver errado. Nunca fabrica, nunca segue
mesmo assim. Cada estágio lê os artefatos do anterior em disco e grava os seus,
o que torna o pipeline resumível e testável estágio a estágio.

Fluxo:
  prepare  -> 1 ingestão+deid | 2 normalização | 3 órgão (auto) | 4a preparar lesão
  finalize -> 4b importar lesão | 5 refino | 6 malha | 7 STL + publicação
"""
from __future__ import annotations

import json
import logging
import shutil
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pydicom
import pyvista as pv
from skimage import measure, morphology

from .core import (
    Case,
    PipelineError,
    array_from,
    array_to_image,
    now_utc,
    read_dicom_series,
    read_image,
    save_image,
    sha256_of,
    world_vertices_from_index,
)

log = logging.getLogger("dtwin")

MIN_SLICES = 3


# --------------------------------------------------------------------------- #
# Helpers internos
# --------------------------------------------------------------------------- #
def _first_dicom(folder: Path):
    files = [p for p in folder.rglob("*") if p.is_file()]
    pool = [p for p in files if p.suffix.lower() == ".dcm"] or files
    for p in pool:
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            return str(p)
        except Exception:  # noqa: BLE001
            continue
    return None


def _make_case_id(policy: str, ds) -> str:
    """Gera o identificador do caso conforme a política de privacidade.

    MVP usa 'anonymize' (UUID, sem vínculo com o paciente). 'pseudonymize' é um
    ponto de extensão RESERVADO (contexto/03_REGULATORIO_LGPD.md): exige um cofre
    de chaves protegido antes do uso clínico, por isso aqui aborta explicitamente
    em vez de simular um vínculo.
    """
    if policy == "anonymize":
        return "anon-" + uuid.uuid4().hex[:12]
    if policy == "pseudonymize":
        raise PipelineError(
            "Pseudonimização ainda não habilitada no MVP. O ponto de extensão "
            "existe (contexto/03_REGULATORIO_LGPD.md), mas exige um cofre de "
            "chaves protegido antes do uso clínico. Rode com --policy anonymize."
        )
    raise PipelineError(f"Política de privacidade desconhecida: {policy}")


def _archive_for_training(case: Case, profile: dict, manifest: dict) -> None:
    """Flywheel: arquiva a anotação humana de lesão para treino futuro.

    Cada lesão marcada à mão vira um dado rotulado. Acumulando, constrói-se o
    conjunto de treino que hoje não existe (contexto/06_SEGMENTACAO.md). Só dados
    anonimizados são arquivados.
    """
    base = (
        Path(profile.get("flywheel", {}).get("dir", "flywheel"))
        / profile["id"]
        / manifest["case_id"]
    )
    base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case.volume, base / "volume.nii.gz")
    shutil.copy2(case.mask_lesion, base / "mask_lesion.nii.gz")
    if case.mask_organ.exists():
        shutil.copy2(case.mask_organ, base / "mask_organ.nii.gz")
    (base / "meta.json").write_text(
        json.dumps(
            {
                "case_id": manifest["case_id"],
                "organ": profile["id"],
                "modality": manifest.get("modality"),
                "policy": manifest.get("policy"),
                "note": "Anotação humana de lesão p/ treino futuro (flywheel). "
                "Apenas dados anonimizados.",
                "created_utc": now_utc(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("Flywheel: caso arquivado para treino futuro em %s", base)


def _refine_mask(mask_zyx, opening: bool, radius: int, min_voxels: int) -> np.ndarray:
    m = mask_zyx.astype(bool)
    fp = np.ones((int(radius),) * 3, dtype=bool)
    if opening:
        m = morphology.binary_opening(m, footprint=fp)  # API atual: footprint (não selem)
    m = morphology.binary_closing(m, footprint=fp)
    if min_voxels and int(min_voxels) > 0:
        m = morphology.remove_small_objects(m, min_size=int(min_voxels))
    return m.astype(np.uint8)


def _mesh_from_mask(mask_path: Path, level: float, smooth_iter: int, feature_angle: float):
    img = read_image(mask_path)
    mask = array_from(img).astype(np.float32)
    if mask.max() < 0.5:
        return None  # máscara vazia
    try:
        verts_zyx, faces, _n, _v = measure.marching_cubes(mask, level=level)
    except (ValueError, RuntimeError) as e:
        raise PipelineError(
            f"Falha no marching cubes de {mask_path.name}: {e}"
        ) from e
    # Vértices em coordenadas físicas LPS (origin+direction+spacing da própria máscara).
    verts_lps = world_vertices_from_index(verts_zyx, img)
    # PyVista exige o contador de vértices por face: [3, i, j, k, 3, i, j, k, ...].
    # (Corrige o reshape(-1,3) do original, que embaralhava as faces.)
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()
    mesh = pv.PolyData(verts_lps, faces_pv)
    if smooth_iter and int(smooth_iter) > 0:
        mesh = mesh.smooth(n_iter=int(smooth_iter), feature_angle=float(feature_angle))
    return mesh


# --------------------------------------------------------------------------- #
# (1) Ingestão + des-identificação
# --------------------------------------------------------------------------- #
def stage1_ingest(case: Case, profile: dict, dicom_dir, policy: str) -> None:
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_dir():
        raise PipelineError(f"Pasta DICOM inexistente: {dicom_dir}")

    first = _first_dicom(dicom_dir)
    if first is None:
        raise PipelineError(f"Nenhum arquivo DICOM legível em {dicom_dir}")

    ds = pydicom.dcmread(first, stop_before_pixels=True, force=True)
    modality = str(getattr(ds, "Modality", "") or "").upper()
    expected = [str(m).upper() for m in profile["modalidade"]]
    if modality and modality not in expected:
        raise PipelineError(
            f"Modalidade do exame ({modality}) não bate com o perfil "
            f"'{profile['id']}' (espera {expected}). Use o perfil correto."
        )

    image = read_dicom_series(dicom_dir)
    if image.GetSize()[2] < MIN_SLICES:
        raise PipelineError(
            f"Série com poucas fatias ({image.GetSize()[2]}); volume 3D inviável."
        )

    case.root.mkdir(parents=True, exist_ok=True)
    # Anonimização (MVP): converter para NIfTI descarta os cabeçalhos DICOM, então
    # nenhum identificador (nome, ID, datas) viaja adiante.
    save_image(image, case.volume)

    case_id = _make_case_id(policy, ds)
    manifest = {
        "case_id": case_id,
        "policy": policy,
        "modality": modality or "DESCONHECIDA",
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": [float(x) for x in image.GetSpacing()],
        "origin_xyz": [float(x) for x in image.GetOrigin()],
        "volume_sha256": sha256_of(case.volume),
        "regulatory_state": profile.get("estado_regulatorio", "PESQUISA"),
        "software": "digital-twin-pipeline (MVP, Nível 1, modo Pesquisa)",
        "created_utc": now_utc(),
        "caveats": [
            "PHI gravada nos pixels (burned-in) NÃO é detectada automaticamente; "
            "exige verificação humana.",
        ],
    }
    case.write_manifest(manifest)
    log.info(
        "Estágio 1: volume %s, spacing %s mm, case_id=%s, modo=%s.",
        tuple(image.GetSize()),
        tuple(round(s, 3) for s in image.GetSpacing()),
        case_id,
        policy,
    )


# --------------------------------------------------------------------------- #
# (2) Normalização (referência/inspeção; NÃO é o que vai para o segmentador)
# --------------------------------------------------------------------------- #
def stage2_normalize(case: Case, profile: dict) -> None:
    img = read_image(case.volume)
    arr = array_from(img).astype(np.float32)
    method = str(profile.get("normalizacao", "zscore")).lower()

    if method == "zscore":
        std = float(arr.std())
        if std < 1e-6:
            raise PipelineError(
                "Volume praticamente constante (std~0); exame possivelmente corrompido."
            )
        norm = (arr - float(arr.mean())) / (std + 1e-8)
    elif method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            raise PipelineError(
                "Volume sem contraste (min==max); exame possivelmente corrompido."
            )
        norm = (arr - lo) / (hi - lo)
    else:
        raise PipelineError(
            f"Normalização '{method}' não suportada (use zscore ou minmax)."
        )

    save_image(array_to_image(norm, img, np.float32), case.volume_zscore)
    # NOTA: este volume normalizado é só referência. O estágio 3 alimenta o
    # TotalSegmentator com o volume ORIGINAL (case.volume); o modelo faz a própria
    # normalização interna. Não duplicar (contexto/05_PIPELINE.md).
    log.info("Estágio 2: volume normalizado (%s) salvo para referência.", method)


# --------------------------------------------------------------------------- #
# (3) Segmentação do ÓRGÃO — automática (TotalSegmentator MRI). GATE CRÍTICO.
# --------------------------------------------------------------------------- #
def stage3_segment_organ(case: Case, profile: dict, device: str, fast: bool) -> None:
    seg = profile["segmentacao_orgao"]
    task = seg.get("motor_task", "total_mr")
    label = seg["rotulo_alvo"]

    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as e:
        raise PipelineError(
            "TotalSegmentator não está instalado. O pipeline ABORTA (regra de ouro "
            "nº 1): jamais gerar máscara aleatória, como o script original fazia. "
            "Instale com: pip install TotalSegmentator"
        ) from e

    case.seg_dir.mkdir(parents=True, exist_ok=True)
    try:
        totalsegmentator(
            input=str(case.volume),
            output=str(case.seg_dir),
            task=task,
            roi_subset=[label],  # só o órgão-alvo: mais rápido e enxuto
            device=device,
            fast=fast,
            quiet=True,
        )
    except Exception as e:  # noqa: BLE001
        raise PipelineError(
            f"Falha na segmentação automática ({task}/{label}): {e}"
        ) from e

    produced = case.seg_dir / f"{label}.nii.gz"
    if not produced.exists():
        raise PipelineError(
            f"Saída de segmentação esperada não encontrada: {produced}. "
            f"Verifique se '{label}' é classe válida da task '{task}' "
            f"(rode: totalseg_info --classes -ta {task})."
        )

    organ_img = read_image(produced)
    if int(array_from(organ_img).sum()) == 0:
        raise PipelineError(
            f"Segmentação automática não encontrou '{label}' no exame. "
            "Revisão humana necessária (não há órgão a modelar)."
        )

    save_image(organ_img, case.mask_organ)
    log.info("Estágio 3: órgão '%s' segmentado automaticamente (task=%s).", label, task)


# --------------------------------------------------------------------------- #
# (4a) Preparar marcação da LESÃO — handoff para o 3D Slicer
# --------------------------------------------------------------------------- #
def stage4a_prepare_lesion(case: Case, profile: dict) -> None:
    if not case.volume.exists() or not case.mask_organ.exists():
        raise PipelineError("Estágios 1–3 incompletos; rode 'prepare' do início.")

    tools = ", ".join(
        profile.get("segmentacao_lesao", {}).get(
            "ferramentas_sugeridas", ["threshold", "region_growing", "paint"]
        )
    )
    msg = textwrap.dedent(
        f"""
    ------------------------------------------------------------------
    REVISÃO HUMANA NECESSÁRIA — 3D Slicer (estágio 4)
    A segmentação do órgão é automática; a LESÃO é marcada por humano.

      1) Abra o 3D Slicer e carregue:
           Volume : {case.volume}
           Órgão  : {case.mask_organ}   (revise e corrija se necessário)
      2) Crie um novo segmento para a LESÃO usando: {tools}.
      3) Salve a máscara da lesão EXATAMENTE em:
           {case.mask_lesion}
      4) (Opcional) Se corrigiu o órgão, sobrescreva:
           {case.mask_organ}

    Depois finalize com:
       python digital_twin.py finalize "{case.root}" --profile <perfil>
    Se o caso REALMENTE não tiver lesão, finalize com --no-lesion.
    ------------------------------------------------------------------
    """
    )
    print(msg)
    log.info("Estágio 4a: aguardando marcação da lesão no 3D Slicer.")


# --------------------------------------------------------------------------- #
# (4b) Importar LESÃO marcada + arquivar para o flywheel
# --------------------------------------------------------------------------- #
def stage4b_import_lesion(case: Case, profile: dict, no_lesion: bool) -> None:
    if not case.mask_organ.exists():
        raise PipelineError(
            "Máscara do órgão ausente; rode 'prepare' antes de 'finalize'."
        )

    if not case.mask_lesion.exists():
        if no_lesion:
            ref = read_image(case.mask_organ)
            empty = array_to_image(
                np.zeros(array_from(ref).shape, dtype=np.uint8), ref, np.uint8
            )
            save_image(empty, case.mask_lesion)
            log.warning(
                "Estágio 4b: caso sem lesão por escolha explícita (--no-lesion)."
            )
        else:
            raise PipelineError(
                f"Máscara de lesão ausente: {case.mask_lesion}\n"
                "Marque a lesão no 3D Slicer (ver instruções do 'prepare') e salve "
                "nesse caminho, ou rode 'finalize' com --no-lesion se não houver lesão."
            )

    lesion = read_image(case.mask_lesion)
    organ = read_image(case.mask_organ)
    if lesion.GetSize() != organ.GetSize():
        raise PipelineError(
            "Máscara de lesão com tamanho diferente do volume/órgão "
            f"({lesion.GetSize()} != {organ.GetSize()}). Refaça a marcação sobre o "
            "volume correto no Slicer."
        )

    l = array_from(lesion).astype(bool)
    o = array_from(organ).astype(bool)
    if l.sum() > 0 and not (l & o).any():
        # Aviso (não-gate): lesão adjacente pode ser legítima, mas costuma indicar erro.
        log.warning("Estágio 4b: a lesão marcada não sobrepõe o órgão. Confira no Slicer.")

    manifest = case.read_manifest()
    if manifest.get("policy") == "anonymize" and l.sum() > 0:
        _archive_for_training(case, profile, manifest)
    log.info("Estágio 4b: lesão importada e validada.")


# --------------------------------------------------------------------------- #
# (5) Refino das máscaras
# --------------------------------------------------------------------------- #
def stage5_refine(case: Case, profile: dict) -> None:
    refino = profile.get("refino", {})

    # Órgão
    organ_img = read_image(case.mask_organ)
    organ = array_from(organ_img)
    oc = refino.get("orgao", {})
    organ_clean = _refine_mask(
        organ, oc.get("opening", True), oc.get("opening_radius", 2),
        oc.get("min_volume_voxels", 300),
    )
    if organ.sum() > 0 and organ_clean.sum() == 0:
        raise PipelineError(
            "Refino zerou a máscara do órgão — parâmetros mal calibrados (refino.orgao)."
        )
    save_image(array_to_image(organ_clean, organ_img, np.uint8), case.mask_organ_clean)
    log.info(
        "Estágio 5: órgão refinado (%d -> %d voxels).",
        int(organ.sum()), int(organ_clean.sum()),
    )

    # Lesão (gentil: não apagar lesões pequenas)
    lesion_img = read_image(case.mask_lesion)
    lesion = array_from(lesion_img)
    if lesion.sum() == 0:
        save_image(array_to_image(lesion, lesion_img, np.uint8), case.mask_lesion_clean)
        log.info("Estágio 5: sem lesão a refinar.")
        return

    lc = refino.get("lesao", {})
    lesion_clean = _refine_mask(
        lesion, lc.get("opening", False), lc.get("opening_radius", 1),
        lc.get("min_volume_voxels", 30),
    )
    if lesion_clean.sum() == 0:
        raise PipelineError(
            "Refino zerou a máscara da lesão — afrouxe refino.lesao "
            "(a lesão pode ser pequena)."
        )
    save_image(array_to_image(lesion_clean, lesion_img, np.uint8), case.mask_lesion_clean)
    log.info(
        "Estágio 5: lesão refinada (%d -> %d voxels).",
        int(lesion.sum()), int(lesion_clean.sum()),
    )


# --------------------------------------------------------------------------- #
# (6) Geração de malha (superfície). FEA/tetraedralização = fase 2.
# --------------------------------------------------------------------------- #
def stage6_mesh(case: Case, profile: dict) -> None:
    mesh_cfg = profile.get("mesh", {})
    level = float(mesh_cfg.get("nivel_marching_cubes", 0.5))
    sm = int(mesh_cfg.get("suavizacao_iteracoes", 30))
    fa = float(mesh_cfg.get("feature_angle", 60.0))

    organ_mesh = _mesh_from_mask(case.mask_organ_clean, level, sm, fa)
    if organ_mesh is None:
        raise PipelineError(
            "Malha do órgão vazia — máscara do órgão sem conteúdo após refino."
        )
    organ_mesh.save(str(case.mesh_organ))
    log.info(
        "Estágio 6: malha do órgão (%d vértices, %d faces).",
        organ_mesh.n_points, organ_mesh.n_cells,
    )

    lesion_mesh = _mesh_from_mask(case.mask_lesion_clean, level, sm, fa)
    if lesion_mesh is not None:
        lesion_mesh.save(str(case.mesh_lesion))
        log.info(
            "Estágio 6: malha da lesão (%d vértices, %d faces).",
            lesion_mesh.n_points, lesion_mesh.n_cells,
        )
    else:
        log.info("Estágio 6: sem malha de lesão (máscara vazia).")


# --------------------------------------------------------------------------- #
# (7) Exportação STL (LPS) + publicação para o visualizador web
# --------------------------------------------------------------------------- #
def stage7_export_publish(case: Case, profile: dict) -> None:
    case.outputs.mkdir(parents=True, exist_ok=True)
    mesh_cfg = profile.get("mesh", {})
    plan = [
        ("orgao", case.mesh_organ, mesh_cfg.get("cor_orgao", "#C8A27D")),
        ("lesao", case.mesh_lesion, mesh_cfg.get("cor_lesao", "#D7263D")),
    ]

    items = []
    for role, vtp, color in plan:
        if not vtp.exists():
            continue
        mesh = pv.read(str(vtp))
        stl = case.outputs / f"{profile['id']}_{role}.stl"
        try:
            mesh.save(str(stl))  # API correta (corrige o pv.save_mesh_as inexistente)
        except Exception as e:  # noqa: BLE001
            raise PipelineError(f"Falha ao exportar STL {stl}: {e}") from e
        items.append({"role": role, "stl": stl.name, "color": color})
        log.info("Estágio 7: STL exportado -> %s", stl)

    if not items:
        raise PipelineError("Nenhuma malha para exportar.")

    manifest = case.read_manifest()
    viewer = {
        "case_id": manifest.get("case_id"),
        "organ": profile["id"],
        "coordinate_system": profile.get("exportacao", {}).get(
            "sistema_coordenadas", "LPS"
        ),
        "regulatory_state": manifest.get("regulatory_state", "PESQUISA"),
        "disclaimer": "Uso em pesquisa/educação. NÃO destinado a decisão clínica.",
        "meshes": items,
    }
    (case.outputs / "viewer_manifest.json").write_text(
        json.dumps(viewer, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "Estágio 7: manifesto do visualizador escrito em %s",
        case.outputs / "viewer_manifest.json",
    )
