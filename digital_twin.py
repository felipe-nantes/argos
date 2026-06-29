#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
digital_twin.py — Entrada (CLI) do pipeline de Digital Twin cirúrgico.

Esta é a camada FINA que obedece o motor (pacote dtwin/). Ela não contém lógica
de pipeline: apenas faz parse de argumentos, imprime o aviso de modo Pesquisa e
chama o motor em duas fases (prepare / finalize). Toda a lógica está no motor;
todo o comportamento por órgão está no perfil (profiles/figado.yaml).

MVP — Nível 1 (modelo anatômico). Ver a pasta contexto/ para a estratégia completa.

Uso:
  # Fase 1: ingestão + normalização + segmentação automática do órgão
  python digital_twin.py prepare <pasta_dicom> \\
         --case-dir casos/paciente001 --profile profiles/figado.yaml

  # >>> marque a LESÃO no 3D Slicer (instruções impressas ao fim do prepare) <<<

  # Fase 2: importa a lesão + refino + malha + STL + manifesto do visualizador
  python digital_twin.py finalize casos/paciente001 --profile profiles/figado.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dtwin import Engine, PipelineError

DEFAULT_PROFILE = "profiles/figado.yaml"


def _banner() -> None:
    line = "=" * 70
    print(line)
    print(" DIGITAL TWIN CIRÚRGICO — PIPELINE (MVP · Nível 1: modelo anatômico)")
    print(" MODO PESQUISA — saída NÃO destinada a decisão clínica.")
    print(line)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="digital_twin.py",
        description="Pipeline órgão-agnóstico: DICOM (RM) -> modelo 3D (órgão + lesão).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Estágios 1–4a (até a marcação da lesão).")
    p.add_argument("dicom_dir", help="Pasta com a série DICOM (RM) de um paciente.")
    p.add_argument("--case-dir", required=True, help="Pasta de trabalho do caso (saída).")
    p.add_argument("--profile", default=DEFAULT_PROFILE, help="Perfil do órgão (YAML).")
    p.add_argument(
        "--policy",
        choices=["anonymize", "pseudonymize"],
        default="anonymize",
        help="Política de privacidade (MVP: anonymize).",
    )
    p.add_argument("--device", default="gpu", help="gpu | cpu | gpu:N (TotalSegmentator).")
    p.add_argument("--fast", action="store_true", help="Modo rápido (recomendado em CPU).")

    f = sub.add_parser("finalize", help="Estágios 4b–7 (após a marcação da lesão).")
    f.add_argument("case_dir", help="Pasta de trabalho do caso (criada no prepare).")
    f.add_argument("--profile", default=DEFAULT_PROFILE, help="Perfil do órgão (YAML).")
    f.add_argument(
        "--no-lesion",
        action="store_true",
        help="Caso sem lesão (escolha explícita; não fabrica nada).",
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _banner()
    try:
        engine = Engine(Path(args.profile))
        if args.cmd == "prepare":
            case = engine.prepare(
                args.dicom_dir,
                args.case_dir,
                policy=args.policy,
                device=args.device,
                fast=args.fast,
            )
            print(f"\n[OK] 'prepare' concluído para {case.root}.")
            print("Marque a lesão no 3D Slicer (instruções acima) e rode 'finalize'.")
        else:
            case = engine.finalize(args.case_dir, no_lesion=args.no_lesion)
            print(f"\n[OK] 'finalize' concluído. Saídas em: {case.outputs}")
            print("STL(s) e viewer_manifest.json prontos para o visualizador web.")
        return 0
    except PipelineError as e:
        print(f"\n[ABORTADO] {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
