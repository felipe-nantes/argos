#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI/orquestração do fluxo RM + fígado -> hipótese visual MedGemma."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .core import PipelineError, load_profile, now_utc, sha256_of
from .medgemma_client import (
    build_medgemma_prompt,
    create_medgemma_client,
    load_screening_config,
    model_trace,
    validate_medgemma_report,
)
from .medgemma_panel import generate_liver_panel
from .medgemma_panel_multiphase import derive_texture_channels, generate_liver_panel_multiphase


REPORT_FILENAME = "medgemma_report.json"


def build_report_envelope(
    *,
    case_id: str,
    config: dict[str, Any],
    panel_filename: str,
    panel_manifest_filename: str,
    panel_manifest: dict[str, Any],
    screening_config_sha256: str,
    report: dict[str, Any],
    durations_seconds: dict[str, float] | None = None,
) -> dict[str, Any]:
    validated = validate_medgemma_report(report, config["report"])
    return {
        "case_id": case_id,
        "status": "pending_review",
        "regulatory_mode": "RESEARCH",
        **model_trace(config),
        "input_panel": panel_filename,
        "input_panel_sha256": panel_manifest["panel_sha256"],
        "input_volume_sha256": panel_manifest["input_volume_sha256"],
        "input_liver_mask_sha256": panel_manifest["input_liver_mask_sha256"],
        "screening_config_sha256": screening_config_sha256,
        "panel_manifest": panel_manifest_filename,
        "lesion_pre_marked": False,
        "report": validated,
        "requires_human_review": True,
        "disclaimer": config["report"]["disclaimer"],
        "created_at": now_utc(),
        "durations_seconds": durations_seconds or {},
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"{label} inválido ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} deve ser um objeto JSON: {path}")
    return value


def run_screening(
    *,
    volume_path: Path | None = None,
    liver_mask_path: Path,
    profile_path: Path,
    medgemma_config_path: Path,
    output_dir: Path,
    case_manifest_path: Path | None = None,
    phase_paths: dict[str, Path] | None = None,
    panel_only: bool = False,
    visible_phi_confirmed: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    total_started = time.monotonic()
    liver_mask_path = Path(liver_mask_path)
    output_dir = Path(output_dir)
    profile = load_profile(profile_path)
    config = load_screening_config(medgemma_config_path)
    screening_config_sha256 = sha256_of(Path(medgemma_config_path))
    mode = str(config.get("panel", {}).get("mode", "single_grayscale"))

    panel_started = time.monotonic()
    if mode == "multiphase_fusion":
        if not phase_paths:
            raise PipelineError(
                "A config selecionada usa panel.mode=multiphase_fusion; informe as fases "
                "com --phase nome=caminho (ex.: --phase art=... --phase pv=... --phase del=...)."
            )
        if case_manifest_path is None:
            raise PipelineError("Modo multifásico exige --case-manifest ou --case-dir.")
        panel = generate_liver_panel_multiphase(
            phase_paths=phase_paths,
            liver_mask_path=liver_mask_path,
            case_manifest_path=Path(case_manifest_path),
            organ_profile=profile,
            screening_config=config,
            output_dir=output_dir,
            model_trace=model_trace(config),
            visible_phi_confirmed=visible_phi_confirmed,
        )
    elif mode == "texture_fusion":
        # Fase única: deriva 3 canais (sinal + CLAHE + heterogeneidade) do próprio
        # volume e reaproveita a fusão RGB (que já faz crop + janela no fígado).
        if volume_path is None:
            raise PipelineError("Modo texture_fusion exige --volume (ou --case-dir).")
        volume_path = Path(volume_path)
        case_manifest_path = (
            Path(case_manifest_path)
            if case_manifest_path
            else volume_path.parent / "manifest.json"
        )
        channels = derive_texture_channels(
            volume_path, liver_mask_path, output_dir / "_channels", config.get("panel", {})
        )
        panel = generate_liver_panel_multiphase(
            phase_paths=channels,
            liver_mask_path=liver_mask_path,
            case_manifest_path=case_manifest_path,
            organ_profile=profile,
            screening_config=config,
            output_dir=output_dir,
            model_trace=model_trace(config),
            visible_phi_confirmed=visible_phi_confirmed,
        )
    else:
        if phase_paths:
            raise PipelineError(
                "Fases (--phase) foram fornecidas, mas a config não é multiphase_fusion. "
                "Use uma config com panel.mode=multiphase_fusion ou remova --phase."
            )
        if volume_path is None:
            raise PipelineError("Modo de fase única exige --volume (ou --case-dir).")
        volume_path = Path(volume_path)
        case_manifest_path = (
            Path(case_manifest_path)
            if case_manifest_path
            else volume_path.parent / "manifest.json"
        )
        panel = generate_liver_panel(
            volume_path=volume_path,
            liver_mask_path=liver_mask_path,
            case_manifest_path=case_manifest_path,
            organ_profile=profile,
            screening_config=config,
            output_dir=output_dir,
            model_trace=model_trace(config),
            visible_phi_confirmed=visible_phi_confirmed,
        )
    panel_duration = time.monotonic() - panel_started
    panel_manifest = _read_json(panel.manifest_path, "Manifesto do painel")
    result = {
        "case_id": panel_manifest["case_id"],
        "status": "panel_ready" if panel_only else "pending_model",
        "panel_path": str(panel.panel_path),
        "panel_manifest_path": str(panel.manifest_path),
        "report_path": None,
        "requires_human_review": True,
        "panel_sha256": panel_manifest["panel_sha256"],
    }
    if panel_only:
        return result

    privacy = config.get("privacy", {})
    if privacy.get("require_visible_phi_confirmation", True) and not visible_phi_confirmed:
        raise PipelineError(
            "Confirmação visual de ausência de PHI queimada nos pixels é obrigatória. "
            "Revise o painel e reexecute com --confirm-no-visible-phi."
        )
    if panel_manifest.get("visible_phi_confirmed") is not True:
        raise PipelineError("O manifest do painel não registrou a confirmação visual de PHI.")
    if sha256_of(panel.panel_path) != panel_manifest.get("panel_sha256"):
        raise PipelineError("O painel mudou após a geração/revisão; inferência abortada.")
    prompt = build_medgemma_prompt(config)
    medgemma_client = client if client is not None else create_medgemma_client(config)
    raw_report = medgemma_client.generate(panel.panel_path, prompt)
    durations = {
        "panel_generation": round(panel_duration, 4),
        **dict(getattr(medgemma_client, "last_timings", {}) or {}),
        "screening_total": round(time.monotonic() - total_started, 4),
    }
    envelope = build_report_envelope(
        case_id=panel_manifest["case_id"],
        config=config,
        panel_filename=panel.panel_path.name,
        panel_manifest_filename=panel.manifest_path.name,
        panel_manifest=panel_manifest,
        screening_config_sha256=screening_config_sha256,
        report=raw_report,
        durations_seconds=durations,
    )
    report_path = output_dir / REPORT_FILENAME
    temp_path = output_dir / f".{REPORT_FILENAME}.tmp"
    temp_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(report_path)
    result.update(
        status="pending_review",
        report_path=str(report_path),
        model_version=config["medgemma"]["model_version"],
        model_parameter_scale=config["medgemma"]["model_parameter_scale"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dtwin.medgemma_screening",
        description=(
            "Triagem visual hepática MedGemma em modo Pesquisa. "
            "Não é diagnóstico nem laudo médico."
        ),
    )
    parser.add_argument(
        "--case-dir",
        help=(
            "pasta de caso criada pelo prepare; usa volume.nii.gz, mask_organ.nii.gz, "
            "manifest.json e outputs/medgemma automaticamente"
        ),
    )
    parser.add_argument("--volume", help="volume.nii.gz des-identificado (fase única)")
    parser.add_argument(
        "--phase",
        action="append",
        metavar="nome=caminho",
        help="fase de RM co-registrada para o painel multifásico "
        "(ex.: --phase art=... --phase pv=... --phase del=...); repetível",
    )
    parser.add_argument("--liver-mask", help="mask_organ.nii.gz revisada")
    parser.add_argument("--case-manifest", help="manifest.json do caso (default: ao lado do volume)")
    parser.add_argument("--profile", default="profiles/figado.yaml", help="perfil do órgão")
    parser.add_argument(
        "--medgemma-config",
        default="configs/medgemma_4b.yaml",
        help="configuração versionada do backend/modelo",
    )
    parser.add_argument("--output", help="diretório dos artefatos MedGemma")
    parser.add_argument(
        "--panel-only",
        action="store_true",
        help="gera/valida painel e manifest sem chamar o modelo",
    )
    parser.add_argument(
        "--confirm-no-visible-phi",
        action="store_true",
        help="confirma revisão humana do painel quanto a PHI queimada nos pixels",
    )
    return parser


def _parse_phases(phase_args) -> dict[str, Path] | None:
    if not phase_args:
        return None
    phases: dict[str, Path] = {}
    for item in phase_args:
        if "=" not in item:
            raise PipelineError(f"--phase deve ser nome=caminho; recebido: {item!r}.")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise PipelineError(f"--phase inválido: {item!r}.")
        if name in phases:
            raise PipelineError(f"Fase duplicada em --phase: {name!r}.")
        phases[name] = Path(path)
    return phases


def _resolve_cli_paths(args, has_phases: bool) -> tuple[Path | None, Path, Path | None, Path]:
    explicit = (args.volume, args.liver_mask, args.case_manifest, args.output)
    if args.case_dir:
        if any(value is not None for value in explicit):
            raise PipelineError(
                "Use --case-dir sozinho ou informe --volume/--liver-mask/--output; "
                "não misture os dois modos."
            )
        case_dir = Path(args.case_dir)
        volume = None if has_phases else case_dir / "volume.nii.gz"
        return (
            volume,
            case_dir / "mask_organ.nii.gz",
            case_dir / "manifest.json",
            case_dir / "outputs" / "medgemma",
        )
    if not args.liver_mask or not args.output:
        raise PipelineError(
            "Informe --case-dir ou o conjunto --liver-mask e --output."
        )
    if not has_phases and not args.volume:
        raise PipelineError("Fase única exige --volume (ou use --phase para multifásico).")
    return (
        Path(args.volume) if args.volume else None,
        Path(args.liver_mask),
        Path(args.case_manifest) if args.case_manifest else None,
        Path(args.output),
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print("=" * 70)
    print(" MEDGEMMA — TRIAGEM VISUAL HEPÁTICA (MODO PESQUISA)")
    print(" NÃO é diagnóstico. NÃO é laudo médico. Revisão humana obrigatória.")
    print("=" * 70)
    try:
        phase_paths = _parse_phases(args.phase)
        volume_path, liver_mask_path, case_manifest_path, output_dir = _resolve_cli_paths(
            args, has_phases=phase_paths is not None
        )
        result = run_screening(
            volume_path=volume_path,
            liver_mask_path=liver_mask_path,
            case_manifest_path=case_manifest_path,
            phase_paths=phase_paths,
            profile_path=Path(args.profile),
            medgemma_config_path=Path(args.medgemma_config),
            output_dir=output_dir,
            panel_only=args.panel_only,
            visible_phi_confirmed=args.confirm_no_visible_phi,
        )
    except PipelineError as exc:
        print(f"\n[ABORTADO] {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
