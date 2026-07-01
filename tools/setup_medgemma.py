#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight, autenticação/licença e cache do modelo MedGemma configurado."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dtwin.medgemma_client import load_screening_config


def _runtime_preflight(config: dict) -> None:
    med = config["medgemma"]
    import torch
    import transformers

    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível; o backend proíbe fallback para CPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A GPU não oferece suporte BF16.")
    if med.get("quantization") == "bitsandbytes-nf4":
        import bitsandbytes  # noqa: F401
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    minimum = float(med.get("minimum_cuda_memory_gb", 6.0))
    if vram < minimum:
        raise RuntimeError(f"VRAM insuficiente ({vram:.1f} GiB < {minimum:.1f} GiB).")
    print(
        f"[OK] Runtime: transformers={transformers.__version__}, "
        f"GPU={torch.cuda.get_device_name(0)}, VRAM={vram:.1f} GiB, BF16=true"
    )


def _verify_local_snapshot(snapshot_path: str | Path) -> None:
    """Confirma processor/config e todos os shards sem carregar o modelo na GPU."""
    from transformers import AutoConfig, AutoProcessor

    snapshot = Path(snapshot_path)
    AutoConfig.from_pretrained(snapshot, local_files_only=True)
    AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    single = snapshot / "model.safetensors"
    index = snapshot / "model.safetensors.index.json"
    if single.is_file():
        return
    if not index.is_file():
        raise RuntimeError("Cache incompleto: pesos safetensors não encontrados.")
    data = json.loads(index.read_text(encoding="utf-8"))
    shards = {str(name) for name in data.get("weight_map", {}).values()}
    missing = sorted(name for name in shards if not (snapshot / name).is_file())
    if not shards or missing:
        detail = f"; faltando: {missing[:3]}" if missing else ""
        raise RuntimeError(f"Cache incompleto: shards do modelo ausentes{detail}.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepara os pesos gated do MedGemma.")
    parser.add_argument("--config", default="configs/medgemma_local_4b.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verifica runtime, autenticação e licença sem baixar pesos",
    )
    mode.add_argument(
        "--local-only",
        action="store_true",
        help="verifica runtime e cache local sem acessar a rede",
    )
    args = parser.parse_args(argv)
    config = load_screening_config(Path(args.config))
    model_id = config["medgemma"]["model_id"]
    try:
        from huggingface_hub import HfApi, snapshot_download

        _runtime_preflight(config)
        if args.local_only:
            path = snapshot_download(repo_id=model_id, local_files_only=True)
            _verify_local_snapshot(path)
            print(f"[OK] Modelo completo no cache local: {path}")
            return 0
        who = HfApi().whoami()
        print(f"[OK] Hugging Face autenticado como: {who.get('name', 'conta autenticada')}")
        info = HfApi().model_info(model_id)
        print(f"[OK] Acesso autorizado ao modelo: {info.id}")
        if args.verify_only:
            return 0
        path = snapshot_download(repo_id=model_id)
        _verify_local_snapshot(path)
        print(f"[OK] Modelo disponível no cache local: {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ABORTADO] Não foi possível acessar {model_id}: {type(exc).__name__}: {exc}")
        print("Aceite os termos no Hugging Face e execute 'hf auth login' nesta máquina.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
