#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gateway HTTP local do MedGemma (contrato dtwin-medgemma-v1).

O modelo é carregado com a API oficial Transformers e quantização NF4. Falhas de
licença, download, CUDA ou memória ficam expostas em /health; nunca há resposta
clínica simulada ou fallback para outro modelo.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from dtwin.core import PipelineError
from dtwin.medgemma_client import load_screening_config

log = logging.getLogger("dtwin.medgemma.server")


class ImagePayload(BaseModel):
    mime_type: Literal["image/png"]
    base64: str


class GenerationPayload(BaseModel):
    max_output_tokens: int = Field(ge=1)


class GeneratePayload(BaseModel):
    contract: Literal["dtwin-medgemma-v1"]
    model_id: str
    model_version: str
    prompt: str
    image: ImagePayload
    generation: GenerationPayload


class MedGemmaRuntime:
    def __init__(self, config: dict):
        self.config = config
        self.med = config["medgemma"]
        self.model = None
        self.processor = None
        self.load_error: str | None = None
        self.lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA não está disponível; CPU fallback é proibido.")
            if self.med.get("device") != "cuda":
                raise RuntimeError("O backend operacional exige device=cuda.")
            minimum_vram = float(self.med.get("minimum_cuda_memory_gb", 6.0))
            available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if available_vram < minimum_vram:
                raise RuntimeError(
                    f"VRAM insuficiente ({available_vram:.1f} GiB < {minimum_vram:.1f} GiB)."
                )
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("A GPU não oferece suporte BF16 exigido pelo backend.")
            quantization = self.med.get("quantization")
            kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
            if quantization == "bitsandbytes-nf4":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            elif quantization not in {None, "none"}:
                raise RuntimeError(f"Quantização não suportada: {quantization!r}")

            model_id = self.med["model_id"]
            local_files_only = bool(self.med.get("local_files_only", False))
            log.info("Carregando %s (%s)...", model_id, quantization or "sem quantização")
            self.processor = AutoProcessor.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, local_files_only=local_files_only, **kwargs
            )
            device_map = getattr(self.model, "hf_device_map", {}) or {}
            forbidden_devices = {
                str(device).lower()
                for device in device_map.values()
                if str(device).lower() in {"cpu", "disk"}
            }
            if forbidden_devices:
                raise RuntimeError(
                    "O modelo foi parcialmente descarregado para CPU/disco; "
                    "fallback é proibido neste backend."
                )
            if not device_map and getattr(self.model.device, "type", None) != "cuda":
                raise RuntimeError("O modelo não foi carregado integralmente na GPU.")
            self.model.eval()
        except AttributeError:
            # Nome correto nas versões atuais; bloco separado mantém a falha clara
            # caso uma versão incompatível de Transformers seja instalada.
            self.model = None
            self.processor = None
            self.load_error = (
                "Transformers incompatível: AutoModelForImageTextToText indisponível. "
                "Reinstale o extra [medgemma]."
            )
            log.exception(self.load_error)
            return
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.processor = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("Falha ao carregar MedGemma")
            return
        self.load_error = None
        log.info("MedGemma carregado com sucesso na GPU.")

    def generate(self, image, prompt: str, max_new_tokens: int) -> str:
        if not self.loaded:
            raise RuntimeError(self.load_error or "Modelo não carregado.")
        import torch

        # O template oficial do MedGemma 1.5 usa uma única mensagem `user`.
        # Manter as salvaguardas no próprio texto evita incompatibilidade com
        # templates Gemma que não aceitam a função `system`.
        safe_prompt = (
            "Você é um assistente de pesquisa em imagem médica. "
            "Não emita diagnóstico definitivo nem recomendação clínica.\n\n" + prompt
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": safe_prompt},
                ],
            }
        ]
        with self.lock, torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )[0][input_len:]
            return self.processor.decode(generated, skip_special_tokens=True).strip()


def create_app(config_path: Path):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise PipelineError("Backend ausente. Instale com: pip install -e .[medgemma]") from exc
    from PIL import Image

    config = load_screening_config(config_path)
    runtime = MedGemmaRuntime(config)

    @asynccontextmanager
    async def lifespan(_app):
        runtime.load()
        yield
        runtime.model = None
        runtime.processor = None

    app = FastAPI(title="Digital Twin MedGemma Gateway", version="1", lifespan=lifespan)

    @app.get("/health")
    def health():
        import torch

        return {
            "status": "ready" if runtime.loaded else "failed",
            "contract": "dtwin-medgemma-v1",
            "model_loaded": runtime.loaded,
            "model_id": runtime.med["model_id"],
            "model_version": runtime.med["model_version"],
            "quantization": runtime.med.get("quantization"),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "load_error": runtime.load_error,
            "research_only": True,
        }

    @app.post("/generate")
    def generate(payload: GeneratePayload):
        if not runtime.loaded:
            raise HTTPException(status_code=503, detail=runtime.load_error or "Modelo não carregado")
        if payload.model_id != runtime.med["model_id"]:
            raise HTTPException(status_code=409, detail="model_id não corresponde ao modelo carregado")
        if payload.model_version != runtime.med["model_version"]:
            raise HTTPException(status_code=409, detail="model_version não corresponde ao modelo carregado")
        if len(payload.prompt) > int(runtime.med.get("max_prompt_chars", 12000)):
            raise HTTPException(status_code=413, detail="Prompt excede max_prompt_chars")
        try:
            raw = base64.b64decode(payload.image.base64, validate=True)
            if len(raw) > int(runtime.med["max_input_bytes"]):
                raise ValueError("Imagem excede max_input_bytes")
            source = Image.open(io.BytesIO(raw))
            if source.format != "PNG":
                raise ValueError("A imagem deve ser um PNG válido")
            width, height = source.size
            if width * height > int(runtime.med.get("max_image_pixels", 4_000_000)):
                raise ValueError("Imagem excede max_image_pixels")
            source.load()
            image = source.convert("RGB")
            max_tokens = min(
                int(payload.generation.max_output_tokens),
                int(runtime.med["max_output_tokens"]),
            )
            output = runtime.generate(image, payload.prompt, max_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("Falha de inferência MedGemma")
            raise HTTPException(status_code=500, detail=f"Inferência falhou: {type(exc).__name__}") from exc
        return {
            "model_id": runtime.med["model_id"],
            "model_version": runtime.med["model_version"],
            "output": output,
        }

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backend local MedGemma (modo Pesquisa).")
    parser.add_argument("--config", default="configs/medgemma_local_4b.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("[ABORTADO] O backend local só pode escutar em loopback.")
        return 1
    try:
        import uvicorn

        app = create_app(Path(args.config))
    except PipelineError as exc:
        print(f"[ABORTADO] {exc}")
        return 1
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
