#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gateway HTTP local do MedGemma (contrato dtwin-medgemma-v1).

O modelo é carregado com a API oficial Transformers. Em GPU (device=cuda) usa
quantização NF4; em Apple Silicon (device=mps, opt-in de Pesquisa) usa bf16 sem
quantização, com carga integral e as mesmas travas anti-offload. Falhas de
licença, download, device ou memória ficam expostas em /health; nunca há resposta
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

            device = self.med.get("device")
            quantization = self.med.get("quantization")
            model_id = self.med["model_id"]
            local_files_only = bool(self.med.get("local_files_only", False))

            if device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA não está disponível; CPU fallback é proibido.")
                minimum_vram = float(self.med.get("minimum_cuda_memory_gb", 6.0))
                available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if available_vram < minimum_vram:
                    raise RuntimeError(
                        f"VRAM insuficiente ({available_vram:.1f} GiB < {minimum_vram:.1f} GiB)."
                    )
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("A GPU não oferece suporte BF16 exigido pelo backend.")
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
            elif device == "mps":
                # Opt-in explícito para Apple Silicon (modo Pesquisa): carga INTEGRAL
                # em MPS, bf16, SEM quantização (bitsandbytes exige CUDA) e SEM
                # device_map/offload. As mesmas travas anti-fallback do caminho CUDA
                # continuam valendo — nenhuma parte do modelo vai para CPU/disco.
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS não está disponível; verifique PyTorch/hardware.")
                if quantization not in {None, "none"}:
                    raise RuntimeError(
                        "Quantização bitsandbytes não é suportada em MPS; use quantization: none."
                    )
                kwargs = {"dtype": torch.bfloat16}
            else:
                raise RuntimeError("device deve ser 'cuda' ou 'mps'.")

            log.info(
                "Carregando %s (%s, device=%s)...",
                model_id, quantization or "sem quantização", device,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, local_files_only=local_files_only, **kwargs
            )
            device_map = getattr(self.model, "hf_device_map", {}) or {}
            forbidden_devices = {
                str(dev).lower()
                for dev in device_map.values()
                if str(dev).lower() in {"cpu", "disk"}
            }
            if forbidden_devices:
                raise RuntimeError(
                    "O modelo foi parcialmente descarregado para CPU/disco; "
                    "fallback é proibido neste backend."
                )
            if device == "mps":
                self.model = self.model.to("mps")
                if getattr(self.model.device, "type", None) != "mps":
                    raise RuntimeError("O modelo não foi carregado integralmente em MPS.")
            elif not device_map and getattr(self.model.device, "type", None) != "cuda":
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
        log.info("MedGemma carregado com sucesso (device=%s).", self.med.get("device"))

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


class OllamaRuntime:
    """Runtime que delega a inferência a um daemon Ollama local (GGUF/Metal).

    Expõe a mesma interface interna de MedGemmaRuntime (``loaded``/``load``/
    ``generate``), mas em vez de carregar o modelo em processo via Transformers,
    encaminha imagem+prompt para a API do Ollama. Continua modo PESQUISA e
    fail-closed: se o daemon não responder, a tag não existir ou não tiver
    capacidade de visão, ``load_error`` é setado e ``/health`` expõe a falha.
    Nunca há resposta clínica simulada nem fallback para outro modelo.
    """

    def __init__(self, config: dict):
        self.config = config
        self.med = config["medgemma"]
        self.load_error: str | None = None
        self._ready = False
        self.lock = threading.Lock()
        self.base_url = str(self.med.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/")
        self.tag = str(self.med.get("ollama_model") or self.med["model_id"])

    @property
    def loaded(self) -> bool:
        return self._ready

    def unload(self) -> None:
        self._ready = False

    def load(self) -> None:
        import json as _json
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        try:
            request = Request(
                f"{self.base_url}/api/show",
                data=_json.dumps({"name": self.tag}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=20) as response:
                info = _json.loads(response.read().decode("utf-8"))
            capabilities = info.get("capabilities") or []
            if "vision" not in capabilities:
                raise RuntimeError(
                    f"Modelo Ollama '{self.tag}' não declara capacidade de visão "
                    f"(capabilities={capabilities}); o painel exige um modelo image-text."
                )
            self._ready = True
            self.load_error = None
            log.info(
                "Ollama runtime pronto: tag=%s em %s (capabilities=%s).",
                self.tag, self.base_url, capabilities,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            self._ready = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("Falha ao preparar o runtime Ollama")

    def generate(self, image, prompt: str, max_new_tokens: int) -> str:
        import base64 as _b64
        import io as _io
        import json as _json
        from urllib.request import Request, urlopen

        if not self._ready:
            raise RuntimeError(self.load_error or "Ollama runtime não pronto.")
        # Mesmas salvaguardas no texto do caminho Transformers (template Gemma sem
        # função `system`).
        safe_prompt = (
            "Você é um assistente de pesquisa em imagem médica. "
            "Não emita diagnóstico definitivo nem recomendação clínica.\n\n" + prompt
        )
        buffer = _io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = _b64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": self.tag,
            "messages": [
                {"role": "user", "content": safe_prompt, "images": [image_b64]}
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": int(max_new_tokens)},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = int(self.med.get("timeout_seconds", 600))
        with self.lock, urlopen(request, timeout=timeout) as response:
            data = _json.loads(response.read().decode("utf-8"))
        return str((data.get("message") or {}).get("content", "")).strip()


def _build_runtime(config: dict):
    """Escolhe o runtime pelo campo medgemma.runtime (transformers|ollama)."""
    kind = str(config["medgemma"].get("runtime", "transformers")).lower()
    if kind == "ollama":
        return OllamaRuntime(config)
    if kind == "transformers":
        return MedGemmaRuntime(config)
    raise PipelineError(f"medgemma.runtime desconhecido: {kind!r} (use transformers ou ollama).")


def create_app(config_path: Path):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise PipelineError("Backend ausente. Instale com: pip install -e .[medgemma]") from exc
    from PIL import Image

    config = load_screening_config(config_path)
    runtime = _build_runtime(config)

    @asynccontextmanager
    async def lifespan(_app):
        runtime.load()
        yield
        if hasattr(runtime, "unload"):
            runtime.unload()
        else:
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
            "device": runtime.med.get("device"),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
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
