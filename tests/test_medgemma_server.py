import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from tools.medgemma_server import MedGemmaRuntime, create_app


def _png_base64():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "black").save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _jpeg_base64():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "black").save(stream, format="JPEG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_local_gateway_health_and_contract(monkeypatch):
    def fake_load(self):
        self.model = object()
        self.processor = object()
        self.load_error = None

    monkeypatch.setattr(MedGemmaRuntime, "load", fake_load)
    monkeypatch.setattr(
        MedGemmaRuntime,
        "generate",
        lambda _self, _image, _prompt, _tokens: '{"resultado_hipotese":"INCONCLUSIVA"}',
    )
    app = create_app("configs/medgemma_local_4b.yaml")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ready"
        response = client.post(
            "/generate",
            json={
                "contract": "dtwin-medgemma-v1",
                "model_id": "google/medgemma-1.5-4b-it",
                "model_version": "MedGemma 1.5 4B Instruction-Tuned",
                "prompt": "research only",
                "image": {"mime_type": "image/png", "base64": _png_base64()},
                "generation": {"max_output_tokens": 32},
            },
        )
        assert response.status_code == 200
        assert response.json()["model_id"] == "google/medgemma-1.5-4b-it"


def test_local_gateway_rejects_wrong_model(monkeypatch):
    monkeypatch.setattr(
        MedGemmaRuntime,
        "load",
        lambda self: (setattr(self, "model", object()), setattr(self, "processor", object())),
    )
    app = create_app("configs/medgemma_local_4b.yaml")
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "contract": "dtwin-medgemma-v1",
                "model_id": "wrong/model",
                "model_version": "Wrong model",
                "prompt": "research only",
                "image": {"mime_type": "image/png", "base64": _png_base64()},
                "generation": {"max_output_tokens": 32},
            },
        )
        assert response.status_code == 409


def test_local_gateway_rejects_non_png_payload(monkeypatch):
    monkeypatch.setattr(
        MedGemmaRuntime,
        "load",
        lambda self: (setattr(self, "model", object()), setattr(self, "processor", object())),
    )
    app = create_app("configs/medgemma_local_4b.yaml")
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "contract": "dtwin-medgemma-v1",
                "model_id": "google/medgemma-1.5-4b-it",
                "model_version": "MedGemma 1.5 4B Instruction-Tuned",
                "prompt": "research only",
                "image": {"mime_type": "image/png", "base64": _jpeg_base64()},
                "generation": {"max_output_tokens": 32},
            },
        )
        assert response.status_code == 400
        assert "PNG" in response.json()["detail"]
