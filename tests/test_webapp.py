from fastapi.testclient import TestClient

from dtwin.core import PipelineError
from webapp import server


def test_graceful_payload_shape():
    g = server._graceful("motivo", "detalhe")
    assert g["status"] == "nao_concluido"
    assert g["requires_human_review"] is True
    assert "pesquisa" in g["disclaimer"].lower()
    # nunca contém um estado clínico fabricado
    assert "resultado_hipotese" not in g


def test_friendly_messages_are_human_and_nonclinical():
    assert "MedGemma" in server._friendly(PipelineError("MedGemma backend not configured. Aborting analysis."))
    assert "fígado" in server._friendly(PipelineError("Falha na segmentação automática (total_mr/liver): x"))
    assert "RM" in server._friendly(PipelineError("Modalidade do exame (CT) não bate com o perfil"))
    assert "segurança" in server._friendly(PipelineError("Resposta MedGemma contém diagnóstico definitivo"))
    # fallback genérico
    assert server._friendly(PipelineError("algo aleatório")) == "Não foi possível concluir a análise deste exame."


def test_find_best_series_empty_when_no_dicom(tmp_path):
    (tmp_path / "leia.txt").write_text("nao é dicom")
    files, n = server.find_best_series(tmp_path)
    assert files == [] and n == 0


def test_load_report_accepts_valid_report_regardless_of_returncode(tmp_path):
    # relatório válido no disco = sucesso, mesmo que o subprocesso tenha crashado no shutdown
    import json
    rp = tmp_path / "medgemma_report.json"
    rp.write_text(json.dumps({"report": {"resultado_hipotese": "NEGATIVA"}, "status": "pending_review"}), "utf-8")
    data = server._load_report(rp)
    assert data is not None and data["report"]["resultado_hipotese"] == "NEGATIVA"


def test_load_report_rejects_missing_or_incomplete(tmp_path):
    import json
    assert server._load_report(tmp_path / "ausente.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"status": "x"}), "utf-8")  # sem report.resultado_hipotese
    assert server._load_report(bad) is None


def test_success_status_overrides_report_envelope_status():
    # o envelope tem status='pending_review'; o frontend detecta sucesso por
    # status=='concluido', então o marcador de conclusão deve prevalecer.
    envelope = {"status": "pending_review", "report": {"resultado_hipotese": "NEGATIVA"},
                "model_version": "MedGemma 1.5 4B Instruction-Tuned", "disclaimer": "..."}
    result = server._success_result(envelope)
    assert result["status"] == "concluido"
    assert result["report"]["resultado_hipotese"] == "NEGATIVA"
    assert result["model_version"].startswith("MedGemma")


def test_seg_done_requires_volume_and_mask(tmp_path):
    assert server._seg_done(tmp_path) is False
    (tmp_path / "volume.nii.gz").write_bytes(b"x")
    assert server._seg_done(tmp_path) is False
    (tmp_path / "mask_organ.nii.gz").write_bytes(b"x")
    assert server._seg_done(tmp_path) is True


def test_analyze_creates_job_and_status_is_queryable(monkeypatch, tmp_path):
    # não roda o pipeline real (GPU/MedGemma): substitui o worker por no-op
    monkeypatch.setattr(server, "process_job", lambda *a, **k: None)
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    client = TestClient(server.app)
    resp = client.post(
        "/api/analyze",
        files=[("files", ("IMG-0001.dcm", b"fake-dicom-bytes", "application/dicom"))],
        data={"relpaths": '["estudo/IMG-0001.dcm"]'},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    status = client.get(f"/api/status/{job_id}")
    assert status.status_code == 200
    assert status.json()["state"] in ("queued", "processing", "done")
    assert client.get("/api/status/inexistente").status_code == 404
    # o upload foi materializado preservando a subpasta
    assert (tmp_path / job_id / "_upload" / "estudo" / "IMG-0001.dcm").is_file()
