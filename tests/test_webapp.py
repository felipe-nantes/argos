from pathlib import Path

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


def test_find_best_series_prefers_profile_modality(tmp_path):
    # Envio misto CT+MR (dataset CHAOS): a série CT é MAIOR, mas o perfil do fígado
    # é MR — find_best_series deve escolher a série MR, não a CT (regressão do bug
    # "Modalidade (CT) não bate" que dava ANÁLISE NÃO CONCLUÍDA).
    import numpy as np

    from tools.make_synthetic_case import write_dicom_series

    write_dicom_series(tmp_path / "ct", np.random.default_rng(0).integers(0, 200, (8, 16, 16)), modality="CT")
    write_dicom_series(tmp_path / "mr", np.random.default_rng(1).integers(0, 200, (5, 16, 16)), modality="MR")
    files, n = server.find_best_series(tmp_path)
    assert n == 5, f"deveria pegar a série MR (5 cortes), não a CT (8); pegou {n}"
    assert server._modality_of(files) == "MR"


def test_find_best_series_empty_when_only_incompatible_modality(tmp_path):
    # Só CT no envio, perfil é MR -> nenhuma série compatível -> vazio (mensagem
    # honesta "não encontramos série de RM", em vez de abortar fundo no stage1).
    import numpy as np

    from tools.make_synthetic_case import write_dicom_series

    write_dicom_series(tmp_path / "ct", np.random.default_rng(2).integers(0, 200, (6, 16, 16)), modality="CT")
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


def test_viewer_result_exposes_review_url_only_when_model_is_ready():
    report = {"report": {"resultado_hipotese": "NEGATIVA"}}
    ready = server._viewer_result(report, "abc123", True)
    assert ready["viewer_ready"] is True
    assert ready["viewer_url"].endswith("&job=abc123")
    assert ready["approval"] == {"status": "pending"}
    unavailable = server._viewer_result(report, "abc123", False)
    assert unavailable["viewer_url"] is None


def test_model_endpoints_and_manual_approval(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    job_id = "abc123"
    outputs = tmp_path / job_id / "case" / "outputs"
    outputs.mkdir(parents=True)
    stl = outputs / "figado_orgao.stl"
    stl.write_bytes(b"solid liver\nendsolid liver\n")
    (outputs / "viewer_manifest.json").write_text(
        json.dumps({"meshes": [{"role": "orgao", "stl": stl.name, "color": "#ffffff"}]}),
        "utf-8",
    )
    server._jobs[job_id] = {
        "state": "done", "step": "concluido", "progress": 100,
        "result": {}, "approval": {"status": "pending"},
    }
    client = TestClient(server.app)
    assert client.get(f"/api/jobs/{job_id}/model/viewer_manifest.json").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/model/{stl.name}").content == stl.read_bytes()
    response = client.post(f"/api/jobs/{job_id}/approval", json={"status": "approved"})
    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    saved = json.loads((outputs / "approval.json").read_text("utf-8"))
    assert saved["review_type"] == "human_visual_review"


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


def test_benchmark_metrics_keep_failures_and_inconclusives_visible():
    results = [
        {"truth": "positive", "prediction": "POSITIVA", "status": "decisive"},
        {"truth": "positive", "prediction": "NEGATIVA", "status": "decisive"},
        {"truth": "negative", "prediction": "NEGATIVA", "status": "decisive"},
        {"truth": "negative", "prediction": "POSITIVA", "status": "decisive"},
        {"truth": "positive", "prediction": "INCONCLUSIVA", "status": "inconclusive"},
        {"truth": "negative", "prediction": None, "status": "failed"},
    ]
    metrics = server.calculate_benchmark_metrics(results)
    assert metrics["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 2, "fn": 2}
    assert metrics["accuracy"] == 0.3333
    assert metrics["sensitivity"] == 0.3333
    assert metrics["specificity"] == 0.3333
    assert metrics["precision"] == 0.3333
    assert metrics["f1_score"] == 0.3333
    assert metrics["coverage_rate"] == 0.6667
    assert metrics["completion_rate"] == 0.8333
    assert metrics["inconclusive_cases"] == 1
    assert metrics["failed_cases"] == 1
    assert metrics["scoring_policy"] == "inconclusive_and_failed_count_as_errors"
    assert metrics["target"]["met"] is False
    assert metrics["decisive_only"]["confusion_matrix"] == {
        "tp": 1, "tn": 1, "fp": 1, "fn": 1,
    }


def test_benchmark_target_requires_both_classes_at_75_percent():
    results = [
        *[{"truth": "positive", "prediction": "POSITIVA", "status": "decisive"}] * 3,
        {"truth": "positive", "prediction": "INCONCLUSIVA", "status": "inconclusive"},
        *[{"truth": "negative", "prediction": "NEGATIVA", "status": "decisive"}] * 3,
        {"truth": "negative", "prediction": None, "status": "failed"},
    ]
    metrics = server.calculate_benchmark_metrics(results)
    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 0.75
    assert metrics["target"]["met"] is True
    assert metrics["confidence_intervals_95"]["sensitivity"] is not None


def test_benchmark_metrics_return_none_when_class_is_absent():
    metrics = server.calculate_benchmark_metrics([
        {"truth": "negative", "prediction": "NEGATIVA", "status": "decisive"},
    ])
    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["sensitivity"] is None
    assert metrics["precision"] is None
    assert metrics["f1_score"] is None
    assert metrics["target"]["met"] is False


def test_benchmark_upload_maps_files_to_cases(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(server, "process_benchmark", lambda *a, **k: None)
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    client = TestClient(server.app)
    manifest = {
        "dataset_name": "Coorte teste",
        "dataset_kind": "mixed",
        "cases": [
            {"id": "caso-a", "label": "positive", "file_indices": [0, 1]},
            {"id": "caso-b", "label": "negative", "file_indices": [2]},
        ],
    }
    response = client.post(
        "/api/benchmarks",
        files=[
            ("files", ("a1.dcm", b"a1", "application/dicom")),
            ("files", ("a2.dcm", b"a2", "application/dicom")),
            ("files", ("b1.dcm", b"b1", "application/dicom")),
        ],
        data={"manifest": json.dumps(manifest)},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_cases"] == 2
    benchmark_id = payload["benchmark_id"]
    status = client.get(f"/api/benchmarks/{benchmark_id}")
    assert status.status_code == 200
    assert status.json()["total"] == 2
    root = tmp_path / "benchmarks" / benchmark_id / "_upload"
    assert len(list((root / "0001").iterdir())) == 2
    assert len(list((root / "0002").iterdir())) == 1


def test_benchmark_upload_accepts_more_than_default_starlette_file_cap(monkeypatch, tmp_path):
    """Starlette limita multipart a max_files=1000 por padrão; um dataset de
    benchmark real (muitos exames x muitas fatias) estoura isso facilmente.
    O endpoint precisa aceitar mais, via MAX_UPLOAD_FILES (ver webapp/server.py)."""
    import json

    monkeypatch.setattr(server, "process_benchmark", lambda *a, **k: None)
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    client = TestClient(server.app)
    n = 1200  # acima do max_files=1000 default do Starlette; abaixo de MAX_UPLOAD_FILES
    manifest = {
        "dataset_name": "Dataset grande",
        "dataset_kind": "positive",
        "cases": [{"id": "caso-a", "label": "positive", "file_indices": list(range(n))}],
    }
    response = client.post(
        "/api/benchmarks",
        files=[("files", (f"slice_{i:05d}.dcm", b"x", "application/dicom")) for i in range(n)],
        data={"manifest": json.dumps(manifest)},
    )
    assert response.status_code == 200
    assert response.json()["total_cases"] == 1


def test_benchmark_case_uses_absolute_case_dir_and_scores(monkeypatch, tmp_path):
    """Regressão: a segmentação do benchmark roda por um launcher com cwd=%TEMP%.
    Se o case_dir for relativo, a saída cai fora do repo e _seg_done() nunca a
    encontra, marcando TODO exame como falha. O case_dir precisa ser absoluto."""
    import subprocess

    # Reproduz produção: WORKSPACE é RELATIVO ("casos/webapp"). Sem .resolve() no
    # código, o case_dir sairia relativo — é exatamente isso que o teste captura.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "WORKSPACE", Path("casos/webapp"))
    seen = {}

    def fake_segment(series_dir, case_dir, device, timeout):
        # o bug real era o case_dir chegar relativo aqui
        seen["absolute"] = Path(case_dir).is_absolute()
        # simula uma segmentação bem-sucedida gravando os artefatos esperados
        Path(case_dir).mkdir(parents=True, exist_ok=True)
        (Path(case_dir) / "volume.nii.gz").write_bytes(b"vol")
        (Path(case_dir) / "mask_organ.nii.gz").write_bytes(b"mask")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def fake_run(cmd, timeout, cwd=None):  # a triagem MedGemma (subprocesso)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def fake_load_report(path):
        return {"report": {"resultado_hipotese": "POSITIVA", "confianca": "alta",
                           "resumo_do_achado": "x"}}

    monkeypatch.setattr(server, "_segment", fake_segment)
    monkeypatch.setattr(server, "_run", fake_run)
    monkeypatch.setattr(server, "_load_report", fake_load_report)

    raw_case = tmp_path / "raw" / "0001"
    raw_case.mkdir(parents=True)
    dcm_files = []
    for i in range(3):
        f = raw_case / f"IMG-{i}.dcm"
        f.write_bytes(b"fake-dicom")
        dcm_files.append(str(f))
    monkeypatch.setattr(server, "find_best_series", lambda d: (dcm_files, 3))

    result = server._run_benchmark_case("bench01", 1, {"id": "c1", "label": "positive"}, raw_case)

    assert seen["absolute"] is True, "case_dir passado à segmentação deve ser absoluto"
    assert result["status"] == "decisive"
    assert result["prediction"] == "POSITIVA"
    assert result["correct"] is True


def test_benchmark_upload_rejects_unmapped_file(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    client = TestClient(server.app)
    manifest = {
        "dataset_name": "Inválido",
        "dataset_kind": "positive",
        "cases": [{"id": "caso-a", "label": "positive", "file_indices": [0]}],
    }
    response = client.post(
        "/api/benchmarks",
        files=[
            ("files", ("a.dcm", b"a", "application/dicom")),
            ("files", ("b.dcm", b"b", "application/dicom")),
        ],
        data={"manifest": json.dumps(manifest)},
    )
    assert response.status_code == 400
    assert "Todos os arquivos" in response.json()["detail"]


def test_benchmark_report_downloads_json_and_csv(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    benchmark_id = "abc123"
    root = tmp_path / "benchmarks" / benchmark_id
    root.mkdir(parents=True)
    report = {
        "benchmark_id": benchmark_id,
        "cases": [{
            "case_id": "caso-a", "truth": "positive", "prediction": "POSITIVA",
            "status": "decisive", "correct": True, "confidence": "alta",
            "duration_seconds": 12.5, "error": None,
        }],
    }
    (root / "benchmark_report.json").write_text(json.dumps(report), "utf-8")
    client = TestClient(server.app)
    assert client.get(f"/api/benchmarks/{benchmark_id}/report.json").status_code == 200
    csv_response = client.get(f"/api/benchmarks/{benchmark_id}/report.csv")
    assert csv_response.status_code == 200
    assert "case_id,truth,prediction" in csv_response.text
    assert "caso-a,positive,POSITIVA" in csv_response.text
