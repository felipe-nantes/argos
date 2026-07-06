import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from dtwin.benchmark.importers import load_dataset_manifest, prepare_inference_case
from dtwin.benchmark.metrics import compute_benchmark_metrics
from dtwin.benchmark.reporting import write_run_outputs
from dtwin.benchmark.runner import ExperimentConfig, recalculate_existing_run
from dtwin.medgemma_benchmark import main
from dtwin.medgemma_client import effective_config_sha256, load_screening_config

# Config real e válida (o runner agora valida a config como o subprocesso real faria).
BASELINE_CONFIG = Path("configs/medgemma_local_4b.yaml")


def _fixture(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    image = sitk.GetImageFromArray(np.ones((4, 5, 6), dtype=np.uint8))
    sitk.WriteImage(image, str(root / "volume.nii.gz"))
    sitk.WriteImage(image, str(root / "mask.nii.gz"))
    (tmp_path / "labels.yaml").write_text(
        """cases:
  - case_id: c1
    label: POSITIVE
    inference: {volume: volume.nii.gz, organ_mask: mask.nii.gz}
    ground_truth: {}
""", encoding="utf-8",
    )
    (tmp_path / "datasets.yaml").write_text(
        """datasets:
  - {name: TEST, format: MIDS, root: dataset, labels_manifest: labels.yaml}
""", encoding="utf-8",
    )
    return tmp_path / "datasets.yaml", BASELINE_CONFIG


def test_cli_dry_run_validates_and_never_calls_inference(tmp_path, capsys):
    datasets, med = _fixture(tmp_path)
    code = main([
        "--datasets-manifest", str(datasets), "--medgemma-config", str(med), "--dry-run",
    ])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready"
    assert output["inference_called"] is False
    assert output["cases"][0]["volume_hash"]


def test_recalculate_existing_run_checks_hashes_without_inference(tmp_path):
    datasets, med = _fixture(tmp_path)
    case = load_dataset_manifest(datasets)[0]
    source_dir = tmp_path / "source-run"
    source_dir.mkdir()
    inference = prepare_inference_case(case.inference, source_dir / "workspace" / "inference")
    rows = [{
        "case_id": "c1", "dataset": "TEST", "input_format": "MIDS",
        "truth": "positive", "prediction": "POSITIVA", "status": "decisive",
        "input_hashes": inference.input_hashes,
    }]
    metrics = compute_benchmark_metrics(rows)
    # Deriva os valores esperados da MESMA config real que o runner valida, para
    # que a checagem de reuso (model, hash de conteúdo e panel_strategy) passe.
    screening = load_screening_config(med)
    source_manifest = {
        "run_id": "source", "code_commit": "abc",
        "model_id": screening["medgemma"]["model_id"],
        "model_version": screening["medgemma"]["model_version"],
        "medgemma_config_hash": effective_config_sha256(screening),
        "experimental_strategy": "current_panel",
        "panel_strategy": screening.get("panel", {}).get("strategy", "uniform_9"),
    }
    write_run_outputs(source_dir, source_manifest, rows, metrics)
    new_dir, new_metrics, results = recalculate_existing_run(
        [case], existing_run=source_dir, repo=tmp_path, out_root=tmp_path / "out",
        medgemma_config=med, experiment_config_path=None,
        experiment=ExperimentConfig(),
    )
    assert new_dir.is_dir()
    assert new_metrics["accuracy"] == 1.0
    assert results[0].extra["inference_reused"] is True
