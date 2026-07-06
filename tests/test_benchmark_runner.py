import json
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from dtwin.benchmark.importers import load_dataset_manifest
from dtwin.benchmark.models import BenchmarkStatus
from dtwin.benchmark.runner import (
    ExperimentConfig,
    classify_screening_failure,
    make_run_id,
    run_case,
)


def _image(path: Path, value=1):
    image = sitk.GetImageFromArray(np.full((4, 5, 6), value, dtype=np.uint8))
    sitk.WriteImage(image, str(path))


def _dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _image(root / "volume.nii.gz", 2)
    _image(root / "mask.nii.gz", 1)
    _image(root / "lesion.nii.gz", 1)
    (tmp_path / "labels.yaml").write_text(
        """cases:
  - case_id: case-001
    label: POSITIVE
    inference:
      volume: volume.nii.gz
      organ_mask: mask.nii.gz
    ground_truth:
      lesion_mask: lesion.nii.gz
""", encoding="utf-8",
    )
    (tmp_path / "datasets.yaml").write_text(
        """datasets:
  - name: TEST
    format: NIFTI
    root: dataset
    labels_manifest: labels.yaml
""", encoding="utf-8",
    )
    return load_dataset_manifest(tmp_path / "datasets.yaml")[0]


def test_run_id_is_stable_and_filesystem_safe():
    from datetime import datetime, timezone

    assert make_run_id("abcdef123", "single phase / gray", datetime(2026, 7, 6, tzinfo=timezone.utc)) == (
        "20260706T000000Z_abcdef12_single-phase-gray"
    )


def test_invalid_model_response_is_not_collapsed_into_generic_failure():
    assert classify_screening_failure("Resposta MedGemma não contém objeto JSON válido") is BenchmarkStatus.INVALID_RESPONSE
    assert classify_screening_failure("Backend inacessível") is BenchmarkStatus.FAILURE


def test_runner_subprocess_never_receives_ground_truth(monkeypatch, tmp_path):
    case = _dataset(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    med_config = tmp_path / "config.yaml"
    med_config.write_text("x: 1", encoding="utf-8")
    seen = {}

    def fake_run(command, **kwargs):
        joined = " ".join(map(str, command))
        seen["command"] = joined
        assert "lesion" not in joined.lower()
        assert "positive" not in joined.lower()
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "panel.png").write_bytes(b"png")
        (output / "medgemma_report.json").write_text(json.dumps({
            "report": {"resultado_hipotese": "POSITIVA", "confianca": "alta"},
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_case(
        case, run_dir=run_dir, medgemma_config=med_config,
        experiment=ExperimentConfig(visible_phi_confirmed=True),
    )
    assert result.status is BenchmarkStatus.DECISIVE
    assert result.is_correct_primary is True
    assert result.protected_ground_truth_hashes["lesion_mask"]
    assert "--confirm-no-visible-phi" in seen["command"]


def test_runner_timeout_is_a_distinct_error(monkeypatch, tmp_path):
    case = _dataset(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    med_config = tmp_path / "config.yaml"
    med_config.write_text("x: 1", encoding="utf-8")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_case(case, run_dir=run_dir, medgemma_config=med_config, experiment=ExperimentConfig())
    assert result.status is BenchmarkStatus.TIMEOUT
    assert result.error_type == "TimeoutExpired"
