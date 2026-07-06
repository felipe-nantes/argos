import json

from dtwin.benchmark.hashing import git_state, input_hashes
from dtwin.benchmark.metrics import compute_benchmark_metrics
from dtwin.benchmark.reporting import write_run_outputs


def test_hashes_and_git_state_are_recordable(tmp_path):
    volume = tmp_path / "volume.nii.gz"
    mask = tmp_path / "mask_organ.nii.gz"
    manifest = tmp_path / "manifest.json"
    volume.write_bytes(b"volume")
    mask.write_bytes(b"mask")
    manifest.write_text("{}", encoding="utf-8")
    hashes = input_hashes(volume, mask, manifest)
    assert set(hashes) == {"volume", "mask_organ", "manifest"}
    assert all(len(value) == 64 for value in hashes.values())
    state = git_state(tmp_path)
    assert state["code_commit"] is None
    assert state["git_dirty"] is True


def test_run_outputs_are_complete_and_machine_readable(tmp_path):
    cases = [
        {"case_id": "a", "truth": "positive", "prediction": "POSITIVA", "status": "decisive"},
        {"case_id": "b", "truth": "negative", "prediction": None, "status": "timeout"},
    ]
    metrics = compute_benchmark_metrics(cases)
    manifest = {
        "run_id": "run-test", "created_at": "2026-07-06T12:00:00Z",
        "code_commit": "abc", "git_dirty": False, "model_id": "medgemma-test",
        "model_parameter_scale": "4B", "experimental_strategy": "current_panel",
    }
    outputs = write_run_outputs(tmp_path, manifest, cases, metrics)
    assert set(outputs) == {
        "run_manifest.json", "cases.jsonl", "metrics_primary.json",
        "metrics_decisions_only.json", "confusion_matrices.json", "summary.md",
    }
    assert json.loads((tmp_path / "metrics_primary.json").read_text("utf-8"))["timeout_count"] == 1
    lines = (tmp_path / "cases.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["case_id"] == "a"
    summary = (tmp_path / "summary.md").read_text("utf-8")
    assert "FAIL" in summary and "Uso exclusivo em Pesquisa" in summary
