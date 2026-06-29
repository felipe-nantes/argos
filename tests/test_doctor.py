# tests/test_doctor.py
import digital_twin


def test_doctor_runs_and_returns_zero(capsys):
    rc = digital_twin.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TotalSegmentator" in out
    assert "SimpleITK" in out
