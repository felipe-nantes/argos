# tests/test_cli.py
"""Integration tests for the digital_twin.py CLI entrypoint (main dispatch),
so the user-facing surface — not just the engine — is exercised end-to-end.
"""
import digital_twin


def test_cli_finalize_returns_zero(synthetic_case):
    rc = digital_twin.main(
        ["finalize", str(synthetic_case.root), "--profile", "profiles/figado.yaml"]
    )
    assert rc == 0
    assert (synthetic_case.outputs / "viewer_manifest.json").exists()


def test_cli_finalize_no_lesion_returns_zero(synthetic_case):
    synthetic_case.mask_lesion.unlink()
    rc = digital_twin.main(
        ["finalize", str(synthetic_case.root), "--profile", "profiles/figado.yaml", "--no-lesion"]
    )
    assert rc == 0


def test_cli_bad_profile_returns_one(synthetic_case):
    rc = digital_twin.main(
        ["finalize", str(synthetic_case.root), "--profile", "profiles/__nope__.yaml"]
    )
    assert rc == 1
