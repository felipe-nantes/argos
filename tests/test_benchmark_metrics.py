from dtwin.benchmark.metrics import compute_benchmark_metrics, wilson_interval


def _case(truth, prediction=None, status="failure", **extra):
    return {
        "case_id": f"{truth}-{status}",
        "truth": truth,
        "prediction": prediction,
        "status": status,
        **extra,
    }


def test_primary_penalizes_every_non_correct_outcome():
    cases = [
        _case("positive", "POSITIVA", "decisive"),
        _case("positive", "NEGATIVA", "decisive"),
        _case("positive", "INCONCLUSIVA", "inconclusive"),
        _case("positive", None, "timeout"),
        _case("negative", "NEGATIVA", "decisive"),
        _case("negative", "POSITIVA", "decisive"),
        _case("negative", None, "invalid_response"),
        _case("negative", None, "failure"),
    ]
    metrics = compute_benchmark_metrics(cases)
    assert metrics["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 3, "fn": 3}
    assert metrics["sensitivity"] == 0.25
    assert metrics["specificity"] == 0.25
    assert metrics["accuracy"] == 0.25
    assert metrics["coverage_rate"] == 0.5
    assert metrics["timeout_count"] == 1
    assert metrics["invalid_response_count"] == 1
    assert metrics["failure_count"] == 1


def test_decisions_only_excludes_non_decisions():
    metrics = compute_benchmark_metrics([
        _case("positive", "POSITIVA", "decisive"),
        _case("negative", "NEGATIVA", "decisive"),
        _case("positive", None, "timeout"),
        _case("negative", "INCONCLUSIVA", "inconclusive"),
    ])
    secondary = metrics["decisions_only"]
    assert secondary["total_cases"] == 2
    assert secondary["accuracy"] == 1.0
    assert "secundárias" in secondary["warning"]
    assert metrics["accuracy"] == 0.5


def test_categorical_matrix_preserves_technical_states():
    metrics = compute_benchmark_metrics([
        _case("positive", None, "failure"),
        _case("positive", None, "timeout"),
        _case("negative", None, "invalid_response"),
        _case("negative", "INCONCLUSIVA", "inconclusive"),
    ])
    matrix = metrics["categorical_confusion_matrix"]
    assert matrix["positive"]["FAILURE"] == 1
    assert matrix["positive"]["TIMEOUT"] == 1
    assert matrix["negative"]["INVALID_RESPONSE"] == 1
    assert matrix["negative"]["INCONCLUSIVA"] == 1


def test_gate_requires_sensitivity_and_specificity():
    metrics = compute_benchmark_metrics([
        *[_case("positive", "POSITIVA", "decisive") for _ in range(3)],
        _case("positive", None, "failure"),
        *[_case("negative", "NEGATIVA", "decisive") for _ in range(2)],
        *[_case("negative", "POSITIVA", "decisive") for _ in range(2)],
    ])
    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 0.5
    assert metrics["gate"]["sensitivity_passed"] is True
    assert metrics["gate"]["specificity_passed"] is False
    assert metrics["gate"]["passed"] is False


def test_wilson_and_undefined_metrics_are_explicit():
    interval = wilson_interval(5, 10)
    assert interval == {"low": 0.2366, "high": 0.7634}
    metrics = compute_benchmark_metrics([_case("negative", "NEGATIVA", "decisive")])
    assert metrics["sensitivity"] is None
    assert metrics["confidence_intervals_95"]["sensitivity"] is None
    assert metrics["undefined_reasons"]["sensitivity"] == "no_positive_cases"
    assert metrics["confidence_intervals_95"]["f1"] is None
    assert metrics["f1_ci_method"] == "not_implemented"


def test_stratified_metrics_separate_benign_variants_and_positive_subtypes():
    metrics = compute_benchmark_metrics([
        _case("negative", "NEGATIVA", "decisive", negative_subtype="normal"),
        _case(
            "negative",
            "POSITIVA",
            "decisive",
            negative_subtype="benign_anatomic_variant",
            phenotype_tags=["prominent_hepatic_vein"],
        ),
        _case("negative", "INCONCLUSIVA", "inconclusive", negative_subtype="benign_anatomic_variant"),
        _case("positive", "POSITIVA", "decisive", positive_subtype="hcc_suspicious"),
        _case("positive", "NEGATIVA", "decisive", positive_subtype="hcc_suspicious"),
    ])

    stratified = metrics["primary"]["stratified"]
    benign = stratified["negative_subtypes"]["benign_anatomic_variant"]
    normal = stratified["negative_subtypes"]["normal"]
    hcc = stratified["positive_subtypes"]["hcc_suspicious"]
    vein = stratified["phenotype_tags"]["prominent_hepatic_vein"]

    assert normal["specificity"] == 1.0
    assert benign["total"] == 2
    assert benign["specificity"] == 0.0
    assert benign["false_positive_rate"] == 0.5
    assert stratified["positive_rate_on_benign_variants"] == 0.5
    assert hcc["sensitivity"] == 0.5
    assert vein["specificity_on_tagged_negatives"] == 0.0
    assert metrics["stratified_metrics"] == stratified
