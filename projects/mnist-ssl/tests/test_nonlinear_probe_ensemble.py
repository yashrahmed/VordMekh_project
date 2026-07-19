from mnist_ssl.ensembles.nonlinear_probe_pair import summarize_plateaus


def test_summarize_plateaus_reports_exact_and_near_best_ranges() -> None:
    rows = []
    for method in ("logit", "probability"):
        for weight, errors in ((0.0, 5), (0.5, 3), (1.0, 4)):
            rows.append(
                {
                    "method": method,
                    "dino_weight": weight,
                    "errors": errors,
                    "test_accuracy": 100.0 - errors,
                }
            )

    result = summarize_plateaus(rows)

    assert result["logit"]["best_errors"] == 3
    assert result["logit"]["exact_best_dino_weights"] == [0.5]
    assert result["logit"]["within_one_error_weight_min"] == 0.5
    assert result["logit"]["within_one_error_weight_max"] == 1.0
