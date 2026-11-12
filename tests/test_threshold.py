import numpy as np
import pandas as pd
import pytest

from maxatac.utilities.peak_tools import get_threshold
from maxatac.utilities.threshold_tools import compute_calibration_curve, PEAK_RECALL


def _actual_metrics(goldstandard, prediction, threshold):
    """Precision and recall at a threshold, with sklearn's >= rule"""
    called = prediction >= threshold
    true_positives = np.count_nonzero(called & goldstandard)
    return true_positives / np.count_nonzero(called), true_positives / np.count_nonzero(goldstandard)


def test_calibration_curve_keeps_only_attained_thresholds():
    # Real precision: 0.667 at threshold 0.1, 0.5 at 0.2, 0 at 0.9 (the top-scoring bin is unbound).
    # 0.2 and 0.9 are beaten by 0.1, and the (Precision=1, Recall=0) sentinel is not a threshold
    curve = compute_calibration_curve(np.array([1, 1, 0]), np.array([0.1, 0.2, 0.9]))

    assert curve["Threshold"].tolist() == [0.1]
    assert curve["Precision"].tolist() == pytest.approx([2 / 3])
    assert curve["Recall"].tolist() == pytest.approx([1.0])
    assert (curve["Precision"] < 1).all()


def test_calibration_curve_rows_are_real_and_monotonic():
    rng = np.random.default_rng(0)
    goldstandard = rng.random(500) < 0.2
    prediction = np.round(goldstandard * 0.3 + rng.random(500), 3)
    peak_max = rng.random(40)

    curve = compute_calibration_curve(goldstandard, prediction, peak_max=peak_max)

    # Every row is the precision/recall actually achieved at its own threshold
    for row in curve.itertuples():
        precision, recall = _actual_metrics(goldstandard, prediction, row.Threshold)
        assert row.Precision == pytest.approx(precision)
        assert row.Recall == pytest.approx(recall)

    # Ascending threshold: precision strictly rises, recall and Peak_Recall never rise
    curve = curve.sort_values("Threshold")
    assert np.all(np.diff(curve["Precision"]) > 0)
    assert np.all(np.diff(curve["Recall"]) <= 0)
    assert np.all(np.diff(curve[PEAK_RECALL]) <= 0)


def _write_table(path, rows):
    pd.DataFrame(rows, columns=["Metric", "Bin", "Precision", "Recall", "Threshold", "F1"]).to_csv(
        path, sep="\t", index=False)
    return str(path)


def test_get_threshold_requires_the_achieved_value(tmp_path):
    # Grid extension: bins past the best precision (0.8) repeat its row, so Bin 1.0 holds Precision 0.8
    table = _write_table(tmp_path / "table.tsv", [
        ["Precision", 0.70, 0.70, 0.60, 0.5, 0.65],
        ["Precision", 0.80, 0.80, 0.40, 0.7, 0.53],
        ["Precision", 0.90, 0.80, 0.40, 0.7, 0.53],
        ["Precision", 1.00, 0.80, 0.40, 0.7, 0.53],
    ])

    with pytest.raises(ValueError, match="highest Precision it reaches is 0.800"):
        get_threshold(table, "Precision", 1.0)

    # Closest achieved value that meets the request
    assert get_threshold(table, "Precision", 0.75) == pytest.approx(0.7)
    assert get_threshold(table, "Precision", 0.65) == pytest.approx(0.5)


def test_get_threshold_breaks_ties_on_f1(tmp_path):
    table = _write_table(tmp_path / "table.tsv", [
        ["Recall", 0.50, 0.60, 0.50, 0.9, 0.545],
        ["Recall", 0.51, 0.70, 0.50, 0.8, 0.583],
    ])

    assert get_threshold(table, "Recall", 0.5) == pytest.approx(0.8)
