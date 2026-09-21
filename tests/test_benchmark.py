import os

import numpy as np
import pytest

from maxatac.utilities.genome_tools import combine_prediction_arrays


DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))


def test_combine_prediction_arrays_mean_uses_available_values():
    primary = np.array([1.0, np.nan, 2.0, np.nan])
    alternative = np.array([3.0, 4.0, np.nan, np.nan])

    combined = combine_prediction_arrays(primary, alternative, combine_operation="mean")

    np.testing.assert_allclose(combined, np.array([2.0, 4.0, 2.0, 0.0]))


def test_combine_prediction_arrays_max_uses_available_values():
    primary = np.array([1.0, np.nan, 2.0, np.nan])
    alternative = np.array([3.0, 4.0, np.nan, np.nan])

    combined = combine_prediction_arrays(primary, alternative, combine_operation="max")

    np.testing.assert_allclose(combined, np.array([3.0, 4.0, 2.0, 0.0]))


def test_parse_benchmark_with_alternative_prediction_arguments():
    parser_module = pytest.importorskip("maxatac.utilities.parser")
    args = parser_module.parse_arguments([
        "benchmark",
        "--bw", "average.bigwig",
        "--alternative_prediction", "predict_signal_cell_GM12878.bigwig",
        "--prediction_combine_operation", "max",
        "--gold_standard", "average.bigwig",
        "--name", "benchmark_test"
    ], DATA_FOLDER)

    assert args.alternative_prediction.endswith("predict_signal_cell_GM12878.bigwig")
    assert args.prediction_combine_operation == "max"


def test_precision_for_recall_interpolates_between_bracketing_points():
    import pandas as pd
    from maxatac.utilities.benchmarking_tools import Precision_for_Recall

    # PR curve given in sklearn order (recall descending), including the (R=0, P=1) sentinel
    curve = pd.DataFrame({
        "Recall": [0.30, 0.20, 0.05, 0.0],
        "Precision": [0.40, 0.60, 0.90, 1.0],
    })

    # 0.1 lies between (0.05, 0.90) and (0.20, 0.60): 0.90 + (0.60 - 0.90) * (0.05 / 0.15)
    assert Precision_for_Recall(curve, 0.10) == pytest.approx(0.80)
    # Exact hits return the point itself
    assert Precision_for_Recall(curve, 0.20) == pytest.approx(0.60)
    # Beyond the curve's max recall, hold the endpoint
    assert Precision_for_Recall(curve, 0.50) == pytest.approx(0.40)


def test_precision_for_recall_collapses_duplicate_recall_to_max_precision():
    import pandas as pd
    from maxatac.utilities.benchmarking_tools import Precision_for_Recall

    curve = pd.DataFrame({
        "Recall": [0.2, 0.1, 0.1, 0.0],
        "Precision": [0.5, 0.7, 0.9, 1.0],
    })

    assert Precision_for_Recall(curve, 0.10) == pytest.approx(0.9)
    assert Precision_for_Recall(curve, 0.15) == pytest.approx(0.7)
