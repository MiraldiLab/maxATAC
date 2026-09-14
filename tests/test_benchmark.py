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
