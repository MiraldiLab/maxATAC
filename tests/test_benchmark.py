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


class _StubGoldStandard:
    """pyBigWig stand-in returning fixed per-bin stats"""

    def __init__(self, bins):
        self.bins = bins

    def stats(self, chromosome, start, end, type, nBins, exact):
        return list(self.bins)


def test_random_auprc_counts_gold_standard_bins_inside_the_evaluated_mask(tmp_path):
    import pandas as pd
    from maxatac.utilities.benchmarking_tools import ChromosomeAUPRC

    # 200 bins, the first 100 bound. The mask (blacklist and whitelist combined) keeps 10 bins, 2 of them bound
    gold_standard = np.zeros(200)
    gold_standard[:100] = 1
    mask = np.zeros(200, dtype=bool)
    mask[98:108] = True

    # Bypass __init__ (bigwig I/O) and run the gold standard import and AUPRC steps on in-memory arrays
    auprc = ChromosomeAUPRC.__new__(ChromosomeAUPRC)
    auprc.chromosome = "chr1"
    auprc.chromosome_length = 200 * 200
    auprc.bin_count = 200
    auprc.bin_size = 200
    auprc.agg_function = "max"
    auprc.agg_threshold = 0.5
    auprc.peak_based = False
    auprc.blacklist_mask = mask
    auprc.goldstandard_stream = _StubGoldStandard(gold_standard)
    auprc.prediction_array = np.linspace(1, 0, 200)
    auprc.results_location = str(tmp_path / "chr1_PRC.tsv")

    auprc.__import_goldstandard_array__()
    auprc.__AUPRC__()

    results = pd.read_csv(auprc.results_location, sep="\t")

    # Before the fix: 100 chromosome-wide positives / 10 evaluated bins = 10
    assert results["Total_GoldStandard_Bins"].iloc[0] == 2
    assert results["Random_AUPRC"].iloc[0] == pytest.approx(0.2)
    assert results["Random_AUPRC"].iloc[0] == pytest.approx(auprc.random_precision)


def _peak_gold_standard(peak_sizes, gap, rng):
    """Boolean bins: peaks of the given sizes separated by random negative gaps of at least 1 bin"""
    bins = []
    for size in peak_sizes:
        bins += [False] * int(rng.integers(1, gap + 1)) + [True] * size
    return np.array(bins + [False] * gap)


def test_random_aup_ucper_matches_mean_of_random_rankings():
    from maxatac.utilities.benchmarking_tools import (peak_gs_array_to_member_array,
                                                      precision_recall_curve_peak_based,
                                                      random_aup_ucper)
    from sklearn import metrics

    # The analytic curve is the expected curve, so it is about 1/peaks below the mean of shuffled
    # rankings (8% at 20 peaks). Use 500 peaks so that bias is well under the tolerance
    rng = np.random.default_rng(0)
    member = peak_gs_array_to_member_array(_peak_gold_standard([1, 2, 3, 5, 8] * 100, 30, rng))

    simulated = []
    for _ in range(100):
        precision, recall, _ = precision_recall_curve_peak_based(member, rng.random(len(member)))
        simulated.append(metrics.auc(x=recall[:-1], y=precision[:-1]))

    assert random_aup_ucper(member) == pytest.approx(np.mean(simulated), rel=0.05)


def test_random_aup_ucper_is_flat_bin_level_baseline_for_single_bin_peaks():
    from maxatac.utilities.benchmarking_tools import peak_gs_array_to_member_array, random_aup_ucper

    gold_standard = _peak_gold_standard([1] * 25, 10, np.random.default_rng(1))
    member = peak_gs_array_to_member_array(gold_standard)

    # Flat at P/N; the curve starts at recall 1/N rather than 0, hence the small tolerance
    assert random_aup_ucper(member) == pytest.approx(gold_standard.mean(), rel=0.01)


def test_random_aup_ucper_lies_between_end_and_start_precision():
    from maxatac.utilities.benchmarking_tools import peak_gs_array_to_member_array, random_aup_ucper

    gold_standard = _peak_gold_standard([4, 10, 20, 6] * 5, 40, np.random.default_rng(2))
    member = peak_gs_array_to_member_array(gold_standard)
    n_peaks = len(np.unique(member[member > 0]))
    n_negative = np.count_nonzero(member < 0)

    assert n_peaks / (n_peaks + n_negative) < random_aup_ucper(member) < gold_standard.mean()


def test_random_precision_at_ucper_matches_mean_of_random_rankings():
    from maxatac.utilities.benchmarking_tools import (Precision_for_Recall,
                                                      peak_gs_array_to_member_array,
                                                      precision_recall_curve_peak_based,
                                                      random_precision_at_ucper)
    import pandas as pd

    rng = np.random.default_rng(3)
    member = peak_gs_array_to_member_array(_peak_gold_standard([1, 2, 3, 5, 8] * 100, 30, rng))

    simulated = []
    for _ in range(100):
        precision, recall, _ = precision_recall_curve_peak_based(member, rng.random(len(member)))
        curve = pd.DataFrame({"Precision": precision[:-1], "Recall": recall[:-1]})
        simulated.append(Precision_for_Recall(curve, 0.1))

    assert random_precision_at_ucper(member, 0.1) == pytest.approx(np.mean(simulated), rel=0.03)


def test_peak_based_auprc_writes_random_baselines(tmp_path):
    import pandas as pd
    from maxatac.utilities.benchmarking_tools import (ChromosomeAUPRC, peak_gs_array_to_member_array,
                                                      random_aup_ucper, random_precision_at_ucper)

    gold_standard = _peak_gold_standard([1, 2, 3, 5, 8] * 20, 30, np.random.default_rng(4)).astype(float)
    n_bins = len(gold_standard)

    auprc = ChromosomeAUPRC.__new__(ChromosomeAUPRC)
    auprc.chromosome = "chr1"
    auprc.chromosome_length = n_bins * 200
    auprc.bin_count = n_bins
    auprc.bin_size = 200
    auprc.agg_function = "max"
    auprc.agg_threshold = 0.5
    auprc.peak_based = True
    auprc.blacklist_mask = np.ones(n_bins, dtype=bool)
    auprc.goldstandard_stream = _StubGoldStandard(gold_standard)
    auprc.prediction_array = np.random.default_rng(5).random(n_bins) + gold_standard
    auprc.results_location = str(tmp_path / "chr1_PRC.tsv")

    auprc.__import_goldstandard_array__()
    auprc.__AUPRC__()

    results = pd.read_csv(auprc.results_location, sep="\t").iloc[0]
    member = peak_gs_array_to_member_array(gold_standard > 0)

    assert results["Total_GoldStandard_Peaks"] == 100
    assert results["Random_AUPRC"] == pytest.approx(random_aup_ucper(member))
    assert results["Random_Precision_at_10_Percent_Recall"] == pytest.approx(random_precision_at_ucper(member, 0.1))
    assert results["log2FC_Precision_at_10_Percent_Recall_Random"] == pytest.approx(
        np.log2(results["Precision_at_10_Percent_Recall"] / results["Random_Precision_at_10_Percent_Recall"]))


class _StubGoldStandardBigWig(_StubGoldStandard):
    """pyBigWig stand-in with a chromosome size, for running ChromosomeAUPRC.__init__"""

    def chroms(self, chromosome):
        return len(self.bins) * 200


def test_random_only_writes_just_the_two_random_baselines_without_reading_the_prediction(tmp_path, monkeypatch):
    import pandas as pd
    from maxatac.utilities import benchmarking_tools
    from maxatac.utilities.benchmarking_tools import (ChromosomeAUPRC, peak_gs_array_to_member_array,
                                                      random_aup_ucper, random_precision_at_ucper)

    gold_standard = _peak_gold_standard([1, 2, 3, 5, 8] * 20, 30, np.random.default_rng(6)).astype(float)
    n_bins = len(gold_standard)
    # Blacklist the first 50 bins: the baselines must use the masked gold standard only
    mask = np.ones(n_bins, dtype=bool)
    mask[:50] = False

    opened = []

    def load_bigwig(path):
        opened.append(path)
        return _StubGoldStandardBigWig(gold_standard)

    monkeypatch.setattr(benchmarking_tools, "load_bigwig", load_bigwig)
    monkeypatch.setattr(benchmarking_tools, "chromosome_blacklist_mask", lambda *args: mask)

    results_location = str(tmp_path / "test_chr1_200bp_random_UCPeR.tsv")
    ChromosomeAUPRC(None, "gold_standard.bw", "blacklist.bw", None, "chr1", 200, "max", 0.5,
                    results_location, 9, peak_based=True, random_only=True)

    results = pd.read_csv(results_location, sep="\t")
    member = peak_gs_array_to_member_array(gold_standard[mask] > 0)

    assert opened == ["gold_standard.bw"]
    assert list(results.columns) == ["Random_AUPRC", "Random_Precision_at_10_Percent_Recall"]
    assert len(results) == 1
    assert results["Random_AUPRC"].iloc[0] == pytest.approx(random_aup_ucper(member))
    assert results["Random_Precision_at_10_Percent_Recall"].iloc[0] == pytest.approx(random_precision_at_ucper(member, 0.1))


def test_random_only_requires_peak_based():
    from maxatac.utilities.benchmarking_tools import ChromosomeAUPRC

    with pytest.raises(ValueError, match="peak_based"):
        ChromosomeAUPRC(None, "gold_standard.bw", None, None, "chr1", 200, "max", 0.5,
                        "out.tsv", 9, peak_based=False, random_only=True)


def test_parse_benchmark_random_needs_no_prediction_but_needs_peak_based():
    parser_module = pytest.importorskip("maxatac.utilities.parser")
    base = ["benchmark", "--gold_standard", "average.bigwig", "--name", "benchmark_test"]

    args = parser_module.parse_arguments(base + ["--peak_based", "--random"], DATA_FOLDER)
    assert args.random and args.peak_based and args.prediction is None

    with pytest.raises(SystemExit):
        parser_module.parse_arguments(base + ["--random"], DATA_FOLDER)

    with pytest.raises(SystemExit):
        parser_module.parse_arguments(base + ["--peak_based"], DATA_FOLDER)


def test_pct_rank_gives_tied_bins_the_lowest_rank_and_zero_or_missing_bins_zero():
    from maxatac.utilities.genome_tools import _pct_rank

    # 6 bins: a zero, a NaN, a tie at 0.5, 0.2 and 0.9
    percentile = _pct_rank(np.array([0.0, np.nan, 0.5, 0.5, 0.2, 0.9]))

    # rankdata(min) over [0, 0, 0.5, 0.5, 0.2, 0.9] -> [1, 1, 4, 4, 3, 6], / 6; zero and NaN bins forced to 0
    np.testing.assert_allclose(percentile, np.array([0, 0, 4, 4, 3, 6]) / 6)


def test_rank_max_takes_the_max_percentile_and_falls_back_to_the_other_track():
    from maxatac.utilities.genome_tools import combine_prediction_arrays

    # Model scores are small, the average ChIP signal is large: a raw max would always pick the average
    model = np.array([0.01, 0.02, 0.03, np.nan, 0.0, 0.04])
    average = np.array([10.0, 50.0, 0.0, 30.0, 20.0, 40.0])

    combined = combine_prediction_arrays(model, average, combine_operation="rank_max")

    # model percentiles:   [3, 4, 5, 0, 0, 6] / 6   (NaN and zero bins -> 0)
    # average percentiles: [2, 6, 0, 4, 3, 5] / 6   (zero bin -> 0)
    np.testing.assert_allclose(combined, np.array([3, 6, 5, 4, 3, 6]) / 6)
    np.testing.assert_allclose(combine_prediction_arrays(model, average, combine_operation="max"),
                               np.array([10.0, 50.0, 0.03, 30.0, 20.0, 40.0]))


def test_rank_max_with_all_zero_or_missing_bins_scores_them_zero():
    from maxatac.utilities.genome_tools import combine_prediction_arrays

    combined = combine_prediction_arrays(np.array([np.nan, 0.0, 0.3]), np.array([0.0, np.nan, np.nan]),
                                         combine_operation="rank_max")

    np.testing.assert_allclose(combined, np.array([0.0, 0.0, 1.0]))


def test_rank_max_logs_how_often_each_track_supplies_the_max(caplog):
    import logging
    from maxatac.utilities.genome_tools import combine_prediction_arrays

    with caplog.at_level(logging.INFO):
        combine_prediction_arrays(np.array([0.01, 0.02, 0.03, np.nan, 0.0, 0.04]),
                                  np.array([10.0, 50.0, 0.0, 30.0, 20.0, 40.0]),
                                  combine_operation="rank_max")

    # From the percentiles above: model wins bins 1, 3, 6 and the average wins bins 2, 4, 5
    assert "primary supplies the max in 50.00% of bins, alternative in 50.00%, tied in 0.00%" in caplog.text


class _StubPredictionBigWig(_StubGoldStandardBigWig):
    """pyBigWig stand-in for a prediction track; keeps NaN for missing bins like import_bigwig_stats_array expects"""


def test_chromosome_auprc_benchmarks_the_rank_max_ensemble(tmp_path, monkeypatch):
    import pandas as pd
    from maxatac.utilities import benchmarking_tools
    from maxatac.utilities.benchmarking_tools import ChromosomeAUPRC
    from maxatac.utilities.genome_tools import combine_prediction_arrays

    rng = np.random.default_rng(7)
    gold_standard = _peak_gold_standard([1, 2, 3] * 30, 20, rng).astype(float)
    n_bins = len(gold_standard)

    # Model: weak signal on peaks, missing on 5% of bins. Average ChIP: large values, zero on 40% of bins
    model = rng.random(n_bins) * 0.1 + 0.05 * gold_standard
    model[rng.random(n_bins) < 0.05] = np.nan
    average = rng.random(n_bins) * 100 * (rng.random(n_bins) > 0.4)

    streams = {"model.bw": _StubPredictionBigWig(model),
               "average.bw": _StubPredictionBigWig(average),
               # --agg sum returns base-pair totals: 200 for a fully covered 200 bp bin
               "gold_standard.bw": _StubGoldStandardBigWig(gold_standard * 200)}
    monkeypatch.setattr(benchmarking_tools, "load_bigwig", lambda path: streams[path])
    # Blacklist bins 0-19, whitelist everything else but the last 10 bins
    keep = np.ones(n_bins, dtype=bool)
    keep[:20] = False
    monkeypatch.setattr(benchmarking_tools, "chromosome_blacklist_mask", lambda *args: keep)
    whitelist = np.ones(n_bins, dtype=bool)
    whitelist[-10:] = False
    monkeypatch.setattr(benchmarking_tools, "chromosome_whitelist_mask", lambda *args: whitelist)

    auprc = ChromosomeAUPRC("model.bw", "gold_standard.bw", "blacklist.bw", "whitelist.bw", "chr1", 200, "sum", 0.5,
                            str(tmp_path / "chr1_PRC.tsv"), 9, peak_based=True,
                            alternative_prediction_bw="average.bw", prediction_combine_operation="rank_max")

    # Combined on all bins before masking, and not divided by bin_size under --agg sum
    expected = np.round(combine_prediction_arrays(model, average, combine_operation="rank_max"), 9)
    np.testing.assert_allclose(auprc.prediction_array, expected)
    assert auprc.prediction_array.max() <= 1.0

    np.testing.assert_array_equal(auprc.blacklist_mask, keep & whitelist)
    results = pd.read_csv(tmp_path / "chr1_PRC.tsv", sep="\t")
    assert 0 < results["AUPRC"].iloc[0] <= 1


def test_quant_benchmark_rejects_rank_max(monkeypatch):
    from maxatac.utilities import benchmarking_tools

    stub = _StubGoldStandardBigWig(np.zeros(10))
    monkeypatch.setattr(benchmarking_tools, "load_bigwig", lambda path: stub)

    with pytest.raises(ValueError, match="rank_max"):
        benchmarking_tools.calculate_R2_pearson_spearman("model.bw", "gs.bw", "quant_gs.bw", "chr1", 200, "mean",
                                                         "out.tsv", None, None, "null.bw",
                                                         alternative_prediction_bw="average.bw",
                                                         prediction_combine_operation="rank_max")


def test_parse_benchmark_rank_max_is_rejected_with_quant():
    parser_module = pytest.importorskip("maxatac.utilities.parser")
    base = ["benchmark", "--bw", "model.bw", "--alternative_prediction", "average.bw",
            "--gold_standard", "gs.bw", "--name", "benchmark_test", "--prediction_combine_operation", "rank_max"]

    assert parser_module.parse_arguments(base + ["--peak_based"], DATA_FOLDER).prediction_combine_operation == "rank_max"

    with pytest.raises(SystemExit):
        parser_module.parse_arguments(base + ["--quant", "--quant_gold_standard", "q.bw", "--quant_gs_null", "n.bw"],
                                      DATA_FOLDER)
