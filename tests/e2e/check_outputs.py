#!/usr/bin/env python3
"""Output checks for the end-to-end test (tests/e2e/run_e2e.sh)

Subcommands:
    data        the reference files and the TF model installed by `maxatac data` are present
    whitelist   write a bigWig that is 1 over the ROI, used to restrict `maxatac benchmark` to it
    prediction  the prediction bigWig and peaks look sane over the ROI
    report      collect the metrics, compare them with the baseline and write a summary
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyBigWig
from scipy import stats

BIN_SIZE = 200


def fail(message):
    # ::error:: is shown as an annotation on GitHub Actions and is plain text elsewhere
    print(f"::error::{message}")
    sys.exit(1)


def read_roi(roi_path):
    """Return (chrom, start, end) of the first interval in a BED file"""
    with open(roi_path) as roi:
        chrom, start, end = roi.readline().split("\t")[:3]

    return chrom, int(start), int(end)


def binned_stats(bigwig_path, chrom, start, end, agg="mean", bin_size=BIN_SIZE):
    """Aggregate a bigWig into fixed bins over [start, end); bins without data are NaN"""
    n_bins = (end - start) // bin_size
    bigwig = pyBigWig.open(bigwig_path)

    try:
        values = bigwig.stats(chrom, start, start + n_bins * bin_size, type=agg, nBins=n_bins)

    finally:
        bigwig.close()

    return np.array(values, dtype=float)


def check_data(args):
    genome_dir = os.path.join(args.data_dir, args.genome)
    required = [os.path.join(genome_dir, f"{args.genome}.chrom.sizes"),
                os.path.join(genome_dir, f"{args.genome}_maxatac_blacklist.bed"),
                os.path.join(genome_dir, f"{args.genome}_maxatac_blacklist.bw"),
                os.path.join(genome_dir, f"{args.genome}.2bit")]

    for path in required:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            fail(f"`maxatac data` did not install {path}")

    model_dir = os.path.join(args.data_dir, "models", args.tf)
    models = glob.glob(os.path.join(model_dir, f"{args.tf}*.h5"))
    tables = glob.glob(os.path.join(model_dir, f"{args.tf}*_cross_celltype.tsv"))

    if not models:
        fail(f"No {args.tf} model (*.h5) in {model_dir}")

    if not tables:
        fail(f"No {args.tf} threshold calibration table (*_cross_celltype.tsv) in {model_dir}")

    print(f"Reference data OK: {genome_dir}")
    print(f"Model: {models[0]}")
    print(f"Calibration table: {tables[0]}")


def write_whitelist(args):
    chrom, start, end = read_roi(args.roi)
    chrom_sizes = pd.read_csv(args.chrom_sizes, sep="\t", header=None, index_col=0)[1]

    whitelist = pyBigWig.open(args.out, "w")
    whitelist.addHeader([(chrom, int(chrom_sizes[chrom]))])
    whitelist.addEntries([chrom], [start], ends=[end], values=[1.0])
    whitelist.close()

    print(f"Wrote {args.out}: {chrom}:{start}-{end}")


def check_prediction(args):
    if not os.path.isfile(args.prediction):
        fail(f"`maxatac predict` did not write {args.prediction}")

    chrom, start, end = read_roi(args.roi)

    prediction = pyBigWig.open(args.prediction)
    chroms = prediction.chroms()
    prediction.close()

    if chrom not in chroms:
        fail(f"{args.prediction} has no {chrom}; chromosomes: {list(chroms)}")

    values = binned_stats(args.prediction, chrom, start, end, agg="max")
    covered = np.isfinite(values)

    # Blacklisted windows are not predicted, so full coverage is not expected
    if covered.mean() < 0.5:
        fail(f"Only {covered.mean():.1%} of the {BIN_SIZE} bp bins in the ROI have a prediction")

    covered_values = values[covered]

    if covered_values.min() < 0:
        fail(f"Negative prediction scores (min {covered_values.min():.4g}); the quant models end in softplus")

    if np.std(covered_values) == 0:
        fail(f"The prediction is constant ({covered_values[0]:.4g}) over the ROI")

    if not os.path.isfile(args.peaks) or os.path.getsize(args.peaks) == 0:
        fail(f"`maxatac predict` called no peaks: {args.peaks} is missing or empty")

    peaks = pd.read_csv(args.peaks, sep="\t", header=None, usecols=[0, 1, 2])

    print(f"Prediction OK: {covered.mean():.1%} of bins covered, "
          f"score range {covered_values.min():.4g}-{covered_values.max():.4g}, {len(peaks)} peaks")


def report(args):
    chrom, start, end = read_roi(args.roi)

    prc = pd.read_csv(args.prc, sep="\t")
    metrics = {"AUPRC": float(prc["AUPRC"].iloc[0])}

    if "Random_AUPRC" in prc.columns:
        metrics["Random_AUPRC"] = float(prc["Random_AUPRC"].iloc[0])

    # Interim quantitative check until a null track for `maxatac benchmark --quant` is in the test data
    prediction = binned_stats(args.prediction, chrom, start, end, agg="mean")
    observed = binned_stats(args.quant_gold_standard, chrom, start, end, agg="mean")
    blacklisted = binned_stats(args.blacklist_bw, chrom, start, end, agg="max") > 0

    # Bins without gold standard signal are 0, not missing
    observed = np.nan_to_num(observed, nan=0.0)
    keep = np.isfinite(prediction) & ~blacklisted

    metrics["pearson"] = float(stats.pearsonr(prediction[keep], observed[keep])[0])
    metrics["spearman"] = float(stats.spearmanr(prediction[keep], observed[keep])[0])
    metrics["n_bins"] = int(keep.sum())

    failures = []

    if "Random_AUPRC" in metrics and metrics["AUPRC"] <= metrics["Random_AUPRC"]:
        failures.append(f"AUPRC {metrics['AUPRC']:.4f} is not above random ({metrics['Random_AUPRC']:.4f})")

    if metrics["pearson"] <= 0:
        failures.append(f"Pearson correlation with the quantitative gold standard is {metrics['pearson']:.4f}")

    with open(args.baseline) as baseline_file:
        baseline = json.load(baseline_file)

    rows = []

    for name, reference in baseline["metrics"].items():
        tolerance = baseline["tolerance"][name]
        value = metrics[name]

        if reference is None:
            status = "no baseline"

        elif value < reference - tolerance:
            status = "REGRESSED"
            failures.append(f"{name} {value:.4f} is below the baseline {reference:.4f} - {tolerance}")

        else:
            status = "ok"

        rows.append(f"| {name} | {value:.4f} | {'-' if reference is None else f'{reference:.4f}'} | {status} |")

    with open(args.metrics_out, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    summary = "\n".join([f"### quant-maxATAC e2e: GM12878 {chrom}:{start:,}-{end:,}",
                         "",
                         "| Metric | Value | Baseline | Status |",
                         "|---|---|---|---|",
                         *rows,
                         "",
                         f"Random AUPRC: {metrics.get('Random_AUPRC', float('nan')):.4f}; "
                         f"{metrics['n_bins']} {BIN_SIZE} bp bins outside the blacklist",
                         ""])

    print(summary)

    # Also kept as a file: release.yml puts it in the GitHub Release notes
    with open(os.path.splitext(args.metrics_out)[0] + ".md", "w") as summary_file:
        summary_file.write(summary)

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as step_summary:
            step_summary.write(summary + "\n")

    for failure in failures:
        print(f"::error::{failure}")

    if failures:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data")
    data.add_argument("--data_dir", required=True)
    data.add_argument("--genome", required=True)
    data.add_argument("--tf", required=True)
    data.set_defaults(func=check_data)

    whitelist = subparsers.add_parser("whitelist")
    whitelist.add_argument("--roi", required=True)
    whitelist.add_argument("--chrom_sizes", required=True)
    whitelist.add_argument("--out", required=True)
    whitelist.set_defaults(func=write_whitelist)

    prediction = subparsers.add_parser("prediction")
    prediction.add_argument("--prediction", required=True)
    prediction.add_argument("--peaks", required=True)
    prediction.add_argument("--roi", required=True)
    prediction.set_defaults(func=check_prediction)

    metrics = subparsers.add_parser("report")
    metrics.add_argument("--prc", required=True)
    metrics.add_argument("--prediction", required=True)
    metrics.add_argument("--quant_gold_standard", required=True)
    metrics.add_argument("--blacklist_bw", required=True)
    metrics.add_argument("--roi", required=True)
    metrics.add_argument("--baseline", required=True)
    metrics.add_argument("--metrics_out", required=True)
    metrics.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
