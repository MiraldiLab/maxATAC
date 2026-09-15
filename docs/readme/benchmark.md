# Benchmark

The `benchmark` function compares a prediction bigwig against a gold standard at a chosen resolution (e.g., 200 bp bins). It has two modes:

* **Binary benchmark (default):** precision-recall analysis (AUPRC) of the prediction against a **binary** gold standard (1 = TF bound, 0 = unbound). This works for both binary and quantitative predictions, since only the ranking of the prediction scores matters.
* **Quantitative benchmark (`--quant`):** regression metrics of the prediction against a **quantitative** gold standard (e.g. ChIP-seq signal on the same scale as the model's training targets): mean absolute error (MAE), `R2_pred`, Pearson and Spearman correlation, plus an observed-vs-predicted scatter plot.

`R2_pred` is computed relative to a null model rather than the mean of the gold standard:

```
R2_pred = 1 - SSE(prediction vs. quant gold standard) / SSE(null model vs. quant gold standard)
```

where the null model (`--quant_gs_null`) is typically the average of the quantitative gold standard tracks across all cell types for the TF, built with [`maxatac average --quant`](./average.md).

## Examples

Binary benchmark (AUPRC):

```bash
maxatac benchmark --bw GM12878_CTCF_chr1.bw --gold_standard GM12878_CTCF_ENCODE_IDR.bw --chromosomes chr1 --bin_size 200 -n GM12878_CTCF
maxatac benchmark --bw primary.bw --alternative_prediction secondary.bw --prediction_combine_operation max --gold_standard GM12878_CTCF_ENCODE_IDR.bw --chromosomes chr1 --bin_size 200 -n GM12878_CTCF_combined
```

Quantitative benchmark:

```bash
maxatac benchmark --quant --bw GM12878_CTCF_chr1.bw --gold_standard GM12878_CTCF_ENCODE_IDR.bw --quant_gold_standard GM12878_CTCF_signal.bw --quant_gs_null CTCF_signal_all_celltypes_avg.bw --chromosomes chr1 --bin_size 32 --agg mean -n GM12878_CTCF
```

## Outputs

Binary benchmark, per chromosome:

| Filename                                | Description                                                                                                                   |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `<prefix>_<chr>_<bin_size>bp_PRC.tsv`   | Precision, recall and threshold for every point on the PR curve, with `AUPRC`, `Total_GoldStandard_Bins`, `Random_AUPRC`, `log2FC_AUPRC_Random_AUPRC` and `Precision_at_10_Percent_Recall` |
| `<prefix>_<chr>_<bin_size>bp_PRC.png`   | PR curve (only with `--plot`)                                                                                                 |

Quantitative benchmark, per chromosome:

| Filename                                                   | Description                                                                                  |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `<prefix>_<chr>_<bin_size>_r2_pearson_spearman.tsv`        | `MAE`, `R2_pred`, `pearson`, `pearson_pval`, `spearman`, `spearman_pval`                     |
| `<prefix>_<chr>_<bin_size>_r2_pearson_spearman_scatterPlot.png` | Observed (x) vs. predicted (y) scatter plot with the best-fit line through the origin     |
| `<prefix>_<chr>_<bin_size>_r2_pearson_spearman_scatterPlot_df.tsv` | Per-bin table (`chrom`, `start`, `end`, `y_pred`, `y_obs`) behind the scatter plot      |
| `<prefix>_<chr>_<bin_size>_R2_yisx_Slope_df.tsv`           | Slope of the best-fit line and its R² against the identity line `y = x`                       |

## Required Arguments

### `--bw, --bigwig, -bw` or `-bed, --bed`

The prediction to benchmark, as a bigwig signal track (`--bw`) or BED file (`--bed`). Any bigwig signal track can be compared against a gold standard this way, not only maxATAC predictions.

### `--gold_standard`

The input gold standard bigwig file. This needs to be a binary signal track that has 1 corresponding to TFBS (e.g., from ChIP-seq peaks) and 0 in positions with no TFBS. It is required in both modes; with `--quant` it only supplies the chromosome lengths.

### `-n, --name, --prefix`

The output filename prefix to use.

## Quantitative benchmark arguments

### `--quant`

Run the quantitative benchmark (MAE, `R2_pred`, Pearson, Spearman, scatter plot) instead of the precision-recall analysis. Requires `--quant_gold_standard` and `--quant_gs_null`.

### `--quant_gold_standard`

The quantitative gold standard bigwig (e.g. ChIP-seq signal). It must be on the same scale as the model's training targets (see [`--target_scale_factor`](./train.md#--target_scale_factor)).

### `--quant_gs_null`

The null-model bigwig for `R2_pred`, typically the average of the quantitative gold standard tracks across all cell types for the TF:

```bash
maxatac average --quant -i CTCF_*_signal.bw -n CTCF_signal_all_celltypes_avg
```

## Optional Arguments

### `--chromosomes`

The chromosomes to benchmark the predictions for. Default: `chr1 chr8`, the held-out test chromosomes.

### `--bin_size`

The size of the bin to use for aggregating the single base-pair predictions. Default: `200` is the size used by the [ENCODE-DREAM in vivo TFBS Prediction Challenge](https://www.synapse.org/#!Synapse:syn6131484/wiki/402026). For quantitative benchmarks `32` (the model resolution) is a natural choice.

### `--agg`

The method used to aggregate the single base-pair values into bins, applied to the prediction and the gold standard(s). Options: `max`, `mean`, `min`, `sum`. Default: `max`, which is the binary default; `mean` is usually more appropriate for quantitative tracks.

See the [pyBigWig documentation](https://github.com/deeptools/pyBigWig#compute-summary-information-on-a-range) for more details.

### `--agg_threshold`

Binary benchmark only. When `--agg sum` is selected, the summed gold standard value is divided by the bin size and then converted to a binary label. Bins with values greater than or equal to this threshold become `1.0`; all others become `0.0`. Default: `0.5`.

### `--round_predictions`

Binary benchmark only. Round the binned prediction values to this number of decimal places. Default: `9`.

### `--peak_based`

Binary benchmark only. Compute the precision-recall curve on gold-standard *peak membership* (a peak counts as recovered once any of its bins is predicted above the threshold) instead of per bin. Default: `False`

### `--plot`

Binary benchmark only. Write the PR curve as a PNG next to the results TSV. The quantitative benchmark always writes a scatter plot. Default: `False`

### `--alternative_prediction`

An optional second prediction bigwig to benchmark together with the primary prediction. Before metrics are computed, the binned values of both files are combined per bin (see `--prediction_combine_operation`). If only one file provides a value for a bin, that value is used. Works in both modes.

### `--prediction_combine_operation`

How to combine bins when both `--bw` and `--alternative_prediction` provide a value: `mean` or `max`. Default: `mean`

### `-o, --output_directory`

The output directory to write the results to. Default: `./benchmarking_results`

### `--blacklist_bw`

The path to the blacklist bigwig signal track of regions that should be excluded. Default: the maxATAC blacklist bigwig for hg38, which contains regions that are specific to ATAC-seq.

### `--whitelist_bw`

The path to a whitelist bigwig signal track of regions that should be included. When provided, benchmarking is restricted to bins that overlap the whitelist track (in addition to blacklist exclusion).

### `--loglevel`

Logging level (`fatal`, `error`, `warning`, `info`, `debug`). Default: `info`.
