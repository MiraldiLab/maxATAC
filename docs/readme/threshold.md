# Threshold

The `threshold` function calibrates prediction-score thresholds for a model against binary gold standards and writes the **threshold calibration table** (`<prefix>_cross_celltype.tsv`) that [`maxatac predict`](./predict.md) and [`maxatac peaks`](./peaks.md) use to turn a requested precision, recall or F1 into a cutoff on the prediction track.

Because quantitative models output predicted ChIP-seq signal rather than probabilities, the calibrated thresholds are on the predicted-signal scale of the model used. A table must therefore be generated from predictions of the model (or an equivalently scaled model) it will be applied to. The tables distributed with the quant-maxATAC models live in `~/opt/maxatac/data/models/<TF>/`.

## Approach

For every cell type listed in the meta file:

1. The prediction bigwig and the binary gold standard bigwig are binned at `--bin_size` (max per bin) on `--chromosomes`, excluding blacklisted bins.
2. A precision-recall curve is computed from the binned prediction scores against the binned gold standard, giving precision, recall, F1 and log2(precision / random precision) for every threshold.
3. The curve is re-binned on a 0.01 grid of Precision, Recall and F1 values so that each metric value maps to the threshold that achieves it.

The per-metric grids are then combined across cell types by taking the median threshold per grid bin (only thresholds are aggregated, never raw signal), producing the cross-cell-type table. `predict`/`peaks` look up the lowest grid bin that still meets the requested `--cutoff_value` (or the max-F1 bin when `--cutoff_type F1` is used without a value).

## Meta file

A tab-separated file with one row per cell type:

| Column Name    | Description                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| `Prediction`   | Path to the prediction bigwig for this cell type (from `maxatac predict`, quantitative or binary)    |
| `Binding_File` | Path to the binary gold standard bigwig for this cell type (1 = TF bound, 0 = unbound)               |

Additional columns (e.g. `Cell_Line`, `TF`) are ignored. Use validation cell types / chromosomes that were not used to train the model.

## Example

```bash
maxatac threshold --prefix CTCF --meta_file CTCF_threshold_meta.tsv --chromosomes chr2 chr19 --bin_size 200 --output ./threshold
```

## Outputs

| Filename                                            | Description                                                                                                              |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `<prefix>_cross_celltype.tsv`                       | The calibration table used by `predict`/`peaks`. Columns: `Metric` (`Precision`, `Recall` or `F1`), `Bin` (metric value on the 0.01 grid), `Precision`, `Recall`, `Threshold`, `F1` |
| `<prediction basename>.tsv`                         | The same table for each individual cell type in the meta file                                                            |
| `*_validationPerformance_vs_thresholdCalibration.png` | Precision, log2FC, recall and F1 vs. threshold for every cell type, with the cross-cell-type median in black          |

## Required Arguments

### `--prefix`

Output filename prefix. For distributed models this should be the TF name, so that `maxatac predict -tf <TF>` can find `<TF>_cross_celltype.tsv`.

### `--meta_file`

The meta file described above.

## Optional Arguments

### `--chromosomes`

The chromosomes used to calibrate the thresholds. Default: `chr2 chr19` (the default validation chromosomes).

### `--bin_size`

The bin size (bp) at which predictions and gold standards are compared. Default: `200`

### `--chrom_sizes`

Chromosome sizes file. Default: hg38 chromosome sizes.

### `--blacklist_bw`

The blacklist bigwig of regions to exclude. A BED file with the same basename must sit next to it (it is used to build the non-blacklisted bin count). Default: the maxATAC blacklist for hg38.

### `--output`

Output directory. Default: `./threshold`

### `--loglevel`

Logging level (`fatal`, `error`, `warning`, `info`, `debug`). Default: `info`.
