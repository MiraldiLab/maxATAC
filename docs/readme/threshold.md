# Threshold

The `threshold` function calibrates prediction-score thresholds for a model against binary gold standards and writes the **threshold calibration table** (`<prefix>_cross_celltype.tsv`) that [`maxatac predict`](./predict.md) and [`maxatac peaks`](./peaks.md) use to turn a requested precision, recall or F1 into a cutoff on the prediction track.

Because quantitative models output predicted ChIP-seq signal rather than probabilities, the calibrated thresholds are on the predicted-signal scale of the model used. A table must therefore be generated from predictions of the model (or an equivalently scaled model) it will be applied to. The tables distributed with the quant-maxATAC models live in `~/opt/maxatac/data/models/<TF>/`.

## Approach

For every cell type listed in the meta file:

1. The prediction bigwig and the binary gold standard bigwig are binned at `--bin_size` (max per bin) on the single held-out chromosome given by `--chromosomes`, excluding blacklisted bins.
2. A precision-recall curve is computed from the binned prediction scores against the binned gold standard: for every threshold, the precision, recall, F1 and log2(precision / random precision) **actually achieved at that threshold**.
3. Only thresholds worth using are kept. A threshold is dropped when a lower threshold reaches the same or a higher precision, because that lower threshold also has at least the recall (and Peak_Recall). On the kept rows, precision rises and recall falls as the threshold increases, and every row is a real threshold with the metrics it achieves: no value is carried over from another threshold. The final point of a precision-recall curve (Precision = 1, Recall = 0) belongs to no threshold and is not included, so a precision of 1 only appears if some threshold really reaches it.
4. Alongside the bin-level Recall, a peak-level **Peak_Recall** is computed per kept threshold: the fraction of unique ChIP-seq peaks with at least one non-blacklisted bin whose prediction is ≥ the threshold. Peaks are the unique intervals of the optional `ChIP_peaks` BED, or runs of consecutive gold-standard bins when it is not given. Fully blacklisted peaks are not counted.
5. The curve is re-binned on a 0.01 grid of Precision, Recall and Peak_Recall values, keeping the highest-F1 row in each bin.

The per-cell-type grids are then combined into the cross-cell-type table:

* Each cell type's grid is first extended to every bin from 0.00 to 1.00, so all cell types cover the same bins. Bins between two values the curve reaches are linearly interpolated. Bins beyond the curve's range repeat its nearest real row, so their `Precision`/`Recall` columns show what the cell type actually reaches, not the bin value.
* Per bin, the median `Precision`, `Recall`, `Threshold` and `Peak_Recall` across cell types is taken (only these per-threshold values are aggregated, never raw signal), and F1 is recomputed from the median precision and recall. A row therefore describes a typical cell type, not one particular cell type.
* The F1 grid is built from the Recall table by re-binning its rows on F1.

`predict`/`peaks` look up a threshold by the **achieved** value of the requested metric, not by the bin label: among the rows of `--cutoff_type` whose achieved `Precision`, `Recall`, `F1` or `Peak_Recall` is ≥ `--cutoff_value`, they use the one closest to the request (ties go to the higher F1). If no row reaches the value, they stop with an error naming the highest value the table reaches. With `--cutoff_type F1` and no `--cutoff_value`, the max-F1 row is used.

## Meta file

A tab-separated file with one row per cell type:

| Column Name    | Description                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| `Prediction`   | Path to the prediction bigwig for this cell type (from `maxatac predict`, quantitative or binary)    |
| `Binding_File` | Path to the binary gold standard bigwig for this cell type (1 = TF bound, 0 = unbound)               |
| `ChIP_peaks`   | *(Optional)* ChIP-seq peak BED for this cell type, defining the unique peaks counted by `Peak_Recall`. If the column is missing or a cell is empty, runs of consecutive gold-standard bins are used as peaks (peaks less than one bin apart then merge) |

Additional columns (e.g. `Cell_Line`, `TF`) are ignored. Use validation cell types and a chromosome that were not used to train the model.

## Example

```bash
maxatac threshold --prefix CTCF --meta_file CTCF_threshold_meta.tsv --chromosomes chr2 --bin_size 200 --output ./threshold
```

## Outputs

| Filename                                            | Description                                                                                                              |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `<prefix>_cross_celltype.tsv`                       | The calibration table used by `predict`/`peaks`. Columns: `Metric` (`Precision`, `Recall`, `F1` or `Peak_Recall`), `Bin` (position on the 0.01 grid of that metric), `Precision`, `Recall`, `Threshold`, `F1`, `Peak_Recall` (fraction of unique ChIP-seq peaks recovered at that threshold). The `Precision`/`Recall`/`F1`/`Peak_Recall` columns are the values achieved at `Threshold`; past the range a curve reaches they can be lower than `Bin` |
| `<prediction basename>.tsv`                         | The same table for each individual cell type in the meta file                                                            |
| `<prefix>_peak_recovery.tsv`                        | Per cell type: peak source, number of unique peaks, and `Peak_Recall` at that cell type's own max-F1 threshold and at the cross-cell-type max-F1 threshold |
| `*_validationPerformance_vs_thresholdCalibration.png` | Precision, recall, F1 and fraction of unique peaks recovered vs. threshold for every cell type, with the cross-cell-type median in black |

## Required Arguments

### `--prefix`

Output filename prefix. For distributed models this should be the TF name, so that `maxatac predict -tf <TF>` can find `<TF>_cross_celltype.tsv`.

### `--meta_file`

The meta file described above.

## Optional Arguments

### `--chromosomes`

The single held-out chromosome used to calibrate the thresholds. Exactly one chromosome must be given; `threshold` exits with an error otherwise. Default: `chr2`.

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
