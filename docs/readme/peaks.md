# Peaks

The `peaks` function takes a quant-maxATAC prediction `.bw` signal track and calls intervals of TFBS whose score meets a calibrated threshold. These TFBS intervals are written as a BED file that can be visualized and used for downstream analysis.

The threshold is looked up in a calibration table written by [`maxatac threshold`](./threshold.md) (`<prefix>_cross_celltype.tsv`), which maps precision, recall and F1 values to prediction-score thresholds. For quantitative models the thresholds are on the predicted-signal scale, so the table must have been calibrated on predictions from the same (or an equivalently scaled) model. Tables from binary maxATAC v1 (`*_validationPerformance_vs_thresholdCalibration.tsv`) are no longer accepted; regenerate them with `maxatac threshold`.

`maxatac predict` runs this step automatically when a calibration table is available; `maxatac peaks` lets you re-call peaks with a different cutoff or bin size without re-predicting.

## Example

```bash
maxatac peaks -i GM12878_CTCF.bw -o ./peaks -bin 32 -cutoff_file CTCF_cross_celltype.tsv -cutoff_type Precision -cutoff_value 0.7
```

## Output

`<prefix>_<bin_size>bp.bed`: bins with a max score ≥ the threshold, merged into intervals. Column 4 holds the max score of the merged bins.

## Required Arguments

### `-i, --input_bigwig`

The input quant-maxATAC prediction bigwig file.

### `-cutoff_file, --cutoff_file`

The threshold calibration table written by `maxatac threshold`, provided in `/data/models/<TF>/` for the distributed TF models. It maps each target metric value to the prediction score threshold that achieves it.

## Optional Arguments

### `-cutoff_type, --cutoff_type`

The metric whose calibration grid is used to pick the threshold (`Precision`, `Recall`, or `F1`). Default: `F1`.

### `-cutoff_value, --cutoff_value`

The cutoff value for the cutoff type provided; precision, recall, and F1-scores range 0-1. Example: `.7`. Optional for `F1`, where omitting it selects the threshold with the highest F1. The lowest calibration bin that still meets the requested value is used.

### `-n, --name, -prefix, --prefix`

The prefix to use for the output file name. Default: the input bigwig filename without `.bw`.

### `-bin, --bin_size`

The bin size (TFBS interval length) used for calling peaks. Default: `32` bp, the resolution of the maxATAC models.

### `-o, --output`

The path to the output directory to write the BED file. Default: `./peaks`

### `-chromosomes`

The chromosomes to limit peak calling to. Default: autosomal chromosomes chr1-22.

### `--loglevel`

Logging level (`fatal`, `error`, `warning`, `info`, `debug`). Default: `info`.
