# Peaks

The `peaks` will take a maxATAC prediction `.bw` signal track and call intervals of TFBS that meet a given confidence threshold. These TFBS intervals will be output as BED intervals that can be visualized and used for downstream analysis.

The peaks function takes as input a bigwig signal track and will output bins that are above a given threshold. 



## Example

`maxatac peaks -i GM12878_CTCF.bw -o ./peaks -bin 32 -cutoff_file ARID3A_cross_celltype.tsv`

## Required Arguments

### `"-i", "--input_bigwig"`

The input maxATAC bigwig file.

### `"-cutoff_file", "--cutoff_file"`

The threshold calibration table written by `maxatac threshold`, provided in /data/models for the TF model. It maps each target metric value to the prediction score threshold that achieves it.

## Optional Arguments
Note on abbreviations: 

* F1 = F1-score

### `"-cutoff_type", "--cutoff_type"`

The metric whose calibration grid is used to pick the threshold (`Precision`, `Recall`, or `F1`). Default: F1.

### `"-cutoff_value", "--cutoff_value"`

The cutoff value for the cutoff type provided; precision, recall, and F1-scores range 0-1. Example: .7. Optional for F1, where omitting it selects the threshold with the highest F1.

### `"-n", "--name", "-prefix", "--prefix"`

The prefix to use for the output file name.

### `"-bin", "--bin_size"`

The bin size (TFBS interval length) used for calling peaks. Default: 32 bp, based on the benchmarking intervals predictions. 32 bp, the resolution of the maxATAC models, is also a good option. 

### `"-o", "--output"`

The path to the output directory to write the bed.

### `"--chromosomes"`

The chromosomes to limit peak calling to. Default: Autosomal chromosomes that are used in training and evaluation.
