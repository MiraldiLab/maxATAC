# Peaks

The `peaks` will take a maxATAC prediction `.bw` signal track and call intervals of TFBS that meet a given confidence threshold. These TFBS intervals will be output as BED intervals that can be visualized and used for downstream analysis.

The peaks function takes as input a bigwig signal track and will output bins that are above a given threshold. 



## Example

`maxatac peaks -i GM12878_CTCF.bw -o ./peaks -bin 32 -cutoff_file ARID3A_validationPerformance_vs_thresholdCalibration.tsv`

## Required Arguments

### `"-i", "--input_bigwig"`

The input maxATAC bigwig file.

### `"-cutoff_file", "--cutoff_file"`

The cutoff file provided in /data/models that corresponds to the average validation performance metrics for the TF model. 

## Optional Arguments
Note on abbreviations: 

* F1 = F1-score
* log2FC = Log2( Precision : Random Precision))

### `"-cutoff_type", "--cutoff_type"`

The cutoff type (i.e. `Precision`, `Recall`, `F1`, `log2FC`). Default: F1.

### `"-cutoff_value", "--cutoff_value"`

The cutoff value for the cutoff type provided. Note precision, recall, and F1-scores range 0-1, while better-than-random log2FC scores range from 0 to infinity. Example: .7

### `"-prefix", "--prefix"`

The prefix to use for the output file name.

### `"-bin", "--bin_size"`

The bin size (TFBS interval length) used for calling peaks. Default: 32 bp, based on the benchmarking intervals predictions. 32 bp, the resolution of the maxATAC models, is also a good option. 

### `"-o", "--output"`

The path to the output directory to write the bed.

### `"--chromosomes"`

The chromosomes to limit peak calling to. Default: Autosomal chromosomes that are used in training and evaluation.
