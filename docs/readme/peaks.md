# Peaks

The `peaks` will take a maxATAC prediction `.bw` signal track and call bins that are above a specific threshold. These bins will be merged and output as BED intervals that can be visualized and used for downstream analysis.

## Example

`maxatac peaks -i GM12878_CTCF.bw -o ./peaks -bin 32 -threshold .75`

## Required Arguments

### `"-i", "--input_bigwig"`

The input maxATAC bigwig file.

### `"-cutoff_value", "--cutoff_value"`

The cutoff value for the cutoff type provided. Example: .7

### `"-cutoff_type", "--cutoff_type"`

The cutoff type (i.e. Precision, Recall, F1, Log2FC). Default: F1.

### `"-cutoff_file", "--cutoff_file"`

The cutoff file provided in /data/models that corresponds to the average validation performance metrics for the TF model.

## Optional Arguments

### `"-prefix", "--prefix"`

The prefix to use for the output file name.

### `"-bin", "--bin_size"`

The bin size to use for calling peaks. Default: 200 bp based on the same sized used for benchmarking predictions.

### `"-o", "--output"`

The path to the output directory to write the bed.

### `"--chromosomes"`

The chromosomes to limit peak calling to. Default: Autosomal chromosomes that are used in training and evaluation.
