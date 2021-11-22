# Peaks

The `peaks` will take a maxATAC prediction `.bw` signal track and call bins that are above a specific threshold. These bins will be merged and output as BED intervals that can be visualized and used for downstream analysis.

## Example

`maxatac peaks -i GM12878_CTCF.bw -o ./peaks -bin 32 -threshold .75`

## Required Arguments

### `"-i", "--input_bigwig"`

The input maxATAC bigwig file.

### `"-threshold", "--threshold"`

The minimum threshold to use for the calling peaks.

## Optional Arguments

### `"-prefix", "--prefix"`

The prefix to use for the output file name.

### `"-bin", "--bin_size"`

The bin size to use for calling peaks.

### `"-o", "--output"`

The path to the output directory to write the bed.

### `"--chromosomes"`

The chromosomes to limit peak calling to.
