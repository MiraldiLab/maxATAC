# Normalize

The `normalize` function will normalize an input bigwig file based on the following approaches:

* `min-max`: Find the genomic min and max values, then scale them between `[0,1]` or some user-defined range. The max value can be calculated as (1) the absolute max value across the genome (traditional definition of min-max) or (2) you can set a percentile cutoff to use as the max value. Option 2 improved robustness to outlying high ATAC-seq signal and maxATAC prediction accuracy. Specifically, we use the 99th-percentile max value instead of the absolute max value, and, given important performance ramifications, is the default. This is the normalization applied to the **ATAC-seq input** by `maxatac prepare`.
* `zscore`: Set the mean value to 0 with a standard deviation of 1.
* Variance-stabilizing transforms, provided for preparing **quantitative ChIP-seq target tracks** for `--quant` training and benchmarking:
  * `arcsinh`: inverse hyperbolic sine, `arcsinh(x)`
  * `log1p`: natural log of one plus the value, `log(1 + x)`
  * `sqrt`: square root, `x^(1/2)`
  * `three_fourths`: `x^(3/4)`
  * `three_eighths`: `x^(3/8)`

<!-- TODO: state which transform (if any) was applied to the ChIP-seq signal used as targets for the released
quant models, so users can prepare matching gold standards. -->

In every method, blacklisted regions are set to 0 in the output.

## Example

```bash
maxatac normalize -i GM12878_RP20M.bw -n GM12878_minmax -o ./test --method min-max --max_percentile 99
```

## Required Arguments

### `-i`, `--signal`

The input bigwig file to be normalized.

### `-n`, `--name`, `--prefix`

The name used to build the output filename. This can be any string; `.bw` is appended.

## Optional Arguments

### `--method`

The method to use for normalization: `min-max`, `zscore`, `arcsinh`, `log1p`, `sqrt`, `three_fourths` or `three_eighths` (see above). Default: `min-max`

### `--max_percentile`

If method is `min-max` this argument will set the percentile value to use as the reported max value. The default is `99`, so that default will be consistent with the ATAC-seq processing for the maxATAC models, where the 99th percentile value was used as the max value.

### `--min`

The minimum value for `min-max` normalization. Default: `0`

### `--max`

The maximum value for `min-max` normalization. Default: `False`, so that max is calculated based on the ATAC signal track.

### `--clip`

Flag. Clip values above the max used in `min-max` normalization to 1 instead of leaving them above 1. Default: `False`

### `-c`, `--chroms`, `--chromosomes`

Define the chromosomes that are normalized. Only the chromosomes in this list will be written to the output file. The current default list of chromosomes are restricted to the autosomal chromosomes:

```pre
chr1, chr2, chr3, chr4, chr5, chr6, chr7, chr8, chr9, chr10, chr11, chr12, chr13, chr14, chr15, chr16, chr17, chr18, chr19, chr20, chr21, chr22
```

### `-cs`, `--chrom_sizes`, `--chromosome_sizes`

Define the chromosome sizes file. The current default file are the chromosome sizes for hg38.

### `--blacklist_bw`

The path to the blacklist bigwig file. This file is used to remove all the regions that are considered to have high technical noise. Default: maxATAC publication-defined blacklist.

### `--max_zooms`

The number of zoom levels to compute for the normalized bigWig file. Zoom levels are pre-computed summary statistics that let genome browsers (IGV, UCSC Genome Browser) zoom in and out of a region quickly; fewer levels mean slower loading in browsers, more levels mean more memory during writing. Valid range: `0`-`10`. Default: `10`. Note: a value of `0` produces a bigWig that is not compatible with other bigWig tools (e.g. deepTools) and cannot be visualized in IGV or the UCSC Genome Browser. See the [pyBigWig README](https://github.com/deeptools/pyBigWig/blob/master/README.md) for details.

### `-o`, `--output`, `--output_dir`

Define the output directory. Default: the current working directory.

### `--loglevel`

Logging level (`fatal`, `error`, `warning`, `info`, `debug`). Default: `info`.
