# Normalize

The `normalize` function will normalize an input bigwig file based on the following approaches:

* `min-max`: Find the genomic min and max values, then scale them between `[0,1]` or some-user defined range. The max value can be calculated as (1) the absolute max value across the genome (traditional definition of min-max) or (2) you can set a percentile cutoff to use as the max value. Option 2 improved robustness to outlying high ATAC-seq signal and maxATAC prediction accuracy. Specifically, we use the 99th-percentile max value instead of the absolute max value, and, given important performance ramifications, is the default.
* `zscore`: Set the mean value to 0 with a standard deviation of 1.
* `arcsinh`: Transform the values using an inverse hyperbolic sin transformation (arcsinh)

## Example

```bash
maxatac normalize -i GM12878_RP20M.bw -name GM12878_minmax -o ./test --method min-max --max_percentile 99
```

## Required Arguments

### `--signal`

The input bigwig file to be normalized.

### `--prefix`

The name used to build the output filename. This can be any string.

## Optional Arguments

### `--method`

The method to use for normalization. Default: `min-max`

* `min-max`: Find the genomic min and max values, then scale them between `[0,1]` or some user-defined range. The max value can be calculated as (1) the absolute max value across the genome (traditional definition of min-max) or (2) you can set a percentile cutoff to use as the max value. Option 2 improved robustness to outlying high ATAC-seq signal and maxATAC prediction accuracy. Specifically, we use the 99th-percentile max value instead of the absolute max value, and, given important performance ramifications, is the default.
* `zscore`: Set the mean value to 0 with a standard deviation of 1.
* `arcsinh`: Transform the values using an inverse hyperbolic sin transformation (arcsinh)

### `--max_percentile`

If method is `min-max` this argument will set the percentile value to use as the reported max value. The default is `99`, so that default will be consistent with the ATAC-seq processing for the maxATAC models, where the 99th percentile value was used as the max value.

### `--min`

The minimum value for `min-max` normalization. Default: `0`

### `--max`

The maximum value for `min-max` normalization. Default: `False`, so that max is calculated based on the ATAC signal track.

### `--clip`

This flag determines whether to clip the values that are above the max value used in `min-max` normalization or to leave them as their real value. Default: `False`

### `-c`, `--chroms`, `--chromosomes`

Define the chromosomes that are normalized. Only the chromosomes in this list will be written to the output file. The current default list of chromosomes are restricted to the autosomal chromosomes:

```pre
chr1, chr2, chr3, chr4, chr5, chr6, chr7, chr8, chr9, chr10, chr11, chr12, chr13, chr14, chr15, chr16, chr17, chr18, chr19, chr20, chr21, chr22
```

### `-cs`, `--chrom_sizes`, `--chromosome_sizes`

Define the chromosome sizes file. The current default file are the chromosome sizes for hg38.

### `--blacklist_bw`

The path to the blacklist bigwig file. This file is used to remove all the regions that are considered to have high technical noise. Default: maxATAC publication-defined blacklist.

### `--output`

Define the output directory. If the output directory is not supplied a directory called `./normalize` will be created in the current working directory.

### `--loglevel`

Set the logging level. Currently, the only working logging level is `ERROR`.
