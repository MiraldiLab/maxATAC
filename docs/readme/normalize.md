# Normalize

The `normalize` function will normalize an input bigwig file based on the following approaches:

* `min-max`: Find the genomic min and max values, then scale them between `[0,1]` or some-user defined range. The max value can be calculated as (1) the absolute max value across the genome (traditional definition of min-max) or (2) you can set a percentile cutoff to use as the max value. Option 2 improved robustness to outlying high ATAC-seq signal and maxATAC prediction accuracy. Specifically, we use the 99th-percentile max value instead of the absolute max value, and, given important performance ramifications, is the default.
* `median-mad`: Find the genomic median and calculate the median absolute deviation.
* `zscore`: Set the mean value to 0 with a standard deviation of 1.
* `arcsinh`: Transform the values using an inverse hyperbolic sin transformation (arcsinh)

## Example

```bash
maxatac normalize --signal GM12878_RP20M.bw --prefix GM12878 --output ./test --method min-max --max_percentile 99
```

## Required Arguments

### `--signal`

The input bigwig file to be normalized.

## Optional Arguments

### `--method`

This argument is used to determine which method to use for normalization. Default: `min-max`

* `min-max`: Find the genomic min and max values, then scale them between `[0,1]` or some user-defined range. The max value can be calculated as (1) the absolute max value across the genome (traditional definition of min-max) or (2) you can set a percentile cutoff to use as the max value. Option 2 improved robustness to outlying high ATAC-seq signal and maxATAC prediction accuracy. Specifically, we use the 99th-percentile max value instead of the absolute max value, and, given important performance ramifications, is the default.
* `median-mad`: Find the genomic median and calculate the median absolute deviation.
* `zscore`: Set the mean value to 0 with a standard deviation of 1.
* `arcsinh`: Transform the values using an inverse hyperbolic sin transformation (arcsinh)

### `--max_percentile`

If method is `min-max` this argument will set the percentile value to use as the reported max value. The default is `99`, so that default will be consistent with the ATAC-seq processing for the maxATAC models, where the 99th percentile value was used as the max value.

### `--min`

The value to use as the minimum value for `min-max` normalization. Default: `0`

### `--max`

The value to use as the maximum value for `min-max` normalization. Default: `False`, so that max is calculated based on the ATAC signal track.

### `--clip`

This flag determines whether to clip the values that are above the max value used in `min-max` normalization or to leave them as their real value. Default: `False`

### `--prefix`

This argument is reserved for the prefix used to build the output filename. This can be any string.

### `--chroms`

This argument is used to define the chromosomes that are averaged together. Only the chromosomes in this list will be written to the output file. The current default list of chromosomes are restricted to the autosomal chromosomes:

```pre
chr1, chr2, chr3, chr4, chr5, chr6, chr7, chr8, chr9, chr10, chr11, chr12, chr13, chr14, chr15, chr16, chr17, chr18, chr19, chr20, chr21, chr22
```

### `--chrom_sizes`

This argument is used to define the chromosome sizes file that is used to calcuate the chromosome ends. The current default file are the chromosome sizes for hg38.

### `--blacklist_bw`

The path to the blacklist bigwig file. This file is used to remove all the regions that are considered to have high technical noise. Default: maxATAC publication-defined blacklist.

### `--output`

This argument is used to define the output directory. If the output directory is not supplied a directory called `./normalize` will be created in the current working directory.

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.
