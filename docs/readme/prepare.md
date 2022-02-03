# Prepare

The `prepare` function will convert a BAM file to Tn5 cut sites that are smoothed with a specific slop size. The files are converted to bigwig signal tracks and then min-max normalized. The `maxatac prepare` function requires `samtools`, `bedtools`, `pigz`, and `bedGraphToBigWig` be installed on your PATH to run.

## Example

```bash
maxatac prepare -i SRX2717911.bam -o ./output -prefix SRX2717911
```

## Required Arguments

### `-i, --input`

The input file to be processed. The input file can be either:

* `.bam`: Bulk ATAC-seq BAM file.
* `.tsv`: 10X scATAC fragments file. Must end in `.tsv` or `.tsv.gz`.

### `-o, --output`

The output directory path.

### `-prefix`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

## Optional Arguments

The default values for the optional arguments are based on the testing performed in the maxATAC publication. See the [Methods](https://www.biorxiv.org/content/10.1101/2022.01.28.478235v1.article-metrics) of our publication for a detailed explanation of each parameter choice.

### '-dedup, --deduplicate'

It is important to remove PCR duplicates from your ATAC-seq data if you have not done so already. Include this flag to perform PCR deduplication of the input BAM file if you know that it has not been deduplicated. Skipping this step will speed up data processing.

### `-slop, --slop`

The slop size to use around the Tn5 cut sites. We use a slop size to smooth the sparse cut site resolved signal using a value that approximates the size of Tn5 transposase. Default: 20 bp.

### `-rpm, --rpm_factor`

The RPM factor to use for scaling your read depth normalized signal. Most groups use 1,000,000 as a scaling factor, but maxATAC uses 20,000,000 because it is approximately the median sequencing depth of the ATAC-seq data used for training. Default: 20000000.

### `--blacklist_bed`

The path to the blacklist bed file. Default: maxATAC defined blacklisted regions for hg38.

### `--blacklist_bw`

The path to the blacklist bigwig file. Default: maxATAC defined blacklist for hg38 as a bigwig file.

### `--chrom_sizes`

The chromosome sizes file. Default: hg38 chrom sizes.

### `-chroms, --chromosomes`

The chromosomes to use for the final output. Default: Autosomal chromosomes chr1-22.

### `-threads`

The number of threads to use. Default: Get available CPU count.

### `--loglevel`

The log level to use for printing messages.
