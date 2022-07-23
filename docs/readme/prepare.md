# Prepare

The `prepare` function will convert a BAM file to Tn5 cut sites that are smoothed with a specific slop size. The files are converted to bigwig signal tracks and then min-max normalized. The `maxatac prepare` function requires `samtools`, `bedtools`, `pigz`, and `bedGraphToBigWig` be installed on your PATH to run.

## Examples

### Bulk ATAC-seq
```bash
maxatac prepare -i GM12878.bam -o ./output -prefix GM12878
```

### Pseudo-bulk scATAC-seq
```bash
maxatac prepare -i GM12878_scatac_1M.tsv -o ./output -prefix GM12878
```

### Outputs

There are multiple outputs from the prepare function. The outputs and a brief description are shown below for an example pseudo-bulk GM12878 scATAC-seq input with the name `-prefix GM12878_scatac_1M`.

| Filename                                                          | Description                                                                                 |
|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| GM12878_scatac_1M_IS_slop20_RP20M_minmax01_chromosome_min_max.txt | Contains the minimum and maximum values per chromosome                                      |
| GM12878_scatac_1M_IS_slop20_RP20M_minmax01_genome_stats.txt       | Contains the min, max, median, and stats on the input file.                                 |
| GM12878_scatac_1M_IS_slop20_RP20M_minmax01.bw                     | The output file that is to be used for prediction. This file has been min-max normalized.   |
| GM12878_scatac_1M_IS_slop20_RP20M.bw                              | The read-depth normalized signal tracks.                                                    |
| GM12878_scatac_1M_IS_slop20.bed.gz                                | The compressed bed file of individual cut sites that have been corrected for the Tn5 shift. |

Description of filename parts:

* `IS` : Insertion sites
* `slop20`: Slop size 20
* `RP20M`: Read depth normalized to 20,000,000 reads
* `minmax01`: Min-max normalized between 0 and 1

## Required Arguments

### `-i, --input`

The input file to be processed. The input file can be either:

* `.bam`: Bulk ATAC-seq BAM file.
* `.tsv`: 10X scATAC fragments file. Must end in `.tsv` or `.tsv.gz`.

### `-o, --output`

The output directory path.

### `-prefix`

This argument is used to set the prefix for setting the filenames. 

## Optional Arguments

The default values for the optional arguments are based on the testing performed in the maxATAC publication. See the [Methods](https://www.biorxiv.org/content/10.1101/2022.01.28.478235v1.article-metrics) of our publication for a detailed explanation of each parameter choice.

### `-skip_dedup, --skip_deduplication`

It is important to remove PCR duplicates from your ATAC-seq data if you have not done so already. Include this flag to perform PCR deduplication of the input BAM file if you know that it has not been deduplicated. Skipping this step will speed up data processing. Defualt: False

### `-slop, --slop`

The slop size used to smooth sparse Tn5 cut sites' signal. Each Tn5 cut site will be extended +/- the slop size (in bp). Because maxATAC models were trained using slop size of 20bp (a value that approximates the size of Tn5 transposase), **this parameter should not be changed from default (20 bp) when using the trained models provided by maxATAC**. Default: 20 bp.

### `-rpm, --rpm_factor`

The reads per million (RPM) factor used for read-depth normalization of signal. Most groups use RPM and therefore 1,000,000 as a scaling factor, but maxATAC uses RP20M and therefore 20,000,000 because it is approximately the median sequencing depth of the ATAC-seq data used for training. Changing from the default (20000000) is not problematic for maxATAC prediction, as this track is only used for visualization. (Predictions are made on a min-max-like normalized signal track, also an output from `maxatac prepare`.) Default: 20000000.

### `--blacklist`

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
