# Variants

The `variants` function is intended for using a maxATAC model to predict TF binding in non-overlapping LD blocks. *The function can perform whole genome prediction, but has not been optimized for that yet.* The `variants` function takes as input a bed file of variants and the nucleotide to use at that position. You must also provide the regions that you want to make predictions in. This function will then merge nearby ROI intervals (+/- 512 bp) and create sliding windows (1,024 bp wide x 256 bp step) along the ROI. Regions at the end of the interval will be trimmmed off if they are less than 1,024 bp. 

## Example

```bash
maxatac variants -m ELF1_99.h5 -signal GM12878__slop20bp_RP20M_minmax01.bw -name GM12878_ELF1 -s hg38.2bit --chromosome chr20 -variants_bed AD_risk_loci.bed
```

## Required Arguments

### `-m, --model`

The trained maxATAC model that will be used to predict TF binding. This is a h5 file produced from `maxatac train`. 

### `--signal`

The ATAC-seq signal bigwig track that will be used to make predictions of TF binding. 

### `--variants_bed`

The bed file of nucleotides to change. The first 3 columns should be the coordinates and the fourth column should be the nucleotide to use.

### `-n, --name`

Output filename prefix to use.

## Optional Arguments

### `-s, --sequence`

This argument specifies the path to the 2bit DNA sequence for the genome of interest. Default: hg38.2bit

### `-roi`

The bed file of intervals to use for prediction windows. Predictions will be limited to these specific regions. Only the first 3 columns of the file will be considered when making the prediction windows. Default: Whole genome prediction. 

### `-o, --output`

Output directory path. Default: `./variantss`

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

### `--blacklist`

The path to a bigwig file that has regions to exclude. Default: maxATAC defined blacklist.

### `-step_size`

The number of base pairs to overlap the 1,024 bp regions during prediction. This should be in multiples of 256. Default: 256

### `-chroms, --chromosomes`

The chromosomes to make predictions on. Default: All chromosomes. chr1-22, X, Y

### `--chrom_sizes`

The path to the chromosome sizes file. This is used to generate the bigwig signal tracks. Default: hg38.chrom.sizes
