# Variants

The `variants` function will use a maxATAC model to predict TF binding centered over a specific genetic loci. The `variants` function can alter the specific nucleotide at the target position.

## Example

```bash
maxatac variants -m ELF1_99.h5 -signal GM12878__slop20bp_RP20M_minmax01.bw -name GM12878_ELF1 -s hg38.2bit --chromosome chr20 -p 20000000 -nuc A
```

## Required Arguments

### `-m, --model`

The trained maxATAC model that will be used to predict TF binding. This is a h5 file produced from `maxatac train`. 

### `--signal`

The ATAC-seq signal bigwig track that will be used to make predictions of TF binding. 

### `-s, --sequence`

This argument specifies the path to the 2bit DNA sequence for the genome of interest

### `--chromosomes`

The chromosomes to make predictions on. Default: `chr1, chr8`

### `-p, --position`

The chromosome start position for the variant or loci to be altered. These coordinates should be the start position of the 0-based coordinate system.

### `-nuc, --target_nucleotide`

The target nucleotide to change the variant position to. Example: A

### `-overhang`

The number of base pairs around a 1,024 bp region to include in prediction. This should be in multiples of 256.

## Optional Arguments

### `-o, --output`

Output directory path. Default: `./variantss`

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

### `-n, --name`

Output filename prefix to use.
