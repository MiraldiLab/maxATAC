# Predict

The `predict` function will use a maxATAC model to predict TF binding in a new condition. The user must provide a model and a bigwig file that corresponds to an ATAC-seq signal track. 

## Example

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig
```

## Required Arguments

### `--sequence`

This argument specifies the path to the 2bit DNA sequence for the genome of interest

### `--models`

The trained maxATAC model that will be used to predict TF binding. This is a h5 file produced from `maxatac train`. 

### `--signal`

The ATAC-seq signal bigwig track that will be used to make predictions of TF binding. 

## Optional Arguments

### `--quant`

Whether the model was used with quantitative data. Default: `False`

### `--output`

Output directory path. Default: `./prediction_results`

### `--blacklist`

The path to a bigwig file that has regions to exclude.

### `--roi`

The path to a bed file that contains the genomic regions to predict TF binding in. These regions should be the same size as the regions that are used for training. 

### `--stranded`

Whether to predict on both the reference strand sequence and the complement strand sequence. This will produce a third file which is also the mean prediction between both strands. Default: `False`

### `--threads`

The number of threads to use for multiprocessing.

### `--batch_size`

The number of regions to predict on per batch. Default `10000`

### `--step_size`

The step size to use for building the prediction intervals. Overlapping prediction bins will be averaged together. Default: `1024`

### `--prefix`

Output filename prefix to use. Default `maxatac_predict

### `--chromosome_sizes`

The path to the hromosome sizes file. This is used to generate the bigwig signal tracks. 

### `--chromosomes`

The chromosomes to make predictions on. Default: `chr1, chr8`

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.
