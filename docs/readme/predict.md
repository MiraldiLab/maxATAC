# Predict

The `predict` function will use a maxATAC model to predict TF binding in a new condition. The user must provide a model and a bigwig file that corresponds to an ATAC-seq signal track. 

## Example

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig
```

## Required Arguments

### `--sequence`

This argument specifies the path to the 2bit DNA sequence for the genome of interest. maxATAC models are trained with hg38 so you will need the correct `.2bit` file.

### `--model`

The trained maxATAC model that will be used to predict TF binding. This is a h5 file produced from `maxatac train`.

### `--signal`

The ATAC-seq signal bigwig track that will be used to make predictions of TF binding.

## Optional Arguments

### `"-cutoff_value", "--cutoff_value"`

The cutoff value for the cutoff type provided. Example: .7

### `"-cutoff_type", "--cutoff_type"`

The cutoff type (i.e. Precision, Recall, F1, Log2FC). Default: F1.

### `"-cutoff_file", "--cutoff_file"`

The cutoff file provided in /data/models that corresponds to the average validation performance metrics for the TF model.

### `--output`

Output directory path. Default: `./prediction_results`

### `--blacklist`

The path to a bigwig file that has regions to exclude. Default: maxATAC defined blacklist.

### `--roi`

The path to a bed file that contains the genomic regions to predict TF binding in. These regions should be the same size as the regions that are used for training (i.e. 1,024 bp).

### `--batch_size`

The number of regions to predict on per batch. Default `10000`. Decrease this value if you are having memory issues.

### `--step_size`

The step size to use for building the prediction intervals. Overlapping prediction bins will be averaged together. Default: `INPUT_LENGTH/4` which corresponds to the input size of 1,024 bp. 

### `--prefix`

Output filename prefix to use. Default `maxatac_predict`.

### `--chromosome_sizes`

The path to the chromosome sizes file. This is used to generate the bigwig signal tracks.

### `--chromosomes`

The chromosomes to make predictions on. Default: Autosomal chromosomes 1-22.

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

### `"-bin", "--bin_size"`

The bin size to use for calling peaks. Default: 200 bp based on the same sized used for benchmarking predictions.
