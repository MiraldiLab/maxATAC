# Predict

The `predict` function will use a maxATAC model to predict TF binding in a new condition. The user must provide a model and a bigwig file that corresponds to an ATAC-seq signal track. 

## Example

```bash
maxatac predict --model CTCF.h5 --signal GM12878.bigwig
```

or

```bash
maxatac predict --tf CTCF --signal GM12878.bigwig
```

## Required Arguments

### `-tf, --tf_name` or `-m, --model`

The user must provide either the TF name that they want to make predictions for or the h5 model file they desire. If the user provides a TF name, the best model will be used and the correct threshold file will be provided for peak calling.

### `-s, --signal, -i`

The ATAC-seq signal bigwig track that will be used to make predictions of TF binding.

### `-n, --name, --prefix`

Output filename prefix to use. Default `maxatac_predict`.

## Optional Arguments

### `--sequence, --seq`

This argument specifies the path to the 2bit DNA sequence for the genome of interest. maxATAC models are trained with hg38 so you will need the correct `.2bit` file.

### `"-cutoff_type", "--cutoff_type"`

The cutoff type (i.e. `Precision`, `Recall`, `F1`, `log2FC`). (F1 = F1-score, and log2FC = Log2( Precision : Random Precision)). Default: F1.

### `"-cutoff_value", "--cutoff_value"`

The cutoff value for the cutoff type provided. Note precision, recall, and F1-scores range 0-1, while better-than-random log2FC scores range from 0 to infinity. Example: .7

### `-cutoff_file, --cutoff_file`

The cutoff file provided in /data/models that corresponds to the average validation performance metrics for the TF model.

### `-o, --output`

Output directory path. Default: `./prediction_results`

### `-bl, --blacklist`

The path to a bigwig file that has regions to exclude. Default: maxATAC-defined blacklist.

### `--bed, --peaks, --regions, , --roi, -roi`

The path to a bed file that contains the genomic regions to focus TF predictions on. These peaks will be used to refine the prediction windows. 

### `--batch_size`

The number of regions to predict on per batch. Default `10000`. Decrease this value if you are having memory issues.

### `--step_size`

The step size to use for building the prediction intervals. Overlapping prediction bins will be averaged together. Default: `INPUT_LENGTH/4`, where INPUT_LENGTH is the maxATAC model input size of 1,024 bp. 

### `-cs, --chrom_sizes, -chrom_sizes, --chromosome_sizes`

The path to the chromosome sizes file. This is used to generate the bigwig signal tracks.

### `-c, -chroms, --chromosomes`

The chromosomes to make predictions on. Our models do not currently considered chromosomes X or Y. This means that most of the files will not contain this information. You should not predict in chrX or chrY unless you know your bigwig contains these chromosomes. Default: Autosomal chromosomes 1-22.

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

### `-w, --windows`

The windows to use for prediction. These windows must be 1,024 bp wide and have a consistent step size.

### `-skip_call_peaks, --skip_call_peaks`

This will skip calling peaks at the end of predictions. 

### `--threads`

Set number of parallel threads in prediction tasks. If GPUs are used, set this value to be the number of GPUs used for the task. Default: 24.   