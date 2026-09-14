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

The user must provide either the TF name that they want to make predictions for OR the h5 model file they desire. If the user provides a TF name, the best model will be used and the correct threshold file will be provided for peak calling.

### `-s, --signal, -i`

The ATAC-seq signal bigWig track that will be used to make TF binding predictions.

### `-n, --name, --prefix`

Output filename prefix (without extension) to use. Default `maxatac_predict`.

## Optional Arguments

### `--batch_size`

The number of regions to predict on per batch. Default `10000`. Decrease this value if you are having memory issues.

### `--bed, --peaks, --regions, , --roi, -roi`

The path to a BED file containing genomic regions to focus TF predictions on. These peaks will be used to refine the prediction windows. Default: whole-chromosome predictions.

### `-bl, --blacklist`

The path to a bigWig file that has regions to exclude. Default: maxATAC-defined blacklist.

### `-c, -chroms, --chromosomes`

The chromosomes to make predictions on. Our models do not currently consider chromosomes X or Y. This means that most of the files will not contain this information. You should not predict in chrX or chrY unless you know your bigWig contains these chromosomes. Default: human autosomal chromosomes 1-22. Note: this argument MUST be specified in conjunction with `--genome`, `-sequence`, and `--chrom_sizes` if the input file was aligned to a genome build other than hg38.

### `-cs, --chrom_sizes, -chrom_sizes, --chromosome_sizes`

The path to the chromosome sizes file. This is used to generate the bigwig signal tracks. Note: this argument MUST be specified in conjunction with `--genome`, `-sequence`, and `--chromosomes` if the input file was aligned to a genome build other than hg38.

### `"-cf", -cutoff_file, --cutoff_file`

The cutoff file provided in /data/models that corresponds to the average validation performance metrics for the TF model.

### `"-ct", "-cutoff_type", "--cutoff_type"`

The cutoff type (i.e. `Precision`, `Recall`, `F1`, `log2FC`). (F1 = F1-score, and log2FC = Log2( Precision : Random Precision)). Default: F1.

### `"-cv", "-cutoff_value", "--cutoff_value"`

The cutoff value for the cutoff type provided. Note: precision, recall, and F1-scores range from 0-1, while better-than-random log2FC scores range from 0 to infinity. Example: 0.7.

### `--genome`

The genome build that was used for alignment of the ATAC-seq signal file. Default: hg38.

### `--max_zooms`

The number of zoom levels that should be computed for the output bigWig file. Zoom levels are pre-computed summary statistics that enable fast zooming into/out of a genomic region in a bigWig file when a visualization tool (e.g., IGV, UCSC Genome Browser). Lower values of this parameter result in slower loading of bigWig files in visualization tools, while higher values of this parameter result in a large memory overhead. The range of potential parameter values is (0-10). Default: 5. Note: if this argument is set to 0, the resulting bigWig files are NOT compatible with other bigWig tools (e.g., deepTools) and cannot be visualized using tools like IGV and the UCSC Genome Browser. Please see: https://github.com/deeptools/pyBigWig/blob/master/README.md for additional details.

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

### `-o, --output`

Output directory path. Default: `./prediction_results`

### `--sequence, --seq`

This argument specifies the path to the 2bit DNA sequence for the genome of interest. maxATAC models are trained with hg38, so you will need the correct `.2bit` file. Note: this argument MUST be specified (with a valid 2bit file) in conjunction with `--genome`, `--chrom_sizes`, and `--chromosomes` if the input file was aligned to a genome build other than hg38.

### `-skip_call_peaks, --skip_call_peaks`

This will skip calling peaks on prediction tracks. Default: `False`

### `--step_size`

The step size to use for building the prediction intervals. Overlapping prediction bins will be averaged together. Default: `INPUT_LENGTH/4`, where INPUT_LENGTH is the maxATAC model input size of 1,024 bp. 

### `-w, --windows`

The windows to use for prediction. These windows must be 1,024 bp wide and have a consistent step size.
