# Benchmark

The `benchmark` function can be used to calculate the area under the precision recall curve (AUPRC) for a bigwig file compared to a gold standard in bigwig format. The user must provide the predictions in bigwig format and specify the resolution of the evaluation (e.g., 200bp).

## Example

```bash
maxatac benchmark --prediction GM12878_CTCF_chr1.bw --gold_standard GM12878_CTCF_ENCODE_IDR.bw --chromosomes chr1 --bin_size 200
```

## Required Arguments

### `--prediction`

The input bigwig file of transcription factor binding predictions. This file can also be any bigwig signal track that you want to compare against a gold standard.

### `--gold_standard`

The input gold standard bigwig file. This file needs to be a binary signal track that has 1 corresponding to TFBS (e.g., from ChIP-seq) and 0 in positions with no TFBS.

### `--prefix`

The output filename prefix to use. Default: `maxatac_benchmark`

## Optional Arguments

### `--quant`

Whether the predictions should be assessed with the Rsquared metric. Default: `False` if your data is binary.

### `--chromosomes`

The chromosomes to benchmark the predictions for. Default: `chr1` is the held out test chromosome.

### `--bin_size`

The size of the bin to use for aggregating the single base-pair predictions. Default: `200` is the size used by the [ENCODE-DREAM in vivo TFBS Prediction Challenge](https://www.synapse.org/#!Synapse:syn6131484/wiki/402026)

### `--agg`

The method to use for aggregating the single base-pair predictions into larger bins. Options include `max`, `min`, `std`, `cov`, and `mean`. Default: `max` score found in the window.

See the [pyBigWig documentation](https://github.com/deeptools/pyBigWig#compute-summary-information-on-a-range) for more details.

### `--round_predictions`

This flag will set the precision of the predictions signal track. Provide an integer that represents the number of floats before rounding. Currently, the predictions go from `0 - .0000000001`. Default: `9` is the limit of precision from TensorFlow.

### `--output_directory`

The output directory to write the results to. Default: `./prediction_results`

### `--blacklist`

The path to the blacklist bigwig signal track of regions that should be excluded. Default: `hg38_maxatac_blacklist.bed` which contains regions that are specific to ATAC-seq.

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.
