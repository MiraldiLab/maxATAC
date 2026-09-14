# Benchmark

The `benchmark` function can be used to calculate the area under the precision recall curve (AUPRC) for a bigwig file compared to a gold standard in bigwig format. The user must provide the predictions in bigwig format and specify the resolution of the evaluation (e.g., 200bp).

## Example

```bash
maxatac benchmark --prediction GM12878_CTCF_chr1.bw --gold_standard GM12878_CTCF_ENCODE_IDR.bw --chromosomes chr1 --bin_size 200
maxatac benchmark --prediction primary.bw --alternative_prediction secondary.bw --prediction_combine_operation max --gold_standard GM12878_CTCF_ENCODE_IDR.bw --chromosomes chr1 --bin_size 200
```

## Required Arguments

### `--prediction`

The input bigwig file of transcription factor binding predictions. This file can also be any bigwig signal track that you want to compare against a gold standard.

### `--gold_standard`

The input gold standard bigwig file. This file needs to be a binary signal track that has 1 corresponding to TFBS (e.g., from ChIP-seq) and 0 in positions with no TFBS.

### `--prefix`

The output filename prefix to use. Default: `maxatac_benchmark`

## Optional Arguments

### `--chromosomes`

The chromosomes to benchmark the predictions for. Default: `chr1` is the held out test chromosome.

### `--bin_size`

The size of the bin to use for aggregating the single base-pair predictions. Default: `200` is the size used by the [ENCODE-DREAM in vivo TFBS Prediction Challenge](https://www.synapse.org/#!Synapse:syn6131484/wiki/402026)

### `--agg`

The method to use for aggregating the single base-pair predictions into larger bins. Options include `max`, `min`, `mean`, and `sum`. Default: `max` score found in the window.

See the [pyBigWig documentation](https://github.com/deeptools/pyBigWig#compute-summary-information-on-a-range) for more details.

### `--agg_threshold`

When `--agg sum` is selected, the summed value is divided by the bin size and then converted to a binary label. Bins with values greater than or equal to this threshold become `1.0`; all others become `0.0`. Default: `0.5`.

### `--round_predictions`

This flag will set the precision of the predictions signal track. Provide an integer that represents the number of floats before rounding. Currently, the predictions go from `0 - .0000000001`. Default: `9` is the limit of precision from TensorFlow.

### `--output_directory`

The output directory to write the results to. Default: `./prediction_results`

### `--blacklist_bw`

The path to the blacklist bigwig signal track of regions that should be excluded. Default: `hg38_maxatac_blacklist.bed` which contains regions that are specific to ATAC-seq.

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

### `--whitelist_bw`

The path to a whitelist bigwig signal track of regions that should be included. When provided, benchmarking is restricted to bins that overlap the whitelist track (in addition to blacklist exclusion).


### Optional alternate prediction bigWig

You can provide `--alternative_prediction` to benchmark a second prediction bigWig together with the primary `--prediction` input. Before downstream metrics are computed, the benchmark command extracts the binned values from both files and combines them per bin. Use `--prediction_combine_operation mean` to average bins when both files provide a value, or `--prediction_combine_operation max` to keep the larger value. If only one file provides a value for a bin, that available value is used.
