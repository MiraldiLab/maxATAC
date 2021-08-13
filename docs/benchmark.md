# Benchmark

The `benchmark` function can be used to calculate the area under the precision recall curve (AUPRc) for a bigwig file compared to a gold standard in bigwig format. The user must provide the predictions in bigwig format and specify the resolution they want their prediction at. 

## Example

```bash
maxatac benchmark --prediction GM12878_CTCF_chr1.bw --gold_standard GM12878_CTCF_ENCODE_IDR.bw --chromosomes chr1 --bin_size 200
```

## Required Arguments

### `--prediction`

The input bigwig file of transcription factor binding predictions. 

### `--gold_standard`

The input gold standard bigwig file.

### `--prefix`

The output filename prefix to use. Default: `maxatac_benchmark`

## Optional Arguments

### `--quant`

Whether the predictions should be assessed with the Rsquared metric. Default: `False`

### `--chromosomes`

The chromosomes to benchmark the predictions for. Default: `chr`

### `--bin_size`

The size of the bin to use for aggregating the single base-pair predictions. Default: `200`

### `--agg`

The method to use for aggregating the single base-pair predictions into larger bins. 

### `--round_predictions`

This flag will set the precision of the predictions signal track. Provide an integer that represents the number of floats before rounding. Currently the predictions go from 0-.0000000001. Default: `9`

### `--output_directory`

The output directory to write the results to. Default: `./prediction_results`

### `--blacklist`

The path to the blacklist bigwig signal track of regions that should be excluded. 

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.
