# Average

The `average` function will average multiple bigwig files into a single bigwig file. Typical uses:

* Averaging replicate ATAC-seq signal tracks before prediction.
* Building the **null-model track** for the quantitative benchmark (`maxatac benchmark --quant_gs_null`): the average of the quantitative ChIP-seq gold standard tracks across all cell types for a TF.
* Averaging quantitative prediction tracks (e.g. across models or replicates).

## Example

Example command using only required flags:

```bash
maxatac average -i *.bw -n IMR-90
```

Example command using all flags:

```bash
maxatac average -i *.bw -n IMR-90 -o ./test -c chr1 -cs hg38.chrom.sizes
```

Building a per-TF null model from quantitative ChIP-seq tracks, rounded to 2 decimals:

```bash
maxatac average --quant --decimal_points 2 -i CTCF_*_signal.bw -n CTCF_signal_all_celltypes_avg
```

## Required Arguments

### `-i`

The input bigwig files. You could use a `*.bw` wildcard to make a list of bigwig files as input or provide the path to each file.

### `-n`, `--name`, `--prefix`

The name string used to build the output filename. The extension `.bw` will be added to the filename.

## Optional Arguments

### `-q`, `--quant`

Round the averaged values to `--decimal_points` decimals. Intended for quantitative prediction or ChIP-seq signal tracks, where full float precision inflates the output bigwig without adding information. Default: `False` (no rounding)

### `--decimal_points`

The number of decimals to round to when `--quant` is set. Default: `2`

### `-cs`, `--chrom_sizes`, `--chromosome_sizes`

The chromosome sizes file for the reference genome used during alignment. The current default is set for hg38.

### `-c`, `--chroms`, `--chromosomes`

The chromosomes that are averaged together and written to output. Only the chromosomes in this list will be written to the output file. The current default list of chromosomes are restricted to the autosomal chromosomes:

```bash
chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22
```

### `--max_zooms`

The number of zoom levels to compute for the averaged bigWig file. Zoom levels are pre-computed summary statistics that let genome browsers (IGV, UCSC Genome Browser) zoom in and out of a region quickly; fewer levels mean slower loading in browsers, more levels mean more memory during writing. Valid range: `0`-`10`. Default: `10`. Note: a value of `0` produces a bigWig that is not compatible with other bigWig tools (e.g. deepTools) and cannot be visualized in IGV or the UCSC Genome Browser. See the [pyBigWig README](https://github.com/deeptools/pyBigWig/blob/master/README.md) for details.

### `-o`, `--output`, `--output_dir`

The output directory. If the output directory is not supplied the file will be created in the current working directory.

### `--loglevel`

Logging level (`fatal`, `error`, `warning`, `info`, `debug`). Default: `info`.
