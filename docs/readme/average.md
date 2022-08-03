# Average

The `average` function will average multiple bigwig files into a single bigwig file.

## Example

Example command using only required flags:

```bash
maxatac average -i *.bw -n IMR-90
```

Example command using all flags:

```bash
maxatac average -i *.bw -n IMR-90 -o ./test -c chr1 -cs hg38.chrom.sizes
```

## Required Arguments

### `-i`

The input bigwig files. You could use a `*.bw` wildcard to make a list of bigwig files as input or provide the path to each file.

### `-n`, `--name`

The name string used to build the output filename. The extension `.bw` will be added to the filename.

## Optional Arguments

### `-cs`, `--chrom_sizes`, `--chromosome_sizes`

The chromosome sizes file for the reference genome used during alignment. The current default is set for hg38.

### `-c`, `--chroms`, `--chromosomes`

The chromosomes that are averaged together and written to output. Only the chromosomes in this list will be written to the output file. The current default list of chromosomes are restricted to the autosomal chromosomes:

```bash
chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22
```

### `-o`, `--output`

The output directory. If the output directory is not supplied the file will be created in the current working directory.

### `--loglevel`

Set the logging level. Currently, the only working logging level is `ERROR`.
