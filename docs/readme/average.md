# Average

The `average` function will average multiple bigwig files into a single output bigwig file.

## Example

```bash
maxatac average --bigwigs *.bw --prefix IMR-90 --output ./test --chroms chr1
```

## Required Arguments

### `--bigwigs`

This argument is reserved for the input bigwig files to be averaged together. You could use a `*.bw` wildcard to make a list of bigwig files as input or provide the path to two bigwig files. If you provide only 1 bigwig file you are wasting your time and resources averaging...

### `--prefix`

This argument is reserved for the prefix used to build the output filename. This can be any string. The extension `.bw` will be added to the filename prefix.

## Optional Arguments

### `--chrom_sizes`

This argument is used to define the chromosome sizes file that is used to calcuate the chromosome ends. The current default file are the chromosome sizes for hg38.

### `--chromosomes`

This argument is used to define the chromosomes that are averaged together. Only the chromosomes in this list will be written to the output file. The current default list of chromosomes are restricted to the autosomal chromosomes: 

```pre
chr1, chr2, chr3, chr4, chr5, chr6, chr7, chr8, chr9, chr10, chr11, chr12, chr13, chr14, chr15, chr16, chr17, chr18, chr19, chr20, chr21, chr22
```

### `--output`

This argument is used to define the output directory. If the output directory is not supplied a directory called `./average` will be created in the current working directory.

### `--loglevel`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.
