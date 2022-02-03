# Prediction Walkthrough

![maxATAC Predict Overview](../figs/maxatac_predict_overview.svg)

## Introduction

MaxATAC is a python package for predicting transcription factor (TF) binding using ATAC-seq signal and DNA-sequence in humans. MaxATAC takes as input 1,024 base pair regions of one-hot encoded DNA sequence and min-max normalized ATAC-seq signal to make predictions at 32 bp resolution. MaxATAC can make predictions on bulk ATAC-seq or pseudo-bulk scATAC-seq signal tracks. Our method requires three inputs:

### Inputs

* DNA sequence: A [`.2bit`](https://genome.ucsc.edu/goldenPath/help/twoBit.html) DNA sequence file.
* ATAC-seq signal: Cell-type specific ATAC-seq signal.
* Trained Model: A trained maxATAC model [`.h5`](https://www.tensorflow.org/tutorials/keras/save_and_load) file.

### Outputs

* Prediction [`.bw`](https://genome.ucsc.edu/FAQ/FAQformat.html#format6.1) file: A raw prediction signal track.
* Prediction [`.bed`](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) file: A bed format file containing TF binding sites, thresholded according to a user-supplied confidence cut off (e.g., corresponding to max F1-score, estimated precision).

## Data

It is important to process the ATAC-seq data to cut sites and min-max normalize the ATAC-seq signal before prediction. See the maxATAC publication [*Methods*](https://www.biorxiv.org/content/10.1101/2022.01.28.478235v1) for more details.

maxATAC was trained and tested on the human hg38 reference genome. Make sure that the correct reference data has been installed in `./maxatac/data/` including the reference `.2bit`, `chrom.sizes`, and blacklist files.

### Preparing the ATAC-seq signal

The current `maxatac predict` function requires a normalized ATAC-seq signal in a bigwig format. Use `maxatac prepare` to generate a normalized signal track from a `.bam` file of aligned reads.

#### Bulk ATAC-seq

The function `maxatac prepare` was designed to take an input BAM file that has aligned to the hg38 reference genome. The inputs to `maxatac prepare` are the input bam file, the output directory, and the filename prefix.

```bash
maxatac prepare -i SRX2717911.bam -o ./output -prefix SRX2717911 -dedup
```

This function took 38 minutes for a sample with 52,657,164 reads in the BAM file. This was tested on a 2019 Macbook Pro with a 2.6 GHz 6-Core Intel Core i7 and 16 GB of memory.

#### Pseudo-bulk scATAC-seq

First, convert the `.tsv.gz` output fragments file from CellRanger into pseudo-bulk specific fragment files. Then, use `maxatac prepare` with each of the fragment files in order to generate a normalized bigwig file for input into `maxatac predict`.

```bash
maxatac prepare -i HighLoading_GM12878.tsv -o ./output -prefix HighLoading_GM12878
```

The prediction parameters and steps are the same for scATAC-seq data after normalization.

## Predicting TF binding in bulk ATAC-seq

First, prepare the input ATAC-seq data as described above. Then, use `maxatac predict` to predict TF binding with a maxATAC model.

maxATAC can predict TF binding in a single chromosome, or across the entire genome. Alternatively, the user can also provide a `.bed` file of regions where predictions will be made.

### Whole genome prediction

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig
```

### Prediction in a specific genomic region(s)

Use the `roi` (regions of interest) argument to make predictions in specific regions.

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig --roi ROI.bed
```

### Prediction on a specific chromosome(s)

Use the `--chromosomes` argument to limit prediction to specific chromosomes.

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig --chromosomes chr3 chr5
```
