![maxATAC_logo](https://user-images.githubusercontent.com/47329147/137503708-86d000ef-d6d4-4f75-99aa-39f8aab6dec5.png)

# maxATAC: transcription factor binding from ATAC-seq with deep neural networks

## Introduction

maxATAC is a Python package for transcription factor (TF) binding prediction from ATAC-seq signal and DNA sequence in *human* cell types. You can use either population-level (bulk) ATAC-seq or pseudobulk derived from single-cell (sc)ATAC-seq with maxATAC. maxATAC makes TF binding site (TFBS) predictions at 32 bp resolution. Our method requires three inputs:

* DNA sequence, in [`.2bit`](https://genome.ucsc.edu/goldenPath/help/twoBit.html) file format.
* ATAC-seq signal, processed as described [below](#Preparing-your-ATAC-seq-signal).
* Trained maxATAC TF Models, in [`.h5`](https://www.tensorflow.org/tutorials/keras/save_and_load) file format.

## Requirements

This version requires Python 3.9, `bedtools`, `samtools`, `pigz`, and `bedGraphToBigWig`.

## Installation

It is best to install maxATAC into a dedicated virtual environment.

First, clone the repository with `git clone https://github.com/MiraldiLab/maxATAC.git` into your local `repo` directory of choice.

### Installing with Conda

1. Create a conda environment for maxATAC with `conda create -n maxatac python=3.9`

2. Install `bedtools`, `samtools`, `bedGraphToBigWig`, and `pigz` or make sure it is found in your `PATH`. We recommend using `conda install` to install these packages.

3. Navigate to your local maxATAC git repository, e.g., with `cd \locationOnMyComputer\maxATAC`, and use `pip install -e .` to install maxATAC.

4. Test installation with `maxatac -h`

## maxATAC Quick Start Overview

![maxATAC Predict Overview](./docs/readme/maxatac_predict_overview.svg)
Schematic: maxATAC prediction of CTCF bindings sites for processed GM12878 ATAC-seq signal

### Inputs

* DNA sequence, in [`.2bit`](https://genome.ucsc.edu/goldenPath/help/twoBit.html) file format.
* ATAC-seq signal, processed as described [below](#Preparing-your-ATAC-seq-signal).
* Trained maxATAC TF Models, in [`.h5`](https://www.tensorflow.org/tutorials/keras/save_and_load) file format.

### Outputs

* Raw maxATAC TFBS scores tracks in [`.bw`](https://genome.ucsc.edu/FAQ/FAQformat.html#format6.1) file format.
* [`.bed`](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) file of TF binding sites, thresholded according to a user-supplied confidence cut off (e.g., corresponding to an estimated precision, recall value or max F1-score) or default (log_2(precision:precision_{random} > 7).

## ATAC-seq Data Requirements

As described in Cazares et al., **maxATAC processing of ATAC-seq signal is critical to maxATAC prediction**. Key maxATAC processing steps, summarized in a single command [`maxatac prepare`](./docs/readme/prepare.md#Prepare), include identification of Tn5 cut sites from ATAC-seq fragments, ATAC-seq signal smoothing, filtering with an extended blacklist, and robust, min-max-like normalization. 

The maxATAC models were trained on paired-end ATAC-seq data in human. For this reason, we recommend  paired-end sequencing with sufficient sequencing depth (e.g., ~20M reads for bulk ATAC-seq). Until these models are benchmarked in other species, we recommend limiting their use to human ATAC-seq datasets.

### Preparing your ATAC-seq signal

maxATAC prediction requires maxATAC-normalized ATAC-seq signal in a bigwig format. You can use [`maxatac prepare`](./docs/readme/prepare.md#Prepare) to generate a maxATAC-normalized signal track from a `.bam` file of aligned reads.

#### Converting a BAM file to bigwig file

[`maxatac prepare`](./docs/readme/prepare.md#Prepare) processes aligned ATAC-seq reads (`.bam` for bulk ATAC-seq or `.tsv` or `tsv.gz` for scATAC-seq) into smoothed, normalized Tn5 cut sites. Below is an example using `maxatac prepare` for bulk ATAC-seq. Inputs are:

1) `-i` : the input bam file
2) `-o` : the output directory
3) `-prefix` : the filename prefix.

```bash
maxatac prepare -i SRX2717911.bam -o ./output -prefix SRX2717911
```

This function took 38 minutes for a sample with 52,657,164 reads in the BAM file. This was tested on a 2019 Macbook Pro with a 2.6 GHz 6-Core Intel Core i7 and 16 GB of memory.

## Predicting TF binding in bulk ATAC-seq

Following maxATAC-specific processing of ATAC-seq signal inputs, use the [`maxatac predict`](./docs/readme/predict.md#Predict) function to predict TF binding with a maxATAC model.

You can predict TF binding in a single chromosome, or you can predict across the entire genome. Alternatively, the user can provide a `.bed` file of genomic intervals for maxATAC predictions to be made.

The reference `.2bit`, `chrom.sizes`, and blacklist files should be downloaded and installed.

### Whole genome prediction

You can predict TF binding across the whole genome with the following command:

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig
```

### Prediction in a specific genomic region(s)

You can also predict TF binding across a specific region of the genome if you provide a BED file to the `roi` (regions of interest) argument.

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig --roi ROI.bed
```

### Prediction on a specific chromosome(s)

You can make a prediction on a single chromosome or a subset of chromosomes by providing the chromosome names to the `--chromosomes` argument. 

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig --chromosomes chr3 chr5
```

## Predicting TF binding in scATAC data

maxATAC prediction on scATAC-seq occurs at the level of pseudobulk ATAC-seq signal inputs, processed using [`maxatac prepare`](./docs/readme/prepare.md#Prepare). Thus, prediction commands with pseudobulk scATAC-seq are identical to prediction the prediction commands with bulk ATAC-seq using [`maxatac predict`](./docs/readme/predict.md#Predict) described above.


## maxATAC Exended Documentation

### [Code to generate most of our figures](https://github.com/MiraldiLab/maxATAC_docs/tree/main/figure_code)

### [Code and snakemake workflow for ChIP-seq data processing/curation](https://github.com/MiraldiLab/maxATAC_dataScraping)

## maxATAC functions
| Subcommand                                          | Description                                    |
|-----------------------------------------------------|------------------------------------------------|
| [`prepare`](./docs/readme/prepare.md#Prepare)       | Prepare input data                             |
| [`average`](./docs/readme/average.md#Average)       | Average ATAC-seq signal tracks                 |
| [`normalize`](./docs/readme/normalize.md#Normalize) | Minmax normalize ATAC-seq signal tracks        |
| [`train`](./docs/readme/train.md#Train)             | Train a model                                  |
| [`predict`](./docs/readme/predict.md#Predict)       | Predict TF binding                             |
| [`benchmark`](./docs/readme/benchmark.md#Benchmark) | Benchmark maxATAC predictions against ChIP-seq |
| [`peaks`](./docs/readme/peaks.md#Peaks)             | Call "peaks" on maxATAC signal tracks          |
| [`variants`](./docs/readme/variants.md#Variants)    | Predict sequence specific TF binding           |
