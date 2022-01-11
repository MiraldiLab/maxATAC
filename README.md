![maxATAC_logo](https://user-images.githubusercontent.com/47329147/137503708-86d000ef-d6d4-4f75-99aa-39f8aab6dec5.png)

# maxATAC: transcription factor binding from ATAC-seq with deep neural networks

## Introduction

MaxATAC is a python package for predicting transcription factor (TF) binding using ATAC-seq signal and DNA-sequence in humans. You can use both bulk and pseudo-bulk scATAC-seq with maxATAC. MaxATAC takes as input a 1,024 base pair regions of DNA paired with an ATAC-seq signal to make predictions at 32 bp resolution. Our method requires three inputs:

## Requirements

This version requires python 3.9, `bedtools`, `samtools`, `pigz`, and `bedGraphToBigWig` in order to run all functions.

## Installation

It is best to install maxATAC into a dedicated virtual environment.

First, clone the repository with `git clone https://github.com/MiraldiLab/maxATAC.git` into your `repo` directory of choice.

### Installing with Conda

1. Create a conda environment for maxATAC with `conda create -n maxatac python=3.9`

2. Install `bedtools`, `samtools`, `bedGraphToBigWig`, and `pigz` or make sure it is found in your `PATH`. I prefer to use the `conda install` method for installing these packages.

3. Change into the maxATAC git repository with `cd maxATAC` and use `pip install -e .` to install maxATAC into the conda environment.

4. Test installation with `maxatac -h`

## maxATAC Quick Start Overview

![maxATAC Predict Overview](./docs/readme/maxatac_predict_overview.svg)

### Inputs

* DNA sequence: A [`.2bit`](https://genome.ucsc.edu/goldenPath/help/twoBit.html) DNA sequence file.
* ATAC-seq signal: Cell-type specific ATACseq signal.
* Trained Model: A trained maxATAC model [`.h5`](https://www.tensorflow.org/tutorials/keras/save_and_load) file.

### Outputs

* Prediction [`.bw`](https://genome.ucsc.edu/FAQ/FAQformat.html#format6.1) file: A raw prediction signal track.
* Prediction [`.bed`](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) file: A bed format file containing TF binding sites, thresholded according to a user-supplied confidence cut off (e.g., corresponding to max F1-score, estimated precision).

## Data Requirements

In our publication, we found that it was important to process the data to cut sites and min-max normalize the ATACseq signal tracks.

### Preparing your ATAC-seq signal

The current `maxatac predict` function requires a normalized ATACseq signal in a bigwig format. You can use `maxatac prepare` to generate a normalized signal track from a `.bam` file of aligned reads.

#### Converting a BAM file to bigwig file

The function `maxatac prepare` was designed to take an input BAM file quality filter the alignments and perform PCR de-duplication. The inputs to `maxatac prepare` are:

1) `-i` : the input bam file
2) `-o` : the output directory
3) `-prefix` : the filename prefix.

```bash
maxatac prepare -i SRX2717911.bam -o ./output -prefix SRX2717911
```

This function took 38 minutes for a sample with 52,657,164 reads in the BAM file. This was tested on a 2019 Macbook Pro with a 2.6 GHz 6-Core Intel Core i7 and 16 GB of memory.

## Predicting TF binding in bulk ATAC-seq

You first need to prepare the input ATAC-seq data. Then you can use the `maxatac predict` function to predict TF binding with a maxATAC model.

You can predict TF binding in a single chromosome, or you can predict across the entire genome. The user can also provide a `.bed` file of regions where predictions will be made.

You will need to make sure that you have downloaded and installed the reference `.2bit`, `chrom.sizes`, and blacklist files.

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

You will first need to convert your fragments file into pseudo-bulk specific fragment files before you can use maxATAC. You will then need to use `maxatac prepare` with each of the fragment files in order to generate a normalized bigwig file for input into `maxatac predict`. The prediction parameters and steps are the same for scATACseq data after normalization.


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