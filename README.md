![maxATAC_logo](https://user-images.githubusercontent.com/47329147/137503708-86d000ef-d6d4-4f75-99aa-39f8aab6dec5.png)

# maxATAC: genome-scale transcription-factor binding prediction from ATAC-seq with deep neural networks

## Introduction

Cellular behavior is the result of complex genomic regulation partially controlled by the activity of DNA binding proteins called transcription factors (TFs). TFs bind DNA in a sequence specific manner to regulate gene transcription. TFs, other DNA binding proteins, nucleosomes, and structural proteins are all involved in the regulation of gene expression. The physical interaction of protein with DNA molecules results in changes in the accessibility of the underlying DNA sequence. The assay for transposase accessible chromatin (ATAC-seq) uses a hyperactive Tn5 transposase to probe for genomic regions that are accessible to cleavage, and in turn, accessible to TF binding. It has been shown that distinct patterns in genomic Tn5 cleavage signal can be used to identify TF binding positions that are partially protected from Tn5 cleavage, known as TF footprints. ATAC-seq can also be used to identify regions of the genome that are generally accessible. Here we present a method to predict TF binding by learning from ATAC-seq accessibility signal and the underlying DNA sequence of TF binding locations identified by ChIP-seq.

The maxATAC package is a collection of tools used for learning to predict TF binding from ATAC-seq and ChIP-seq data. MaxATAC also provides functions for interpreting trained models and preparing the input data. maxATAC is trained and tested in the human hg38 reference genome. maxATAC requires that all experimental data is aligned to the hg38 reference.

___

## Installation

It is best to install maxATAC into a dedicated virtual environment.

### Requirements

This version requires python 3.9, `bedtools`, `samtools`, `pigz`, and `bedGraphToBigWig` in order to run all functions.

### Installing with Conda

1. Create a conda environment for maxATAC with `conda create -n maxatac python=3.9 maxatac samtools bedtools bedGraphToBigWig pigz`

2. Test installation with `maxatac -h`

### Installing with pip

1. Create a virtual environment for maxATAC (conda is shown in the example) with `conda create -n maxatac python=3.9`.

2. Install required packages and make sure they are on your PATH: samtools, bedtools, bedGraphToBigWig, pigz.

3. Install maxatac with `pip install maxatac`

4. Test installation with `maxatac -h`

___

## maxATAC Tutorials and Walkthroughs

[Introduction and Walkthrough](./docs/readme/prediction_walkthrough.md)

___

## maxATAC Functions

### 1. Prepare Input Data

* [`prepare`](./docs/readme/prepare.md#Prepare)
* [`average`](./docs/readme/average.md#Average)
* [`normalize`](./docs/readme/normalize.md#Normalize)

### 2. Train a model

* [`train`](./docs/readme/train.md#Train)
 
### 3. Predict in new cell type

* [`predict`](./docs/readme/predict.md#Predict)

### 4. Benchmark models against experimental data

* [`benchmark`](./docs/readme/benchmark.md#Benchmark)

### 5. Call "peaks" on maxATAC signal tracks

* [`peaks`](./docs/readme/peaks.md#Peaks)

### 6. Predict sequence specific TF binding

* [`variants`](./docs/readme/variants.md#Variants)

___

## Publication

The maxATAC pre-print is currently available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.01.28.478235v1.article-metrics). 

```pre
maxATAC: genome-scale transcription-factor binding prediction from ATAC-seq with deep neural networks
Tareian Cazares, Faiz W. Rizvi, Balaji Iyer, Xiaoting Chen, Michael Kotliar, Joseph A. Wayman, Anthony Bejjani, Omer Donmez, Benjamin Wronowski, Sreeja Parameswaran, Leah C. Kottyan, Artem Barski, Matthew T. Weirauch, VB Surya Prasath, Emily R. Miraldi
bioRxiv 2022.01.28.478235; doi: https://doi.org/10.1101/2022.01.28.478235
```

### [Code to generate most of our figures](https://github.com/MiraldiLab/maxATAC_docs/tree/main/figure_code)
