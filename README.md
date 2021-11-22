![maxATAC_logo](https://user-images.githubusercontent.com/47329147/137503708-86d000ef-d6d4-4f75-99aa-39f8aab6dec5.png)

# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

## Introduction

Cellular behavior is the result of complex genomic regulation partially controlled by the activity of DNA binding proteins called transcription factors (TFs). TFs bind DNA in a sequence specific manner to regulate gene transcription. TFs, other DNA binding proteins, nucleosomes, and structural proteins are all involved in the regulation of gene expression. The physical interaction of protein with DNA molecules results in changes in the accessibility of the underlying DNA sequence. The assay for transposase accessible chromatin (ATAC-seq) uses a hyperactive Tn5 transposase to probe for genomic regions that are accessible to cleavage, and in turn, accessible to TF binding. It has been shown that distinct patterns in genomic Tn5 cleavage signal can be used to identify TF binding positions that are partially protected from Tn5 cleavage, known as TF footprints. ATAC-seq can also be used to identify regions of the genome that are generally accessible. Here we present a method to predict TF binding by learning from ATAC-seq accessibility signal and the underlying DNA sequence of TF binding locations identified by ChIP-seq. 


The maxATAC package is a collection of tools used for learning to predict TF binding from ATAC-seq and ChIP-seq data. MaxATAC also provides functions for interpreting trained models and preparing the input data.

## Requirements

This version requires python 3.9 and BEDTools.

## Installation

It is best to install maxATAC into a dedicated virtual environment. 

First, clone the repository with `git clone https://github.com/MiraldiLab/maxATAC.git` into your `repo` directory of choice.

### Installing with Conda

1. Create a conda environment for maxATAC with `conda create -n maxatac python=3.9`

2. Activate the conda environment for maxATAC with `conda activate maxatac` or `source activate maxatac` if you have an error using a HPC.

3. Install bedtools or make sure it is found in your `PATH` with `conda install bedtools`

4. Install pysam with `conda install pysam`

5. Change into the maxATAC git repository with `cd maxATAC` and use `pip install -e .` to install maxATAC into the conda environment.

6. Test installation with `maxatac -h`

## maxATAC Workflow Overview

Steps in training and assessing a maxATAC model. Relevant functions are listed below each step.

### 1. Prepare Input Data
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