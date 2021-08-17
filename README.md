# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

## Introduction

Cellular behavior is the result of complex genomic regulation partially controlled by the activity of DNA binding proteins called transcription factors (TFs). TFs bind DNA in a sequence specific manner to regulate gene transcription. TFs, other DNA binding proteins, nucleosomes, and structural proteins are all involved in the regulation of gene expression. The physical interaction of protein with DNA molecules results in changes in the accessibility of the underlying DNA sequence. The assay for transposase accessible chromatin (ATAC-seq) uses a hyperactive Tn5 transposase to probe for genomic regions that are accessible to cleavage, and in turn, accessible to TF binding. It has been shown that distinct patterns in genomic Tn5 cleavage signal can be used to identify TF binding positions that are partially protected from Tn5 cleavage, known as TF footprints. ATAC-seq can also be used to identify regions of the genome that are generally accessible. Here we present a method to predict TF binding by learning from ATAC-seq accessibility signal and the underlying DNA sequence of TF binding locations identified by ChIP-seq. 


The maxATAC package is a collection of tools used for learning to predict TF binding from ATAC-seq and ChIP-seq data. MaxATAC also provides functions for interpreting trained models and preparing the input data.

## Requirements

This version requires python 3.6 and BEDTools.

## Installation

It is best to install maxATAC into a dedicated virtual environment. Clone the repository with `git clone` into your `repo` directory of choice. 

Change into the maxATAC repository with `cd maxATAC` and use `pip3 install -e .` to install the package.

You will also need to have BEDtools installed or loaded on your PATH.

*Note: sometimes SHAP will produce an error when installing with `pip3 install -e .` due to conflicts with numpy. In this case, you will need to install numpy into your virtual env BEFORE installing maxATAC*

## maxATAC Workflow Overview

Steps in training and assessing a maxATAC model. Relevant functions are listed below each step.

### 1. Prepare Input Data
   * [`average`](./docs/average.md#Average)
   * [`normalize`](./docs/normalize.md#Normalize)
   
### 2. Train a model
   * [`train`](./docs/train.md#Train)
    
### 3. Predict in new cell type
   * [`predict`](./docs/predict.md#Predict)
   
### 4. Benchmark models against experimental data
   * [`benchmark`](./docs/benchmark.md#Benchmark)
    
### 5. Learn features import to predict TF binding with a neural network
   * `interpret`
