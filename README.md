# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

The maxATAC package is a collection of tools used for learning to predict TF binding from ATAC-seq data. MaxATAC also provides functions for interpreting trained models and preparing the input data.
## TODO

- [ ] Learn how to create tests
- [ ] Expand prediction to multi-chromosome
- [ ] Clean up code
- [ ] Document code
- [ ] Expand benchmarking multi-chromosome
- [ ] Organize the utilities
- [ ] Test all architectures

## Predicting TF Binding Workflow Overview

Steps in training and assessing a maxATAC model. Relevant functions are listed below each step. 

1. Prepare Input Data
   * `average`
    * `normalize`
    
2. Train a model
   * `roi`
    * `train`
    
3. Predict in new cell type
   * `predict`
    
4. Benchmark models against experimental data
    * `benchmark`
    
5. Learn features import to predict TF binding with a neural network
    * `interpret`

## Functions

The maxATAC bundle has several useful functions needed for building a deep learning model of TF binding.

**Data Pre-processing:**

* average
* normalize
  
**TF Binding Prediction Functions:**

* train
* predict
* benchmark
* roi
* interpret

### Average

The `average` function will average multiple bigwig files into a single output bigwig file.

This function can take a list of input bigwig files and average their scores using pyBigWig. The only requirement for the bigwig files is that they contain the same chromosomes or there might be an error about retrieving scores.

**Workflow Overview**

1) Create directories and set up filenames
2) Build a dictionary of chromosome sizes and filter it based on desired chromosomes to average
3) Open the bigwig file for writing
4) Loop through each entry in the chromosome sizes dictionary and calculate the average across all inputs
5) Write the bigwig file

### Normalize

The `normalize` function will take an input bigwig file and minmax normalize the values genome wide.

This function will min-max a bigwig file based on the minimum and maximum values in the chromosomes of interest. The code will loop through each chromosome and find the min and max values. It will then create a dataframe of the values per chromosome. It will then scale all other values between [0,1].

**Workflow Overview**

1) Create directories and set up filenames
2) Build a dictionary of the chromosomes sizes.
3) Find the genomic min and max values by looping through each chromosome
4) Loop through each chromosome and minmax normalize the values based on the genomic values.

### Train

The `train` function takes as input ATAC-seq signal, DNA sequence, Delta ATAC-seq signal, and ChIP-seq signal to train a neural network with the architecture of choice.

The primary input to the training function is a meta file that contains all of the information for the locations of ATAC-seq signal, ChIP-seq signal, TF, and Cell type.

Example header for meta file. The meta file must be a tsv file, but the order of the columns does not matter. As long as the column names are the same:

`TF | Cell_Type | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks`

**Workflow Overview**

1) Create directories and set up filenames
2) Initialize the model based on the desired architectures
3) Read in the meta table
4) Read in training and validation pool
5) Initialize the training generator
6) Initialize the validation generator
7) Fit the models with the specific parameters

### Predict

The `predict` function takes as input BED formatted genomic regions to predict TF binding using a trained maxATAC model.

BED file requirements for prediction. You must have at least a 3 column file with chromosome, start, and stop coordinates. The interval distance has to be the same as the distance used to train the model. If you trained a model with a resolution 1024, you need to make sure your intervals are spaced 1024 bp apart for prediction with your model.

Example input BED file for prediction:

`chr1   1000    2024`

**Workflow Overview**

1) Create directories and set up filenames
2) Make predictions
3) Convert predictions to bigwig format and write results

### Benchmark

The `benchmark` function takes as input a prediction bigwig signal track and a ChIP-seq gold standard bigwig track to calculate precision and recall.

The inputs need to be in bigwig format to use this function. You can also provide a custom blacklist to filter out regions that you do not want to include in your comparison. We use a np.mask to exclude these regions.

Currently, benchmarking is set up for one chromosome at a time. The most time-consuming step is importing and binning the input bigwig files to resolutions smaller than 100bp. We are also only benchmarking on whole chromosomes at the moment so everything not in the blacklist will be considered a potential region.

**Workflow Overview**

1) Create directories and set up filenames
2) Get the blacklist mask using the input blacklist and bin it at the same resolution as the predictions and GS
3) Calculate the AUPR

### ROI

The `roi` function will generate regions of interest based on input ChIP-seq, ATAC-seq, or randomly generated regions. 

This method will use the run meta file to merge the ATAC-seq peaks and ChIP-seq peaks into BED files where each entry is an example region. ONE MAJOR ASSUMPTION IS THAT THE TEST CELL LINE IS NOT INCLUDED IN THE META FILE!!!!!

The input meta file must have the columns in any order:

`TF | Cell_Type | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks`

**Workflow Overview**

1) Import the ATAC-seq and ChIP-seq and filter for training chromosomes
2) Write the ChIP, ATAC, and combined ROI pools with stats
3) Import the ATAC-seq and ChIP-seq and filter for validation chromosomes
4) Write the ChIP, ATAC, and combined ROI pools with stats

### Interpret

The `interpret` function will interpret a trained maxATAC model using TFmodisco, DeepLift, and DeepShap. 