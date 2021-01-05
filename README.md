# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq 

## Dependencies

maxATAC uses python 3.6 and can be run with or without a GPU. Package requirements include:

<pre>
  bedtools           bioconda/osx-64::bedtools-2.29.2-h37cfd92_0
  bzip2              conda-forge/osx-64::bzip2-1.0.8-hc929b4f_4
  curl               conda-forge/osx-64::curl-7.71.1-hcb81553_8
  libdeflate         bioconda/osx-64::libdeflate-1.0-h1de35cc_1
  python-dateutil    conda-forge/noarch::python-dateutil-2.8.1-py_0
  pytz               conda-forge/noarch::pytz-2020.5-pyhd8ed1ab_0 
</pre>

The following python packages are required:
<pre>
    "tensorflow==1.14.0",
    "keras==2.2.5",
    "pyBigWig==0.3.16",
    "py2bit==0.3.0",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "pandas==1.1.5",
    "pysam==0.15.3",
    "pybedtools==0.8.1"
</pre>

## Functions

There are four main functions:

1. Normalization
2. Training
3. Prediction
4. Benchmarking

### Normalization

The normalization function will take an input bigwig file and minmax normalize the values genome wide.

### Training

The training function takes as input ATAC-seq signal, DNA sequence, .....

### Prediction

### Benchmarking