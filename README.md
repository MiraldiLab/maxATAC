# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

A working version of maxATAC for binary TF binding predictions. This is the base DCNN model that uses peak-centric, pan-cell, and random regions for training. This model can also take in multiple cell types and is formatted to be used with a meta file. 

This version lacks documentation of functions and code still.

## TODO

- [ ] Learn how to create tests 
- [ ] Expand prediction to multi-chromosome
- [ ] Clean up code
- [ ] Document code
- [ ] Expand benchmarking multi-chromosome
- [ ] Organize the utilities

## Dependencies

maxATAC uses python 3.6 and can be run with or without a GPU. Package requirements include:

<pre>
bedtools           bioconda/osx-64::bedtools-2.29.2-h37cfd92_0
</pre>
# Work in progress

The following python packages are required:

<pre>
"tensorflow-gpu==1.14.0",
"tensorboard==1.14.0",
"keras==2.2.5",
"pyBigWig==0.3.16",
"py2bit==0.3.0",
"numpy==1.19.4",
"matplotlib",
"scikit-learn",
"pandas==1.1.5",
"pysam==0.15.3",
"h5py<3.0.0",
"pybedtools==0.8.1",
"tqdm"
</pre>

        "tensorflow-gpu==1.14.0",
        "keras==2.2.5",
        "pyBigWig==0.3.16",
        "py2bit==0.3.0",
        "numpy==1.16.2",
        "pydot==1.4.1",
        "matplotlib==3.2.1",
        "scikit-learn==0.22.2",
        "pandas==1.0.3"

## Functions

These are the main functions:

* Average
* Normalize
* Train
* Predict
* Benchmark

### Average

The average function will average multiple bigwig files into a single output bigwig file. 

### Normalize

The normalize function will take an input bigwig file and minmax normalize the values genome wide.

### Train

The training function takes as input ATAC-seq signal, DNA sequence, Delta-ATAC-seq signal, and ChIP-seq signal to train a dilated convolutional neural network. 

### Predict

The predict function takes as input BED formatted genomic regions to predict TF binding using a trained maxATAC model.

### Benchmark

The benchmark function takes as input a prediction bigwig signal track and a ChIP-seq gold standard bigwig track to calculate precision and recall.