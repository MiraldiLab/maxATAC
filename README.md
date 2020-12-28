# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq 

## Dependencies

maxATAC uses python 3.6 and can be run with or without a GPU

The following packages are required:

        tensorflow-gpu==1.14.0,
        keras==2.2.5,
        pyBigWig==0.3.16,
        py2bit==0.3.0,
        numpy==1.16.2,
        pydot==1.4.1,
        matplotlib==3.2.1,
        scikit-learn==0.22.2,
        enum34>=1.0.4,
        pandas==1.0.3,
        dask==2.30.0,
        pybedtools

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