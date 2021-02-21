# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

## TODO

- [ ] Learn how to create tests
- [ ] Expand prediction to multi-chromosome
- [ ] Clean up code
- [ ] Document code
- [ ] Expand benchmarking multi-chromosome
- [ ] Organize the utilities

## Functions

These are the main functions:

* average
* normalize
* train
* predict
* benchmark
* roi

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