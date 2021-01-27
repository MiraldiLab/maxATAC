# Calculating the AUPR curve for chromosome-wide predictions against a ChIP-seq gold standard

## Rational

Many methods have been developed to predict TF binding along the genome, but how do you test if any of them are good? How can you be sure that when you apply your predictions to a new setting that you will be able to make reliable predictions? Here I discuss how we evaluate our TF binding predictions against known experimentally derived binding sites to gauge our model performance. 

## Types of benchmarking

The type of benchmarking that you need to perform depends on the type of data that you are comparing. There are multiple considerations like whether the predictions have been made genome-wide, chromosome-wide, or in specific regions of interest like peaks. You must also consider the type of predictions that are being benchmarked. Are they binary predictions where the target binding site is denoted as a 0 or 1. Or quantitative predictions that are meant to predict read counts or other continuous data? The different methods are explained below in each section.

### Whole chromosome prediction

Predicting TF binding chromosome-wide is the approach many of the DREAM challenge competitors used to benchmark their methods. The rational is that a whole chromosome would contain a mix of bound/unbound regions that would closely resemble the task of predicting across a genome and looking for TF binding locations.

For the whole chromosome approach, a chromosome of interest is split into windows that are the size of the inputs for the model (i.e. 1024 bp). 

This can be accomplished using bedtools to window the genome into non-overlapping intervals based on the chromosome sizes.

```bedtools makewindows -w 1024 -s 1024 -g hg38.chrom.sizes > hg38_w1024_s1024.bed```

The windowed chromosome regions are then input into your model for prediction. The output to our maxATAC model is a bigwig file that corresponds to the input genomic intervals along the chromosome. If you are using another method like peak based prediction, then you will need to convert your predictions to bigwig format to use our AUPR method.

#### Binary Predictions

A binary benchmarking approach uses metrics like precision, recall, and AUPR to benchmark a model against a binary gold standard. In the context of genomics, a binary gold standard would have a 1 at each base position for which there is an experimental ChIP-seq peak found. The predictions output from our method and other methods usually do not provide the score in a binary format, instead it is usually in the form of float between [0,1]. If we were to round these values to 1 using some cutoff we would end up with a single precision recall point. 

### Peak based predictions

The output to our model is a bigwig file containing chromosome wide predictions of TF binding. 
