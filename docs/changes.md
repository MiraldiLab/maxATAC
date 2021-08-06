## Change Logs

***Version 0.1.0***

Stable merge of all forks

***Version 0.1.1***

- Expanding documentation and  benchmarking features
- Benchmarking outputs AUPRC plots and quantitative data
- Removed the average signal track that was not used
- Spearman, R2, precision, and recall for training
- Whole-genome, ROI, and chromosome prediction

***Version 0.1.2***

- Scripts for thresholding, "peak calling", and mean-combine signal generation.
- Code for balancing training data sets by experimental type.  
- Make predictions using a sliding window approach if user desires
- Normalization using a specific percentile

***Version 0.1.3***

- Documentation update
- Update to TF 2.5
- Update normalization and average code
- Training using reverse complement sequences and orientation
- Added a workaround for bigwig files that do not have all chromosomes
- Updated training method to use the `Keras.Sequence` and `Keras.OrderedEnqueuer`