# Training

To train a maxATAC model you need to set up a lot of inputs and a meta table to organize those input files.

___

## Requirements

### Meta Table

The large number of examples, targets, inputs, and peaks are tracked using a meta file. The meta file should be in the following format: 

| Cell_Type | TF   | Output type                        | Experiment date released | File accession | priority | CHIP_Peaks                                                                               | ATAC_Peaks                                                                                   | ATAC_Signal_File                                                                                      | Binding_File                                                                            | Peak_Counts | tuple      | Train_Test_Label |
|-----------|------|------------------------------------|--------------------------|----------------|----------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-------------|------------|------------------|
| A549      | CTCF | conservative IDR thresholded peaks | 2012-08-20               | ENCFF277ZAR    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/A549__CTCF.bed    | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/A549_ATAC.bed    | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/A549_RPM_minmax01.bw    | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/A549__CTCF.bw    | 36415       | CTCF_36415 | Train            |
| GM12878   | CTCF | conservative IDR thresholded peaks | 2011-02-10               | ENCFF017XLW    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/GM12878__CTCF.bed | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/GM12878_ATAC.bed | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/GM12878_RPM_minmax01.bw | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/GM12878__CTCF.bw | 39892       | CTCF_39892 | Train            |
| HCT116    | CTCF | conservative IDR thresholded peaks | 2012-01-17               | ENCFF832GBA    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HCT116__CTCF.bed  | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/HCT116_ATAC.bed  | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/HCT116_RPM_minmax01.bw  | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HCT116__CTCF.bw  | 49964       | CTCF_49964 | Train            |
| HepG2     | CTCF | conservative IDR thresholded peaks | 2011-03-17               | ENCFF704ECS    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HepG2__CTCF.bed   | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/HepG2_ATAC.bed   | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/HepG2_RPM_minmax01.bw   | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HepG2__CTCF.bw   | 44930       | CTCF_44930 | Train            |



You will need to have ATAC-seq and ChIP-seq data in a bigwig format. You will also need peak file for both ATAC-seq and ChIP-seq. If no ATAC-seq or ChIP-seq files are used then you will get an error when building the ROI based training regions. 

___

