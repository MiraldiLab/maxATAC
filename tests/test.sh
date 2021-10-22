#!/bin/bash

module load bedtools/2.30.0
module load anaconda3
source activate maxATAC_IS59 


mkdir -p /scratch/war9qi/data/aronow/Balaji_Iyer/Projects/Maxatac_IS59/Maxatac_Predictions/20211015_DCNN_ATAC_DELTA/ARNT1
cd /scratch/war9qi/data/aronow/Balaji_Iyer/Projects/Maxatac_IS59/Maxatac_Predictions/20211015_DCNN_ATAC_DELTA/ARNT1

maxatac train \
--rand_ratio .30 \
--arch DCNN_V2 \
--atac_delta \
--loss_name clipped_cross_entropy \
--output_activation 'softmax' \
--sequence /data/miraldiLab/databank/maxatac/maxATAC_inputs/genome_inf/hg38.2bit \
--meta_file /scratch/war9qi/ARNT_revcomp_99percentile_Binary_Train.tsv \
--output /scratch/war9qi/data/aronow/Balaji_Iyer/Projects/Maxatac_IS59/Maxatac_Predictions/20211015_DCNN_ATAC_DELTA/ARNT1/training_results \
--prefix ARNT_binary_archs \
--batch_size 1000 \
--val_batch_size 1000 \
--batches 100 \
--epochs 30 \
--threads 8 \
--shuffle_cell_type


declare -a best_model_path=$(cat /scratch/war9qi/data/aronow/Balaji_Iyer/Projects/Maxatac_IS59/Maxatac_Predictions/20211015_DCNN_ATAC_DELTA/ARNT1/training_results/*txt)
for best_model in ${best_model_path[@]}; do
    maxatac predict \
    --models ${best_model} \
    --sequence /data/miraldiLab/databank/maxatac/maxATAC_inputs/genome_inf/hg38.2bit \
    --signal /data/miraldiLab/databank/maxatac/maxATAC_inputs/ATAC/hg38/normalization/GM12878__minmax_percentile99.bw  \
    --chromosomes chr1 \
    --prefix ARNT_binary_revcomp99_RR30 \
    --output /scratch/war9qi/data/aronow/Balaji_Iyer/Projects/Maxatac_IS59/Maxatac_Predictions/20211015_DCNN_ATAC_DELTA/ARNT1/prediction_results \
    --step_size 256
done

declare -a bp_RES=(200)

for item in ${bp_RES[@]}; do
    maxatac benchmark \
    --prediction /scratch/war9qi/data/aronow/Balaji_Iyer/Projects/Maxatac_IS59/Maxatac_Predictions/20211015_DCNN_ATAC_DELTA/ARNT1/prediction_results/ARNT_binary_revcomp99_RR30.bw \
    --gold_standard /data/miraldiLab/databank/maxatac/maxatac_goldstandards/hg38/binary/GM12878__ARNT.bw  \
    --prefix ARNT_binary_revcomp99_RR30_benchmark \
    --bin_size ${item} \
    --agg max \
    --chromosomes chr1 \
    --output /scratch/war9qi/data/aronow/Balaji_Iyer/Projects/Maxatac_IS59/Maxatac_Predictions/20211015_DCNN_ATAC_DELTA/ARNT1/benchmarking_results
done
