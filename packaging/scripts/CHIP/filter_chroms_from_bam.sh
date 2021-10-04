#!/bin/bash

################### filter_chroms_from_bam.sh ###################
# This workflow will filter a BAM file based on the chroms

# INPUT:

# 1: Input BAM file 
# example: GM12878.bam

# 2: Number of cores to use
# example: 6

# OUTPUT:

# Filtered, sorted, and indexed BAM file
###########################################################

### Rename Input Variables ###
bam=${1}
cores=${2}

final_bam=`basename ${bam} .bam`_chromFiltered.bam

samtools index -@ ${cores} ${bam}

samtools view -@ ${cores} -b ${bam} chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr1 chr20 chr21 chr22 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chrX chrY | \
samtools sort -@ ${cores} -o ${final_bam} -

samtools index -@ ${cores} ${final_bam}