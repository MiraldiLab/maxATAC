#!/bin/bash

################### align_reads_STAR.sh ###################
# This script will align reads in a fastq file to a reference genome using STAR. 
# This script was written for ATAC-seq data anlysis of paired end sequencing data.

# INPUT:

# 1: Input fastq1
# example: Sample1_R1_001_val_1.fq.gz

# 2: Input fastq2
# example: Sample1_R1_001_val_2.fq.gz

# 3: STAR Index:
# example: /data/miraldiLab/databank/genome_inf/mm10/STAR_index

# 4: Prefix including the outdir and sample name:
# example: /data/miraldiNB/Th17_resources_PECA/ATACseq_Th17_48h/tmp/Sample1_

# 5: Number of Threads to use:
# example: 32

# OUTPUT:

# Aligned reads in BAM file. 
# Log of alignment stats.
###########################################################

### Rename Input Variables ###
fq1=${1}
fq2=${2}
STAR_INDEX=${3}
Prefix=${4}
THREADS=${5}

### Align Reads ###

# Parameter Description

# runThreadN: Number of threads to use
# readFilesIn: Fastq files
# outFileNamePrefix: Prefix of the outputDIR and the name to use
# outSAMtype: Output the file as a BAM file that is sorted
# outSAMunmapped: Output unmapped reads in the SAM file with special flag
# outSAMattributes: Standard SAM attributes
# alignIntronMax: Allow only 1 max intron. This is specific to ATAC-seq
# STAR was designed for RNA transcripts so we want to ignore some parameters
# alignMatesGapMax: Allow a maximum of 2000 bp gap.
# alignEndsType: This aligns the full read and considers the whole read in score calculation.

echo "Align Files with STAR: In Progress"

STAR --genomeDir "${STAR_INDEX}" \
--runThreadN "${THREADS}" \
--readFilesIn "${fq1}" "${fq2}" \
--outFileNamePrefix "${Prefix}" \
--outSAMtype BAM SortedByCoordinate \
--outSAMunmapped Within \
--outSAMattributes Standard \
--alignIntronMax 1 \
--alignMatesGapMax 2000 \
--alignEndsType EndToEnd
