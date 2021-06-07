#!/bin/bash

################### sra2fastq.sh ###################
# This workflow will download an SRA file and convert to fastq

# INPUT:

# 1:    A file of SRA IDs

# 2:    cores to use

# OUTPUT:

# Two fastq files that have been validated, filtered, and adapters removed
###################################################

for i in `cat ${1}`;
    do
        echo "Downloading Data from SRA"
        prefetch ${i}

        echo "Converting SRA to Fastq"
        fasterq-dump ${i} -e 6
        
        echo "Compressing Fastq"
        pigz ${i}*

        echo "Trimming adapters from reads"
        trim_galore -q 30 -paired -j 4 ${i}_1.fastq.gz ${i}_2.fastq.gz
    done
