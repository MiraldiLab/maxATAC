#!/bin/bash

################### trim_reads.sh ###################
# This workflow will trim adapters off reads in a fastq file 

# INPUT:

# 1:    Fastq1

# 2:    Fastq2

# OUTPUT:

# Two fastq files that have been validated, filtered, and adapters removed
###################################################

echo "Trimming adapters from reads"
trim_galore -q 30 -paired -j 4 ${1} ${2}
