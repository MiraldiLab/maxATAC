#!/bin/bash

################### infer_5prime_BAM.sh ###################
# This workflow will infer the 5prime site of the reads

# INPUT:

# 1: The filtered bam file
# example: GM12878_final.bam

# 2: Output directory
# example: ./test

# 3: Blacklisted regions
# example: hg38.composite.blacklist.bed

# OUTPUT:

# Inferred 5' read sites
###########################################################

### Rename Input Variables ###
bam=${1}
output_directory=${2}
blacklist=${3}

### Build names ###
five_prime=`basename ${bam} .bam`_5prime.bed.gz

# Make directory and change into it
mkdir -p ${output_directory}

### Process ###

# Convert bam to bed and then infer 1 bp resolved 5' sites
# I use pigz at the end of the command to compress the file since they can be very large.
# Make sure to +1 on the (+) strand and to use the 5' end. The 5' end is on the left side (+) strand.
# Make sure to -1 on the (-) strand and to use the 5' end. The 5' end is on the right side (-) strand.

# Because BED intervals are 0-based, half-open coordinate systems we want to have atleast 1 base where the interval covers so we add 1 bp to the end position.
# For the (+) strand add 1 bp for the end position.
# For the (-) strand add 1 bp for the end position. 
bedtools bamtobed -i ${bam} | \
awk 'BEGIN {OFS = "\t"} ; {if ($6 == "+") print $1, $2, $2 + 1, $4, $5, $6; else print $1, $3, $3 + 1, $4, $5, $6}' | \
bedtools intersect -a - -b ${blacklist} -v | \
pigz > ${output_directory}/${five_prime}