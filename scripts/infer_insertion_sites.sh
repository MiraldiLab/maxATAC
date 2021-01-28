#!/bin/bash

################### infer_insertion_sites.sh ###################
# This workflow will infer the Tn5 insertions site based on the reads location

# INPUT:

# 1: The filtered ATAC-seq bam file
# example: GM12878_final.bam

# 2: Output directory
# example: ./test

# 3: Blacklisted regions
# example: hg38.composite.blacklist.bed

# OUTPUT:

# Inferred Tn5 insertion sites
# These sites are single base pair resolved Tn5 cut sites based on the 5' end of the read
###########################################################

### Rename Input Variables ###
bam=${1}
output_directory=${2}
blacklist=${3}

### Build names ###
insertion_sites=`basename ${bam} .bam`_IS.bed.gz

# Make directory and change into it
mkdir -p ${output_directory}

### Process ###

# Convert bam to bed and then infer 1 bp resolved insertion sites. 
# I use pigz at the end of the command to compress the file since they can be very large.
# Make sure to +4 on the (+) strand and to use the 5' end. The 5' end is on the left side (+) strand.
# Make sure to -5 on the (-) strand and to use the 5' end. The 5' end is on the right side (-) strand.

# Because BED intervals are 0-based, half-open coordinate systems we want to have atleast 1 base where the interval covers so we add 1 bp to the end position.
# For the (+) strand add 1 bp (+4bp + 1) for the end position.
# For the (-) strand add 1 bp (-5bp + 1) for the end position. 
bedtools bamtobed -i ${bam} | \
awk 'BEGIN {OFS = "\t"} ; {if ($6 == "+") print $1, $2 + 4, $2 + 5, $4, $5, $6; else print $1, $3 - 5, $3 - 4, $4, $5, $6}' | \
bedtools intersect -a - -b ${blacklist} -v | \
pigz > ${output_directory}/${insertion_sites}