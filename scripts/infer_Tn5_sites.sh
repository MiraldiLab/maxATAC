#!/bin/bash

################### infer_Tn5_sites.sh ###################
# This workflow will infer the Tn5 binding site based on the insertion site + a flank size

# INPUT:

# 1: insertion sites
# example: GM12878_Tn5.bed

# 2: flanking size
# example: 10

# 3: chromosome sizes file
# example: hg38.chrom.sizes

# 4: blacklisted
# example: hg38.composite.blacklist.bed

# 5: Output directory
# example: ./test

# OUTPUT:

# Inferred Tn5 dimer sites based on windowing around the base pair resolved Tn5 cut site
###########################################################

### Rename Input Variables ###
insertion_sites=${1}
flanking_size=${2}
chrom_sizes=${3}
blacklist=${4}
output_directory=${5}

### Build names ###
Tn5_sites=`basename ${insertion_sites} .bed.gz`_Tn5_slop${flanking_size}bp.bed.gz

### Process ###

# Window around the input insertion size and then compress the file
# Then remove the sites overlapping blacklisted regions. -v option
# Bedtools can take as input a compressed or uncompressed BED file. It's awesome
bedtools slop  -i ${insertion_sites} -g ${chrom_sizes} -b ${flanking_size} | \
bedtools intersect -a - -b ${blacklist} -v | \
pigz > ${output_directory}/${Tn5_sites}
