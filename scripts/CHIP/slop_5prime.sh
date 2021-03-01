#!/bin/bash

################### slop_5prime.sh ###################
# This workflow will window the 5' end of the read by a specific interval

# INPUT:

# 1: 5' sites
# example: GM12878_5prime.bed

# 2: flanking size
# example: 10

# 3: chromosome sizes file
# example: hg38.chrom.sizes

# 4: blacklisted
# example: hg38.composite.blacklist.bed

# 5: Output directory
# example: ./test

# OUTPUT:

# Windowed 5' sites based on windowing around the base pair resolved 5' end of the read
###########################################################

### Rename Input Variables ###
five_prime=${1}
flanking_size=${2}
chrom_sizes=${3}
blacklist=${4}
output_directory=${5}

### Build names ###
five_prime_slop=$(basename "${five_prime}" .bed.gz)_slop${flanking_size}bp.bed.gz

### Process ###

# Window around the input insertion size and then compress the file
# Then remove the sites overlapping blacklisted regions. -v option
# Bedtools can take as input a compressed or uncompressed BED file. It's awesome
bedtools slop  -i "${five_prime}" -g "${chrom_sizes}" -b "${flanking_size}" | \
bedtools intersect -a - -b "${blacklist}" -v | \
pigz > "${output_directory}"/"${five_prime_slop}"
