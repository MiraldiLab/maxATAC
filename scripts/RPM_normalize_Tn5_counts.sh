#!/bin/bash

########### RPM_normalize_Tn5_counts.sh ###########
# This script will take a BAM file and create a bigwig file. 

# INPUT:

# 1: Filtered BAM file
# example: GM12878_final.bam

# 2: chromosome sizes file
# example: hg38.chrom.sizes

# 3: Millions factor
# example: 1000000

# 4: Name
# example: GM12878_slop10_RP20M

# 5: BED of Tn5 Sites
# example: GM12878_IS_slop10.bed

# OUTPUT: Bigwig file of Tn5 counts that have been normalized by sequencing depth

### Rename Input Variables ###
bedgraph=${4}.bg
bigwig=${4}.bw

### Process ###

mapped_reads=$(samtools view -c -F 260 ${1})
reads_factor=$(bc -l <<< "1/${mapped_reads}")
rpm_factor=$(bc -l <<< "${reads_factor} * ${3}")

echo "Scale factor: " $rpm_factor

# Use bedtools to obtain genome coverage
echo "Using Bedtools to convert BAM to bedgraph"

bedtools genomecov -i ${5} -g ${2} -bg -scale ${rpm_factor} | LC_COLLATE=C sort -k1,1 -k2,2n > ${bedgraph}

# Use bedGraphToBigWig to convert bedgraph to bigwig
echo "Using bedGraphToBigWig to convert bedgraph to bigwig"

bedGraphToBigWig ${bedgraph} ${2} ${bigwig}

rm ${bedgraph}