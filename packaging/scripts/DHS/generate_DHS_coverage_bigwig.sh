#!/bin/bash

################### generate_DHS_coverage_bigwig.sh ###################
# This workflow will window Dnase I cut site and generate a coverage track that is RP20M normalized

# INPUT:

# 1: The filtered DnaseI-seq bam file
# example: GM12878_final.bam

# 2: The DHS cut sites
# example: GM12878_DHS_IS.bed.gz

# 2: Output directory
# example: ./test

# 3: Blacklisted regions
# example: hg38.composite.blacklist.bed

# 4: Slop size
# example: 10

# 5: Chromosome sizes file
# example: hg38.chrom.sizes

# 6: Scale Factor
# example: 20000000

# 7: Threads
# example: 6

# OUTPUT:

# DHS cut site bigwig track

###########################################################

### Rename Input Variables ###
bam=${1}
DHS_bed=${2}
output_directory=${3}
blacklist=${4}
slop=${5}
chrom_sizes=${6}
scale_factor=${7}
threads=${8}

### Build names ###
base_filename=`basename ${DHS_bed} _IS.bed.gz`
prefix=${base_filename}_RP20M_slop${slop}
bedgraph=${prefix}.bg
bigwig=${prefix}.bw

# Make directory and change into it
mkdir -p ${output_directory}

### Process ###
# http://www.metagenomics.wiki/tools/samtools/number-of-reads-in-bam-file
mapped_reads=$(samtools view -F 260 -@ ${threads} -c ${bam})
reads_factor=$(bc -l <<< "1/${mapped_reads}")
rpm_factor=$(bc -l <<< "${reads_factor} * ${scale_factor}")

echo "Scale factor: " ${rpm_factor}

# Use bedtools to window the signal around the cut site then remove blacklisted regions and generate coverage
# Use the total mapped reads to scale the DHS signal to RP20M, sort the bedgraph
bedtools slop -b ${slop} -g ${chrom_sizes} -i ${DHS_bed} | \
bedtools intersect -a - -b ${blacklist} -v | \
bedtools genomecov -i - -g ${chrom_sizes} -bg -scale "${rpm_factor}" | \
LC_COLLATE=C sort -k1,1 -k2,2n > ${output_directory}/${bedgraph}

# Use bedGraphToBigWig to convert bedgraph to bigwig
echo "Using bedGraphToBigWig to convert bedgraph to bigwig"

bedGraphToBigWig ${output_directory}/${bedgraph} ${chrom_sizes} ${output_directory}/${bigwig}

rm "${output_directory}/${bedgraph}"