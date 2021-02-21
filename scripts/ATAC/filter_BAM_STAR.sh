#!/bin/bash

################### filter_BAM.sh ###################
# This workflow will filter a BAM file based on the blacklist, autosomal chroms, and quality

# INPUT:

# 1: Input BAM file 
# example: GM12878.bam

# 2: Output Directory
# example: ./test

# 3: Number of cores to use
# example: 6

# OUTPUT:

# Filtered BAM file that has properly paired, deduplicated, and chromosome filtered files
###########################################################

### Rename Input Variables ###
bam=${1}
output_directory=${2}
cores=${3}

### Build Names ###
base_filename=$(basename "${bam}" .bam)

### Print Information ###
echo "Input filename: ${bam}"
echo "Base filename: ${base_filename}"
echo "Output directory: ${output_directory}"

## Make Directory ##
mkdir -p "${output_directory}"
cd "${output_directory}" || exit

## Build Filenames ##
name_sort="${base_filename}_namesort.bam"
fix_mate="${base_filename}_fixmate.bam"
deduped="${base_filename}_deduped.bam"
final_bam="${base_filename}_final.bam"

### Process ###
echo "Sort input bam file"

# Align and filter file by quality then sort by NAME
samtools sort -@ "${cores}" -n -o "${name_sort}" "${bam}"

echo "Fixmate positions and index bam file"

# Fixmate and sort by POSITION then index. This is required for mark duplicates to work
samtools fixmate -@ "${cores}" -m "${name_sort}" - | \
samtools sort -@ "${cores}" -o "${fix_mate}" -

# Index the fixmate bam file
samtools index -@ "${cores}" "${fix_mate}"

# Remove the previous file: this is to save memory
rm "${name_sort}"

echo "Removing PCR duplicates, sorting, and indexing"

# Mark duplicates, remove, sort, index
samtools markdup -@ "${cores}" -r -s "${fix_mate}" - | \
samtools sort -@ "${cores}" -o "${deduped}" -

# Index the deduped bam file
samtools index -@ "${cores}" "${deduped}"

rm "${fix_mate}" "${fix_mate}".bai

echo "Filter for autosomal chromosomes + X, Y. Removing unknown contigs and chromM."
echo "Filter for unique reads from STAR: i.e. quality score 255."
echo "Filter for properly paired and oriented reads: i.e. samflag 3."

# Remove unwanted chroms, sort, index
# The -f 3 flag will select reads that are properly aligned and paired. read paired (0x1) + read mapped in proper pair (0x2) = 3
# The -q 255 corresponds to selecting the MAPQ score == 255 which is STAR specific for uniquely mapped reads
# https://physiology.med.cornell.edu/faculty/skrabanek/lab/angsd/lecture_notes/STARmanual.pdf
samtools view -@ "${cores}" -f 3 -b -q 255 "${deduped}" chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr1 chr20 chr21 chr22 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chrX chrY | \
samtools sort -@ "${cores}" -o "${final_bam}" -

samtools index -@ "${cores}" "${final_bam}"

rm "${deduped}" "${deduped}".bai