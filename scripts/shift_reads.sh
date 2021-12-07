#!/bin/bash

# $1: Input BAM
# $2: Chromosome sizes text file
# $3: Slop size to use
# $4: Path to blacklist
# $5: Output filename prefix
# $6: Output directory
# $7: Scaling factor

bed_name=${6}/${5}_IS_slop${3}.bed
bedgraph_name=${6}/${5}_IS_slop${3}.bg
bigwig_name=${6}/${5}_IS_slop${3}_RP20M.bw

bedtools bamtobed -i ${1} | \
awk 'BEGIN {OFS = "\t"} ; {if ($6 == "+") print $1, $2 + 4, $2 + 5, $4, $5, $6; else print $1, $3 - 5, $3 - 4, $4, $5, $6}' | \
bedtools slop  -i - -g ${2} -b ${3} | \
bedtools intersect -a - -b ${4} -v | \
tee ${bed_name} | \
bedtools genomecov -i - -g ${2} -bg -scale ${7} | \
LC_COLLATE=C sort -k1,1 -k2,2n > ${bedgraph_name}

echo "Converting bedgraph to bigwig"
bedGraphToBigWig ${bedgraph_name} ${2} ${bigwig_name}

echo "Compressing files"

cd ${6}

pigz ${bed_name}
pigz ${bedgraph_name}

echo "Done!"