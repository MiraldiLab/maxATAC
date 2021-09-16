# ATAC-seq data processing

## Retrieving Public ATAC-seq Data: From SRA to Fastq

### SRA Files and File Requirements

For all of my analysis that were not performed at CCHMC I used SRAtoolkit to download the SRA file and then convert it to a fastq file. A `.sra` file is an archival format of the sequencing data used by NCBI to store the output of sequencing machines. We want to download the the publicly available sequencing data from the Short Read Archive (SRA) maintained by NCBI. The Gene Expression Omnibus (GEO) is a database that catalogues experimental data such as sequencing data, data analysis, and expression data to study gene expression. It helps to organize the SRA files into coherent experiments.

### Paired-End Sequencing vs Single-End Sequencing

For our analysis we want to use paired end data. This is because paired-end data provides us information about sequencing fragment size distributions that can be used to infer nucleosome positioning. In the original ATAC-seq paper from Buenrostro et al., the position of nucleosomes could be inferred based on the lengths of sequencing fragments produced from ATAC-seq data. PE sequencing is becoming the standard for ATAC-seq and is the data type that we will focus on. My other rationale is that paired-end sequencing will help resolve ambiguous reads or reads that are harder to place. This will be an advantage to ATAC-seq data analysis because open regions of chromatin can fall in repetitive regions or regions with structural variations making SE reads harder to place. Here is the Illumina reference for PE vs SE sequencing: [Paired-end video explanation](https://www.illumina.com/science/technology/next-generation-sequencing/plan-experiments/paired-end-vs-single-read.html).

### SRA Prefetch and Fasterq-dump

The SRA database has an associated tool called the SRAtoolkit. This toolkit has commands like `prefetch` and `fastq-dump` that can be used to retrieve publicly available sequencing data. We use SRA `prefetch` to pre-download the SRA file to our system. Then we use `fasterq-dump` to convert the SRA file to fastq format. The `prefetch` to `fastq-dump` workflow allows for pre-downloading SRA files and converting those files locally as needed.

## Read Alignment and Quality Control: Fastq to BAM

### Trimming Adapters from Reads using Trim-Galore!

Reads produced from sequencing experiments use adapters to help index, prime, and keep track of reads in an experiment. [These sequencing primers need to be removed before downstream data analysis](https://support.illumina.com/bulletins/2016/04/adapter-trimming-why-are-adapter-sequences-trimmed-from-only-the--ends-of-reads.html).

The adapters contain the sequencing primer binding sites, the index sequences, and the sites that allow library fragments to attach to the flow cell lawn. Libraries prepared with Illumina library prep kits require adapter trimming only on the 3’ ends of reads, because adapter sequences are not found on the 5’ ends.

Most of our experiments use paired-end Illumina sequencing technology, so we must process the fastq file that is downloaded with that adapter trimming algorithm. We use [Trim-Galore!](https://github.com/FelixKrueger/TrimGalore) for this purpose. Trim-Galore! uses the popular Cutadapt algorithm with FastQC to apply adapter trimming.

We use the following command to trim adapters from reads:

```bash
trim_galore -q 30 -paired -j 4 ${Fastq1} ${Fastq2}
```

The `-q` flag is saying to keep only reads with a [PHRED](https://en.wikipedia.org/wiki/Phred_quality_score) score 30 or greater.

The `-paired` flag is used to specify that the data is from paired-end sequencing experiments.

The `-j 4` flag is used to specify that the jobs 4 batches, but at least 16 cores are needed. See documentation

### Using STAR vs Bowtie

Our original pipeline used Bowtie2 to align reads to a reference genome. Bowtie2 takes several hours to align reads. Since we are developing workflows meant for high-throughput analysis we wanted to use a faster aligner like STAR. STAR is typically used for aligning reads to a reference transcriptome, but it can be used to align reads to a reference genome with modification. The alignment takes less than an hour per sample. We reference the following project for detailed explanations of quality statistics [HPC guide to STAR](https://github.com/hbctraining/Intro-to-rnaseq-hpc-O2/tree/master/lessons).

We used the following code for our ATAC-seq alignment with STAR:

```bash
STAR --genomeDir STAR_index
--runThreadN 16
--readFilesIn {fq1} {fq2}
--outFileNamePrefix {Filename}_
--outSAMtype BAM SortedByCoordinate
--outSAMattributes Standard --alignIntronMax 1 --alignMatesGapMax 2000 --alignEndsType EndToEnd
```

## Quality Control of Read Alignment

### PHRED Scores and STAR default output

We used STAR so the quality scores are different than those produced by Bowtie2. We can see this when looking at the `bedtools bamtobed` output:
<pre>
# The fifth column is the quality score
chr1|9988|10063|SRR5427886.6419238/1|255|+
chr1|9989|10064|SRR5427886.21797648/1|255|+
chr1|9989|10047|SRR5427886.23684048/2|255|+
chr1|9992|10068|SRR5427886.18590314/1|255|+
chr1|10039|10112|SRR5427886.8368275/1|255|+
</pre>

A score of 255 is for unique mapping reads. [STAR Manual](https://physiology.med.cornell.edu/faculty/skrabanek/lab/angsd/lecture_notes/STARmanual.pdf)

<pre>
|Flag|Description                              |
|--- |---                                      |
|1   |read is mapped                           |
|2   |read is mapped as part of a pair         |
|4   |read is unmapped                         |
|8   |mate is unmapped                         |
|16  |read reverse strand                      |
|32  |mate reverse strand                      |
|64  |first in pair                            |
|128 |second in pair                           |
|256 |not primary alignment                    |
|512 |read fails platform/vendor quality checks|
|1024|read is PCR or optical duplicate         |
</pre>

For now, we will focus on unique mapping reads only.

### Multi-mapped reads

We currently do not have a method for dealing with multi-mapped reads. We will eventually works towards implementing a method similar to [Basenji](https://genome.cshlp.org/content/28/5/739.full.html).

### Deduplication

We want to remove reads that are the product of PCR duplicates. These are reads that have the exact same mapping position and barcode. This helps reduce the library specific bias caused by PCR amplification of random sequences. To do this we use `samtools` to sort and then fixmate to fix named pairs. We then use `samtools markdup` to remove the duplicated read names. 

```bash
echo "Filter Files with Samtools: In Progress"

# Align and filter file by quality then sort by NAME
samtools sort -@ {cores} -n -o {name_sort} {file}

# Fixmate and sort by POSITION then index
samtools fixmate -@ {cores} -m {name_sort} - | \
samtools sort -@ {cores} -o {fix_mate} -

samtools index -@ {cores} {fix_mate}

rm {name_sort}

# Mark duplicates, remove, sort, index
samtools markdup -@ {cores} -r -s {fix_mate} - | \
samtools sort -@ {cores} -o {deduped} -
```

### Removing Specific Chromosomes and Singletons

The mitochondrial genome is abundant in every cell and in Standard ATAC-seq experiments there is a lot of mitochondrial contamination. We remove all reads associated with the mitochondrial genome.

### Properly paired and oriented reads without singletons removal with STAR specific samflags

During the filtering process we want to double check that no singleton reads are stored in the BAM file. This is because bedtools will output all reads in a BAM file. The pre-filtering ensures you only have the paired reads left and convert those to a BED file. We use samtools to do this filtering for us. We specify the -f 3 flag to filter for [properly paired and oriented reads](https://broadinstitute.github.io/picard/explain-flags.html).

This is an example code:

```bash
samtools view -@ ${cores} -f 3 -b -q 255 ${deduped} chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr1 chr20 chr21 chr22 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chrX chrY | \
samtools sort -@ ${cores} -o ${final_bam} -
```

## Calling regions of open chromatin: BAM to BED

### Tn5 Bias Correction and Inferring Tn5 binding sites

The Tn5 transposase has a bias in how far it inserts the sequencing adapter compared to where it cut the DNA strand. This bias has been shown to result in a 4 bp difference on the (+) strand and a 5 bp difference on the (-) strand at the 5' position. I take the BAM file and convert each read to an interval in a BED file. I then use awk to shift the reads the appropriate distance based on the strand. Another step is to take the Tn5 bias corrected reads and then find the insertion site and then center the signal over the insertion site. The signal should be centered over the 5' end of the read. In [BED intervals](https://angus.readthedocs.io/en/2013/_static/interval-datatypes.pdf) the 5' of the (+) strand is found as the chromosome start position (column 2) and the (-) strand is found as the chromosome end position (column 3). We also center the signal in between 20 bp regions that represent the width of the Tn5 transposase based on x-ray crystallography. This will provide us with regions of the genome that are occupied by Tn5 which are regions that are representative of open chromatin.

I used the following example code to generate the Tn5 binding sites:

```bash
bedtools bamtobed -i {Input_BAM} > ATAC_reads.bed
```

This is an example of raw reads converted to BED:

<pre>
chr1	10002	10099	J00118:569:HGKLCBBXY:6:1125:30847:1244/1	255	+
chr1	10002	10101	J00118:569:HGKLCBBXY:6:2114:14813:44042/1	255	+
chr1	10003	10104	J00118:569:HGKLCBBXY:6:1104:14570:44711/1	255	+
chr1	10003	10104	J00118:569:HGKLCBBXY:6:1103:14702:36816/1	255	+
chr1	10003	10077	J00118:569:HGKLCBBXY:6:1109:26971:13060/2	255	+
chr1	10003	10104	J00118:569:HGKLCBBXY:6:1109:27539:27971/2	255	+
chr1	10003	10081	J00118:569:HGKLCBBXY:6:1111:27082:45361/2	255	+
chr1	10003	10104	J00118:569:HGKLCBBXY:6:1116:19126:8154/1	255	+
chr1	10003	10104	J00118:569:HGKLCBBXY:6:1115:16457:35251/2	255	+
chr1	10003	10104	J00118:569:HGKLCBBXY:6:1116:15493:33545/1	255	+
</pre>

We then shift the reads by +4 or -5 to get the bias corrected read location:


```bash
awk 'BEGIN {OFS = "\t"} ; {if ($6 == "+") print $1, $2 + 4, $3 + 4, $4, $5, $6; else print $1, $2 - 5, $3 - 5, $4, $5, $6}' ATAC_reads.bed > ATAC_reads_BiasCorrected.bed
```

This is an example of reads shifted by +4 or -5:

<pre>
chr1	10006	10103	J00118:569:HGKLCBBXY:6:1125:30847:1244/1	255	+
chr1	10006	10105	J00118:569:HGKLCBBXY:6:2114:14813:44042/1	255	+
chr1	10007	10108	J00118:569:HGKLCBBXY:6:1104:14570:44711/1	255	+
chr1	10007	10108	J00118:569:HGKLCBBXY:6:1103:14702:36816/1	255	+
chr1	10007	10081	J00118:569:HGKLCBBXY:6:1109:26971:13060/2	255	+
chr1	10007	10108	J00118:569:HGKLCBBXY:6:1109:27539:27971/2	255	+
chr1	10007	10085	J00118:569:HGKLCBBXY:6:1111:27082:45361/2	255	+
chr1	10007	10108	J00118:569:HGKLCBBXY:6:1116:19126:8154/1	255	+
chr1	10007	10108	J00118:569:HGKLCBBXY:6:1115:16457:35251/2	255	+
chr1	10007	10108	J00118:569:HGKLCBBXY:6:1116:15493:33545/1	255	+
</pre>

The final step is to center the signal over the 5' end of the read that represents the Tn5 transposase binding event. You can use different window sizes depending on your needs. The Tn5 transposase is ~40 bp wide so I used a 40 bp width in the example below. The original Miraldi lab protocol was to use a 50bp window over the insertion size. Other groups have also used larger window sizes or only a single base pair. The window around the Tn5 cut size helps to smooth the signal out. It also provides biological insight into what is bound as the site.


```bash
awk 'BEGIN {OFS = "\t"} ; {if ($6 == "+") print $1, $2 - 20, $2 + 20, $4, $5, $6; else print $1, $3 - 20, $3 + 20, $4, $5, $6}' ATAC_reads_BiasCorrected.bed > Tn5BindingSites.bed
```


This is an example of inferring Tn5 binding sizes from bias corrected reads:

<pre>
chr1	9986	10026	J00118:569:HGKLCBBXY:6:1125:30847:1244/1	255	+
chr1	9986	10026	J00118:569:HGKLCBBXY:6:2114:14813:44042/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1104:14570:44711/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1103:14702:36816/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1109:26971:13060/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1109:27539:27971/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1111:27082:45361/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1116:19126:8154/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1115:16457:35251/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1116:15493:33545/1	255	+
</pre>

The problem with the above approach is that the chromosome sizes are ignored and you have a chance of getting entries that are beyond the chromosome position. So you can use slop in place of awk to get Tn5 sites centered around the 5' end.

```bash
bedtools bamtobed -i {Input_BAM} | \
awk 'BEGIN {OFS = "\t"} ; {if ($6 == "+") print $1, $2 + 4, $2 + 4, $4, $5, $6; else print $1, $3 - 5, $3 - 5, $4, $5, $6}' | \
bedtools slop -i - -g hg38.chrom.sizes -b 20 | \
sort -k 1,1 > ${Tn5_binding_sites_bed_filename}
```

This is an example of using the one line code with the above data. The results are the same as going through each step.

<pre>
chr1	9986	10026	J00118:569:HGKLCBBXY:6:1125:30847:1244/1	255	+
chr1	9986	10026	J00118:569:HGKLCBBXY:6:2114:14813:44042/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1104:14570:44711/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1103:14702:36816/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1109:26971:13060/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1109:27539:27971/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1111:27082:45361/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1116:19126:8154/1	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1115:16457:35251/2	255	+
chr1	9987	10027	J00118:569:HGKLCBBXY:6:1116:15493:33545/1	255	+
</pre>

### Generating bigwig files based on read normalized Tn5 tag counts

I used bedtools genomecov with the -bg flag. I performed the Tn5 BED to BEDgraph conversion for each replicate. The results should be a 4 column file that count in the 4th column representing the Tn5 count in that interval. I used the original BAM to get the mapped read count from the header.

This is an example workflow for read normalizing your inferred Tn5 bed files. This code will take the inferred Tn5 sites and build the genome wide coverage track using information in the bam header:

```bash
#!/bin/bash

########### RPM_normalize_Tn5_counts.sh ###########
# This script will take a BAM file and a BED file of insertion sites to create a bigwig file. 

# This script requires that bedtools be installed and bedGraphToBigWig from UCSC

# INPUT1: BAM
# INPUT2: chromosome sizes file (hg38.chromsizes.txt)
# INPUT3: Millions factor (1000000 or 20000000)
# INPUT4: Name
# INPUT5: BED of Tn5 Sites

# OUTPUT: Sequencing depth normalized bigwig file scaled by # of reads of interest

# Set up Names
bedgraph=${4}.bg
bigwig=${4}.bw

mapped_reads=$(samtools view -c -F 260 ${1})
reads_factor=$(bc -l <<< "1/${mapped_reads}")
rpm_factor=$(bc -l <<< "${reads_factor} * ${3}")
echo "Scale factor: " ${rpm_factor}

# Use bedtools to obtain genome coverage
echo "Using Bedtools to convert BAM to bedgraph"
bedtools genomecov -i ${5} -g ${2} -bg -scale ${rpm_factor} | LC_COLLATE=C sort -k1,1 -k2,2n > ${bedgraph}

# Use bedGraphToBigWig to convert bedgraph to bigwig
echo "Using bedGraphToBigWig to convert bedgraph to bigwig"
bedGraphToBigWig ${bedgraph} ${2} ${bigwig}

rm ${bedgraph}
```