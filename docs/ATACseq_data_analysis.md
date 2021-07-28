# ATAC-seq Data Analysis

## Replicate Analysis

We have multiple biological replicates for some of the ATAC-seq data. We want to look at the Pearson correlation of the alignments from each of the BAM files associated with each replicate. We will use `deeptools multiBamSummary bin` to compare the genomes at 200, 1000, and 10,000 resolution. We are looking for replicates to have high signal correlation across all fo the biological replicates. 

## Principal Component Analysis

We wanted to perform principal component anlaysis on the ATAC-seq data to see if the replicate cluster together using dimensionality reduction. 

### Create a union peak list

We will create a union of all peaks called for all replicates. This will provide us with genomic regions that are high signal for each cell type. We take the merged union of all peaks to create a union peak list. 

```bash
cat *Peaks | cut -f1,2,3 | bedtools sort | bedtools merge > /data/miraldiNB/Tareian/scratch/20210726_ATAC_64samples_PCA/maxATAC_reference_peaks_64samples.bed
```

### Intersect union peaks with tags

I `for` looped through the list of files I used to create the union peak list to get counts at each peak. I then stored those files as gzipped bed files where the 4th column is the number of Tn5 tags that overlap the peak.

```bash
bedtools intersect -a /data/miraldiNB/Tareian/scratch/20210726_ATAC_64samples_PCA/maxATAC_reference_peaks_64samples.bed -b ${BED_file} -c -sorted | pigz
```

Example for loop:

```bash
for BED_file in /scratch/caz3so/ATAC_processing/tags/*.gz;
do
basename_file=`basename ${BED_file} _R2_uniquely_mapped_sorted_filtered_deduped_tn5_cut_wo_blacklisted.bed.gz`

echo ${BED_file} ${basename_file}

bedtools intersect -a /data/miraldiNB/Tareian/scratch/20210726_ATAC_64samples_PCA/maxATAC_reference_peaks_64samples.bed -b ${BED_file} -c -sorted | pigz > /scratch/caz3so/ATAC_processing/peak_counts/${basename_file}_counts.bed.gz

done
```

### Collect read statistics from txt log files

```bash
for stats_file in /scratch/caz3so/ATAC_processing/stat/*.txt;
do
file_name=`basename ${stats_file} _R2_uniquely_mapped_sorted_bam_statistics_report.txt`
mapped_reads=$(grep "reads mapped:" ${stats_file} | cut -f 2 -d ":")
echo ${file_name} ${mapped_reads}
done
```