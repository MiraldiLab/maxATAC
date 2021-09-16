# CD4+ Tcell Data Processing

The CD4+ Tcell fastq files were generated in-house. Each samples has two technical replicates that were sequenced at different times.

## Concatenating replicate fastq files

Technical replicates were concatenated together at the fastq level. You can use `cat` to do this.

Example:

```bash
cat ABSB049_UPSTR.fq.gz ABSB049_fastq_file_upstream.fastq.gz > TCM5h_Svetlana_1.fq.gz