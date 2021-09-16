# MOODS

We used MOODS to perform motif matching in our ATAC-seq data. 

## Prepare inputs

MOODS has specific requirements for the motif format and the format of the input files. 

### Fasta file of peaks

MOODS will scan a DNA sequence and look for matches to a library of input motifs. The input format must be in fasta format. We must retrieve the sequence under the genomic location of interest and then input that to MOODS. We use `bedtools getfasta` to input a `.bed` file of ATAC-seq peaks and get a `.fa` file of sequences.

Example:
`bedtools getfasta -fi Homo_sapiens.GRCh38.fa -bed ATAC_peaks.bed > ../fasta/output.fa`

### Format motif library

The CisBP motifs that you download from the site are not in the format that MOODS accepts.

Each motif appears as:

```
Pos	A	C	G	T
1	0.282958950502589	0.248464961579856	0.245992693905045	0.22258339401251
2	0.316659023962677	0.184532492829531	0.236944410987863	0.261864072219928
3	0.377777871501523	0.112652333533587	0.209412581763086	0.300157213201805
4	0.616393033452309	0.0689847041998677	0.111484414591218	0.203137847756605
5	0.675764369206464	0.0427703219414491	0.0631709440515372	0.21829436480055
6	0.326032179952778	0.133886990300119	0.0820227041025529	0.458058125644551
7	0.321153050529938	0.137265609257083	0.112383575642555	0.429197764570424
```

We need the motifs in a transposed format. I chose to add the .pfm extension to note the matrices are position frequency matrices.

```bash
for i in *.txt;
    do
        tail -n +2 $i | datamash transpose | tail -n +2 > ./pfm/`basename $i .txt`.pfm
    done
```

#### CISBP v2 format: 

``` 
Pos	A	C	G	T
1	1	0	0	0
2	0	0	0	1
3	1	0	0	0
4	1	0	0	0
5	0	0	0	1
```

#### MOODS format:

```
1	0	1	1	0
0	0	0	0	0
0	0	0	0	0
0	1	0	0	1
```

## Run MOODS

The next step is to run MOODS with the motif library, pvalue, and input fasta file. 

```bash
moods-dna.py -m * -s ${INPUT_FASTA} -p ${PVAL} --batch -o ${MOODS_OUTPUT}
```

The output will be a csv file that has every motif match. 

## Parse MOODS output

## Convert motif matches to signal track for benchmarking