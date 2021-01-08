import logging
import numpy as np
import pyBigWig
import pandas as pd
import os

from os import path
from maxatac.utilities.genome_tools import build_chrom_sizes_dict


def find_genomic_minmax(x):
    """Load the genome bigwig file and find the min and max values
    """
    bw = pyBigWig.open(x)    

    minmax_results = []

    logging.error("Finding min and max values per chromosome")    
    for chrom in bw.chroms():
        chr_vals = np.nan_to_num(bw.values(chrom, 0, bw.chroms(chrom), numpy=True))

        minmax_results.append([chrom, np.min(chr_vals), np.max(chr_vals)])

    logging.error("Finding genome min and max values")    

    minmax_results_df = pd.DataFrame(minmax_results) 

    minmax_results_df.columns = ["chromosome", "min", "max"]

    basename = os.path.basename(x)

    minmax_results_df.to_csv(str(basename) + "_chromosome_min_max.txt", sep="\t", index=False)

    return minmax_results_df["min"].min(), minmax_results_df["max"].max()

def normalize_signal(chrom_array, genome_min, genome_max):
    """This function will normalize the numpy array based on the parameters of the min and max values"""
    minmax = lambda x: ((x-genome_min)/(genome_max-genome_min))
    
    return minmax(chrom_array)

def normalize_write_bigwig(input_bigwig, OUT_BIGWIG_FILENAME, chromosome_length_dictionary, chromosome_list, genome_min, genome_max):
    with pyBigWig.open(input_bigwig) as input_bw, pyBigWig.open(OUT_BIGWIG_FILENAME, "w") as output_bw:
        header = [(x, chromosome_length_dictionary[x]) for x in sorted(chromosome_list)]

        output_bw.addHeader(header)

        for chrom_name, chrom_length in header:
            chr_vals = np.nan_to_num(input_bw.values(chrom_name, 0, chrom_length, numpy=True))

            normalized_signal = normalize_signal(chr_vals, genome_min, genome_max)
            
            logging.error("Add Entries for " + str(chrom_name))

            output_bw.addEntries(
                chroms = chrom_name,
                starts = 0,      # [0, 1, 2, 3, 4]
                ends = chrom_length,  # [1, 2, 3, 4, 5]
                span=1,
                step=1,
                values = normalized_signal.tolist()
            )


def run_normalization(args):

    logging.error(
        "Normalization" +
        "\n  Target signal(s): \n   - " + "\n   - ".join(args.signal) +
        "\n  Reference Genome Build: " + args.GENOME + 
        "\n  Output prefix: " + args.prefix + 
        "\n  Output directory: " + args.output
    )

    logging.error("Find the min and max values across the genome")
    
    chromosome_length_dictionary = build_chrom_sizes_dict(args.GENOME)
    
    genome_min, genome_max = find_genomic_minmax(args.signal)

    logging.error("Normalize and Write BigWig file")

    OUTPUT_FILENAME = args.output + "/" + args.prefix + "_minmax01.bw"

    logging.error("Output BigWig Filename: " + OUTPUT_FILENAME)

    normalize_write_bigwig(args.signal, OUTPUT_FILENAME, chromosome_length_dictionary, DEFAULT_CHRS, genome_min, genome_max)