import logging
import os

import numpy as np
import pandas as pd
import pyBigWig
import tqdm


def find_genomic_min_max(bigwig_path):
    """
    Find the genomic minimum and maximum values in the chromosomes of interest

    :param bigwig_path: (str) Path to the input bigwig file

    :return: Genomic minimum and maximum values
    """
    logging.error("Finding min and max values per chromosome")

    with pyBigWig.open(bigwig_path) as input_bigwig:
        minmax_results = []

        # Create a status bar for to look fancy and count what chromosome you are on
        chromosome_status_bar = tqdm.tqdm(total=len(input_bigwig.chroms()), desc='Chromosomes Processed', position=0)

        for chromosome in input_bigwig.chroms():
            chr_vals = np.nan_to_num(input_bigwig.values(chromosome, 0, input_bigwig.chroms(chromosome), numpy=True))

            minmax_results.append([chromosome, np.min(chr_vals), np.max(chr_vals)])

            chromosome_status_bar.update(1)

        logging.error("Finding genome min and max values")

        minmax_results_df = pd.DataFrame(minmax_results)

        minmax_results_df.columns = ["chromosome", "min", "max"]

        basename = os.path.basename(bigwig_path)

        minmax_results_df.to_csv(str(basename) + "_chromosome_min_max.txt", sep="\t", index=False)

    return minmax_results_df["min"].min(), minmax_results_df["max"].max()


def minmax_normalize_array(array, minimum_value, maximum_value):
    """
    MinMax normalize the numpy array based on the genomic min and max

    :param array: Input array of bigwig values
    :param minimum_value: Minimum value to use
    :param maximum_value: Maximum value to use

    :return: MinMax normalized array
    """
    return (array - minimum_value) / (maximum_value - minimum_value)