import logging
import os
import numpy as np
import pandas as pd
import pyBigWig
from scipy import stats

from maxatac.utilities.genome_tools import chromosome_blacklist_mask


def get_genomic_stats(bigwig_path: str,
                      chrom_sizes_dict: dict,
                      blacklist_path: str,
                      max_percentile: int,
                      name: str,
                      output_dir: str):
    """Find the genomic minimum and maximum values in the chromosomes of interest

    Additional statistics for standard deviation, mean, and median are calculated genome-wide.

    Args:
        output_dir (str): Output directory
        bigwig_path (str): Path to the input bigwig file
        chrom_sizes_dict (dict): A dictionary of chromosome sizes filtered for the chroms of interest
        blacklist_path (str): Path to the input blacklist file in bigwig format
        max_percentile (int): Percentile value to use as the max for normalization
        name (str): File prefix

    Returns:
        Any: Genomic minimum and maximum values
    """
    # Create filename for the stats txt output
    chromosome_stats = os.path.join(output_dir, name + "_chromosome_min_max.txt")
    genome_stats = os.path.join(output_dir, name + "_genome_stats.txt")

    # Open bigwig file
    with pyBigWig.open(bigwig_path) as input_bigwig:
        # Create an empty list to store results
        minmax_results = []

        # Create an empty array to store the genomic values
        genome_values_array = np.zeros(0, dtype=np.float32)

        for chromosome in chrom_sizes_dict:
            logging.error(f"Calculating statistics for {chromosome}")
            # Get the chromosome values. Convert nan to 0.
            chr_vals = np.nan_to_num(input_bigwig.values(chromosome, 0, input_bigwig.chroms(chromosome), numpy=True))

            # Import the blacklist mask for the specific chromosome
            blacklist_mask = chromosome_blacklist_mask(blacklist_path,
                                                       chromosome,
                                                       chrom_sizes_dict[chromosome]
                                                       )

            # Append minmax results to list
            minmax_results.append([chromosome,
                                   np.min(chr_vals[blacklist_mask]),
                                   np.max(chr_vals[blacklist_mask]),
                                   np.median(chr_vals[blacklist_mask] > 0)
                                   ])

            # Append chrom values to an array with genome-wide values
            genome_values_array = np.append(genome_values_array, chr_vals[blacklist_mask])

        logging.error("Calculating genome-wide statistics.")

        # Create a dataframe from the minmax results
        minmax_results_df = pd.DataFrame(minmax_results)

        # Add column names to the dataframe of stats
        minmax_results_df.columns = ["chromosome", "min", "max", "median"]

        # Write genome stats to text file
        minmax_results_df.to_csv(chromosome_stats, sep="\t", index=False)

        # Find the max value based on percentile
        max_value = np.percentile(genome_values_array[genome_values_array > 0], max_percentile)

        mean_value = np.mean(genome_values_array[genome_values_array > 0])

        std_value = np.std(genome_values_array[genome_values_array > 0])

        # Find the median value
        median_value = np.median(genome_values_array[genome_values_array > 0])

        # Find the min value based on genome min.
        min_value = minmax_results_df["min"].min()

        with open(genome_stats, 'w') as f:
            f.write("Genomic minimum value: " + str(min_value) +
                    "\nGenomic max value: " + str(max_value) +
                    "\nGenomic median (non-zero): " + str(median_value) +
                    "\nGenomic mean: " + str(mean_value) +
                    "\nGenomic standard deviation: " + str(std_value))

        return min_value, max_value, median_value, mean_value, std_value


def minmax_normalize_array(array: np.array, min_value: int, max_value: int, clip: bool=False):
    """MinMax normalize the numpy array based on the genomic min and max

    Args:
        array (np.array): Input array of bigwig values
        min_value (int): Max value for normalization
        max_value (int): Min value for normalization
        clip (bool, optional): Clip the values above the max value. Defaults to False.

    Returns:
        min-max normalized array: An array that has been min-max normalized
        
    Examples:
    
    >>> normalized_array = minmax_normalize_array(chr1_array, 0, 1, False)
    """
    normalized_array = (array - min_value) / (max_value - min_value)

    if clip:
        normalized_array = np.clip(normalized_array, 0, 1)

    return normalized_array


def zscore_normalize_array(array, mean, std_dev):
    """
    Zscore normalize the numpy array based on the genomic mean and standard deviation

    :param std_dev:
    :param mean:
    :param array: Input array of bigwig values

    :return: Zscore normalized array
    """
    return (array - mean) / std_dev


def arcsinh_normalize_array(array):
    """
    Arcsinh normalize the numpy array

    :param array: Input array of bigwig values

    :return: Arcsinh normalized array
    """
    return np.arcsinh(array)
