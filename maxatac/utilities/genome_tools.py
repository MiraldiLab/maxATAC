import logging
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pyBigWig
import py2bit
import random

from maxatac.utilities.system_tools import get_absolute_path


def build_chrom_sizes_dict(chromosome_list: list, 
                           chrom_sizes_filename: str):
    """Build a dictionary of chromosome sizes filtered for chromosomes in the input chromosome_list.
    
    The dictionary takes the form of: 
    
        {"chr1": 248956422, "chr2": 242193529}

    Args:
        chromosome_list (list): A list of chromosome to filter dictionary by
        chrom_sizes_filename (str): A path to the chromosome sizes file

    Returns:
        dict: A dictionary of chromosome sizes filtered by chromosome list.
        
    Example:
    
    >>> chrom_dict = build_chrom_sizes_dict(["chr1", "chr2"], "hg38.chrom.sizes")
    """
    # Import the data as pandas dataframe
    chrom_sizes_df = pd.read_csv(chrom_sizes_filename, header=None, names=["chr", "len"], sep="\t")

    # Filter the dataframe for the chromosomes of interest
    chrom_sizes_df = chrom_sizes_df[chrom_sizes_df["chr"].isin(chromosome_list)]

    return pd.Series(chrom_sizes_df.len.values, index=chrom_sizes_df.chr).to_dict()

    
def dump_bigwig(location : str):
    """Write a bigwig file to the location

    Args:
        location (str): The path to desired file location

    Returns:
        bigwig stream: An opened bigwig for writing
    """
    return pyBigWig.open(get_absolute_path(location), "w")


def get_one_hot_encoded(sequence, target_bp):
    """Convert a 2bit DNA sequence to a one-hot encoded sequence.

    Args:
        sequence ([type]): path to the 2bit DNA sequence
        target_bp ([type]): resolution of the bp sequence

    Returns:
        array: one-hot encoded DNA sequence
    """
    one_hot_encoded = []
    for s in sequence:
        if s.lower() == target_bp.lower():
            one_hot_encoded.append(1)
        else:
            one_hot_encoded.append(0)
    return one_hot_encoded


def load_2bit(location):
    """
    Load a 2bit file.

    :param location: path to the 2bit DNA sequence

    :return: opened 2bit file
    """
    return py2bit.open(get_absolute_path(location))


def load_bigwig(location):
    """
    Load a bigwig file

    :param location: path to the bigwig file

    :return: opened bigwig file
    """
    return pyBigWig.open(get_absolute_path(location))


def get_bigwig_values(bigwig_path, chrom_name, chrom_end, chrom_start=0):
    """
    Get the values for a genomic region of interest from a bigwig file.

    :param bigwig_path: Path to the bigwig file
    :param chrom_name: Chromosome name
    :param chrom_end: chromosome end
    :param chrom_start: chromosome start

    :return: Bigwig values from the region given
    """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(input_bw.values(chrom_name, chrom_start, chrom_end))


def get_bigwig_stats(bigwig_path, chrom_name, chrom_end, bin_count, agg_function="max"):
    """
    Get the values for a genomic region of interest from a bigwig file.

    :param bin_count:
    :param agg_function:
    :param bigwig_path: Path to the bigwig file
    :param chrom_name: Chromosome name
    :param chrom_end: chromosome end

    :return: Bigwig values from the region given
    """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(np.array(input_bw.stats(chrom_name,
                                                     0,
                                                     chrom_end,
                                                     type=agg_function,
                                                     nBins=bin_count,
                                                     exact=True),
                                      dtype=float  # need it to have NaN instead of None
                                      ))


class EmptyStream():
    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        pass


def safe_load_bigwig(location):
    try:
        return pyBigWig.open(get_absolute_path(location))
    except (RuntimeError, TypeError):
        return EmptyStream()


def chromosome_blacklist_mask(blacklist, chromosome, chromosome_length, nBins=False, agg_method="max"):
    """
    Import the chromosome signal from a blacklist bigwig file and convert to a numpy array to use to generate the array
    to use to exclude regions. If a number of bins are provided, then the function will use the stats method from 
    pyBigWig to bin the data. 

    :return: blacklist_mask: A np.array the has True for regions that are NOT in the blacklist.
    """
    with load_bigwig(blacklist) as blacklist_bigwig_stream:
        if nBins:
            return np.array(blacklist_bigwig_stream.stats(chromosome,
                                                          0,
                                                          chromosome_length,
                                                          type=agg_method,
                                                          nBins=nBins
                                                          ),
                            dtype=float  # need it to have NaN instead of None
                            ) != 1  # Convert to boolean array, select areas that are not 1

        else:
            return blacklist_bigwig_stream.values(chromosome,
                                                  0,
                                                  chromosome_length,
                                                  numpy=True) != 1  # Convert to boolean array, select areas that are not 1


def chromosome_whitelist_mask(whitelist, chromosome, chromosome_length, nBins=False, agg_method="max"):
    """
    Import the chromosome signal from a whitelist bigwig file and convert to a numpy array to use to include regions.
    If a number of bins are provided, then the function will use the stats method from pyBigWig to bin the data.

    :return: whitelist_mask: A np.array that has True for regions that are in the whitelist.
    """
    with load_bigwig(whitelist) as whitelist_bigwig_stream:
        if nBins:
            return np.array(whitelist_bigwig_stream.stats(chromosome,
                                                          0,
                                                          chromosome_length,
                                                          type=agg_method,
                                                          nBins=nBins
                                                          ),
                            dtype=float
                            ) > 0

        else:
            return whitelist_bigwig_stream.values(chromosome,
                                                  0,
                                                  chromosome_length,
                                                  numpy=True) > 0


def filter_chrom_sizes(chrom_sizes_path, chromosomes, target_chrom_sizes_file):
    df = pd.read_table(chrom_sizes_path, header=None, names=["chr", "length"])
    
    df = df[df['chr'].isin(chromosomes)]
    
    df.to_csv(target_chrom_sizes_file, sep="\t", header=False, index=False)
    
    return target_chrom_sizes_file

def import_bigwig_stats_array(bigwig_stream, chromosome, chromosome_length, agg_function, bin_count):
    """
    Import chromosome-level bigWig stats as a float array while preserving missing values as NaN.
    """
    return np.array(bigwig_stream.stats(chromosome,
                                        0,
                                        chromosome_length,
                                        type=agg_function,
                                        nBins=bin_count,
                                        exact=True),
                    dtype=float)


def _pct_rank(array):
    """
    Percentile rank of each bin across the whole array: rankdata(method="min") / number of bins.
    NaN (missing) bins are set to 0 first, and every zero-valued bin gets percentile 0, so the
    many zero bins of a sparse track cannot outrank low but nonzero bins of the other track.
    """
    array = np.nan_to_num(np.asarray(array, dtype=float))
    if array.size == 0:
        return array

    percentile = rankdata(array, method="min") / array.size
    percentile[array == 0] = 0.0

    return percentile


def _log_max_source(primary_array, alternative_array, combine_operation):
    """
    Sanity check for a max combine: how often each track supplies the max. Near 0% or 100% means
    the ensemble is effectively just one of the two inputs.
    """
    n_bins = primary_array.size
    if n_bins == 0:
        return

    primary_wins = np.count_nonzero(primary_array > alternative_array) / n_bins
    alternative_wins = np.count_nonzero(alternative_array > primary_array) / n_bins

    logging.info("%s combine: primary supplies the max in %.2f%% of bins, alternative in %.2f%%, tied in %.2f%%",
                 combine_operation, 100 * primary_wins, 100 * alternative_wins,
                 100 * (1 - primary_wins - alternative_wins))


def combine_prediction_arrays(primary_array, alternative_array=None, combine_operation="mean"):
    """
    Combine two prediction arrays element-wise.

    mean / max: whenever both arrays contain values for the same bin, combine them using either the
    element-wise mean or max. If only one array has a value for a bin, use the available
    value. If neither has a value, return 0 for that position.

    rank_max: convert each array to percentile ranks across all of its bins (_pct_rank), then take the
    element-wise max. Missing bins get percentile 0, so the other track's value is used, which is the
    same missing-value rule as max. Rank before any masking: the result does not depend on the
    blacklist or whitelist.
    """
    primary_array = np.array(primary_array, dtype=float)
    if alternative_array is None:
        return np.nan_to_num(primary_array)

    alternative_array = np.array(alternative_array, dtype=float)

    if primary_array.shape != alternative_array.shape:
        raise ValueError("Prediction arrays must have the same shape to be combined.")

    if combine_operation not in {"mean", "max", "rank_max"}:
        raise ValueError("combine_operation must be one of 'mean', 'max' or 'rank_max'.")

    if combine_operation == "rank_max":
        primary_percentile = _pct_rank(primary_array)
        alternative_percentile = _pct_rank(alternative_array)
        _log_max_source(primary_percentile, alternative_percentile, combine_operation)

        return np.maximum(primary_percentile, alternative_percentile)

    primary_available = ~np.isnan(primary_array)
    alternative_available = ~np.isnan(alternative_array)
    both_available = primary_available & alternative_available

    combined = np.full(primary_array.shape, np.nan, dtype=float)
    combined[primary_available & ~alternative_available] = primary_array[primary_available & ~alternative_available]
    combined[alternative_available & ~primary_available] = alternative_array[alternative_available & ~primary_available]

    if combine_operation == "mean":
        combined[both_available] = (primary_array[both_available] + alternative_array[both_available]) / 2.0
    else:
        combined[both_available] = np.maximum(primary_array[both_available], alternative_array[both_available])
        _log_max_source(np.nan_to_num(primary_array, nan=-np.inf), np.nan_to_num(alternative_array, nan=-np.inf),
                        combine_operation)

    return np.nan_to_num(combined)


def import_prediction_array_fn(prediction_stream,
                               chromosome,
                               chromosome_length,
                               agg_function,
                               bin_count,
                               alternative_prediction_stream=None,
                               combine_operation="mean"):
    """
    Import the chromosome signal from one or two prediction bigWig files and convert to a numpy array.
    """

    prediction_array = import_bigwig_stats_array(prediction_stream,
                                                 chromosome,
                                                 chromosome_length,
                                                 agg_function,
                                                 bin_count)

    alternative_prediction_array = None
    if alternative_prediction_stream is not None:
        alternative_prediction_array = import_bigwig_stats_array(alternative_prediction_stream,
                                                                chromosome,
                                                                chromosome_length,
                                                                agg_function,
                                                                bin_count)

    return combine_prediction_arrays(prediction_array,
                                     alternative_prediction_array,
                                     combine_operation=combine_operation)


def import_quant_goldstandard_array_fn(quant_goldstandard_stream, chromosome, chromosome_length, agg_function, bin_count):
    """
    Import the chromosome signal from the gold standard bigwig file and convert to a numpy array.
    """

    # Get the bin stats from the gold standard array

    quant_goldstandard_array = np.nan_to_num(np.array(quant_goldstandard_stream.stats(chromosome,
                                                                                           0,
                                                                                           chromosome_length,
                                                                                           type=agg_function,
                                                                                           nBins=bin_count,
                                                                                           exact=True
                                                                                           ),
                                                           dtype=float  # need it to have NaN instead of None
                                                           )
                                                  ) # Commented out to keep values non-boolean:  > 0  # to convert to boolean array

    return quant_goldstandard_array
