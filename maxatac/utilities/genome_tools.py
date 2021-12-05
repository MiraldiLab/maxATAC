import pandas as pd
import numpy as np
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
        return np.nan_to_num(input_bw.values(chrom_name, chrom_start, chrom_end, numpy=True))


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


def get_target_matrix(binding,
                      chromosome,
                      start,
                      end,
                      bp_resolution):
    """
    Get the values from a ChIP-seq signal file

    :param binding: ChIP-seq signal file
    :param chromosome: chromosome name: chr1
    :param start: start
    :param end: end
    :param bp_resolution: Prediction resolution in base pairs

    :return: Returns a vector of binary values
    """
    target_vector = np.array(binding.values(chromosome, start, end)).T

    target_vector = np.nan_to_num(target_vector, 0.0)

    n_bins = int(target_vector.shape[0] / bp_resolution)

    split_targets = np.array(np.split(target_vector, n_bins, axis=0))

    bin_sums = np.sum(split_targets, axis=1)

    return np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
    # TODO play with this parameter of 0.5


def get_synced_chroms(chroms, ignore_regions=None):
    """
    This function will generate a nested dictionary of chromosome sizes and the regions available for training.

        {
            "chr2": {"length": 243199373, "region": [0, 243199373]},
            "chr3": {"length": 198022430, "region": [0, 198022430]}
        }

    If ignore_regions is True, set regions to the whole chromosome length
    Returns something like this

    """
    chroms_and_regions = {}
    for chrom in chroms:
        chrom_name, *region = chrom.replace(",", "").split(":")  # region is either [] or ["start-end", ...]
        chroms_and_regions[chrom_name] = None
        if not ignore_regions:
            try:
                chroms_and_regions[chrom_name] = [int(i) for i in region[0].split("-")]
            except (IndexError, ValueError):
                pass

    loaded_chroms = set()

    synced_chroms = {}
    for chrom_name, chrom_length in loaded_chroms:
        if chrom_name not in chroms_and_regions: continue
        region = chroms_and_regions[chrom_name]
        if not region or \
                region[0] < 0 or \
                region[1] <= 0 or \
                region[0] >= region[1] or \
                region[1] > chrom_length:
            region = [0, chrom_length]
        synced_chroms[chrom_name] = {
            "length": chrom_length,
            "region": region
        }
    return synced_chroms


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

def filter_chrom_sizes(chrom_sizes_path, chromosomes, target_chrom_sizes_file):
    df = pd.read_table(chrom_sizes_path, header=None, names=["chr", "length"])
    
    df = df[df['chr'].isin(chromosomes)]
    
    df.to_csv(target_chrom_sizes_file, sep="\t", header=False, index=False)
    
    return target_chrom_sizes_file
