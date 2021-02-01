import pandas as pd
import pybedtools
import numpy as np
import pyBigWig
import py2bit
import random

from maxatac.utilities.system_tools import get_absolute_path


def build_chrom_sizes_dict(chromosome_list,
                           chrom_sizes_filename
                           ):
    """
    Build a dictionary of chromosome sizes.

    :param chromosome_list: (str) path to the chromosome sizes file.
    :param chrom_sizes_filename: (str) a list of chromosomes of interest.
    :return: A dictionary of chromosome sizes filtered by chromosome list.
    """
    chrom_sizes_df = pd.read_csv(chrom_sizes_filename, header=None, names=["chr", "len"], sep="\t")

    chrom_sizes_df = chrom_sizes_df[chrom_sizes_df["chr"].isin(chromosome_list)]

    return pd.Series(chrom_sizes_df.len.values, index=chrom_sizes_df.chr).to_dict()


def dump_bigwig(location):
    """
    Write a bigwig file to the location

    :param location: the path to desired file location

    :return: an opened bigwig for writing
    """
    return pyBigWig.open(get_absolute_path(location), "w")


def get_one_hot_encoded(sequence, target_bp):
    """
    Convert a 2bit DNA sequence to a one-hot encoded sequence.

    :param sequence: path to the 2bit DNA sequence
    :param target_bp: resolution of the bp sequence

    :return: one-hot encoded DNA sequence
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

    @param bigwig_path: Path to the bigwig file
    @param chrom_name: Chromosome name
    @param chrom_end: chromosome end
    @param chrom_start: chromosome start

    @return: Bigwig values from the region given
    """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(input_bw.values(chrom_name, chrom_start, chrom_end, numpy=True))


def import_bed(bed_file,
               region_length,
               chromosomes,
               chromosome_sizes_dictionary,
               blacklist,
               ROI_type_tag,
               ROI_cell_tag):
    """
    Import a BED file and format the regions to be compatible with our maxATAC models

    :param ROI_cell_tag:
    :param ROI_type_tag:
    :param bed_file: Input BED file to format
    :param region_length: Length of the regions to resize BED intervals to
    :param chromosomes: List of chromosomes to limit the input
    :param chromosome_sizes_dictionary: A dictionary of chromosome sizes to make sure intervals fall in bounds
    :param blacklist: A BED file of regions to exclude from our analysis
    :param ROI_type_tag: Tag to use in the description column
    :param ROI_cell_tag: Tag to use in the description column

    :return: A dataframe of BED regions compatible with our model
    """
    # Import dataframe
    df = pd.read_csv(bed_file,
                     sep="\t",
                     usecols=[0, 1, 2],
                     header=None,
                     names=["chr", "start", "stop"],
                     low_memory=False)

    # Make sure the chromosomes in the ROI file frame are in the target chromosome list
    df = df[df["chr"].isin(chromosomes)]

    # Find the length of the regions
    df["length"] = df["stop"] - df["start"]

    # Find the center of each peak.
    # TODO Finding the center of the peak might not be the best approach to finding the ROI.
    # We might want to use bedtools to window the regions of interest around the peak.
    df["center"] = np.floor(df["start"] + (df["length"] / 2)).apply(int)

    # The start of the interval will be the center minus 1/2 the desired region length.
    df["start"] = np.floor(df["center"] - (region_length / 2)).apply(int)

    # the end of the interval will be the center plus 1/2 the desired region length
    df["stop"] = np.floor(df["center"] + (region_length / 2)).apply(int)

    # The chromosome end is defined as the chromosome length
    df["END"] = df["chr"].map(chromosome_sizes_dictionary)

    # Make sure the stop is less than the end
    df = df[df["stop"].apply(int) < df["END"].apply(int)]

    # Make sure the start is greater than the chromosome start of 0
    df = df[df["start"].apply(int) > 0]

    # Select for the first three columns to clean up
    df = df[["chr", "start", "stop"]]

    # Import the dataframe as a pybedtools object so we can remove the blacklist
    BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

    # Import the blacklist as a pybedtools object
    blacklist_bedtool = pybedtools.BedTool(blacklist)

    # Find the intervals that do not intersect blacklisted regions.
    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    # Convert the pybedtools object to a pandas dataframe.
    df = blacklisted_df.to_dataframe()

    # Rename the columns
    df.columns = ["chr", "start", "stop"]

    df["ROI_Type"] = ROI_type_tag

    df["ROI_Cell"] = ROI_cell_tag

    return df


def get_input_matrix(rows,
                     cols,
                     signal_stream,
                     sequence_stream,
                     bp_order,
                     chromosome,
                     start,  # end - start = cols
                     end,
                     scale_signal
                     ):
    """
    Generate the matrix of values from the signal, sequence, and average data tracks

    :param rows: (int) The number of channels or rows
    :param cols: (int) The number of columns or length
    :param signal_stream: (str) ATAC-seq signal
    :param sequence_stream: (str) One-hot encoded sequence
    :param bp_order: (list) Order of the bases in matrix
    :param chromosome: (str) Chromosome name
    :param start: (str) Chromosome start
    :param end: (str) Chromosome end
    :param scale_signal: (tuple) Randomly scale input signal by these values

    :return: A matrix that is rows x columns with the values from each file
    """
    input_matrix = np.zeros((rows, cols))

    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chromosome, start, end),
            bp
        )

    signal_array = np.array(signal_stream.values(chromosome, start, end))
    input_matrix[4, :] = signal_array

    if scale_signal is not None:
        scaling_factor = random.random() * (scale_signal[1] - scale_signal[0]) + scale_signal[0]
        input_matrix[4, :] = input_matrix[4, :] * scaling_factor

    return input_matrix.T


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
