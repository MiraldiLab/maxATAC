import pandas as pd
import pybedtools
import numpy as np
import pyBigWig
import py2bit

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


def import_bed(bed_file, region_length, chromosomes, chromosome_sizes_dictionary, blacklist):
    """
    Import a BED file and format the regions to be compatible with our maxATAC models

    @param bed_file: Input BED file to format
    @param region_length: Length of the regions to resize BED intervals to
    @param chromosomes: Chromosomes to filter the BED file for
    @param chromosome_sizes_dictionary: A dictionary of chromosome sizes to make sure intervals fall in bounds
    @param blacklist: A BED file of regions to exclude from our analysis

    @return: A dataframe of BED regions compatible with our model
    """
    df = pd.read_csv(bed_file,
                     sep="\t",
                     usecols=[0, 1, 2],
                     header=None,
                     names=["chr", "start", "stop"],
                     low_memory=False)

    df = df[df["chr"].isin(chromosomes)]

    df["length"] = df["stop"] - df["start"]

    df["center"] = np.floor(df["start"] + (df["length"] / 2)).apply(int)

    df["start"] = np.floor(df["center"] - (region_length / 2)).apply(int)

    df["stop"] = np.floor(df["center"] + (region_length / 2)).apply(int)

    df["END"] = df["chr"].map(chromosome_sizes_dictionary)

    df = df[df["stop"].apply(int) < df["END"].apply(int)]

    df = df[df["start"].apply(int) > 0]

    df = df[["chr", "start", "stop"]]

    BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

    blacklist_bedtool = pybedtools.BedTool(blacklist)

    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    df = blacklisted_df.to_dataframe()

    df.columns = ["chr", "start", "stop"]

    return df


def get_input_matrix(rows,
                     cols,
                     signal_stream,
                     average_stream,
                     sequence_stream,
                     bp_order,
                     chromosome,
                     start,  # end - start = cols
                     end
                     ):
    """
    Generate the matrix of values from the signal, sequence, and average data tracks

    :param rows: (int) The number of channels or rows
    :param cols: (int) The number of columns or length
    :param batch_size: (int) The number of examples per batch
    :param signal_stream: (str) ATAC-seq signal
    :param average_stream: (str) Average ATAC-seq signal
    :param sequence_stream: (str) One-hot encoded sequence
    :param bp_order: (list) Order of the bases in matrix
    :param chromosome: (str) Chromosome name
    :param start: (str) Chromosome start
    :param end: (str) Chromosome end

    :return: A matrix that is rows x columns with the values from each file
    """
    input_matrix = np.zeros((rows, cols))

    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chromosome, start, end),
            bp
        )

    signal_array = np.array(signal_stream.values(chromosome, start, end))
    avg_array = np.array(average_stream.values(chromosome, start, end))
    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array

    return input_matrix.T
