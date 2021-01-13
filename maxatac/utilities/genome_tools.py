import logging
import os

import pandas as pd
import pybedtools
import numpy as np
import pyBigWig
import py2bit
import tqdm

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


def get_input_matrix(rows,
                     cols,
                     batch_size,  # make sure that cols % batch_size == 0
                     signal_stream,
                     average_stream,
                     sequence_stream,
                     bp_order,
                     chrom,
                     start,  # end - start = cols
                     end,
                     reshape=True,

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
    :param chrom: (str) Chromosome name
    :param start: (str) Chromosome start
    :param end: (str) Chromosome end
    :param reshape: (bool) Whether to transpose the matrix

    :return: A matrix that is rows x columns with the values from each file
    """
    input_matrix = np.zeros((rows, cols))

    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chrom, start, end),
            bp
        )

    signal_array = np.array(signal_stream.values(chrom, start, end))
    avg_array = np.array(average_stream.values(chrom, start, end))
    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array
    input_matrix = input_matrix.T

    if reshape:
        input_matrix = np.reshape(
            input_matrix,
            (batch_size, round(cols / batch_size), rows)
        )

    return input_matrix


def get_significant(data, min_threshold):
    selected = np.concatenate(([0], np.greater_equal(data, min_threshold).view(np.int8), [0]))
    breakpoints = np.abs(np.diff(selected))
    ranges = np.where(breakpoints == 1)[0].reshape(-1, 2)  # [[s1,e1],[s2,e2],[s3,e3]]
    expanded_ranges = list(map(lambda a: list(range(a[0], a[1])), ranges))
    mask = sum(expanded_ranges, [])  # to flatten
    starts = mask.copy()  # copy list just in case
    ends = [i + 1 for i in starts]
    return mask, starts, ends


def window_prediction_intervals(df, number_intervals=32):
    # Create BedTool object from the dataframe
    df_css_bed = pybedtools.BedTool.from_dataframe(df[['chr', 'start', 'stop']])

    # Window the intervals into 32 bins
    pred_css_bed = df_css_bed.window_maker(b=df_css_bed, n=number_intervals)

    # Create a dataframe from the BedTool object
    return pred_css_bed.to_dataframe()


def write_df2bigwig(output_filename, interval_df, chromosome_length_dictionary, chrom):
    with dump_bigwig(output_filename) as data_stream:
        header = [(chrom, int(chromosome_length_dictionary[chrom]))]
        data_stream.addHeader(header)

        data_stream.addEntries(
            chroms=interval_df["chr"].tolist(),
            starts=interval_df["start"].tolist(),
            ends=interval_df["stop"].tolist(),
            values=interval_df["score"].tolist()
        )


def GetBigWigValues(bigwig_path, chrom_name, chrom_end, chrom_start=0):
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(input_bw.values(chrom_name, chrom_start, chrom_end, numpy=True))


def FindGenomicMinMax(bigwig_path):
    """

    :param bigwig_path: (str) Path to the input bigwig file

    :return: Genomic minimum and maximum values
    """
    logging.error("Finding min and max values per chromosome")

    with pyBigWig.open(bigwig_path) as input_bigwig:
        minmax_results = []

        # Create a status bar for to look fancy and count what chromosome you are on
        chrom_status_bar = tqdm.tqdm(total=len(input_bigwig.chromosomes()), desc='Chromosomes Processed', position=0)

        for chrom in input_bigwig.chromosomes():
            chr_vals = np.nan_to_num(input_bigwig.values(chrom, 0, input_bigwig.chromosomes(chrom), numpy=True))

            minmax_results.append([chrom, np.min(chr_vals), np.max(chr_vals)])

            chrom_status_bar.update(1)

        logging.error("Finding genome min and max values")

        minmax_results_df = pd.DataFrame(minmax_results)

        minmax_results_df.columns = ["chromosome", "min", "max"]

        basename = os.path.basename(bigwig_path)

        minmax_results_df.to_csv(str(basename) + "_chromosome_min_max.txt", sep="\t", index=False)

    return minmax_results_df["min"].min(), minmax_results_df["max"].max()


def MinMaxNormalizeArray(array, minimum_value, maximum_value):
    """
    MinMax normalize the numpy array based on the genomic min and max

    :param array: Input array of bigwig values
    :param minimum_value: Minimum value to use
    :param maximum_value: Maximum value to use

    :return: MinMax normalized array
    """
    return (array - minimum_value) / (maximum_value - minimum_value)

def get_batch(
        signal,
        sequence,
        roi_pool,
        bp_resolution=1
):
    inputs_batch, targets_batch = [], []
    roi_size = roi_pool.shape[0]
    with \
            load_bigwig(signal) as signal_stream, \
            load_2bit(sequence) as sequence_stream:
        for row_idx in range(roi_size):
            row = roi_pool.loc[row_idx,:]
            chrom_name = row[0]
            start = int(row[1])
            end = int(row[2])
            input_matrix = get_input_matrix(
                rows=INPUT_CHANNELS,
                cols=INPUT_LENGTH,
                batch_size=1,                  # we will combine into batch later
                reshape=False,
                scale_signal=TRAIN_SCALE_SIGNAL,
                bp_order=BP_ORDER,
                signal_stream=signal_stream,
                sequence_stream=sequence_stream,
                chrom=chrom_name,
                start=start,
                end=end
            )
            inputs_batch.append(input_matrix)
    return (np.array(inputs_batch))


def GetPredictionROI(seq_len=None, roi=None, shuffle=False):
    roi_df = pd.read_csv(roi, sep="\t", header=None)

    roi_df.columns = ['chr', 'start', 'stop']

    temp = roi_df['stop'] - roi_df['start']

    roi_ok = (temp == seq_len).all()

    if not roi_ok:
        sys.exit("ROI Length Does Not Match Input Length")

    if shuffle:
        roi_df = roi_df.sample(frac=1)

    return roi_df
