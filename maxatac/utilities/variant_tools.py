import logging
import numpy as np
import pandas as pd
import pybedtools
import allel
from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH
from maxatac.utilities.training_tools import get_input_matrix
from maxatac.utilities.genome_tools import load_bigwig, load_2bit, get_one_hot_encoded

def import_bed(bed_file,
               blacklist):
    """Import a bed file and exclude regions overlapping blacklist

    Args:
        bed_file: Path to target regions BED file
        blacklist: Path to blacklist BED file

    Example Usage:
    >>> targets_df = import_bed("targets.bed", "blacklist.bed")
    """
    df = pd.read_csv(bed_file,
                     sep="\t",
                     usecols=[0, 1, 2],
                     header=None,
                     names=["chr", "start", "stop"],
                     low_memory=False)
    
    # Create a bedtool object from dataframe
    BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

    # Create a blacklist object form the blacklist bed file
    blacklist_bedtool = pybedtools.BedTool(blacklist)

    # Create an object that has the blacklisted regions removed
    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    # Create a dataframe from the bedtools object
    df = blacklisted_df.to_dataframe()

    # Rename columns
    df.columns = ["chr", "start", "stop"]

    return df

def intersect_bins_targets(windowed_genome,
                           target_regions,
                           blacklist,
                           window=1024
                           ):
    """Intersect target regions with bins for prediction

    Args:
        windowed_genome (dataframe): The windowed genome dataframe
        target_regions (dataframe): The target regions dataframe
        window (int, optional): The window around the target regions to expand the predictions to. Defaults to 768.
    
    Example Usage:
    >>> prediction_regions = intersect_bins_targets(windows_df, targets_df, "hg38_blacklist.bed")
    """
        # Create a bedtool object from dataframe
    windowed_genome_bedtool = pybedtools.BedTool.from_dataframe(windowed_genome)
    target_regions_bedtools = pybedtools.BedTool.from_dataframe(target_regions)

    overlap_bedtools = windowed_genome_bedtool.window(target_regions_bedtools, w=window, u=True)
    
    # Create a blacklist object form the blacklist bed file
    blacklist_bedtool = pybedtools.BedTool(blacklist)

    # Create an object that has the blacklisted regions removed
    blacklisted_df = overlap_bedtools.intersect(blacklist_bedtool, v=True)

    # Create a dataframe from the bedtools object
    df = blacklisted_df.to_dataframe()

    df.drop_duplicates(subset=["chrom", "start", "end"], inplace=True)
    
    df.columns = ["chr", "start", "stop"]
    
    return df

class VariantsDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 signal,
                 sequence,
                 input_channels,
                 input_length,
                 predict_roi_df,
                 batch_size=32,
                 use_complement=False
                 ):
        """
        Initialize the variants generator. This is a keras sequence class object. It is used
        to make sure each batch is unique and sequentially generated.

        :param signal: ATAC-seq signal track
        :param sequence: 2Bit DNA sequence track
        :param input_channels: How many channels there are
        :param input_length: length of receptive field
        :param predict_roi_df: Dataframe that contains the BED intervals to predict on
        :param batch_size: Size of each training batch or # of examples per batch
        :param use_complement: Whether to use the forward or reverse (complement) strand
        """
        self.batch_size = batch_size
        self.predict_roi_df = predict_roi_df
        self.indexes = np.arange(self.predict_roi_df.shape[0])
        self.signal = signal
        self.sequence = sequence
        self.input_channels = input_channels
        self.input_length = input_length
        self.use_complement = use_complement

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(self.predict_roi_df.shape[0] / self.batch_size)

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index: current index of batch that we are on
        """
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self.__data_generation__(batch_indexes)

        return X

    def __data_generation__(self, batch_indexes):
        """
        Generates data containing batch_size samples

        :param batch_indexes: list of indexes of to use for batch
        """
        # Store sample
        batch_roi_df = self.predict_roi_df.loc[batch_indexes, :]

        batch_roi_df.reset_index(drop=True, inplace=True)

        batch = self.__get_region_values__(roi_pool=batch_roi_df)

        return batch

    def __get_region_values__(self, roi_pool):
        """
        Get the bigwig values for each ROI in the ROI pool

        :param roi_pool: Pool of regions to make predictions on
        """
        # Create the lists to hold the batch data
        inputs_batch = []

        # Calculate the size of the predictions pool
        roi_size = roi_pool.shape[0]

        # With the files loaded get the data
        with load_bigwig(self.signal) as signal_stream, load_2bit(self.sequence) as sequence_stream:
            for row_idx in range(roi_size):
                # Get the single row
                row = roi_pool.loc[row_idx, :]

                # Get the matric of values for the entry
                input_matrix = get_input_matrix(signal_stream=signal_stream,
                                                sequence_stream=sequence_stream,
                                                chromosome=row[0],
                                                start=int(row[1]),
                                                end=int(row[2]),
                                                use_complement=self.use_complement,
                                                reverse_matrix=self.use_complement)

                # Append the matrix of values to the batch list
                inputs_batch.append(input_matrix)

        return np.array(input_matrix)
