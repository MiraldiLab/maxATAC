import logging

import numpy as np
import pandas as pd
import pybedtools
import tensorflow as tf

from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH
from maxatac.utilities.system_tools import Mute

with Mute():
    from tensorflow.keras.models import load_model
    from maxatac.utilities.genome_tools import load_bigwig, load_2bit, dump_bigwig
    from maxatac.utilities.training_tools import get_input_matrix


def write_predictions_to_bigwig(df,
                                output_filename,
                                chrom_sizes_dictionary,
                                chromosomes,
                                agg_mean=True
                                ):
    """
    Write the predictions dataframe into a bigwig file

    :param df: The dataframe of BED regions with prediction scores
    :param output_filename: The output bigwig filename
    :param chrom_sizes_dictionary: A dictionary of chromosome sizes used to form the bigwig file
    :param chromosomes: A list of chromosomes that you are predicting in
    :param agg_mean: use aggregation method of mean

    :return: Writes a bigwig file
    """
    if agg_mean:
        # Sort dataframe to make sure that all intervals are in order
        bedgraph_df = df.groupby(["chr", "start", "stop"],
                                 as_index=False).mean()
    else:
        bedgraph_df = df.groupby(["chr", "start", "stop"],
                                 as_index=False).max()

    with dump_bigwig(output_filename) as data_stream:
        # Make the bigwig header using the chrom sizes dictionary
        header = [(x, chrom_sizes_dictionary[x]) for x in chromosomes]

        # Add header to bigwig
        data_stream.addHeader(header)

        for chromosome in chromosomes:
            # Create a tmp df from the predictions dataframe for each chrom
            tmp_chrom_df = bedgraph_df[bedgraph_df["chr"] == chromosome].copy()

            # Bigwig files need sorted intervals as input
            tmp_chrom_df.sort_values(by=["chr", "start", "stop"], inplace=True)

            # Write all entries for the chromosome
            data_stream.addEntries(chroms=tmp_chrom_df["chr"].tolist(),
                                   starts=tmp_chrom_df["start"].tolist(),
                                   ends=tmp_chrom_df["stop"].tolist(),
                                   values=tmp_chrom_df["score"].tolist()
                                   )


def import_prediction_regions(bed_file,
                              region_length,
                              chromosomes,
                              chrom_sizes_dictionary,
                              blacklist):
    """
    Import a BED file and format the regions to be compatible with our maxATAC models

    :param bed_file: Input BED file to format
    :param region_length: Length of the regions to resize BED intervals to
    :param chromosomes: Chromosomes to filter the BED file for
    :param chrom_sizes_dictionary: A dictionary of chromosome sizes to make sure intervals fall in bounds
    :param blacklist: A BED file of regions to exclude from our analysis

    :return: A dataframe of BED regions compatible with our model
    """
    df = pd.read_csv(bed_file,
                     sep="\t",
                     usecols=[0, 1, 2],
                     header=None,
                     names=["chr", "start", "stop"],
                     low_memory=False)

    # Filter for chroms in the desired set
    df = df[df["chr"].isin(chromosomes)]

    # Find the length of the regions
    df["length"] = df["stop"] - df["start"]

    # Find the center of the regions
    df["center"] = np.floor(df["start"] + (df["length"] / 2)).apply(int)

    # Find the start coordinates based on center
    df["start"] = np.floor(df["center"] - (region_length / 2)).apply(int)

    # Find the stop coordinates based on center
    df["stop"] = np.floor(df["center"] + (region_length / 2)).apply(int)

    # Map chrom end
    df["END"] = df["chr"].map(chrom_sizes_dictionary)

    # Make sure none of the stops are out of bounds
    df = df[df["stop"].apply(int) < df["END"].apply(int)]

    # Make sure none of the starts are out of bounds
    df = df[df["start"].apply(int) > 0]

    # Subset to 3-col BED format
    df = df[["chr", "start", "stop"]]

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


def create_prediction_regions(region_length,
                              chromosomes,
                              chrom_sizes,
                              blacklist,
                              step_size):
    """
    Window the genome into regions compatible with the model input sizes

    :param region_length: Length of the region in basepairs
    :param chromosomes: The chromosomes to limit prediction to
    :param chrom_sizes: The chomosomes sizes file for use with pybedtools
    :param blacklist: The blacklist regions to remove from the analysis
    :param step_size: The step size of the window to use.

    :return: A dataframe of regions that are compatible with the model for making predictions
    """
    # Create a bedtools object that is a windowed genome
    BED_df_bedtool = pybedtools.BedTool().window_maker(g=chrom_sizes, w=region_length, s=step_size)

    # Create a blacklist object form the blacklist bed
    blacklist_bedtool = pybedtools.BedTool(blacklist)

    # Remove the blacklisted regions from the windowed genome object
    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    # Create a dataframe from the BedTools object
    df = blacklisted_df.to_dataframe()

    # Rename the columns
    df.columns = ["chr", "start", "stop"]

    # Filter for specific chroms
    df = df[df["chr"].isin(chromosomes)]

    # Reset index so that it goes from 0-end in order
    df = df.reset_index(drop=True)

    return df


class PredictionDataGenerator(tf.keras.utils.Sequence):
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
        Initialize the training generator. This is a keras sequence class object. It is used
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
                input_matrix = get_input_matrix(rows=self.input_channels,
                                                cols=self.input_length,
                                                bp_order=["A", "C", "G", "T"],
                                                signal_stream=signal_stream,
                                                sequence_stream=sequence_stream,
                                                chromosome=row[0],
                                                start=int(row[1]),
                                                end=int(row[2]),
                                                use_complement=self.use_complement,
                                                reverse_matrix=self.use_complement)

                # Append the matrix of values to the batch list
                inputs_batch.append(input_matrix)

        return np.array(inputs_batch)


def make_stranded_predictions(signal,
                              sequence,
                              models,
                              predict_roi_df,
                              batch_size,
                              use_complement,
                              number_intervals=32,
                              input_channels=INPUT_CHANNELS,
                              input_length=INPUT_LENGTH):
    """
    This function will make prediction based on the forward or reverse strand sequence.

    :param number_intervals: Number of intervals to split array into
    :param signal: Input ATAC-seq signal
    :param sequence: Input 2bit DNA sequence
    :param models: Input pre-trained model
    :param predict_roi_df: DataFrame of BED intervals to predict on
    :param batch_size: The number of example per batch to predict on
    :param use_complement: Whether to use complement sequence or reference
    :param input_channels: Number of input channels to use
    :param input_length: Length of input region to use for training

    :return: A dataframe of predictions for the strand of choice
    """
    logging.error("Load pre-trained model")

    nn_model = load_model(models, compile=False)

    logging.error("Start Prediction Generator")

    data_generator = PredictionDataGenerator(signal=signal,
                                             sequence=sequence,
                                             input_channels=input_channels,
                                             input_length=input_length,
                                             predict_roi_df=predict_roi_df,
                                             batch_size=batch_size,
                                             use_complement=use_complement)

    logging.error("Making predictions")

    predictions = nn_model.predict(data_generator)

    logging.error("Parsing results into pandas dataframe")

    predictions_df = pd.DataFrame(data=predictions, index=None, columns=None)

    if use_complement:
        # If reverse + complement used, reverse the columns of the pandas df in order to
        predictions_df = predictions_df[predictions_df.columns[::-1]]

    predictions_df["chr"] = predict_roi_df["chr"]
    predictions_df["start"] = predict_roi_df["start"]
    predictions_df["stop"] = predict_roi_df["stop"]

    # Create BedTool object from the dataframe
    coordinates_dataframe = pybedtools.BedTool.from_dataframe(predictions_df[['chr', 'start', 'stop']])

    # Window the intervals into 32 bins
    windowed_coordinates = coordinates_dataframe.window_maker(b=coordinates_dataframe, n=number_intervals)

    # Create a dataframe from the BedTool object
    windowed_coordinates_dataframe = windowed_coordinates.to_dataframe()

    # Drop all columns except those that have scores in them
    scores_dataframe = predictions_df.drop(['chr', 'start', 'stop'], axis=1)

    # Take the scores and reshape them into a column of the pandas dataframe
    windowed_coordinates_dataframe['score'] = scores_dataframe.to_numpy().flatten()

    # Rename the columns of the dataframe
    windowed_coordinates_dataframe.columns = ['chr', 'start', 'stop', 'score']

    # Get the mean of all sliding window predicitons
    windowed_coordinates_dataframe = windowed_coordinates_dataframe.groupby(["chr", "start", "stop"],
                                                                            as_index=False).mean()

    return windowed_coordinates_dataframe
