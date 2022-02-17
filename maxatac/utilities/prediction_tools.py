import logging

import numpy as np
import pandas as pd
import pybedtools
import tensorflow as tf

from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH, ALL_CHRS
from maxatac.utilities.system_tools import Mute

with Mute():
    from tensorflow.keras.models import load_model
    from maxatac.utilities.genome_tools import load_bigwig, load_2bit, dump_bigwig
    from maxatac.utilities.training_tools import get_input_matrix
 
def sortChroms(chrom):
    """Sort a list of chromosomes based on a specific order

    Args:
        chrom (str): Chromosome name

    Returns:
        int: Position in list based on predefined order
    """
    order = dict(zip(ALL_CHRS, range(0, len(ALL_CHRS))))
    return order[chrom] 

def write_predictions_to_bigwig(df: pd.DataFrame,
                                output_filename: str,
                                chrom_sizes_dictionary: dict,
                                chromosomes: list,
                                agg_mean: bool = True
                                ) -> object:
    """Write the predictions dataframe into a bigwig file

    Args:
        df (pd.DataFrame): The dataframe of BED regions with prediction scores
        output_filename (str): The output bigwig filename
        chrom_sizes_dictionary (dict): A dictionary of chromosome sizes used to form the bigwig file
        chromosomes (list): A list of chromosomes that you are predicting in
        agg_mean (bool, optional): use aggregation method of mean. Defaults to True.

    Returns:
        object: Writes a bigwig file
        
    Example:
    
    >>> write_predictions_to_bigwig(preds_df, "GM12878_CTCF.bw", chrom_sizes_dict, "chr20")
    """
    if agg_mean:
        bedgraph_df = df.groupby(["chr", "start", "stop"],
                                 as_index=False).mean()
    else:
        bedgraph_df = df.groupby(["chr", "start", "stop"],
                                 as_index=False).max()

    chromosomes.sort(key=sortChroms)
    
    with dump_bigwig(output_filename) as data_stream:
        # Make the bigwig header using the chrom sizes dictionary
        header = [(x, chrom_sizes_dictionary[x]) for x in chromosomes]

        # Add header to bigwig
        data_stream.addHeader(header)

        for chromosome in chromosomes:
            # Create a tmp df from the predictions dataframe for each chrom
            tmp_chrom_df = bedgraph_df[bedgraph_df["chr"] == chromosome].copy()

            # Bigwig files need sorted intervals as input
            tmp_chrom_df = tmp_chrom_df.sort_values(by=["chr", "start"])
            
            # Write all entries for the chromosome
            data_stream.addEntries(chroms=tmp_chrom_df["chr"].tolist(),
                                   starts=tmp_chrom_df["start"].tolist(),
                                   ends=tmp_chrom_df["stop"].tolist(),
                                   values=tmp_chrom_df["score"].tolist()
                                   )


def import_prediction_regions(bed_file: str,
                              chromosomes: list,
                              chrom_sizes_dictionary: dict,
                              blacklist: str,
                              region_length: int = INPUT_LENGTH,
                              step_size: int = 256):
    """Import BED file of regions of interest and format for maxATAC

    Args:
        bed_file (str): Path to the ROI BED file
        region_length (int): Prediction region length (1,024 bp)
        chromosomes (list): List of chromosomes for prediction
        chrom_sizes_dictionary (dict): Dictionary of chromosome sizes
        blacklist (str): Path to the blacklist BED file

    Returns:
        pd.DataFrame: A BED formatted dataframe of maxATAC compatible intevals
    
    Example:
    
    >>> roi_df = import_prediction_regions("test.bed", 1024, ["chr1"], chrom_sizes_dict, "hg38.blacklist.bed")
    """
    df = pd.read_csv(bed_file,
                     sep="\t",
                     usecols=[0, 1, 2],
                     header=None,
                     names=["chr", "start", "stop"],
                     low_memory=False)

    # Filter for chroms in the desired set
    df = df[df["chr"].isin(chromosomes)]
    
    # Create a bedtool object from dataframe for merging
    df_bedtool = pybedtools.BedTool.from_dataframe(df)

    # Sort the bedtools object
    sorted_bedtool = df_bedtool.sort()

    # Merge the bedtool object if they are half the region length away or less
    merged_bedtool = sorted_bedtool.merge(d=int(region_length/2))
    
    # Create a blacklist object form the blacklist bed file
    blacklist_bedtool = pybedtools.BedTool(blacklist)

    # Create an object that has the blacklisted regions removed
    blacklisted_bedtool = merged_bedtool.intersect(blacklist_bedtool, v=True)

    windowed_regions = pybedtools.BedTool().makewindows(b=blacklisted_bedtool, w=1024, s=step_size)

    # Create a dataframe from the bedtools object
    df = windowed_regions.to_dataframe()

    # Rename columns
    df.columns = ["chr", "start", "stop"]

    # Find the length of the regions
    df["length"] = df["stop"] - df["start"]

    df = df[df["length"] == 1024]

    return df[["chr", "start", "stop"]]

    
def create_prediction_regions(chromosomes: list,
                              chrom_sizes: dict,
                              blacklist: str,
                              step_size: int = 256,
                              region_length: int = INPUT_LENGTH):
    """Create whole genome or chromosome prediction regions

    Args:
        chromosomes (list): List of chromosomes to create prediction regions for
        chrom_sizes (dict): A dictionary of chromosome sizes
        blacklist (str): Path to the blacklist BED file
        step_size (int, optional): The step size to use for sliding windows. Defaults to 256.
        region_length (int, optional): The region length for prediction. Defaults to INPUT_LENGTH.

    Returns:
        pd.DataFrame : A dataframe of regions that are compatible with the model for making predictions
    
    Example:
    
    >>> roi_df = create_prediction_regions(["chr1"], "hg38.chrom.sizes", "hg38.blacklist.bed")
    """
    # Create a temp chrom.sizes file for the chromosomes of interest only

    
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
                 predict_roi_df,
                 input_channels: int = INPUT_CHANNELS,
                 input_length: int = INPUT_LENGTH,
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

        self.predict_roi_df.reset_index(inplace=True, drop=True)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        if self.predict_roi_df.shape[0] < self.batch_size:
            num_batches = 1  
            
        else:
            num_batches = int(self.predict_roi_df.shape[0] / self.batch_size)
        
        return num_batches
    
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

        return np.array(inputs_batch)


def make_stranded_predictions(roi_pool: pd.DataFrame,
                              signal: str,
                              sequence: str,
                              model: str,
                              batch_size: int,
                              use_complement: bool,
                              chromosome: str,
                              number_intervals: int =32,
                              input_channels: int =INPUT_CHANNELS,
                              input_length: int =INPUT_LENGTH):

    chr_roi_pool = roi_pool[roi_pool["chr"] == chromosome].copy()

    logging.error("Load pre-trained model")

    nn_model = load_model(model, compile=False)

    logging.error("Start Prediction Generator")

    data_generator = PredictionDataGenerator(signal=signal,
                                             sequence=sequence,
                                             input_channels=input_channels,
                                             input_length=input_length,
                                             predict_roi_df=chr_roi_pool,
                                             batch_size=batch_size,
                                             use_complement=use_complement)
    
    logging.error("Making predictions")

    predictions = nn_model.predict(data_generator)
  
    logging.error("Parsing results into pandas dataframe")

    predictions_df = pd.DataFrame(data=predictions, index=None, columns=None)
    
    if use_complement:
        # If reverse + complement used, reverse the columns of the pandas df in order to
        predictions_df = predictions_df[predictions_df.columns[::-1]]

    predictions_df["chr"] = chr_roi_pool["chr"]
    predictions_df["start"] = chr_roi_pool["start"]
    predictions_df["stop"] = chr_roi_pool["stop"]

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
