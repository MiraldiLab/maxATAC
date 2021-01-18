from keras.models import load_model
import logging
import numpy as np
import pandas as pd
import pybedtools
import keras

from maxatac.utilities.genome_tools import load_bigwig, load_2bit, get_input_matrix, dump_bigwig, import_bed


def make_predictions(
        signal,
        sequence,
        average,
        model,
        predict_roi_df,
        batch_size,
        round_predictions,
        input_channels,
        input_length,
        predictions=""
):
    """
    Make predictions on the input ROIs

    :param signal: The ATAC-seq signal file
    :param sequence: The 2bit DNA sequence file
    :param average: The average ATAC-seq signal file
    :param model: The trained maxATAC model(s)
    :param predict_roi_df: A dataframe containing a BED formatted dataframe
    :param batch_size: The number of examples to predict in each batch
    :param round_predictions: Number of decimal places to round prediction scores to
    :param input_channels: Number of input channels
    :param input_length: Length of the input regions of interest
    :param predictions: empty and used to hold predictions.

    :return: A dataframe of scores associated with the regions of interest
    """
    logging.error("Load Model")

    # Load the neural network model that has been trained.
    nn_model = load_model(model, compile=False)

    # Determine the number of batches based on the total number of regions and batch size
    n_batches = int(predict_roi_df.shape[0] / batch_size)

    # Get batches of indexes
    all_batch_idxs = np.array_split(np.arange(predict_roi_df.shape[0]), n_batches)

    logging.error("Making predictions")

    # Loop through all of the batches and make predictions.
    # TODO Make this loop into a parallel loop
    for idx, batch_idxs in enumerate(all_batch_idxs):
        batch_roi_df = predict_roi_df.loc[batch_idxs, :]

        batch_roi_df.reset_index(drop=True, inplace=True)

        input_batch = get_roi_values(signal=signal,
                                     sequence=sequence,
                                     average=average,
                                     input_channels=input_channels,
                                     input_length=input_length,
                                     roi_pool=batch_roi_df
                                     )

        pred_output_batch = nn_model.predict(input_batch)

        if idx == 0:
            predictions = pred_output_batch

        else:
            predictions = np.vstack((predictions, pred_output_batch))

    predictions = np.round(predictions, round_predictions)

    logging.error("Parsing results into pandas dataframe")

    predictions_df = pd.DataFrame(data=predictions, index=None, columns=None)

    predictions_df["chr"] = predict_roi_df["chr"]
    predictions_df["start"] = predict_roi_df["start"]
    predictions_df["stop"] = predict_roi_df["stop"]

    return predictions_df


def write_predictions_to_bigwig(df,
                                output_filename,
                                chromosome_length_dictionary,
                                chromosomes,
                                number_intervals=32
                                ):
    """
    Write the predictions dataframe into a bigwig file

    :param df: The dataframe of BED regions with prediction scores
    :param output_filename: The output bigwig filename
    :param chromosome_length_dictionary: A dictionary of chromosome sizes used to form the bigwig file
    :param chromosomes: A list of chromosomes that you are predicting in
    :param number_intervals: The number of 32 bp intervals found in the sequence

    :return: Writes a bigwig file
    """
    # Create BedTool object from the dataframe
    coordinates_dataframe = pybedtools.BedTool.from_dataframe(df[['chr', 'start', 'stop']])

    # Window the intervals into 32 bins
    windowed_coordinates = coordinates_dataframe.window_maker(b=coordinates_dataframe, n=number_intervals)

    # Create a dataframe from the BedTool object
    windowed_coordinates_dataframe = windowed_coordinates.to_dataframe()

    # Drop all columns except those that have scores in them
    scores_dataframe = df.drop(['chr', 'start', 'stop'], axis=1)

    # Take the scores and reshape them into a column of the pandas dataframe
    windowed_coordinates_dataframe['score'] = scores_dataframe.to_numpy().flatten()

    # Rename the columns of the dataframe
    windowed_coordinates_dataframe.columns = ['chr', 'start', 'stop', 'score']

    # TODO parallel write many chromosomes. Hardcoded now to first chromosome in list
    with dump_bigwig(output_filename) as data_stream:
        header = [(chromosomes[0], int(chromosome_length_dictionary[chromosomes[0]]))]
        data_stream.addHeader(header)

        data_stream.addEntries(
            chroms=windowed_coordinates_dataframe["chr"].tolist(),
            starts=windowed_coordinates_dataframe["start"].tolist(),
            ends=windowed_coordinates_dataframe["stop"].tolist(),
            values=windowed_coordinates_dataframe["score"].tolist()
        )


def get_roi_values(
        signal,
        sequence,
        average,
        roi_pool,
        input_channels,
        input_length
):
    """
    Get the bigwig values for each ROI in the ROI pool

    @param signal: ATAC-seq signal file
    @param sequence: 2bit DNA file
    @param average: Average ATAC-seq signal file
    @param roi_pool: Pool of regions to predict on
    @param input_channels: Number of input channels
    @param input_length: Length of the input regions

    @return: Array of examples
    """
    inputs_batch, targets_batch = [], []

    roi_size = roi_pool.shape[0]

    with load_bigwig(signal) as signal_stream, \
            load_2bit(sequence) as sequence_stream, \
            load_bigwig(average) as average_stream:
        for row_idx in range(roi_size):
            row = roi_pool.loc[row_idx, :]

            chromosome = row[0]

            start = int(row[1])

            end = int(row[2])

            input_matrix = get_input_matrix(rows=input_channels,
                                            cols=input_length,
                                            bp_order=["A", "C", "G", "T"],
                                            average_stream=average_stream,
                                            signal_stream=signal_stream,
                                            sequence_stream=sequence_stream,
                                            chromosome=chromosome,
                                            start=start,
                                            end=end
                                            )

            inputs_batch.append(input_matrix)

    return np.array(inputs_batch)


class DataGenerator(keras.utils.Sequence):
    """
    Generate batches of examples for the chromosomes of interest

    This generator will create individual batches of examples from a pool of regions of interest and/or a pool of
    random regions from the genome. This class object helps keep track of all of the required inputs and how they are
    processed.

    This generator expects a meta_table with the header:

    TF | Cell Type | ATAC signal | ChIP signal | ATAC peaks | ChIP peaks
    """

    def __init__(self,
                 meta_dataframe,
                 chromosomes,
                 blacklist,
                 region_length,
                 average,
                 sequence,
                 batch_size,
                 bp_resolution,
                 input_channels,
                 chromosome_pool_size,
                 chromosome_sizes,
                 random_ratio,
                 peak_paths,
                 cell_types,
                 batches_per_epoch,
                 shuffle=True):
        """
        :param meta_dataframe: Path to meta table
        :param chromosomes: List of chromosome to restrict data to
        :param blacklist: Path to blacklist BED regions to exclude
        :param region_length: Length of the inputs to use
        :param average: Path to the average signal track
        :param sequence: Path to the 2bit DNA sequence
        :param batch_size: Number of examples to generate per batch
        :param bp_resolution: Resolution of the output prediction
        :param input_channels: Number of input channels
        :param chromosome_pool_size: Size of the chromosome pool used for generating random regions
        :param chromosome_sizes: A txt file of chromosome sizes
        :param random_ratio: The proportion of the total batch size that will be from randomly generated examples
        :param batches_per_epoch: The number of batches to use per epoch
        """
        self.chromosome_pool_size = chromosome_pool_size
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution
        self.batch_size = batch_size
        self.sequence = sequence
        self.average = average
        self.region_length = region_length
        self.blacklist = blacklist
        self.chromosomes = chromosomes
        self.meta_dataframe = meta_dataframe
        self.random_ratio = random_ratio
        self.peak_paths = peak_paths
        self.cell_types = cell_types
        self.batch_per_epoch = batches_per_epoch
        self.shuffle = shuffle

        # Calculate the number of ROI and Random regions needed based on batch size and random ratio desired
        self.number_roi = round(batch_size * (1. - random_ratio))
        self.number_random_regions = round(batch_size - self.number_roi)

        self.chromosome_sizes_dictionary = build_chrom_sizes_dict(chromosomes, chromosome_sizes)

        # Get the ROIPool and/or RandomRegionsPool
        if random_ratio < 1:
            self.ROI_pool = self.__get_ROIPool()

        if random_ratio > 0:
            self.RandomRegions_pool = self.__get_RandomRegionsPool()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.batch_per_epoch

    def __getitem__(self, index):
        """Generate one batch of data"""
        current_batch = self.__mix_regions()

        # Generate data
        X, y = self.__data_generation(current_batch)

        return X, y

    def __get_ROIPool(self):
        """
        Passes the attributes to the ROIPool class to build a pool of regions of interest

        :return: Initializes the object used to generate batches of peak centered training examples.
        """
        return ROIPool(meta_dataframe=self.meta_dataframe,
                       chromosomes=self.chromosomes,
                       chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                       blacklist=self.blacklist,
                       region_length=self.region_length,
                       peak_paths=self.peak_paths
                       )

    def __data_generation(self, current_batch):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param current_batch: Current batch of regions

        :return: A batch of training examples centered on regions of interest
        """
        inputs_batch = []

        for region in current_batch:
            with \
                    load_bigwig(self.average) as average_stream, \
                    load_2bit(self.sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:
                inputs_batch.append(get_input_matrix(rows=self.input_channels,
                                                     cols=self.region_length,
                                                     bp_order=["A", "C", "G", "T"],
                                                     signal_stream=signal_stream,
                                                     average_stream=average_stream,
                                                     sequence_stream=sequence_stream,
                                                     chromosome=region[0],
                                                     start=region[1],
                                                     end=region[2]
                                                     )
                                    )

            return np.array(inputs_batch)


class ROIPool(object):
    """
    This class will generate a pool of examples based on regions of interest defined by ATAC-seq and ChIP-seq peaks

    The RandomRegionsGenerator will generate a random region of interest based on the input reference genome
    chromosome sizes, the desired size of the chromosome pool, the input length, and the method used to calculate the
    frequency of chromosome examples.
    """

    def __init__(self,
                 meta_dataframe,
                 chromosomes,
                 chromosome_sizes_dictionary,
                 blacklist,
                 region_length,
                 regions
                 ):
        """
        :param meta_dataframe: Path to the meta file
        :param chromosomes: List of chromosomes to use
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes
        :param blacklist: The blacklist file of BED regions to exclude
        :param peak_paths: List of paths to ATAC and ChIP-seq peaks
        :param region_length: Length of the input regions
        """
        self.meta_dataframe = meta_dataframe
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length
        self.regions = regions

        self.roi_pool = self.__get_roi_pool()

        self.roi_size = self.roi_pool.shape[0]

    def __get_roi_pool(self):
        """
        Build a ROI pool from all of the peak files of interest

        :return: A dataframe of BED regions
        """
        return import_bed(self.regions,
                          region_length=self.region_length,
                          chromosomes=self.chromosomes,
                          chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                          blacklist=self.blacklist)

    def get_regions_list(self,
                         n_roi):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param n_roi: Number of regions to generate per batch

        :return: A batch of training examples centered on regions of interest
        """
        random_roi_pool = self.roi_pool.sample(n=n_roi, replace=True, random_state=1)

        return random_roi_pool.to_numpy().tolist()
