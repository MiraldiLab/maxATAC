import random
import sys
from os import path

import tensorflow as tf
import numpy as np
import pandas as pd
from Bio.Seq import Seq
import threading
import pybedtools
import os
import glob

from maxatac.architectures.dcnn import get_dilated_cnn
from maxatac.utilities.constants import BP_RESOLUTION, BATCH_SIZE, CHR_POOL_SIZE, INPUT_LENGTH, INPUT_CHANNELS, \
    BP_ORDER, TRAIN_SCALE_SIGNAL, BLACKLISTED_REGIONS, DEFAULT_CHROM_SIZES
from maxatac.utilities.genome_tools import load_bigwig, load_2bit, get_one_hot_encoded, build_chrom_sizes_dict
from maxatac.utilities.system_tools import get_dir, remove_tags, replace_extension


class MaxATACModel(object):
    """
    This object will organize the input model parameters and initialize the maxATAC model

    The methods are:

    __get_interpretation_attributes: This will import the interpretation inputs if interpretation module is being used.

    __get_model: This will get the correct architecture and parameters based on the user input
    """

    def __init__(self,
                 arch,
                 seed,
                 output_directory,
                 prefix,
                 threads,
                 meta_path,
                 weights,
                 dense=False,
                 target_scale_factor=TRAIN_SCALE_SIGNAL,
                 output_activation="sigmoid",
                 interpret=False,
                 interpret_cell_type=""
                 ):
        """
        Initialize the maxATAC model with the input parameters and architecture

        :param arch: Neural network architecture to use: DCNN, resNet, UNet, multi-modal
        :param seed: Random seed to use
        :param output_directory: Path to output directory
        :param prefix: Prefix to use for filename
        :param threads: Number of threads to use
        :param meta_path: Path to the meta file associated with the run
        :param output_activation: The activation function to use in the output layer
        :param dense: Whether to use a dense layer on output
        :param weights: Input weights to use for model
        :param interpret: Boolean for whether this is training or interpretation
        """
        self.arch = arch
        self.seed = seed
        self.output_directory = get_dir(output_directory)
        self.model_filename = prefix + "_{epoch}" + ".h5"
        self.results_location = path.join(self.output_directory, self.model_filename)
        self.log_location = replace_extension(remove_tags(self.results_location, "_{epoch}"), ".csv")
        self.tensor_board_log_dir = get_dir(path.join(self.output_directory, "tensorboard"))
        self.threads = threads
        self.training_history = ""
        self.meta_path = meta_path
        self.output_activation = output_activation
        self.dense = dense
        self.weights = weights
        self.target_scale_factor = target_scale_factor

        # Set the random seed for the model
        random.seed(seed)

        # Import meta txt as dataframe
        self.meta_dataframe = pd.read_csv(self.meta_path, sep='\t', header=0, index_col=None)

        # Find the unique number of cell types in the meta file
        self.cell_types = self.meta_dataframe["Cell_Line"].unique().tolist()

        self.train_tf = self.meta_dataframe["TF"].unique()[0]

        self.nn_model = self.__get_model()

        if interpret:
            assert (interpret_cell_type is not None, "Set the interpretation cell type argument")
            self.interpret_cell_type = interpret_cell_type
            self.__get_interpretation_attributes()

    def __get_interpretation_attributes(self):
        self.interpret_location = get_dir(path.join(self.output_directory, 'interpret'))
        self.metacluster_patterns_location = get_dir(path.join(self.interpret_location, 'metacluster_patterns'))
        self.meme_query_pattern_location = get_dir(path.join(self.interpret_location, 'meme_query'))
        self.interpret_model_file = path.join(self.interpret_location, 'tmp.model')

    def __get_model(self):
        # Get the neural network model based on the specified model architecture
        if self.arch == "DCNN_V2":
            return get_dilated_cnn(output_activation=self.output_activation,
                                   target_scale_factor=self.target_scale_factor,
                                   dense_b=self.dense,
                                   weights=self.weights
                                   )
        else:
            sys.exit("Model Architecture not specified correctly. Please check")


def DataGenerator(
        sequence,
        meta_table,
        roi_pool,
        cell_type_list,
        rand_ratio,
        chroms,
        bp_resolution=BP_RESOLUTION,
        target_scale_factor=1,
        batch_size=BATCH_SIZE,
        shuffle_cell_type=False,
        rev_comp_train=False

):
    """
    Initiate a data generator that will yield a batch of examples for training. This generator will mix samples from a
    pool of random regions and a pool of regions of interest based on the user defined ratio. The examples will be
    returned as a list of numpy arrays.

    _________________
    Workflow Overview

    1) Create the random regions pool
    2) Create the roi generator
    3) Create the random regions generator
    4) Combine the roi  and random regions batches according to the rand_ratio value

    :param sequence: The input 2bit DNA sequence
    :param meta_table: The run meta table with locations to ATAC and ChIP-seq data
    :param roi_pool: The pool of regions to use centered on peaks
    :param cell_type_list: The training cell lines to use
    :param rand_ratio: The number of random examples to use per batch
    :param chroms: The training chromosomes
    :param bp_resolution: The resolution of the predictions to use
    :param batch_size: The number of examples to use per batch of training
    :param shuffle_cell_type: Shuffle the ROI cell type labels if True
    :param rev_comp_train: use the reverse complement to train

    :return A generator that will yield a batch with number of examples equal to batch size

    """
    # Calculate the number of ROIs to use based on the total batch size and proportion of random regions to use
    n_roi = round(batch_size * (1. - rand_ratio))

    # Calculate number of random regions to use each batch
    n_rand = round(batch_size - n_roi)

    if n_rand > 0:
        # Generate the training random regions pool
        train_random_regions_pool = RandomRegionsPool(chroms=build_chrom_sizes_dict(chroms, DEFAULT_CHROM_SIZES),
                                                    chrom_pool_size=CHR_POOL_SIZE,
                                                    region_length=INPUT_LENGTH,
                                                    preferences=False  # can be None
                                                    )

        # Initialize the random regions generator
        rand_gen = create_random_batch(sequence=sequence,
                                       meta_table=meta_table,
                                       cell_type_list=cell_type_list,
                                       n_rand=n_rand,
                                       regions_pool=train_random_regions_pool,
                                       bp_resolution=bp_resolution,
                                       target_scale_factor=target_scale_factor,
                                       rev_comp_train=rev_comp_train
                                       )

    # Initialize the ROI generator
    roi_gen = create_roi_batch(sequence=sequence,
                               meta_table=meta_table,
                               roi_pool=roi_pool,
                               n_roi=n_roi,
                               cell_type_list=cell_type_list,
                               bp_resolution=bp_resolution,
                               target_scale_factor=target_scale_factor,
                               shuffle_cell_type=shuffle_cell_type,
                               rev_comp_train=rev_comp_train
                               )

    while True:
        # roi_batch.shape = (n_samples, 1024, 6)
        if 0. < rand_ratio < 1.:
            roi_input_batch, roi_target_batch = next(roi_gen)
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = np.concatenate((roi_input_batch, rand_input_batch), axis=0)
            targets_batch = np.concatenate((roi_target_batch, rand_target_batch), axis=0)

        elif rand_ratio == 1.:
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = rand_input_batch
            targets_batch = rand_target_batch

        else:
            roi_input_batch, roi_target_batch = next(roi_gen)
            inputs_batch = roi_input_batch
            targets_batch = roi_target_batch

        yield inputs_batch, targets_batch  # change to yield


def get_input_matrix(signal_stream,
                     sequence_stream,
                     chromosome,
                     start,  # end - start = cols
                     end,
                     rows=INPUT_CHANNELS,
                     cols=INPUT_LENGTH,
                     bp_order=BP_ORDER,
                     use_complement=False,
                     reverse_matrix=False
                     ):
    """
    Get a matrix of values from the corresponding genomic position. You can supply whether you want to use the
    complement sequence. You can also choose whether you want to reverse the whole matrix.

    :param rows: Number of rows == channels
    :param cols: Number of cols == region length
    :param signal_stream: Signal bigwig stream
    :param sequence_stream: 2bit DNA sequence stream
    :param bp_order: BP order
    :param chromosome: chromosome
    :param start: start
    :param end: end
    :param use_complement: use complement strand for training
    :param reverse_matrix: reverse the input matrix

    :return: a matrix (rows x cols) of values from the input bigwig files
    """

    input_matrix = np.zeros((rows, cols))
    for n, bp in enumerate(bp_order):
        # Get the sequence from the interval of interest
        target_sequence = Seq(sequence_stream.sequence(chromosome, start, end))

        if use_complement:
            # Get the complement of the sequence
            target_sequence = target_sequence.complement()

        # Get the one hot encoded sequence
        input_matrix[n, :] = get_one_hot_encoded(target_sequence, bp)

    signal_array = np.array(signal_stream.values(chromosome, start, end))

    input_matrix[4, :] = signal_array

    # If reverse_matrix then reverse the matrix. This changes the left to right orientation.
    if reverse_matrix:
        input_matrix = input_matrix[::-1]

    return input_matrix.T


def create_roi_batch(sequence,
                     meta_table,
                     roi_pool,
                     n_roi,
                     cell_type_list,
                     bp_resolution=1,
                     target_scale_factor=1,
                     shuffle_cell_type=False,
                     rev_comp_train=False
                     ):
    """
    Create a batch of examples from regions of interest. The batch size is defined by n_roi. This code will randomly
    generate a batch of examples based on an input meta file that defines the paths to training data. The cell_type_list
    is used to randomly select the cell type that the training signal is drawn from.

    :param sequence: The input 2bit DNA sequence
    :param meta_table: The meta file that contains the paths to signal and peak files
    :param roi_pool: The pool of regions that we want to sample from
    :param n_roi: The number of regions that go into each batch
    :param cell_type_list: A list of unique training cell types
    :param bp_resolution: The resolution of the output bins. i.e. 32 bp
    :param shuffle_cell_type: Whether to shuffle cell types during training
    :param rev_comp_train: use reverse complement for training

    :return: np.array(inputs_batch), np.array(targets_batch)
    """
    while True:
        # Create empty lists that will hold the signal tracks
        inputs_batch, targets_batch = [], []

        # Get the shape of the ROI pool
        roi_size = roi_pool.shape[0]

        # Randomly select n regions from the pool
        curr_batch_idxs = random.sample(range(roi_size), n_roi)

        # Extract the signal for every sample
        for row_idx in curr_batch_idxs:
            roi_row = roi_pool.iloc[row_idx, :]

            # If shuffle_cell_type the cell type will be randomly chosen
            if shuffle_cell_type:
                cell_line = random.choice(cell_type_list)

            else:
                cell_line = roi_row['Cell_Line']

            # Get the paths for the cell type of interest.
            meta_row = meta_table[(meta_table['Cell_Line'] == cell_line)]
            meta_row = meta_row.reset_index(drop=True)

            # Rename some variables. This just helps clean up code downstream
            chrom_name = roi_row['Chr']
            start = int(roi_row['Start'])
            end = int(roi_row['Stop'])

            signal = meta_row.loc[0, 'ATAC_Signal_File']
            binding = meta_row.loc[0, 'Binding_File']

            # Choose whether to use the reverse complement of the region
            if rev_comp_train:
                rev_comp = random.choice([True, False])

            else:
                rev_comp = False

            with \
                    load_2bit(sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:

                # Get the input matrix of values and one-hot encoded sequence
                input_matrix = get_input_matrix(signal_stream=signal_stream,
                                                sequence_stream=sequence_stream,
                                                chromosome=chrom_name,
                                                start=start,
                                                end=end,
                                                use_complement=rev_comp,
                                                reverse_matrix=rev_comp
                                                )

                # Append the sample to the inputs batch.
                inputs_batch.append(input_matrix)

                # Some bigwig files do not have signal for some chromosomes because they do not have peaks
                # in those regions
                # Our workaround for issue#42 is to provide a zero matrix for that position
                try:
                    # Get the target matrix
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T

                except:
                    # TODO change length of array
                    target_vector = np.zeros(1024)

                # change nan to numbers
                target_vector = np.nan_to_num(target_vector, 0.0)

                # If reverse compliment, reverse the matrix
                if rev_comp:
                    target_vector = target_vector[::-1]

                # get the number of 32 bp bins across the input sequence
                n_bins = int(target_vector.shape[0] / bp_resolution)

                # Split the data up into 32 x 32 bp bins.
                split_targets = np.array(np.split(target_vector, n_bins, axis=0))

                # TODO we might want to test what happens if we change the
                bin_sums = np.sum(split_targets, axis=1)
                bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)

                # Append the sample to the target batch
                targets_batch.append(bin_vector)

        yield np.array(inputs_batch), np.array(targets_batch)  # change to yield


def create_random_batch(
        sequence,
        meta_table,
        cell_type_list,
        n_rand,
        regions_pool,
        bp_resolution=1,
        target_scale_factor=1,
        rev_comp_train=False
):
    """
    This function will create a batch of examples that are randomly generated. This batch of data is created the same
    as the roi batches.
    """
    while True:
        inputs_batch, targets_batch = [], []

        for idx in range(n_rand):
            cell_line = random.choice(cell_type_list)  # Randomly select a cell line

            chrom_name, seq_start, seq_end = regions_pool.get_region()  # returns random region (chrom_name, start, end)

            meta_row = meta_table[(meta_table['Cell_Line'] == cell_line)]  # get meta row for selected cell line
            meta_row = meta_row.reset_index(drop=True)

            signal = meta_row.loc[0, 'ATAC_Signal_File']
            binding = meta_row.loc[0, 'Binding_File']

            with \
                    load_2bit(sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:

                if rev_comp_train:
                    rev_comp = random.choice([True, False])

                else:
                    rev_comp = False

                input_matrix = get_input_matrix(signal_stream=signal_stream,
                                                sequence_stream=sequence_stream,
                                                chromosome=chrom_name,
                                                start=seq_start,
                                                end=seq_end,
                                                use_complement=rev_comp,
                                                reverse_matrix=rev_comp
                                                )

                inputs_batch.append(input_matrix)

                try:
                    # Get the target matrix
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T

                except:
                    # TODO change length of array
                    target_vector = np.zeros(1024)

                target_vector = np.nan_to_num(target_vector, 0.0)

                if rev_comp:
                    target_vector = target_vector[::-1]

                n_bins = int(target_vector.shape[0] / bp_resolution)
                split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                bin_sums = np.sum(split_targets, axis=1)
                bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
                targets_batch.append(bin_vector)


        yield np.array(inputs_batch), np.array(targets_batch)  # change to yield


class RandomRegionsPool:
    """
    Generate a pool of random genomic regions
    """

    def __init__(
            self,
            chroms,  # in a form of {"chr1": {"length": 249250621, "region": [0, 249250621]}}, "region" is ignored
            chrom_pool_size,
            region_length,
            preferences=None  # bigBed file with ranges to limit random regions selection
    ):

        self.chroms = chroms
        self.chrom_pool_size = chrom_pool_size
        self.region_length = region_length
        self.preferences = preferences

        # self.preference_pool = self.__get_preference_pool()  # should be run before self.__get_chrom_pool()
        self.preference_pool = False

        self.chrom_pool = self.__get_chrom_pool()
        # self.chrom_pool_size is updated to ensure compatibility between HG19 and HG38
        self.chrom_pool_size = min(chrom_pool_size, len(self.chrom_pool))

        self.__idx = 0

    def get_region(self):

        if self.__idx == self.chrom_pool_size:
            random.shuffle(self.chrom_pool)

            self.__idx = 0

        chrom_name, chrom_length = self.chrom_pool[self.__idx]

        self.__idx += 1

        if self.preference_pool:
            preference = random.sample(self.preference_pool[chrom_name], 1)[0]

            start = round(random.randint(preference[0], preference[1] - self.region_length))

        else:
            start = round(random.randint(0, chrom_length - self.region_length))

        end = start + self.region_length

        return chrom_name, start, end

    def __get_preference_pool(self):
        preference_pool = {}

        if self.preferences is not None:
            with load_bigwig(self.preferences) as input_stream:
                for chrom_name, chrom_data in self.chroms.items():
                    for entry in input_stream.entries(chrom_name, 0, chrom_data["length"], withString=False):
                        if entry[1] - entry[0] < self.region_length:
                            continue

                        preference_pool.setdefault(chrom_name, []).append(list(entry[0:2]))

        return preference_pool

    def __get_chrom_pool(self):
        """
        TODO: rewrite to produce exactly the same number of items
        as chrom_pool_size regardless of length(chroms) and
        chrom_pool_size
        """

        sum_lengths = sum(self.chroms.values())

        frequencies = {
            chrom_name: round(chrom_length / sum_lengths * self.chrom_pool_size)

            for chrom_name, chrom_length in self.chroms.items()
        }
        labels = []

        for k, v in frequencies.items():
            labels += [(k, self.chroms[k])] * v

        random.shuffle(labels)

        return labels


class ROIPool(object):
    """
    Import genomic regions of interest for training
    """

    def __init__(self,
                 chroms,
                 roi_file_path,
                 meta_file,
                 prefix,
                 output_directory,
                 shuffle,
                 tag
                 ):
        """
        :param chroms: Chromosomes to limit the analysis to
        :param roi_file_path: User provided ROI file path
        :param meta_file: path to meta file
        :param prefix: Prefix for saving output file
        :param output_directory: Output directory to save files to
        :param shuffle: Whether to shuffle the input ROI file
        :param tag: Tag to use for writing the file.
        """
        self.chroms = chroms
        self.roi_file_path = roi_file_path
        self.meta_file = meta_file
        self.prefix = prefix
        self.output_directory = output_directory
        self.tag = tag

        # If an ROI path is provided import it as the ROI pool
        if self.roi_file_path:
            self.ROI_pool = self.__import_roi_pool__(shuffle=shuffle)

        # Import the data from the meta file.
        else:
            regions = GenomicRegions(meta_path=self.meta_file,
                                     region_length=1024,
                                     chromosomes=self.chroms,
                                     chromosome_sizes_dictionary=build_chrom_sizes_dict(self.chroms,
                                                                                        DEFAULT_CHROM_SIZES),
                                     blacklist=BLACKLISTED_REGIONS)

            regions.write_data(self.prefix,
                               output_dir=self.output_directory,
                               set_tag=tag)

            self.ROI_pool = regions.combined_pool

    def __import_roi_pool__(self, shuffle=False):
        """
        Import the ROI file containing the regions of interest. This file is similar to a bed file, but with a header

        The roi DF is read in from a TSV file that is formatted similarly as a BED file with a header. The following
        columns are required:

        Chr | Start | Stop | ROI_Type | Cell_Line

        The chroms list is used to filter the ROI df to make sure that only training chromosomes are included.

        :param shuffle: Whether to shuffle the dataframe upon import

        :return: A pool of regions to use for training or validation
        """
        roi_df = pd.read_csv(self.roi_file_path, sep="\t", header=0, index_col=None)

        roi_df = roi_df[roi_df['Chr'].isin(self.chroms)]

        if shuffle:
            roi_df = roi_df.sample(frac=1)

        return roi_df


class SeqDataGenerator(tf.keras.utils.Sequence):
    # ‘Generates data for Keras’

    def __init__(self, batches, generator):
        # ‘Initialization’
        self.batches = batches
        self.generator = generator

    def __len__(self):
        # ‘Denotes the number of batches per epoch’
        return self.batches

    def __getitem__(self, index):
        # ‘Generate one batch of data’
        # Generate indexes of the batch
        # Generate data
        return next(self.generator)


def model_selection(training_history, output_dir):
    """
    This function will take the training history and output the best model based on the dice coefficient value.
    """
    # Create a dataframe from the history object
    df = pd.DataFrame(training_history.history)

    epoch = df['val_dice_coef'].idxmax() + 1

    # Get the realpath to the best model
    out = pd.DataFrame([glob.glob(output_dir + "/*" + str(epoch) + ".h5")], columns=['Best_Model_Path'])

    # Write the location of the best model to a file
    out.to_csv(output_dir + "/" + "best_epoch.txt", sep='\t', index=None, header=None)

    return epoch


class GenomicRegions(object):
    """
    This class will generate a pool of examples based on regions of interest defined by ATAC-seq and ChIP-seq peaks.
    """

    def __init__(self,
                 meta_path,
                 chromosomes,
                 chromosome_sizes_dictionary,
                 blacklist,
                 region_length
                 ):
        """
        When the object is initialized it will import all of the peaks in the meta files and parse them into training
        and validation regions of interest. These will be output in the form of TSV formatted file similar to a BED
        file.

        :param meta_path: Path to the meta file
        :param chromosomes: List of chromosomes to use
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes
        :param blacklist: The blacklist file of BED regions to exclude
        :param region_length: Length of the input regions
        """
        self.meta_path = meta_path
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length

        # Import meta txt as dataframe
        self.meta_dataframe = pd.read_csv(self.meta_path, sep='\t', header=0, index_col=None)
        # Select Training Cell lines
        self.meta_dataframe = self.meta_dataframe[self.meta_dataframe["Train_Test_Label"] == 'Train']

        # Get a dictionary of {Cell Types: Peak Paths}
        self.atac_dictionary = pd.Series(self.meta_dataframe.ATAC_Peaks.values,
                                         index=self.meta_dataframe.Cell_Line).to_dict()

        self.chip_dictionary = pd.Series(self.meta_dataframe.CHIP_Peaks.values,
                                         index=self.meta_dataframe.Cell_Line).to_dict()

        # You must generate the ROI pool before you can get the final shape
        self.atac_roi_pool = self.__get_roi_pool(self.atac_dictionary, "ATAC", )
        self.chip_roi_pool = self.__get_roi_pool(self.chip_dictionary, "CHIP")

        self.combined_pool = pd.concat([self.atac_roi_pool, self.chip_roi_pool])

        self.atac_roi_size = self.atac_roi_pool.shape[0]
        self.chip_roi_size = self.chip_roi_pool.shape[0]

    def __get_roi_pool(self, dictionary, roi_type_tag):
        """
        Build a pool of regions of interest from BED files.

        :param dictionary: A dictionary of Cell Types and their associated BED files
        :param roi_type_tag: Tag used to name the type of ROI being generated. IE Chip or ATAC

        :return: A dataframe of BED regions that are formatted for maxATAC training.
        """
        bed_list = []

        for roi_cell_tag, bed_file in dictionary.items():
            bed_list.append(self.__import_bed(bed_file,
                                              ROI_type_tag=roi_type_tag,
                                              ROI_cell_tag=roi_cell_tag))

        return pd.concat(bed_list)

    def write_data(self, prefix="ROI_pool", output_dir="./ROI", set_tag="training"):
        """
        Write the ROI dataframe to a tsv and a bed for for ATAC, CHIP, and combined ROIs

        :param set_tag: Tag for training or validation
        :param prefix: Prefix for filenames to use
        :param output_dir: Directory to output the bed and tsv files

        :return: Write BED and TSV versions of the ROI data
        """
        output_directory = get_dir(output_dir)

        combined_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ROI.bed.gz")

        stats_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ROI_stats")
        total_regions_stats_filename = os.path.join(output_directory,
                                                    prefix + "_" + set_tag + "_ROI_totalregions_stats")

        self.combined_pool.to_csv(combined_BED_filename, sep="\t", index=False, header=False)

        group_ms = self.combined_pool.groupby(["Chr", "Cell_Line", "ROI_Type"], as_index=False).size()
        len_ms = self.combined_pool.shape[0]
        group_ms.to_csv(stats_filename, sep="\t", index=False)

        file = open(total_regions_stats_filename, "a")
        file.write('Total number of regions found for ' + set_tag + ' are: {0}\n'.format(len_ms))
        file.close()

    def get_regions_list(self,
                         n_roi):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param n_roi: Number of regions to generate per batch

        :return: A batch of training examples centered on regions of interest
        """
        random_roi_pool = self.combined_pool.sample(n=n_roi, replace=True, random_state=1)

        return random_roi_pool.to_numpy().tolist()

    def __import_bed(self,
                     bed_file,
                     ROI_type_tag,
                     ROI_cell_tag):
        """
        Import a BED file and format the regions to be compatible with our maxATAC models

        :param bed_file: Input BED file to format
        :param ROI_type_tag: Tag to use in the description column
        :param ROI_cell_tag: Tag to use in the description column

        :return: A dataframe of BED regions compatible with our model
        """
        # Import dataframe
        df = pd.read_csv(bed_file,
                         sep="\t",
                         usecols=[0, 1, 2],
                         header=None,
                         names=["Chr", "Start", "Stop"],
                         low_memory=False)

        # Make sure the chromosomes in the ROI file frame are in the target chromosome list
        df = df[df["Chr"].isin(self.chromosomes)]

        # Find the length of the regions
        df["length"] = df["Stop"] - df["Start"]

        # Find the center of each peak.
        # We might want to use bedtools to window the regions of interest around the peak.
        df["center"] = np.floor(df["Start"] + (df["length"] / 2)).apply(int)

        # The start of the interval will be the center minus 1/2 the desired region length.
        df["Start"] = np.floor(df["center"] - (self.region_length / 2)).apply(int)

        # the end of the interval will be the center plus 1/2 the desired region length
        df["Stop"] = np.floor(df["center"] + (self.region_length / 2)).apply(int)

        # The chromosome end is defined as the chromosome length
        df["END"] = df["Chr"].map(self.chromosome_sizes_dictionary)

        # Make sure the stop is less than the end
        df = df[df["Stop"].apply(int) < df["END"].apply(int)]

        # Make sure the start is greater than the chromosome start of 0
        df = df[df["Start"].apply(int) > 0]

        # Select for the first three columns to clean up
        df = df[["Chr", "Start", "Stop"]]

        # Import the dataframe as a pybedtools object so we can remove the blacklist
        BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

        # Import the blacklist as a pybedtools object
        blacklist_bedtool = pybedtools.BedTool(self.blacklist)

        # Find the intervals that do not intersect blacklisted regions.
        blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

        # Convert the pybedtools object to a pandas dataframe.
        df = blacklisted_df.to_dataframe()

        # Rename the columns
        df.columns = ["Chr", "Start", "Stop"]

        df["ROI_Type"] = ROI_type_tag

        df["Cell_Line"] = ROI_cell_tag

        return df
