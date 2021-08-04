import random
import sys
from os import path

import tensorflow as tf
import numpy as np
import pandas as pd
from Bio.Seq import Seq
import threading

from maxatac.architectures.dcnn import get_dilated_cnn
from maxatac.architectures.multi_modal_models import MM_DCNN_V2
from maxatac.architectures.res_dcnn import get_res_dcnn
from maxatac.utilities.constants import BP_RESOLUTION, BATCH_SIZE, CHR_POOL_SIZE, INPUT_LENGTH, INPUT_CHANNELS, \
    BP_ORDER, TRAIN_SCALE_SIGNAL, BLACKLISTED_REGIONS, DEFAULT_CHROM_SIZES
from maxatac.utilities.genome_tools import load_bigwig, load_2bit, get_one_hot_encoded, build_chrom_sizes_dict
from maxatac.utilities.roi_tools import GenomicRegions
from maxatac.utilities.session import configure_session
from maxatac.utilities.system_tools import get_dir, remove_tags, replace_extension


class MaxATACModel(object):
    """
    This object will organize the input model parameters and initialize the maxATAC model
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
                 quant=False,
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
        :param quant: Whether to perform quantitative predictions
        :param output_activation: The activation function to use in the output layer
        :param target_scale_factor: The scale factor to use for quantitative data
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
        self.quant = quant
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
                                   quant=self.quant,
                                   target_scale_factor=self.target_scale_factor,
                                   dense_b=self.dense,
                                   weights=self.weights
                                   )

        elif self.arch == "RES_DCNN_V2":
            return get_res_dcnn(output_activation=self.output_activation,
                                weights=self.weights,
                                quant=self.quant,
                                target_scale_factor=self.target_scale_factor,
                                dense_b=self.dense
                                )

        elif self.arch == "MM_DCNN_V2":
            return MM_DCNN_V2(output_activation=self.output_activation,
                              weights=self.weights,
                              quant=self.quant,
                              res_conn=False,
                              target_scale_factor=self.target_scale_factor,
                              dense_b=self.dense
                              )

        elif self.arch == "MM_Res_DCNN_V2":
            return MM_DCNN_V2(output_activation=self.output_activation,
                              weights=self.weights,
                              quant=self.quant,
                              res_conn=True,
                              target_scale_factor=self.target_scale_factor,
                              dense_b=self.dense
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
        quant=False,
        target_scale_factor=1,
        batch_size=BATCH_SIZE,
        shuffle_cell_type=False

):
    """
    Initiate a generator

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
    :param quant: Whether to use quantitative predictions
    :param target_scale_factor: Scaling factor to use for scaling target values (quantitative specific)
    :param batch_size: The number of examples to use per batch of training
    :param shuffle_cell_type: Shuffle the ROI cell type labels if True

    :return A generator that will yield a batch with number of examples equal to batch size

    """
    # Calculate the number of ROIs to use based on the total batch size and proportion of random regions to use
    n_roi = round(batch_size * (1. - rand_ratio))

    # Calculate number of random regions to use each batch
    n_rand = round(batch_size - n_roi)

    # Generate the training random regions pool
    train_random_regions_pool = RandomRegionsPool(chroms=chroms,
                                                  chrom_pool_size=CHR_POOL_SIZE,
                                                  region_length=INPUT_LENGTH,
                                                  preferences=False  # can be None
                                                  )

    # Initialize the ROI generator
    roi_gen = create_roi_batch(sequence=sequence,
                               meta_table=meta_table,
                               roi_pool=roi_pool,
                               n_roi=n_roi,
                               cell_type_list=cell_type_list,
                               bp_resolution=bp_resolution,
                               quant=quant,
                               target_scale_factor=target_scale_factor,
                               shuffle_cell_type=shuffle_cell_type
                               )

    # Initialize the random regions generator
    rand_gen = create_random_batch(sequence=sequence,
                                   meta_table=meta_table,
                                   cell_type_list=cell_type_list,
                                   n_rand=n_rand,
                                   regions_pool=train_random_regions_pool,
                                   bp_resolution=bp_resolution,
                                   quant=quant,
                                   target_scale_factor=target_scale_factor
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

        yield inputs_batch, targets_batch # change to yield


def get_input_matrix(rows,
                     cols,
                     signal_stream,
                     sequence_stream,
                     bp_order,
                     chromosome,
                     start,  # end - start = cols
                     end,
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
    :param chrom: chromosome
    :param start: start
    :param end: end
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
                     quant=False,
                     target_scale_factor=1,
                     shuffle_cell_type=False
                     ):
    """
    Create a batch of examples from regions of interest

    :param sequence:
    :param meta_table:
    :param roi_pool:
    :param n_roi:
    :param cell_type_list:
    :param bp_resolution:
    :param quant:
    :param target_scale_factor:
    :param shuffle_cell_type:

    :return: np.array(inputs_batch), np.array(targets_batch
    """
    while True:
        inputs_batch, targets_batch = [], []
        roi_size = roi_pool.shape[0]
        

        curr_batch_idxs = random.sample(range(roi_size), n_roi)

        # Here I will process by row, if performance is bad then process by cell line
        for row_idx in curr_batch_idxs:
            roi_row = roi_pool.iloc[row_idx, :]

            if shuffle_cell_type:
                cell_line = random.choice(cell_type_list)

            else:
                cell_line = roi_row['Cell_Line']

            chrom_name = roi_row['Chr']

            start = int(roi_row['Start'])
            end = int(roi_row['Stop'])

            meta_row = meta_table[(meta_table['Cell_Line'] == cell_line)]
            meta_row = meta_row.reset_index(drop=True)

            signal = meta_row.loc[0, 'ATAC_Signal_File']
            binding = meta_row.loc[0, 'Binding_File']
            

            with \
                    load_2bit(sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:
                
                rev_comp = random.choice([True, False])

                input_matrix = get_input_matrix(rows=INPUT_CHANNELS,
                                                cols=INPUT_LENGTH,
                                                bp_order=BP_ORDER,
                                                signal_stream=signal_stream,
                                                sequence_stream=sequence_stream,
                                                chromosome=chrom_name,
                                                start=start,
                                                end=end,
                                                use_complement=rev_comp,
                                                reverse_matrix=rev_comp
                                                )
                
                inputs_batch.append(input_matrix)

                # TODO we might want to test what happens if we change the
                if not quant:
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    if rev_comp:
                        target_vector = target_vector[::-1]
                    

                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
                    targets_batch.append(bin_vector)

                else:
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_vector = np.mean(split_targets, axis=1)  # Perhaps we can change np.mean to np.median.
                    targets_batch.append(bin_vector)

        if quant:
            targets_batch = np.array(targets_batch)
            targets_batch = targets_batch * target_scale_factor

        yield np.array(inputs_batch), np.array(targets_batch) #change to yield


def create_random_batch(
        sequence,
        meta_table,
        cell_type_list,
        n_rand,
        regions_pool,
        bp_resolution=1,
        quant=False,
        target_scale_factor=1
):
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
                
                rev_comp = random.choice([True, False])

                input_matrix = get_input_matrix(rows=INPUT_CHANNELS,
                                                cols=INPUT_LENGTH,
                                                bp_order=BP_ORDER,
                                                signal_stream=signal_stream,
                                                sequence_stream=sequence_stream,
                                                chromosome=chrom_name,
                                                start=seq_start,
                                                end=seq_end,
                                                use_complement=rev_comp,
                                                reverse_matrix=rev_comp
                                                )

                inputs_batch.append(input_matrix)

                if not quant:
                    target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    
                    if rev_comp:
                        target_vector = target_vector[::-1]
                    
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
                    targets_batch.append(bin_vector)

                else:
                    target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_vector = np.mean(split_targets,
                                         axis=1)  # Perhaps we can change np.mean to np.median. Something to think about.
                    targets_batch.append(bin_vector)

        if quant:
            targets_batch = np.array(targets_batch)
            targets_batch = targets_batch * target_scale_factor

        yield np.array(inputs_batch), np.array(targets_batch) # change to yield


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

        chroms = {chrom_name: chrom_data for chrom_name, chrom_data in self.chroms.items()}

        sum_lengths = sum(map(lambda v: v["length"], chroms.values()))

        frequencies = {
            chrom_name: round(chrom_data["length"] / sum_lengths * self.chrom_pool_size)

            for chrom_name, chrom_data in chroms.items()
        }
        labels = []

        for k, v in frequencies.items():
            labels += [(k, chroms[k]["length"])] * v

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
            self.ROI_pool = self.__import_roi_pool__(filepath=self.roi_file_path,
                                                     chroms=self.chroms,
                                                     shuffle=shuffle
                                                     )
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

        The roi DF is read in from a TSV file that is formatted similarly as a BED file with a header. The following columns
        are required:

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