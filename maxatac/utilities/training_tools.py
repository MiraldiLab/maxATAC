import random
import sys
from os import path
import os
import keras
import numpy as np
import pandas as pd

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

        configure_session(1)

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

    #Here we need to specify how many samples are coming from each Set
    
    #Region1    (50%)   = Set1: True Positives -- Cell Type Regions that have a ChIP-seq Peak
    #Region2    (15%)   = Set2: True Negatives -- For the TP regions, if that region is not a ChIP peak in a different CT included in this bin
    #Region3    (25%)   = Set3: ATAC Only: -- Removing the first two Sets, all regions with an ATAC peak and no ChIP peak
    #Region4    (10%)   = Set4: Random: -- Regions that are not a TP
    
    
    # Calculate the number of ROIs to use based on the total batch size and proportion of Regions to use
    reg1_ratio= 0.50
    reg2_ratio= 0.15
    reg3_ratio= 0.25
    reg4_ratio= 0.10
    
    n_reg1 = round(batch_size * reg1_ratio)
    n_reg2 = round(batch_size * reg2_ratio)
    n_reg3 = round(batch_size * reg3_ratio)
    n_reg4 = round(batch_size * reg4_ratio)
    n_roi = [n_reg1, n_reg2, n_reg3, n_reg4]
    
    # Calculate the number of ROIs to use based on the total batch size and proportion of random regions to use
    N_roi = round(batch_size * (1. - rand_ratio))

    # Calculate number of random regions to use each batch
    n_rand = round(batch_size - N_roi)

    # Generate the training random regions pool
    train_random_regions_pool = RandomRegionsPool(chroms=chroms,
                                                  chrom_pool_size=CHR_POOL_SIZE,
                                                  region_length=INPUT_LENGTH,
                                                  preferences=False  # can be None
                                                  )

    # Initialize the ROI generator
    roi_gen = create_roi_batch(sequence=sequence,
                               meta_table=meta_table,
                               roi_pool=roi_pool, # should contain regions ## roi_pool.regions1 etc
                               n_roi=n_roi,
                               cell_type_list=cell_type_list,
                               bp_resolution=bp_resolution,
                               quant=quant,
                               target_scale_factor=target_scale_factor,
                               shuffle_cell_type=shuffle_cell_type
                               )

    while True:
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

        yield inputs_batch, targets_batch #change back to yield


def get_input_matrix(rows,
                     cols,
                     signal_stream,
                     sequence_stream,
                     bp_order,
                     chromosome,
                     start,  # end - start = cols
                     end
                     ):
    """
    Get the matrix of values from the corresponding genomic position

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
        input_matrix[n, :] = get_one_hot_encoded(sequence_stream.sequence(chromosome, start, end), bp)

    signal_array = np.nan_to_num(np.array(signal_stream.values(chromosome, start, end)))

    input_matrix[4, :] = signal_array

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
        return_all_regions_inputs_batch, return_all_regions_targets_batch, inputs_batch, targets_batch = [], [], [], []
        
        all_regions = [roi_pool.regions1, roi_pool.regions2, roi_pool.regions3, roi_pool.regions4]
        k=0
        for reg in all_regions:
            roi_size = reg.shape[0]
            tot_n_roi=n_roi[k]

            k=k+1
            #roi_size = roi_pool.shape[0] #this needs to be the size of each region
    
            curr_batch_idxs = random.sample(range(roi_size), tot_n_roi)
    
            # Here I will process by row, if performance is bad then process by cell line
            for row_idx in curr_batch_idxs:
                roi_row = reg.iloc[row_idx, :]
                
                
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
    
                    input_matrix = get_input_matrix(rows=INPUT_CHANNELS,
                                                    cols=INPUT_LENGTH,
                                                    bp_order=BP_ORDER,
                                                    signal_stream=signal_stream,
                                                    sequence_stream=sequence_stream,
                                                    chromosome=chrom_name,
                                                    start=start,
                                                    end=end
                                                    )
    
                    inputs_batch.append(input_matrix)
    
                    # TODO we might want to test what happens if we change the
                    if not quant:
                        target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
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
            
            
        yield np.array(inputs_batch), np.array(targets_batch)  

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
                 tag,
                 window_sequence,
                 ):
        """
        :param chroms: Chromosomes to limit the analysis to
        :param roi_file_path: User provided ROI file path
        :param meta_file: path to meta file
        :param prefix: Prefix for saving output file
        :param output_directory: Output directory to save files to
        :param shuffle: Whether to shuffle the input ROI file
        :param tag: Tag to use for writing the file.
        :param window_sequence: Windowed hg38 sequence w1024 sliding 256 bp
        :param chromosomes: List of chromosomes to use
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes
        :param blacklist: The blacklist file of BED regions to exclude
        :param region_length: Length of the input regions
        :param window_sequence: Windowed hg38 sequence w1024 sliding 256 bp
        """
        self.chroms = chroms
        self.roi_file_path = roi_file_path
        self.meta_file = meta_file
        self.prefix = prefix
        self.output_directory = output_directory
        self.tag = tag
        self.window_sequence = window_sequence
     

        # If an ROI path is provided import it as the ROI pool
        if self.roi_file_path:
            self.ROI_pool = self.__import_roi_pool__(filepath=self.roi_file_path,
                                                     chroms=self.chroms,
                                                     shuffle=shuffle
                                                     )
        # Import the data from the meta file.
        else:
            
            
            self.ROI_pool = GenomicRegions(meta_path=self.meta_file,
                                     region_length=1024,
                                     chromosomes=self.chroms,
                                     chromosome_sizes_dictionary=build_chrom_sizes_dict(self.chroms,
                                                                                        DEFAULT_CHROM_SIZES),
                                     blacklist=BLACKLISTED_REGIONS,
                                     window_sequence=self.window_sequence)

            

            ROIPool.regions1 = self.ROI_pool.regions1
            ROIPool.regions2 = self.ROI_pool.regions2
            ROIPool.regions3 = self.ROI_pool.regions3
            ROIPool.regions4 = self.ROI_pool.regions4
            
            ROIPool.regions1_counts = self.ROI_pool.regions1_counts
            ROIPool.regions2_counts = self.ROI_pool.regions2_counts
            ROIPool.regions3_counts = self.ROI_pool.regions3_counts
            ROIPool.regions4_counts = self.ROI_pool.regions4_counts
            
            ROIPool.write_data(self.prefix, output_dir=self.output_directory, set_tag=tag)


    def write_data(self, prefix="BalancedROI_pool", output_dir="./ROI", set_tag="training"):
        """
        Write the ROI dataframe to a tsv and a bed for for ATAC, CHIP, and combined ROIs

        :param set_tag: Tag for training or validation
        :param prefix: Prefix for filenames to use
        :param output_dir: Directory to output the bed and tsv files

        :return: Write BED and TSV versions of the ROI data
        """
        output_directory = get_dir(output_dir)
        
        #Create names for Region files
        region1_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region1_ROI.bed.gz")
        region2_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region2_ROI.bed.gz")
        region3_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region3_ROI.bed.gz")
        region4_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region4_ROI.bed.gz")
        
        #Write the Region files
        ROIPool.regions1.to_csv(region1_BED_filename, sep="\t", index=False, header=False)
        ROIPool.regions2.to_csv(region2_BED_filename, sep="\t", index=False, header=False)
        ROIPool.regions3.to_csv(region3_BED_filename, sep="\t", index=False, header=False)
        ROIPool.regions4.to_csv(region4_BED_filename, sep="\t", index=False, header=False)
        
        #Create names for Stats of Regions Files
        region1_stats_TSV_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region1_stats.tsv.gz")
        region2_stats_TSV_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region2_stats.tsv.gz")
        region3_stats_TSV_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region3_stats.tsv.gz")
        region4_stats_TSV_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_Region4_stats.tsv.gz")
        
        #Write the Stats of the Region Files
        ROIPool.regions1_counts.to_csv(region1_stats_TSV_filename, sep="\t", index=False, header=False)
        ROIPool.regions2_counts.to_csv(region2_stats_TSV_filename, sep="\t", index=False, header=False)
        ROIPool.regions3_counts.to_csv(region3_stats_TSV_filename, sep="\t", index=False, header=False)
        ROIPool.regions4_counts.to_csv(region4_stats_TSV_filename, sep="\t", index=False, header=False)
        
class TrainingDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 signal,
                 sequence,
                 input_channels,
                 input_length,
                 predict_roi_df,
                 batch_size=32
                 ):
        'Initialization'
        self.batch_size = batch_size
        self.predict_roi_df = predict_roi_df
        self.indexes = np.arange(self.predict_roi_df.shape[0])
        self.signal = signal
        self.sequence = sequence
        self.input_channels = input_channels
        self.input_length = input_length

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.predict_roi_df.shape[0] / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self.__data_generation(batch_indexes)

        return X

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        # Generate data
        # Store sample
        batch_roi_df = self.predict_roi_df.loc[batch_indexes, :]

        batch_roi_df.reset_index(drop=True, inplace=True)

       # batch = get_region_values(signal=self.signal,
        #                          sequence=self.sequence,
        #                          input_channels=self.input_channels,
        #                          input_length=self.input_length,
        #                          roi_pool=batch_roi_df
        #                          )

        return batch_roi_df

